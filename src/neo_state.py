"""Neo's durable state — the loop guard, thrash guard, and audit trail.

Lives in the shared VPS Postgres (`reviews_state` db) in Neo-owned tables so it
never collides with rhobear-reviews. Keyed on (repo, pr_number, head_sha): Neo
acts at most once per problem-class per SHA, counts builder rounds, and escalates
instead of thrashing. Also the source of truth for the per-install auto-merge
button and the credit meter.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import psycopg  # psycopg3

SCHEMA = """
CREATE TABLE IF NOT EXISTS neo_actions (
    id          BIGSERIAL PRIMARY KEY,
    repo        TEXT NOT NULL,
    pr_number   INTEGER NOT NULL,
    head_sha    TEXT NOT NULL,
    phase       TEXT NOT NULL,              -- triage|fix_forward|builder|merged|bounced|escalated
    builder_round INTEGER NOT NULL DEFAULT 0,
    verdict     TEXT,                       -- ACCEPT|FIX-FORWARD|BOUNCE|ESCALATE
    detail      JSONB NOT NULL DEFAULT '{}',
    credits     BIGINT NOT NULL DEFAULT 0,  -- credits debited for this action
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS neo_actions_pr ON neo_actions(repo, pr_number, head_sha);

CREATE TABLE IF NOT EXISTS neo_installs (
    install_id  TEXT PRIMARY KEY,           -- github app installation id (or org)
    org         TEXT NOT NULL,
    auto_merge  BOOLEAN NOT NULL DEFAULT false,
    plan        TEXT,                       -- starter|pro|business
    credits_balance BIGINT NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _started_at(detail: dict, created_at) -> float:
    """Epoch seconds when the escalation first started (detail's
    escalation_started, else the row's created_at)."""
    try:
        started = float(detail.get("escalation_started"))
        if started > 0:
            return started
    except (TypeError, ValueError):
        pass
    return created_at.timestamp()


# r4 (HIGH C, authorized Zeus interop change — Morpheus ruling 2026-11-02):
# take_escalation() no longer DELETES the consumed advice's rows. They are
# re-phased to this marker phase with a consumed_at timestamp, so there is
# never a window with no marker between the consume and the recorded outcome
# (the gap Kronos's judge could previously fall into). The marker is removed
# only when the outcome records (escalate_pr's escalated row / the park row),
# and is reclaimable after ESCALATION_INFLIGHT_STALE_SECS so a crashed Zeus
# cannot block a PR forever. Kronos defers whenever the marker exists.
ESCALATION_INFLIGHT_PHASE = "escalation-inflight"
ESCALATION_INFLIGHT_STALE_SECS = 3600


_TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "on", "success", "green"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "n", "off", "failure", "failed", "red"})


def verdict_color(raw: object) -> bool | None:
    """Normalize a wake payload's verdict color to True / False / None.

    None means "unknown" — the key was absent, null, or unparseable — and is
    NOT collapsed into red: `bool(raw)` is wrong twice over, turning a missing
    key into a red verdict and the string "false" into True (a non-empty string
    is truthy), silently inverting red verdicts
    (fix/verdict-flip-redelivery / fix-pr-5)."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):                 # JSON 0/1
        return bool(raw) if raw in (0, 1) else None
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in _TRUE_STRINGS:
            return True
        if v in _FALSE_STRINGS:
            return False
    return None


@dataclass
class NeoState:
    """Thread-safe Postgres state store. Every method opens a fresh
    psycopg.connect(dsn, autocommit=True) — no shared connection, cursor, or
    mutable state. Safe to call from concurrent ThreadPoolExecutor threads."""
    dsn: str

    def _conn(self):
        return psycopg.connect(self.dsn, autocommit=True)

    def ensure_schema(self) -> None:
        with self._conn() as c:
            c.execute(SCHEMA)
        # ONE schema definition for neo_jobs lives in jobs.py (including the
        # in-place migration of the pre-B1 table).  NeoState delegates so the
        # two modules can never drift.
        from .jobs import Jobs
        Jobs(self.dsn).ensure_schema()

    # --- loop / thrash guard ------------------------------------------------
    def builder_rounds(self, repo: str, pr: int) -> int:
        """How many builder rounds Neo has already spent on this PR (any SHA)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(builder_round),0) FROM neo_actions "
                "WHERE repo=%s AND pr_number=%s AND phase='builder'",
                (repo, pr),
            ).fetchone()
        return int(row[0] or 0)

    def already_acted(self, repo: str, pr: int, head_sha: str, phase: str,
                      green: bool | None = None) -> bool:
        """Dedup guard. When `green` is given, only a prior action recorded for
        the SAME verdict color counts as already-acted: a green verdict arriving
        after a red triage of the same head (the reviewer re-ran and passed it)
        is new decision input and must be triaged again. Rows with no recorded
        color predate the field and count as red, so a first green wake still
        re-triggers. `green=None` keeps the old any-phase behavior."""
        with self._conn() as c:
            if green is None:
                row = c.execute(
                    "SELECT 1 FROM neo_actions WHERE repo=%s AND pr_number=%s "
                    "AND head_sha=%s AND phase=%s LIMIT 1",
                    (repo, pr, head_sha, phase),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT 1 FROM neo_actions WHERE repo=%s AND pr_number=%s "
                    "AND head_sha=%s AND phase=%s "
                    "AND COALESCE((detail->>'green')::boolean, false) = %s LIMIT 1",
                    (repo, pr, head_sha, phase, bool(green)),
                ).fetchone()
        return row is not None

    def record(self, repo: str, pr: int, head_sha: str, phase: str, *,
               verdict: str = "", builder_round: int = 0, credits: int = 0,
               detail: dict | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO neo_actions(repo,pr_number,head_sha,phase,builder_round,"
                "verdict,credits,detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (repo, pr, head_sha, phase, builder_round, verdict, credits,
                 json.dumps(detail or {})),
            )

    def clear(self, repo: str, pr: int, head_sha: str, phase: str) -> int:
        """Remove a phase marker so the next wake for this head re-runs it."""
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM neo_actions WHERE repo=%s AND pr_number=%s "
                "AND head_sha=%s AND phase=%s",
                (repo, pr, head_sha, phase),
            )
            return cur.rowcount

    # --- escalation (async advisor) --------------------------------------
    def escalating_prs(self) -> list[dict]:
        """PRs currently awaiting an advisor (phase='escalating'), newest row
        per PR.  Each carries advice_id/installation_id from the detail and
        the row's created_at (used for the give-up timeout)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT ON (repo, pr_number) "
                "repo, pr_number, head_sha, detail, created_at "
                "FROM neo_actions WHERE phase='escalating' "
                "ORDER BY repo, pr_number, id DESC"
            ).fetchall()
        out = []
        for r in rows:
            detail = r[3]
            if isinstance(detail, str):
                try:
                    detail = json.loads(detail)
                except (TypeError, ValueError):
                    detail = {}
            elif detail is None:
                detail = {}
            out.append({
                "repo": r[0], "pr": r[1], "head_sha": r[2],
                "advice_id": str(detail.get("advice_id", "") or ""),
                "installation_id": detail.get("installation_id"),
                "created_at": r[4],
                # When this escalation first started: a re-recorded marker (retry
                # after a failed terminal action) keeps the original start, so
                # the give-up timeout is never pushed out by retries.
                "started_at": _started_at(detail, r[4]),
                "detail": detail,
            })
        return out

    def take_escalation(self, repo: str, pr: int, advice_id: str) -> bool:
        """Atomically consume the escalating marker for `advice_id` IF it is
        still the NEWEST escalation for the PR.

        One transaction under the per-PR advisory lock
        (pg_advisory_xact_lock(hashtext(repo||':'||pr))): re-checks that no
        newer escalating row appeared since the recovery loop's snapshot, then
        consumes exactly this advice's rows (never a newer escalation's
        marker). The lock serializes this check-then-consume-all against a
        concurrent submit_escalation / phase-row INSERT, so a freshly-inserted
        newer escalation is never clobbered (defect 1).

        r4 (HIGH C, authorized interop change): a successful consume does NOT
        delete the rows — it re-phases them to ESCALATION_INFLIGHT_PHASE with
        a consumed_at timestamp, so no gap without a marker exists between
        the consume and escalate_pr recording its outcome. The marker is
        removed only when the outcome records (clear_inflight_markers, called
        by the escalation flow after act()) and is reclaimable after
        ESCALATION_INFLIGHT_STALE_SECS (reclaim_stale_inflight) so a crashed
        Zeus cannot block a PR forever.

        Returns True when the advice was current and is consumed (caller acts
        on it), False when it was superseded (caller drops it, logged, without
        creating a round). In the superseded case this advice's own rows are
        removed — a newer escalation's marker exists, so there is no gap.
        """
        with psycopg.connect(self.dsn, autocommit=False) as c:
            try:
                c.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"{repo}:{pr}",),
                )
                newest = c.execute(
                    "SELECT COALESCE(detail->>'advice_id', '') FROM neo_actions "
                    "WHERE repo=%s AND pr_number=%s AND phase='escalating' "
                    "ORDER BY id DESC LIMIT 1",
                    (repo, pr),
                ).fetchone()
                newest_id = newest[0] if newest else None
                if newest is not None and newest_id == (advice_id or ""):
                    # Only the advice that won the take becomes in-flight.
                    # Older queued advice is superseded, not consumed: making
                    # it in-flight would let it resurrect after this outcome.
                    c.execute(
                        "DELETE FROM neo_actions WHERE repo=%s AND pr_number=%s "
                        "AND phase='escalating' "
                        "AND COALESCE(detail->>'advice_id', '')<>%s",
                        (repo, pr, advice_id or ""),
                    )
                    c.execute(
                        "UPDATE neo_actions SET phase=%s, "
                        "detail = detail || %s::jsonb "
                        "WHERE repo=%s AND pr_number=%s "
                        "AND phase='escalating' "
                        "AND COALESCE(detail->>'advice_id', '')=%s",
                        (ESCALATION_INFLIGHT_PHASE,
                         json.dumps({"consumed_at": time.time()}),
                         repo, pr, advice_id or ""),
                    )
                    c.commit()
                    return True
                # Superseded by a newer escalation: remove ONLY this advice's
                # rows — a newer escalation's marker is never touched.
                c.execute(
                    "DELETE FROM neo_actions WHERE repo=%s AND pr_number=%s "
                    "AND phase='escalating' AND COALESCE(detail->>'advice_id', '')=%s",
                    (repo, pr, advice_id or ""),
                )
                c.commit()
                return False
            except Exception:
                c.rollback()
                raise

    def clear_inflight_markers(
            self, repo: str, pr: int, advice_id: str | None = None) -> int:
        """Remove only the advice whose terminal outcome just recorded.

        The live flow supplies advice_id. For legacy singleton callers, an
        omitted id is accepted only with exactly one marker under the PR lock;
        multiple owners must never be removed without a supplied identity.
        """
        with psycopg.connect(self.dsn, autocommit=False) as c:
            try:
                c.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"{repo}:{pr}",),
                )
                if advice_id is None:
                    # Preserve the pre-r5 single-marker test/consumer contract,
                    # but fail closed on ambiguous multi-advice cleanup.
                    owners = c.execute(
                        "SELECT COALESCE(detail->>'advice_id', '') "
                        "FROM neo_actions WHERE repo=%s AND pr_number=%s "
                        "AND phase=%s LIMIT 2",
                        (repo, pr, ESCALATION_INFLIGHT_PHASE),
                    ).fetchall()
                    if len(owners) > 1:
                        raise ValueError("advice_id required for multiple in-flight owners")
                    if not owners:
                        c.rollback()
                        return 0
                    advice_id = owners[0][0]
                cur = c.execute(
                    "DELETE FROM neo_actions WHERE repo=%s AND pr_number=%s "
                    "AND phase=%s AND COALESCE(detail->>'advice_id', '')=%s",
                    (repo, pr, ESCALATION_INFLIGHT_PHASE, advice_id),
                )
                removed = cur.rowcount
                c.commit()
                return removed
            except Exception:
                c.rollback()
                raise

    def reclaim_stale_inflight(
            self, *,
            stale_secs: int = ESCALATION_INFLIGHT_STALE_SECS) -> list[dict]:
        """r4 (HIGH C): the in-flight marker is reclaimable — a crashed Zeus
        cannot block a PR forever. Every in-flight marker whose consumed_at
        is older than `stale_secs` is re-phased back to 'escalating' (detail
        restored, consumed_at stripped) so the advice re-enters the normal
        recovery pipeline (escalate_pr is idempotent per advice_id) — UNLESS
        a newer escalating marker already owns the PR, in which case the
        stale marker is simply dropped. Process newest first and compare
        durable ids under each PR lock: an older recovered marker cannot
        supersede a newer advice. Returns one row per reclaimed marker."""
        now = time.time()
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, repo, pr_number, detail FROM neo_actions "
                "WHERE phase=%s ORDER BY id DESC",
                (ESCALATION_INFLIGHT_PHASE,),
            ).fetchall()
        out: list[dict] = []
        for row_id, repo, pr, detail in rows:
            if isinstance(detail, str):
                try:
                    detail = json.loads(detail)
                except (TypeError, ValueError):
                    detail = {}
            detail = detail or {}
            try:
                consumed = float(detail.get("consumed_at") or 0)
            except (TypeError, ValueError):
                consumed = 0.0
            if consumed <= 0 or now - consumed < stale_secs:
                continue  # fresh marker: Zeus may still be mid-act
            with psycopg.connect(self.dsn, autocommit=False) as c:
                try:
                    c.execute(
                        "SELECT pg_advisory_xact_lock(hashtext(%s))",
                        (f"{repo}:{pr}",),
                    )
                    still = c.execute(
                        "SELECT detail FROM neo_actions "
                        "WHERE id=%s AND phase=%s",
                        (row_id, ESCALATION_INFLIGHT_PHASE),
                    ).fetchone()
                    if not still:
                        c.rollback()
                        continue
                    # ABA guard: the staleness decision was made on a
                    # snapshot taken before the lock. Re-read the LIVE
                    # consumed_at; a marker that was reclaimed and re-taken
                    # meanwhile (changed / fresh consumed_at) is untouched.
                    live = still[0]
                    if isinstance(live, str):
                        try:
                            live = json.loads(live)
                        except (TypeError, ValueError):
                            live = {}
                    try:
                        live_consumed = float((live or {}).get("consumed_at") or 0)
                    except (TypeError, ValueError):
                        live_consumed = 0.0
                    if (live_consumed != consumed
                            or live_consumed <= 0
                            or time.time() - live_consumed < stale_secs):
                        c.rollback()
                        continue
                    # MEDIUM 2: the outcome may have committed before the
                    # cleanup failed/crashed. A settled advice (its escalated
                    # row, or any later parked row) is cleared without
                    # re-acting, so no contradictory decision is appended.
                    settled = c.execute(
                        "SELECT 1 FROM neo_actions WHERE repo=%s "
                        "AND pr_number=%s AND id>%s AND ((phase='parked' AND head_sha=(SELECT head_sha FROM neo_actions WHERE id=%s)) "
                        "OR (phase='escalated' AND "
                        "COALESCE(detail->>%s, '')=%s)) LIMIT 1",
                        (repo, pr, row_id, row_id, "advice_id",
                         str(detail.get("advice_id") or "")),
                    ).fetchone()
                    if settled:
                        c.execute(
                            "DELETE FROM neo_actions WHERE id=%s", (row_id,))
                        c.commit()
                        out.append({"repo": repo, "pr": pr,
                                    "action": "cleared (outcome already "
                                              "recorded)"})
                        continue
                    # Supersession is ordered by durable row id, not merely
                    # the presence of any escalating row: older A must never
                    # supersede newer B. Include live in-flight owners too.
                    newer = c.execute(
                        "SELECT 1 FROM neo_actions WHERE repo=%s "
                        "AND pr_number=%s AND id>%s "
                        "AND (phase='escalating' OR phase=%s) LIMIT 1",
                        (repo, pr, row_id, ESCALATION_INFLIGHT_PHASE),
                    ).fetchone()
                    if newer:
                        c.execute(
                            "DELETE FROM neo_actions WHERE id=%s", (row_id,))
                        action = "dropped (a newer escalation owns the PR)"
                    else:
                        c.execute(
                            "UPDATE neo_actions SET phase='escalating', "
                            "detail = detail - 'consumed_at' WHERE id=%s",
                            (row_id,))
                        action = "re-recorded"
                    c.commit()
                    out.append({"repo": repo, "pr": pr, "action": action})
                except Exception:
                    c.rollback()
                    raise
        return out

    def record_escalating(self, repo: str, pr: int, head_sha: str, *,
                          verdict: str = "ESCALATE", builder_round: int = 0,
                          credits: int = 0, advice_id: str = "",
                          installation_id=None, extra: dict | None = None) -> None:
        """Record an escalating marker row under the per-PR advisory lock.

        The lock serializes the INSERT against take_escalation's
        check-then-delete-all, so a consume can never clobber a
        concurrently-inserted newer escalation's marker (defect 1).  `extra`
        merges into the detail (e.g. green/wake_job from the worker's phase
        row).
        """
        detail = {
            "advice_id": str(advice_id or ""),
            "installation_id": installation_id,
        }
        if extra:
            detail.update(extra)
        with psycopg.connect(self.dsn, autocommit=False) as c:
            c.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"{repo}:{pr}",),
            )
            c.execute(
                "INSERT INTO neo_actions(repo,pr_number,head_sha,phase,builder_round,"
                "verdict,credits,detail) VALUES (%s,%s,%s,'escalating',%s,%s,%s,%s)",
                (repo, pr, head_sha, builder_round, verdict, credits,
                 json.dumps(detail)),
            )
            c.commit()

    def advisor_escalation_for(self, repo: str, pr: int, advice_id: str):
        """The builder_round of the escalated row recorded for `advice_id`, or
        None.  Used to make escalate_pr idempotent per advice (defect 2): a
        retry after a transient failure must not mint a second escalated round.
        """
        if not advice_id:
            return None
        with self._conn() as c:
            row = c.execute(
                "SELECT builder_round FROM neo_actions WHERE repo=%s AND "
                "pr_number=%s AND phase='escalated' AND "
                "detail->>'advice_id'=%s ORDER BY id DESC LIMIT 1",
                (repo, pr, str(advice_id)),
            ).fetchone()
        if not row:
            return None
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return None

    def already_escalated(self, repo: str, pr: int) -> bool:
        """True once this PR has had its ONE genuine advisor-escalated round.

        Counts only rows written by `escalate_pr` (detail->'advisor_round'
        present).  The overloaded phase='escalated' rows written for the
        agent's canonical ESCALATE verdict or the old out-of-credits marker
        carry no advisor_round and must NOT lock the PR out of builder /
        escalation automation.  A further BOUNCE after a genuine escalation
        parks the PR (no second escalation).
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM neo_actions WHERE repo=%s AND pr_number=%s "
                "AND phase='escalated' AND detail->'advisor_round' IS NOT NULL "
                "LIMIT 1",
                (repo, pr),
            ).fetchone()
        return row is not None

    # --- per-install settings + credits ------------------------------------
    def install(self, install_id: str, org: str) -> dict:
        with self._conn() as c:
            c.execute(
                "INSERT INTO neo_installs(install_id,org) VALUES (%s,%s) "
                "ON CONFLICT (install_id) DO NOTHING",
                (install_id, org),
            )
            row = c.execute(
                "SELECT install_id,org,auto_merge,plan,credits_balance "
                "FROM neo_installs WHERE install_id=%s", (install_id,),
            ).fetchone()
        return dict(zip(("install_id", "org", "auto_merge", "plan", "credits_balance"), row))

    def debit(self, install_id: str, credits: int) -> int:
        """Debit credits; returns remaining balance (may go negative — enforcement
        is a separate gate so a mid-run debit never half-breaks a merge)."""
        with self._conn() as c:
            row = c.execute(
                "UPDATE neo_installs SET credits_balance = credits_balance - %s, "
                "updated_at = now() WHERE install_id=%s RETURNING credits_balance",
                (credits, install_id),
            ).fetchone()
        return int(row[0]) if row else 0
