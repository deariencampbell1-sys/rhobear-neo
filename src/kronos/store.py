"""Kronos's ONLY writer — the two state shapes Zeus already consumes.

r2 (HIGH 2 + HIGH 6): the r1 judge called db.record()/db.create_builder() on
a READ-ONLY rows adapter that has neither, so neither Summit decision could
ever be written. r2 gives the judge one writer with the same deconfliction
semantics Zeus's own escalation path uses (src/escalation.py — read, never
touched): ONE Postgres transaction per decision, under the per-PR advisory
lock `pg_advisory_xact_lock(hashtext(repo||':'||pr))`, which re-checks
BEFORE inserting:

  - no `escalating` marker (Zeus's advisor path owns the PR),
  - no `escalated` marker with an advisor_round (already escalated/stopped),
  - no `merged` row (nothing left to judge — HIGH 2, r3),
  - no `ready` row (Zeus accepted the PR — HIGH 2, r3),
  - Zeus was not recently active on the PR (r3 HIGH 4: the consume-then-act
    gap — Zeus's take_escalation commits the marker's deletion BEFORE
    escalate_pr records the outcome, so in that window neither marker
    exists; recent activity is the durable evidence that the PR is
    mid-flight),
  - MAX(builder_round) over builder rows < 6 (the round-six ceiling: 5
    normal rounds + at most 1 escalated round = 6, NEVER a 7th),
  - the new advisor_round itself <= 6,
  - an injected `authority` callable (r3 HIGH 2) re-proves the PR's
    GitHub-side state inside the lock before the write.

If any check fails the decision is DEFERRED — (False, one-line reason), no
row, no error — so a race with Zeus or a legacy/inconsistent history can
never produce a second escalation or a 7th round.

r3 (HIGH 1): the job write is a FRESH-ONLY
`INSERT ... ON CONFLICT DO NOTHING` — never the re-arming upsert
Jobs.create_builder uses. Kronos NEVER re-queues, re-arms or releases an
existing job (the read-only ruling): if any job row already exists for the
round — parked, expired-running, done, anything — the decision defers and
the conflict is alerted by the caller. `escalate` inserts the neo_actions
escalated row AND the round-N builder job in that same transaction; `stop`
inserts only the stop marker Zeus's already_escalated() consumes.
"""
from __future__ import annotations

import json

import psycopg

#: 5 normal Zeus fix rounds + at most 1 Kronos escalated round.
ROUND_CEILING = 6

#: Default quiet window (seconds): a PR Zeus touched more recently than this
#: may be inside the consume-then-act gap (HIGH 4) — defer.
DEFAULT_QUIET_SECS = 3600

#: The fresh-only job insert (HIGH 1, r3): ON CONFLICT DO NOTHING — a
#: pre-existing row of the same round is NEVER re-armed. The caller alerts
#: on the conflict.
_JOBS_INSERT_FRESH = (
    "INSERT INTO neo_jobs "
    "(kind, repo, pr_number, head_sha, wake_color, builder_round, "
    " state, payload, max_attempts) "
    "VALUES ('builder', %s, %s, %s, false, %s, 'queued', %s, %s) "
    "ON CONFLICT (repo, pr_number, builder_round) WHERE kind = 'builder' "
    "DO NOTHING "
    "RETURNING id"
)

JOB_EXISTS_REASON = (
    "deferred: builder job for this round already exists "
    "(fresh-only insert — never re-armed an existing job)")


def _defer_reason(cur, repo: str, pr: int, advisor_round: int, *,
                  quiet_secs: int = DEFAULT_QUIET_SECS) -> str:
    """The reason the CAS must defer, or '' when the write may proceed.
    Runs INSIDE the advisory-locked transaction (compare-and-set)."""
    row = cur.execute(
        "SELECT 1 FROM neo_actions WHERE phase='escalating' "
        "AND repo = %s AND pr_number = %s LIMIT 1", (repo, pr)).fetchone()
    if row:
        return "deferred: Zeus advisor escalation is in flight"
    # r4 (HIGH C): Zeus's take_escalation now leaves a durable in-flight
    # marker (authorized interop change) until its outcome records. The old
    # quiet-window recency check could NOT see a just-consumed marker over
    # stale history; this one can, and Kronos defers whenever it exists.
    row = cur.execute(
        "SELECT 1 FROM neo_actions WHERE phase='escalation-inflight' "
        "AND repo = %s AND pr_number = %s LIMIT 1", (repo, pr)).fetchone()
    if row:
        return ("deferred: Zeus escalation is mid-flight (the consumed "
                "advice's in-flight marker is present)")
    row = cur.execute(
        "SELECT 1 FROM neo_actions WHERE phase='escalated' "
        "AND detail->'advisor_round' IS NOT NULL "
        "AND repo = %s AND pr_number = %s LIMIT 1", (repo, pr)).fetchone()
    if row:
        return "deferred: already escalated or stopped"
    row = cur.execute(
        "SELECT 1 FROM neo_actions WHERE repo = %s AND pr_number = %s "
        "AND phase = 'merged' LIMIT 1", (repo, pr)).fetchone()
    if row:
        return "deferred: PR is merged (nothing left to judge)"
    row = cur.execute(
        "SELECT 1 FROM neo_actions WHERE repo = %s AND pr_number = %s "
        "AND phase = 'ready' LIMIT 1", (repo, pr)).fetchone()
    if row:
        return "deferred: PR is ready (Zeus accepted it)"
    # Round-six ceiling: a HARD safety check, ahead of the liveness defers.
    max_round = cur.execute(
        "SELECT COALESCE(MAX(builder_round), 0) FROM neo_actions "
        "WHERE repo = %s AND pr_number = %s AND phase = 'builder'",
        (repo, pr)).fetchone()[0]
    if int(max_round or 0) >= ROUND_CEILING:
        return (f"deferred: round-six ceiling — {int(max_round)} builder "
                f"rounds already recorded, never a 7th")
    if advisor_round > ROUND_CEILING:
        return (f"deferred: refusing to create round {advisor_round} "
                f"(ceiling {ROUND_CEILING})")
    # HIGH 4 (r3): Zeus deletes the 'escalating' marker (take_escalation)
    # BEFORE escalate_pr records the outcome. In that window neither marker
    # exists; recent activity on the PR is the durable evidence that a Zeus
    # escalation is mid-flight. The advisory lock does NOT serialize Zeus's
    # whole action, so Kronos defers on recency instead.
    row = cur.execute(
        "SELECT EXTRACT(EPOCH FROM (now() - MAX(created_at))) FROM neo_actions "
        "WHERE repo = %s AND pr_number = %s", (repo, pr)).fetchone()
    age = row[0] if row and row[0] is not None else None
    if age is not None and float(age) < quiet_secs:
        return (f"deferred: Zeus touched this PR {int(float(age))}s ago "
                f"(an escalation may be mid-flight; quiet window "
                f"{quiet_secs}s)")
    row = cur.execute(
        "SELECT EXTRACT(EPOCH FROM (now() - MAX(updated_at))) FROM neo_jobs "
        "WHERE kind='builder' AND repo = %s AND pr_number = %s",
        (repo, pr)).fetchone()
    age = row[0] if row and row[0] is not None else None
    if age is not None and float(age) < quiet_secs:
        return (f"deferred: this PR's Zeus job moved {int(float(age))}s ago "
                f"(an escalation may be mid-flight; quiet window "
                f"{quiet_secs}s)")
    return ""


class SqlWrites:
    """The judge's writes against the shared reviews DSN. Every decision is
    one transaction: lock -> re-check -> insert -> commit."""

    def __init__(self, dsn: str, *, quiet_secs: int = DEFAULT_QUIET_SECS,
                 connect_timeout: int = 15,
                 statement_timeout_ms: int = 15000):
        self.dsn = dsn
        self.quiet_secs = int(quiet_secs)
        self.connect_timeout = max(1, int(connect_timeout))
        self.statement_timeout_ms = max(1, int(statement_timeout_ms))

    def _connect(self):
        conn = psycopg.connect(self.dsn, autocommit=False,
                               connect_timeout=self.connect_timeout)
        # SET cannot take a bound parameter: inline the validated int.
        conn.execute(f"SET LOCAL statement_timeout = "
                     f"{int(self.statement_timeout_ms)}")
        return conn

    def escalate(self, repo: str, pr: int, head_sha: str, *,
                 advisor_round: int, detail: dict,
                 builder_payload: dict,
                 max_attempts: int = 3,
                 authority=None) -> tuple[bool, str]:
        """`yes` -> the ONE escalated Zeus round whose brief is the plan.

        `authority` (r3 HIGH 2) re-proves the PR's GitHub-side state INSIDE
        the lock: zero-arg callable -> (ok, reason). A refusal defers."""
        # The CAS re-check keys on detail->'advisor_round' — inject it so the
        # idempotency never depends on the caller remembering it.
        detail = dict(detail or {})
        detail.setdefault("advisor_round", advisor_round)
        with self._connect() as conn:
            try:
                cur = conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"{repo}:{pr}",))
                reason = _defer_reason(cur, repo, pr, advisor_round,
                                       quiet_secs=self.quiet_secs)
                if not reason and authority is not None:
                    ok, why = authority()
                    if not ok:
                        reason = f"deferred: {why}"
                if reason:
                    conn.rollback()
                    return False, reason
                cur.execute(
                    "INSERT INTO neo_actions(repo,pr_number,head_sha,phase,"
                    "builder_round,verdict,credits,detail) "
                    "VALUES (%s,%s,%s,'escalated',%s,'ESCALATE',0,%s)",
                    (repo, pr, head_sha, advisor_round,
                     json.dumps(detail or {})))
                cur.execute(_JOBS_INSERT_FRESH,
                            (repo, pr, head_sha, advisor_round,
                             json.dumps(builder_payload or {}),
                             max_attempts))
                job_row = cur.fetchone()
                if job_row is None:
                    # HIGH 1 (r3): a job row already exists for this round —
                    # never re-arm it. The whole decision rolls back.
                    conn.rollback()
                    return False, JOB_EXISTS_REASON
                conn.commit()
                return True, (f"escalated round {advisor_round} recorded "
                              f"(5 normal + 1 escalated = 6, never a 7th)")
            except Exception:
                conn.rollback()
                raise

    def stop(self, repo: str, pr: int, head_sha: str, *,
             advisor_round: int, detail: dict,
             authority=None) -> tuple[bool, str]:
        """`no` / bad / timeout -> the stop marker Zeus checks before any
        further round (already_escalated) — no builder job, ever."""
        # Same defense as escalate(): the CAS re-check keys on
        # detail->'advisor_round' — never depend on the caller for it.
        detail = dict(detail or {})
        detail.setdefault("advisor_round", advisor_round)
        with self._connect() as conn:
            try:
                cur = conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"{repo}:{pr}",))
                reason = _defer_reason(cur, repo, pr, advisor_round,
                                       quiet_secs=self.quiet_secs)
                if not reason and authority is not None:
                    ok, why = authority()
                    if not ok:
                        reason = f"deferred: {why}"
                if reason:
                    conn.rollback()
                    return False, reason
                cur.execute(
                    "INSERT INTO neo_actions(repo,pr_number,head_sha,phase,"
                    "builder_round,verdict,credits,detail) "
                    "VALUES (%s,%s,%s,'escalated',0,'PARKED',0,%s)",
                    (repo, pr, head_sha, json.dumps(detail or {})))
                conn.commit()
                return True, "stop marker recorded (Zeus is off this PR)"
            except Exception:
                conn.rollback()
                raise

    def mark_notified(self, repo: str, pr: int) -> int:
        """r4 HIGH D — the durable delivered-flag on this PR's committed
        Kronos decision rows (stop marker and/or escalated marker). The
        notification reconstruction (src/kronos/judge.py) skips rows that
        carry it, so a committed decision is re-notified at most once more
        (at-least-once alerting). Kronos's OWN rows only — the WHERE clause
        pins detail.kronos."""
        with self._connect() as conn:
            try:
                cur = conn.execute(
                    "UPDATE neo_actions SET detail = jsonb_set("
                    "detail, '{kronos_notified}', 'true'::jsonb) "
                    "WHERE repo=%s AND pr_number=%s AND phase='escalated' "
                    "AND COALESCE(detail->'kronos_notified',"
                    "'false'::jsonb) <> 'true'::jsonb "
                    "AND (detail->'kronos_stop' = 'true'::jsonb "
                    "     OR (detail->'kronos' = 'true'::jsonb "
                    "         AND position('kronos:' in "
                    "                      detail->>'advice_id') = 1))",
                    (repo, pr))
                marked = cur.rowcount
                conn.commit()
                return marked
            except Exception:
                conn.rollback()
                raise
