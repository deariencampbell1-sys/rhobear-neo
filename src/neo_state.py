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

-- Verdict color: NULL = unknown (pre-color / legacy rows). Kept as a real column
-- so it can be indexed and uniquified; mirrored into detail->'green' for readers.
ALTER TABLE neo_actions ADD COLUMN IF NOT EXISTS color BOOLEAN;
-- Set only by NeoState.claim(), i.e. only on the atomic dedup-claim row. Audit
-- rows appended afterwards (the post-run verdict) leave it NULL.
ALTER TABLE neo_actions ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ;

-- At most one claim per (pr, head, phase, color) — the DB-level half of the
-- thrash guard, so concurrent redeliveries cannot both pass the check-then-act
-- race and double-run triage. Partial (claimed_at IS NOT NULL) so the ordinary
-- audit trail can still hold many rows per tuple, and expression-based
-- (COALESCE + color IS NULL) so an unknown-color claim still collides with
-- itself instead of being treated as always-distinct NULLs in any PG version.
CREATE UNIQUE INDEX IF NOT EXISTS neo_actions_claim_uniq ON neo_actions(
    repo, pr_number, head_sha, phase, COALESCE(color, false), (color IS NULL)
) WHERE claimed_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS neo_installs (
    install_id  TEXT PRIMARY KEY,           -- github app installation id (or org)
    org         TEXT NOT NULL,
    auto_merge  BOOLEAN NOT NULL DEFAULT false,
    plan        TEXT,                       -- starter|pro|business
    credits_balance BIGINT NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


# Verdict-color vocabulary. The wake payload's `green` field arrives from several
# producers (a commit-status `state`, a check-run `conclusion`, the local watcher),
# so it may be a bool, a JSON 0/1, or a string. These sets are the ONLY values we
# are willing to interpret; anything else is "unknown color" (None), never red.
_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "off"})

# The dedup predicate for a color, expressed WITHOUT a ::boolean cast. Postgres's
# text->boolean cast raises on values like '' or 'maybe', and one malformed legacy
# row would then poison every dedup lookup for that head with a SQL error instead
# of a clean skip. This CASE never raises: only a recognized true-ish value counts
# as green, and everything else (NULL, absent key, junk, legacy rows) is red.
_COLOR_SQL = (
    "CASE lower(detail->>'green') "
    "WHEN 'true' THEN true WHEN 't' THEN true WHEN '1' THEN true "
    "WHEN 'yes' THEN true WHEN 'y' THEN true WHEN 'on' THEN true "
    "ELSE false END"
)


def verdict_color(raw: object) -> bool | None:
    """Normalize a wake payload's verdict color to True / False / None.

    None means "unknown" — the key was absent, null, or an unparseable value —
    and must NOT be collapsed into red. `bool(raw)` is wrong twice over: it turns
    a missing key into a red verdict, and the string "false" into True (a
    non-empty string is truthy), silently inverting red verdicts.
    """
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
    dsn: str

    def _conn(self):
        return psycopg.connect(self.dsn, autocommit=True)

    def ensure_schema(self) -> None:
        with self._conn() as c:
            c.execute(SCHEMA)

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
        """Dedup guard. When `green` is given, only a prior action recorded for the
        SAME verdict color counts as already-acted: a green verdict arriving after a
        red one (or vice versa) is new information and must be triaged again.

        `green=None` means the verdict color is UNKNOWN (absent/unparseable wake
        field) and matches any prior action for the phase — the pre-color,
        sha-only dedup. It deliberately does not mean "red": a caller that cannot
        tell the color must not be able to claim a red action was already taken.
        """
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
                    f"AND head_sha=%s AND phase=%s AND {_COLOR_SQL} = %s LIMIT 1",
                    (repo, pr, head_sha, phase, green),
                ).fetchone()
        return row is not None

    def claim(self, repo: str, pr: int, head_sha: str, phase: str,
              green: bool | None = None, **fields) -> bool:
        """Atomic dedup+record: INSERT the action and report whether we claimed it.

        Closes the check-then-act window in `already_acted` + `record`: GitHub
        redelivers storm-style after an outage, so two worker threads can both
        pass `already_acted` before either records and double-run triage (with
        two credit debits). A partial unique index makes the second insert a
        no-op; the caller must NOT act when this returns False.
        """
        with self._conn() as c:
            row = c.execute(
                "INSERT INTO neo_actions(repo,pr_number,head_sha,phase,builder_round,"
                "verdict,credits,detail,color,claimed_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,now()) "
                "ON CONFLICT DO NOTHING RETURNING id",
                (repo, pr, head_sha, phase, fields.get("builder_round", 0),
                 fields.get("verdict", ""), fields.get("credits", 0),
                 json.dumps(fields.get("detail") or {}), green),
            ).fetchone()
        return row is not None

    def record(self, repo: str, pr: int, head_sha: str, phase: str, *,
               verdict: str = "", builder_round: int = 0, credits: int = 0,
               green: bool | None = None, detail: dict | None = None) -> None:
        """Append an action row.

        `green` is the verdict COLOR this action was taken on, and it is written
        both as a real boolean column (for uniqueness + indexing) and merged into
        `detail["green"]` (the legacy read path). Storing it is what makes the
        color-keyed dedup in `already_acted` able to match at all: without this
        write every green triage looked like an unrecorded red one and re-ran on
        every redelivery.
        """
        payload = dict(detail or {})
        if green is not None:
            payload["green"] = green
        if "green" not in payload:               # keep the detail key present,
            payload["green"] = None              # explicit null == unknown color
        with self._conn() as c:
            c.execute(
                "INSERT INTO neo_actions(repo,pr_number,head_sha,phase,builder_round,"
                "verdict,credits,detail,color) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (repo, pr, head_sha, phase, builder_round, verdict, credits,
                 json.dumps(payload), green),
            )

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
