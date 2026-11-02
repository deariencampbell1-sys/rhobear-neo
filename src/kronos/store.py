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
  - MAX(builder_round) over builder rows < 6 (the round-six ceiling: 5
    normal rounds + at most 1 escalated round = 6, NEVER a 7th),
  - the new advisor_round itself <= 6.

If any check fails the decision is DEFERRED — (False, one-line reason), no
row, no error — so a race with Zeus or a legacy/inconsistent history can
never produce a second escalation or a 7th round. `escalate` inserts the
neo_actions escalated row AND the round-N builder job in that same
transaction (mirroring Jobs.create_builder's dedupe SQL); `stop` inserts
only the stop marker Zeus's already_escalated() consumes.
"""
from __future__ import annotations

import json

import psycopg

#: 5 normal Zeus fix rounds + at most 1 Kronos escalated round.
ROUND_CEILING = 6

# Mirrors Jobs.create_builder's exact upsert (src/jobs.py): dedupe on
# (repo, pr_number, builder_round) WHERE kind='builder'; an ACTIVE job
# suppresses, a terminal/expired one re-arms. Inlined so marker + job are
# one transaction — Jobs' own methods open their own connection, which
# would break the atomicity this module exists for.
_JOBS_UPSERT = (
    "INSERT INTO neo_jobs "
    "(kind, repo, pr_number, head_sha, wake_color, builder_round, "
    " state, payload, max_attempts) "
    "VALUES ('builder', %s, %s, %s, false, %s, 'queued', %s, %s) "
    "ON CONFLICT (repo, pr_number, builder_round) WHERE kind = 'builder' "
    "DO UPDATE SET "
    "  state = 'queued', outcome = NULL, lease_expires = NULL, "
    "  attempt = 0, payload = EXCLUDED.payload, updated_at = now() "
    "WHERE neo_jobs.state IN ('done', 'parked', 'superseded') "
    "   OR (neo_jobs.state = 'running' AND neo_jobs.lease_expires < now()) "
    "RETURNING id"
)


def _defer_reason(cur, repo: str, pr: int, advisor_round: int) -> str:
    """The reason the CAS must defer, or '' when the write may proceed.
    Runs INSIDE the advisory-locked transaction (compare-and-set)."""
    row = cur.execute(
        "SELECT 1 FROM neo_actions WHERE phase='escalating' "
        "AND repo = %s AND pr_number = %s LIMIT 1", (repo, pr)).fetchone()
    if row:
        return "deferred: Zeus advisor escalation is in flight"
    row = cur.execute(
        "SELECT 1 FROM neo_actions WHERE phase='escalated' "
        "AND detail->'advisor_round' IS NOT NULL "
        "AND repo = %s AND pr_number = %s LIMIT 1", (repo, pr)).fetchone()
    if row:
        return "deferred: already escalated or stopped"
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
    return ""


class SqlWrites:
    """The judge's writes against the shared reviews DSN. Every decision is
    one transaction: lock -> re-check -> insert -> commit."""

    def __init__(self, dsn: str):
        self.dsn = dsn

    def escalate(self, repo: str, pr: int, head_sha: str, *,
                 advisor_round: int, detail: dict,
                 builder_payload: dict,
                 max_attempts: int = 3) -> tuple[bool, str]:
        """`yes` -> the ONE escalated Zeus round whose brief is the plan."""
        with psycopg.connect(self.dsn, autocommit=False) as conn:
            try:
                cur = conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"{repo}:{pr}",))
                reason = _defer_reason(cur, repo, pr, advisor_round)
                if reason:
                    conn.rollback()
                    return False, reason
                cur.execute(
                    "INSERT INTO neo_actions(repo,pr_number,head_sha,phase,"
                    "builder_round,verdict,credits,detail) "
                    "VALUES (%s,%s,%s,'escalated',%s,'ESCALATE',0,%s)",
                    (repo, pr, head_sha, advisor_round,
                     json.dumps(detail or {})))
                cur.execute(_JOBS_UPSERT,
                            (repo, pr, head_sha, advisor_round,
                             json.dumps(builder_payload or {}),
                             max_attempts))
                conn.commit()
                return True, (f"escalated round {advisor_round} recorded "
                              f"(5 normal + 1 escalated = 6, never a 7th)")
            except Exception:
                conn.rollback()
                raise

    def stop(self, repo: str, pr: int, head_sha: str, *,
             advisor_round: int, detail: dict) -> tuple[bool, str]:
        """`no` / bad / timeout -> the stop marker Zeus checks before any
        further round (already_escalated) — no builder job, ever."""
        with psycopg.connect(self.dsn, autocommit=False) as conn:
            try:
                cur = conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"{repo}:{pr}",))
                reason = _defer_reason(cur, repo, pr, advisor_round)
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
