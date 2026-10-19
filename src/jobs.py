"""Durable job store for Neo — wake jobs and leased Builder rounds.

Job keys
--------
* wakes:      (kind='wake', repo, head_sha, wake_color)
  The PR number is resolved later, so it may be NULL at create time (a
  `status` event carries no PR).  Including `wake_color` means a green
  verdict arriving after a red triage of the *same* head is NEW decision
  input and gets its own job — never swallowed by the red one.
* builder:    (kind='builder', repo, pr_number, builder_round)
  A newer round for the same PR is a distinct row; the oldest are superseded
  by the newest (round cap enforced at the claim).

Dedupe is on OUTCOME, not on a row merely existing: `create_*` returns the
id of an already-active job (queued/running with a live lease) or None for a
duplicate, but re-arms a terminal job (done/parked/superseded, or running
with an expired lease) so a verdict that was already processed is never
silently dropped on the same head.

Every claimed job ends in one of: done | no_change | failed(retryable) |
parked | superseded.  `retry()` sends a retryable failure back to `queued`
(bounded by `max_attempts`, then `parked` — visible and owner-actionable).

Migrations
----------
`ensure_schema()` creates the table if absent and, on an already-deployed
table that still has the old `UNIQUE (repo, pr_number, head_sha)` constraint
or a NOT NULL `pr_number`, upgrades it in place (CREATE TABLE IF NOT EXISTS
alone would never change a constraint).  There is ONE schema definition,
here; `neo_state` no longer defines `neo_jobs`.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import psycopg  # psycopg3

log = logging.getLogger("rhobear_neo.jobs")

# Terminal outcomes for resolve().  'failed' is handled by retry() (retryable,
# goes back to 'queued'); a retryable failure that exhausts max_attempts parks.
# 'no_installation' is terminal: a repo the customer has not installed the
# GitHub App on is refused, never retried.
TERMINAL_OUTCOMES = ("done", "no_change", "no_installation", "superseded", "parked")


def _new_token() -> str:
    """Random claimant identity (kept in the DB only; never logged)."""
    return secrets.token_hex(16)


def _token_where(worker_token: str | None) -> tuple[str, list[Any]]:
    """SQL fragment + params for the claimant gate.  When no token is given
    (backward-compatible one-shot callers / tests that never mutate later)
    the gate is empty — every resolve/retry/lease still requires the row to be
    'running', so an unclaimed row can never be mutated."""
    if worker_token is None:
        return "", []
    return " AND worker_token = %s", [worker_token]

SCHEMA = """
CREATE TABLE IF NOT EXISTS neo_jobs (
    id             BIGSERIAL PRIMARY KEY,
    kind           TEXT NOT NULL,                -- wake | builder
    repo           TEXT NOT NULL,
    pr_number      INTEGER,                      -- NULL until resolved (wakes)
    head_sha       TEXT NOT NULL,
    wake_color     BOOLEAN NOT NULL DEFAULT false,   -- wake verdict color (key component)
    builder_round  INTEGER NOT NULL DEFAULT 0,       -- builder round number (key component)
    state          TEXT NOT NULL DEFAULT 'queued',   -- queued | running | done
    outcome        TEXT,                             -- done|no_change|failed|superseded|parked
    attempt        INTEGER NOT NULL DEFAULT 0,
    max_attempts   INTEGER NOT NULL DEFAULT 3,
    lease_expires  TIMESTAMPTZ,
    payload        JSONB NOT NULL DEFAULT '{}',
    worker_token   TEXT,                          -- claimant identity (resolve/retry/lease gate)
    submitted_at   TIMESTAMPTZ,                   -- last hand-off to the worker executor (wakes)
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# Wake uniqueness: (repo, head_sha, wake_color). PR is resolved later and is
# deliberately NOT in the key.  Builder uniqueness: (repo, pr_number,
# builder_round) — two rounds of the same PR that share a head SHA are distinct
# rows.  Indexes are created AFTER the migration so an already-deployed table
# gets its new columns first.
INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS neo_jobs_wake_key
    ON neo_jobs (repo, head_sha, wake_color) WHERE kind = 'wake';

CREATE UNIQUE INDEX IF NOT EXISTS neo_jobs_builder_key
    ON neo_jobs (repo, pr_number, builder_round) WHERE kind = 'builder';

CREATE INDEX IF NOT EXISTS neo_jobs_queued ON neo_jobs (state, lease_expires)
    WHERE state = 'queued';
CREATE INDEX IF NOT EXISTS neo_jobs_expired ON neo_jobs (lease_expires)
    WHERE state = 'running';
"""

# Applied to an already-deployed table that predates the key redesign.
MIGRATION = """
-- pr_number becomes nullable: a `status` wake has no PR until resolve_pr.
ALTER TABLE neo_jobs ALTER COLUMN pr_number DROP NOT NULL;
-- New key columns.
ALTER TABLE neo_jobs ADD COLUMN IF NOT EXISTS wake_color BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE neo_jobs ADD COLUMN IF NOT EXISTS builder_round INTEGER NOT NULL DEFAULT 0;
-- Lower-severity round-3 fixes: claimant identity (stale claimant can never
-- mutate a replacement run) and the executor submitted stamp (a wake waiting
-- in the executor is never re-submitted every recovery pass).
ALTER TABLE neo_jobs ADD COLUMN IF NOT EXISTS worker_token TEXT;
ALTER TABLE neo_jobs ADD COLUMN IF NOT EXISTS submitted_at TIMESTAMPTZ;

-- Drop the legacy UNIQUE (repo, pr_number, head_sha) constraint that (a) had no
-- `kind`, so a builder job for the same head as its wake job collided and was
-- dropped by ON CONFLICT DO NOTHING, and (b) had no color, so a green verdict
-- after a red triage of the same head was swallowed.  PostgreSQL names an
-- inline UNIQUE (...) constraint <table>_<col>_<col>_key; find it by columns so
-- a renamed copy is still caught.
DO $$
DECLARE
    cname text;
BEGIN
    SELECT conname INTO cname
      FROM pg_constraint c
      JOIN pg_class t ON t.oid = c.conrelid
     WHERE t.relname = 'neo_jobs' AND c.contype = 'u'
       AND array_length(c.conkey, 1) = 3
       AND NOT EXISTS (
           SELECT 1 FROM unnest(c.conkey) AS u(attno)
           LEFT JOIN pg_attribute a
                  ON a.attrelid = c.conrelid AND a.attnum = u.attno
            WHERE a.attname NOT IN ('repo', 'pr_number', 'head_sha'));
    IF cname IS NOT NULL THEN
        EXECUTE format('ALTER TABLE neo_jobs DROP CONSTRAINT %I', cname);
    END IF;
END $$;

-- Existing wake rows predate wake_color and all default to false; two rows for
-- the same (repo, head_sha) would collide on the new unique index.  A wake is
-- re-creatable from a webhook, so keep the newest of each key.
DELETE FROM neo_jobs a USING neo_jobs b
 WHERE a.kind = 'wake' AND b.kind = 'wake'
   AND a.repo = b.repo AND a.head_sha = b.head_sha
   AND a.wake_color = b.wake_color AND a.id < b.id;
-- Existing builder rows get round 1 (no round recorded before); keep newest.
UPDATE neo_jobs SET builder_round = 1 WHERE kind = 'builder' AND builder_round = 0;
DELETE FROM neo_jobs a USING neo_jobs b
 WHERE a.kind = 'builder' AND b.kind = 'builder'
   AND a.repo = b.repo AND a.pr_number = b.pr_number
   AND a.builder_round = b.builder_round AND a.id < b.id;
"""


@dataclass
class Jobs:
    """Thread-safe Postgres job store.

    Every method opens a fresh psycopg.connect(dsn, autocommit=True) — no
    shared connection, cursor, or mutable state. Safe to call from concurrent
    ThreadPoolExecutor workers and recovery threads.
    """
    dsn: str

    def _conn(self):
        return psycopg.connect(self.dsn, autocommit=True)

    # ------------------------------------------------------------------
    # Schema / migration
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the table if absent, then migrate an existing table in place
        (drop the legacy constraint, make pr_number nullable, add key columns,
        rebuild the unique indexes). Idempotent."""
        with self._conn() as c:
            c.execute(SCHEMA)      # CREATE TABLE IF NOT EXISTS (no-op if present)
            c.execute(MIGRATION)   # in-place upgrade of an old table
            c.execute(INDEXES)     # unique + btree indexes (need new columns)

    # ------------------------------------------------------------------
    # Create (dedupe on OUTCOME)
    # ------------------------------------------------------------------

    def create_wake(self, repo: str, head_sha: str, green: bool, *,
                    pr_number: int | None = None, payload: dict | None = None,
                    max_attempts: int = 3) -> int | None:
        """Persist a wake BEFORE acknowledging it (webhook returns 200 after).

        Returns the job id, or None if an ACTIVE wake for the same
        (repo, head_sha, color) already exists (duplicate — the other job
        owns the SHA).  A terminal or expired-running wake is RE-ARMED so a
        re-delivered verdict is never dropped.

        The wake is stamped `submitted_at = now()`: create_wake is always
        called by a submitter that immediately hands the wake to the worker
        executor (on_verdict, or run_neo's manual replay).  The recovery loop
        re-submits ONLY wakes whose submitted_at is NULL (never submitted) so
        a wake that IS submitted and merely waiting in the executor is never
        re-submitted every recovery pass (defect: recovery amplification)."""
        color = bool(green)
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO neo_jobs "
                "(kind, repo, pr_number, head_sha, wake_color, state, payload, "
                " max_attempts, submitted_at) "
                "VALUES ('wake', %s, %s, %s, %s, 'queued', %s, %s, now()) "
                "ON CONFLICT (repo, head_sha, wake_color) WHERE kind = 'wake' "
                "DO UPDATE SET "
                "  state = 'queued', outcome = NULL, lease_expires = NULL, "
                "  attempt = 0, payload = EXCLUDED.payload, "
                "  submitted_at = now(), "
                "  pr_number = COALESCE(neo_jobs.pr_number, EXCLUDED.pr_number), "
                "  updated_at = now() "
                "WHERE neo_jobs.state IN ('done', 'parked', 'superseded') "
                "   OR (neo_jobs.state = 'running' AND neo_jobs.lease_expires < now()) "
                "RETURNING id",
                (repo, pr_number, head_sha, color,
                 json.dumps(payload or {}), max_attempts),
            )
            row = cur.fetchone()
            return int(row[0]) if row else None

    def create_builder(self, repo: str, pr_number: int, head_sha: str,
                       builder_round: int, *, payload: dict | None = None,
                       max_attempts: int = 3) -> int | None:
        """Create a builder round job.  Returns the id, or None if an active
        builder job for the same (repo, pr, round) already exists.  Terminal /
        expired-running jobs of the same round are re-armed (dedupe on
        outcome), so a re-delivered bounce still reaches the builder lane."""
        with self._conn() as c:
            cur = c.execute(
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
                "RETURNING id",
                (repo, pr_number, head_sha, builder_round,
                 json.dumps(payload or {}), max_attempts),
            )
            row = cur.fetchone()
            return int(row[0]) if row else None

    # ------------------------------------------------------------------
    # Claim
    # ------------------------------------------------------------------

    def claim(self, job_id: int, *, pr_number: int | None = None,
              lease_secs: int = 600, worker_token: str | None = None) -> int | None:
        """Atomically claim a specific queued job by id (attempt + lease).

        Used by the Neo worker to claim the wake job created by on_verdict,
        after the PR number has been resolved.  Returns the id, or None if
        another worker already claimed it.

        `worker_token` names THIS claimant.  Every later resolve/retry/lease
        must present the same token, so a stale claimant (whose lease expired
        under a DB outage and was re-claimed by another worker) can never
        mutate the replacement run's row.  When omitted a token is generated
        here (tests / one-shot callers that never mutate later)."""
        token = worker_token or _new_token()
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET state = 'running', "
                "attempt = attempt + 1, "
                "pr_number = COALESCE(%s, pr_number), "
                "lease_expires = now() + make_interval(secs => %s), "
                "worker_token = %s, "
                "updated_at = now() "
                "WHERE id = %s AND state = 'queued' "
                "RETURNING id",
                (pr_number, lease_secs, token, job_id),
            )
            row = cur.fetchone()
            return int(row[0]) if row else None

    def claim_queued(self, kind: str = "builder", *, lease_secs: int = 600,
                     repo_filter: str = "", allowlist: tuple[str, ...] = (),
                     order: str = "desc", worker_token: str | None = None,
                     exclude_keys: set[tuple[str, int]] | None = None) -> dict | None:
        """Atomically claim one queued job of a kind, NEWEST first (created_at
        DESC — the newest round wins; older rounds are stale by definition).

        Returns the full row (including a fresh `worker_token` the caller must
        present on every later resolve/retry/lease) or None.  Used by the
        builder lane.  `order` accepts 'desc' (default, newest first) or 'asc'
        for callers that want oldest first.

        `exclude_keys` is a set of (repo, pr_number) pairs that must NOT be
        claimed (the builder lane passes its inflight PRs: prepare_worktree()
        keys its dest on repo+pr, so a second round of a PR whose round is
        already running would share — and clobber — one worktree)."""
        if order not in ("asc", "desc"):
            raise ValueError(f"order must be 'asc' or 'desc', got {order!r}")
        token = worker_token or _new_token()
        where = "state = 'queued' AND kind = %s"
        params: list[Any] = [kind]
        if repo_filter:
            where += " AND repo = %s"
            params.append(repo_filter)
        elif allowlist:
            where += " AND repo = ANY(%s)"
            params.append(list(allowlist))
        if exclude_keys:
            where += (" AND (repo, pr_number) NOT IN "
                      "(SELECT * FROM unnest(%s::text[], %s::bigint[]))")
            params.append([k[0] for k in exclude_keys])
            params.append([k[1] for k in exclude_keys])
        with self._conn() as c:
            cur = c.execute(
                f"UPDATE neo_jobs SET state = 'running', "
                f"attempt = attempt + 1, "
                f"lease_expires = now() + make_interval(secs => %s), "
                f"worker_token = %s, "
                f"updated_at = now() "
                f"WHERE id = ("
                f"  SELECT id FROM neo_jobs WHERE {where} "
                f"  AND (lease_expires IS NULL OR lease_expires < now()) "
                f"  ORDER BY created_at {order.upper()}, id DESC LIMIT 1 "
                f"  FOR UPDATE SKIP LOCKED"
                f") RETURNING id, repo, pr_number, head_sha, kind, "
                f"state, outcome, attempt, max_attempts, lease_expires, "
                f"wake_color, builder_round, payload, worker_token, "
                f"created_at, updated_at",
                [lease_secs, token] + params,
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))

    def peek_queued(self, kind: str = "builder", *, limit: int = 20,
                    repo_filter: str = "", allowlist: tuple[str, ...] = ()) -> list[dict]:
        """READ-ONLY peek at the queued jobs (no claim, no lease, no attempt
        burn).  Used by --dry-run so an operator can inspect the queue without
        mutating durable job state."""
        where = "state = 'queued' AND kind = %s"
        params: list[Any] = [kind]
        if repo_filter:
            where += " AND repo = %s"
            params.append(repo_filter)
        elif allowlist:
            where += " AND repo = ANY(%s)"
            params.append(list(allowlist))
        with self._conn() as c:
            cur = c.execute(
                f"SELECT id, repo, pr_number, head_sha, kind, state, outcome, "
                f"attempt, max_attempts, lease_expires, wake_color, builder_round, "
                f"payload, created_at, updated_at "
                f"FROM neo_jobs WHERE {where} "
                f"ORDER BY created_at DESC, id DESC LIMIT %s",
                params + [limit],
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Outcome transitions
    # ------------------------------------------------------------------

    def resolve(self, job_id: int, outcome: str, *,
                worker_token: str | None = None,
                payload: dict | None = None) -> bool:
        """Terminal resolution of a RUNNING job.  Only the current claimant
        (matching worker_token) may resolve.  `failed` is NOT allowed here —
        use retry() for a retryable failure."""
        assert outcome in TERMINAL_OUTCOMES, outcome
        token_cond, token_params = _token_where(worker_token)
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET state = 'done', outcome = %s, "
                "lease_expires = NULL, "
                "worker_token = NULL, "
                "payload = payload || %s::jsonb, "
                "updated_at = now() "
                "WHERE id = %s AND state = 'running'" + token_cond,
                [outcome, json.dumps(payload or {}), job_id] + token_params,
            )
            return cur.rowcount > 0

    def retry(self, job_id: int, *, reason: str = "",
              worker_token: str | None = None,
              payload: dict | None = None) -> str:
        """Mark a RUNNING job failed-retryable and return it to `queued`.

        Only the current claimant (matching worker_token) may retry.  Bounded
        by max_attempts: at the cap the job is `parked` instead (visible,
        owner-actionable — never silently re-armed).  Returns the resulting
        state, 'queued' or 'parked'.  Clears submitted_at so the recovery loop
        re-submits a retried wake exactly once (it is stamped again on
        submission — never every pass while waiting in the executor)."""
        note = dict(payload or {})
        if reason:
            note["reason"] = reason
        token_cond, token_params = _token_where(worker_token)
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET state = 'queued', outcome = 'failed', "
                "lease_expires = NULL, worker_token = NULL, submitted_at = NULL, "
                "payload = payload || %s, updated_at = now() "
                "WHERE id = %s AND state = 'running'" + token_cond + " "
                "RETURNING attempt, max_attempts",
                [json.dumps(note), job_id] + token_params,
            )
            row = cur.fetchone()
            if not row:
                return "?"   # not running (e.g. already resolved) — nothing to retry
            attempt, max_attempts = int(row[0]), int(row[1])
        if attempt >= max_attempts:
            self.park(job_id, reason=f"retryable failure exhausted {attempt}/{max_attempts} attempts")
            return "parked"
        return "queued"

    def park(self, job_id: int, *, reason: str = "", payload: dict | None = None) -> bool:
        """Terminal `parked` from any state (used by recovery/retry paths where
        the row was already reset to queued)."""
        note = dict(payload or {})
        if reason:
            note["reason"] = reason
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET state = 'done', outcome = 'parked', "
                "lease_expires = NULL, "
                "payload = payload || %s, updated_at = now() "
                "WHERE id = %s",
                (json.dumps(note), job_id),
            )
            return cur.rowcount > 0

    def note(self, job_id: int, payload: dict, *,
             worker_token: str | None = None) -> bool:
        """Attach a progress note to a RUNNING job (state stays 'running').
        Only the current claimant (matching worker_token) may write notes."""
        token_cond, token_params = _token_where(worker_token)
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET payload = payload || %s, updated_at = now() "
                "WHERE id = %s AND state = 'running'" + token_cond,
                [json.dumps(payload), job_id] + token_params,
            )
            return cur.rowcount > 0

    def lease(self, job_id: int, lease_secs: int = 600,
              worker_token: str | None = None) -> bool:
        """Extend a running job's lease.  Returns False if the job is no longer
        running (already expired/recovered/parked by recovery) or the caller is
        no longer the claimant (worker_token mismatch)."""
        token_cond, token_params = _token_where(worker_token)
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET lease_expires = "
                "now() + make_interval(secs => %s), updated_at = now() "
                "WHERE id = %s AND state = 'running'" + token_cond,
                [lease_secs, job_id] + token_params,
            )
            return cur.rowcount > 0

    @contextmanager
    def lease_renewal(self, job_id: int, *, lease_secs: int = 600,
                      interval: int | None = None,
                      worker_token: str | None = None) -> Iterator[None]:
        """Renew a running job's lease on a timer while the caller does long
        work (an agent run can outlive a static lease — without renewal a
        second worker would claim the same job and two agents would work one
        worktree).  Renews at least every lease/3.  The claimant's
        worker_token is required so a lost lease (re-claimed by another worker)
        stops renewing instead of silently extending the replacement's lease."""
        interval = interval or max(60, lease_secs // 3)
        stop = threading.Event()

        def _renew() -> None:
            while not stop.wait(interval):
                try:
                    if not self.lease(job_id, lease_secs=lease_secs,
                                      worker_token=worker_token):
                        log.warning("lease renewal lost for job %s — another "
                                    "worker may have claimed it", job_id)
                except Exception:  # noqa: BLE001
                    log.exception("lease renewal failed for job %s", job_id)

        t = threading.Thread(target=_renew, daemon=True,
                             name=f"lease-{job_id}", )
        t.start()
        try:
            yield
        finally:
            stop.set()

    # ------------------------------------------------------------------
    # Recovery / supersede
    # ------------------------------------------------------------------

    def recover_expired(self, kind: str = "wake") -> list[dict]:
        """Reset expired RUNNING leases of a kind back to `queued`.

        Does not increment attempt (a run already counted via claim).  The
        caller re-submits the recovered payload to the executor, or parks the
        job when attempt has reached max_attempts.  Clears worker_token (the
        old claimant is presumed dead) and submitted_at (the recovered wake
        needs a fresh hand-off), so the stale claimant can never mutate the
        re-claimed run."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET state = 'queued', lease_expires = NULL, "
                "worker_token = NULL, submitted_at = NULL, updated_at = now() "
                "WHERE kind = %s AND state = 'running' "
                "AND lease_expires IS NOT NULL AND lease_expires < now() "
                "RETURNING id, repo, pr_number, head_sha, kind, "
                "state, outcome, attempt, max_attempts, lease_expires, "
                "wake_color, builder_round, payload, worker_token, "
                "created_at, updated_at",
                (kind,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def stale_queued(self, kind: str = "wake", *, older_than_secs: int = 0) -> list[dict]:
        """Queued jobs with no consumer (created before `older_than_secs`).

        After a process restart the executor tasks die but the job rows stay
        queued; this sweep re-submits them.  Only rows that were NEVER
        submitted (`submitted_at IS NULL`) are returned: a wake that IS
        submitted and merely waiting in a saturated executor is not re-submitted
        every recovery pass (defect: recovery amplification).  Retried wakes
        clear submitted_at on retry(), so they become eligible again exactly
        once; the re-submission stamps them again."""
        with self._conn() as c:
            cur = c.execute(
                "SELECT id, repo, pr_number, head_sha, kind, state, outcome, "
                "attempt, max_attempts, lease_expires, wake_color, builder_round, "
                "payload, worker_token, submitted_at, created_at, updated_at "
                "FROM neo_jobs "
                "WHERE kind = %s AND state = 'queued' "
                "AND submitted_at IS NULL "
                "AND created_at < now() - make_interval(secs => %s) "
                "ORDER BY created_at DESC LIMIT 100",
                (kind, older_than_secs),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def release(self, job_id: int, *, worker_token: str | None = None) -> bool:
        """Return a RUNNING job to `queued` WITHOUT burning an attempt.

        Used when a claim is voided before the agent starts — e.g. another
        worker is already running the same repo+PR and they share one worktree
        (prepare_worktree keys its dest on repo+pr), so a second agent would
        rm -rf the live worktree.  Unlike retry(), this does not increment
        attempt or record a failure: the row was never actually worked.  Clears
        worker_token/submitted_at so a later claim starts clean."""
        token_cond, token_params = _token_where(worker_token)
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET state = 'queued', lease_expires = NULL, "
                "worker_token = NULL, submitted_at = NULL, updated_at = now() "
                "WHERE id = %s AND state = 'running'" + token_cond,
                [job_id] + token_params,
            )
            return cur.rowcount > 0

    def mark_submitted(self, job_id: int) -> bool:
        """Stamp that a wake has been handed to the worker executor.
        Every submission calls this so the recovery loop's stale_queued sweep
        (submitted_at IS NULL) never re-submits a wake that is already waiting
        in a saturated executor."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET submitted_at = now(), updated_at = now() "
                "WHERE id = %s AND kind = 'wake'",
                (job_id,),
            )
            return cur.rowcount > 0

    def reset_queued_submissions(self, kind: str = "wake") -> int:
        """Clear the submitted stamp on all queued jobs of a kind.

        Called ONCE at process startup: after a restart every executor task
        died, so any queued wake is an orphan and must be re-submitted.  The
        recovery loop's stale_queued sweep (submitted_at IS NULL) then re-arms
        them.  In steady state submissions are stamped again and never
        re-submitted while waiting."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET submitted_at = NULL, updated_at = now() "
                "WHERE kind = %s AND state = 'queued'",
                (kind,),
            )
            return cur.rowcount

    def supersede_older(self, repo: str, pr_number: int, round_no: int) -> list[int]:
        """Mark queued builder rounds OLDER than the claimed round for the same
        PR as superseded.  Newest round wins; the older rounds are stale by
        definition and must never run after a newer one exists.  Running jobs
        are left alone (an in-flight agent cannot be stopped safely)."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET state = 'done', outcome = 'superseded', "
                "updated_at = now() "
                "WHERE kind = 'builder' AND repo = %s AND pr_number = %s "
                "AND builder_round < %s AND state = 'queued' "
                "RETURNING id",
                (repo, pr_number, round_no),
            )
            return [int(r[0]) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get(self, job_id: int) -> dict | None:
        """Fetch one job row."""
        with self._conn() as c:
            cur = c.execute(
                "SELECT id, repo, pr_number, head_sha, kind, state, outcome, "
                "attempt, max_attempts, lease_expires, wake_color, builder_round, "
                "payload, worker_token, submitted_at, created_at, updated_at "
                "FROM neo_jobs WHERE id = %s",
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))

    def count_queued(self, kind: str = "wake") -> int:
        """Count of queued jobs (for monitoring)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM neo_jobs WHERE state = 'queued' AND kind = %s",
                (kind,),
            ).fetchone()
            return int(row[0]) if row else 0
