"""Durable wake jobs for Neo.

Neo wake jobs are keyed by (repository, pr_number, head_sha). They are persisted
before acknowledging the wake, leased with an expiry, and have explicit outcomes:

  done | no_change | failed | superseded | parked

The jobs table is created alongside neo_actions in the same database.  Every
method opens a fresh psycopg connection — thread-safe for concurrent workers.

Architecture:
  - create()  INSERT … ON CONFLICT DO NOTHING — concurrent duplicate wakes
    produce one job.
  - claim()   UPDATE … WHERE state='queued' AND (lease_expires IS NULL OR
    lease_expires < now()) … RETURNING — atomic lease with expiry recovery.
  - resolve() UPDATE state + outcome — terminal transition.
  - lease()   UPDATE lease_expires — extend a running lease.
  - recover_expired()  UPDATE queued WHERE lease_expires < now() — batch
    recovery of expired leases.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import psycopg  # psycopg3

SCHEMA = """
CREATE TABLE IF NOT EXISTS neo_jobs (
    id            BIGSERIAL PRIMARY KEY,
    repo          TEXT NOT NULL,
    pr_number     INTEGER NOT NULL,
    head_sha      TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'wake',   -- wake | builder
    state         TEXT NOT NULL DEFAULT 'queued', -- queued | running | done
    outcome       TEXT,                           -- done | no_change | failed | superseded | parked
    attempt       INTEGER NOT NULL DEFAULT 0,
    max_attempts  INTEGER NOT NULL DEFAULT 3,
    lease_expires TIMESTAMPTZ,
    payload       JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repo, pr_number, head_sha)
);
CREATE INDEX IF NOT EXISTS neo_jobs_queued ON neo_jobs(state, lease_expires)
    WHERE state = 'queued';
CREATE INDEX IF NOT EXISTS neo_jobs_expired ON neo_jobs(lease_expires)
    WHERE lease_expires IS NOT NULL AND state = 'running';
"""


@dataclass
class Jobs:
    """Thread-safe Postgres job store. Every method opens a fresh
    psycopg.connect(dsn, autocommit=True) — no shared connection, cursor, or
    mutable state. Safe to call from concurrent ThreadPoolExecutor threads."""
    dsn: str

    def _conn(self):
        return psycopg.connect(self.dsn, autocommit=True)

    def create(self, repo: str, pr_number: int, head_sha: str, *,
               kind: str = "wake", payload: dict | None = None,
               max_attempts: int = 3) -> int | None:
        """Insert a job, or no-op on conflict. Returns the job id or None if
        the job already existed (duplicate)."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO neo_jobs (repo, pr_number, head_sha, kind, "
                "state, payload, max_attempts) "
                "VALUES (%s, %s, %s, %s, 'queued', %s, %s) "
                "ON CONFLICT (repo, pr_number, head_sha) DO NOTHING "
                "RETURNING id",
                (repo, pr_number, head_sha, kind,
                 json.dumps(payload or {}), max_attempts),
            )
            row = cur.fetchone()
            return int(row[0]) if row else None

    def claim(self, repo: str, pr_number: int, head_sha: str, *,
              lease_secs: int = 600) -> int | None:
        """Atomically claim a specific queued job. Returns job id or None if
        already claimed/absent. Used by the worker loop for lease recovery."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET state = 'running', "
                "attempt = attempt + 1, "
                "lease_expires = now() + make_interval(secs => %s), "
                "updated_at = now() "
                "WHERE repo = %s AND pr_number = %s AND head_sha = %s "
                "AND state = 'queued' "
                "RETURNING id",
                (lease_secs, repo, pr_number, head_sha),
            )
            row = cur.fetchone()
            return int(row[0]) if row else None

    def claim_queued(self, kind: str = "wake", *,
                     lease_secs: int = 600,
                     repo_filter: str = "",
                     allowlist: tuple[str, ...] = ()) -> dict | None:
        """Atomically claim one queued job, oldest first. Returns the full row
        or None. Used by the builder for claiming builder rounds."""
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
                f"UPDATE neo_jobs SET state = 'running', "
                f"attempt = attempt + 1, "
                f"lease_expires = now() + make_interval(secs => %s), "
                f"updated_at = now() "
                f"WHERE id = ("
                f"  SELECT id FROM neo_jobs WHERE {where} "
                f"  AND (lease_expires IS NULL OR lease_expires < now()) "
                f"  ORDER BY created_at ASC LIMIT 1 "
                f"  FOR UPDATE SKIP LOCKED"
                f") RETURNING id, repo, pr_number, head_sha, kind, "
                f"state, outcome, attempt, max_attempts, lease_expires, "
                f"payload, created_at, updated_at",
                [lease_secs] + params,
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))

    def resolve(self, job_id: int, outcome: str, *,
                payload: dict | None = None) -> bool:
        """Mark a job done with a terminal outcome. Returns True if updated."""
        assert outcome in ("done", "no_change", "failed", "superseded", "parked")
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET state = 'done', outcome = %s, "
                "lease_expires = NULL, "
                "payload = CASE WHEN %s IS NULL THEN payload ELSE payload || %s END, "
                "updated_at = now() "
                "WHERE id = %s AND state = 'running'",
                (outcome,
                 "1" if payload else None,
                 json.dumps(payload or {}),
                 job_id),
            )
            return cur.rowcount > 0

    def lease(self, job_id: int, lease_secs: int = 600) -> bool:
        """Extend a running job's lease. Returns True if the job is still
        running (not yet expired/claimed by recovery)."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET lease_expires = "
                "now() + make_interval(secs => %s), updated_at = now() "
                "WHERE id = %s AND state = 'running'",
                (lease_secs, job_id),
            )
            return cur.rowcount > 0

    def recover_expired(self, kind: str = "wake", *,
                        lease_secs: int = 600) -> list[dict]:
        """Recover all expired leases for the given kind, resetting them to
        'queued' for re-claim. Returns the recovered rows."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET state = 'queued', lease_expires = NULL, "
                "updated_at = now() "
                "WHERE kind = %s AND state = 'running' "
                "AND lease_expires IS NOT NULL AND lease_expires < now() "
                "RETURNING id, repo, pr_number, head_sha, kind, "
                "state, outcome, attempt, max_attempts, lease_expires, "
                "payload, created_at, updated_at",
                (kind,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def supersede(self, repo: str, pr_number: int, head_sha: str) -> list[int]:
        """Mark all older queued jobs for the same PR as superseded.  Returns
        the list of superseded job ids.  The newest job (by head_sha or
        created_at) remains untouched."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE neo_jobs SET state = 'done', outcome = 'superseded', "
                "updated_at = now() "
                "WHERE repo = %s AND pr_number = %s AND head_sha != %s "
                "AND state = 'queued' "
                "RETURNING id",
                (repo, pr_number, head_sha),
            )
            return [int(r[0]) for r in cur.fetchall()]

    def get_by_key(self, repo: str, pr_number: int, head_sha: str) -> dict | None:
        """Fetch a job by its unique key."""
        with self._conn() as c:
            cur = c.execute(
                "SELECT id, repo, pr_number, head_sha, kind, "
                "state, outcome, attempt, max_attempts, lease_expires, "
                "payload, created_at, updated_at "
                "FROM neo_jobs "
                "WHERE repo = %s AND pr_number = %s AND head_sha = %s",
                (repo, pr_number, head_sha),
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
