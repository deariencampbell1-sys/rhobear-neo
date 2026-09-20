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

    def already_acted(self, repo: str, pr: int, head_sha: str, phase: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM neo_actions WHERE repo=%s AND pr_number=%s "
                "AND head_sha=%s AND phase=%s LIMIT 1",
                (repo, pr, head_sha, phase),
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
