"""Per-attempt provider usage for Neo and Hermes Builder.

One record per attempt (route, model, uncached/cached input, output, reasoning
tokens, latency, outcome, error class).  Unknown stays None.  Stored in Neo's
DB next to the job (idempotent per attempt — same job_id + attempt_number
upserts rather than duplicating).

The route chain is the default NEO_ROUTES value (same order as Reviews B4):

    GLM 5.3 Flash on DeepInfra → Z.ai direct GLM 5.3 Flash →
    DeepInfra V4 Flash → V4 Flash 0731 → DeepSeek direct Flash →
    DeepInfra V4.1 Flash 1017, maximum reasoning

Routes fall through only on busy/429/402/5xx/timeout-before-first-token.
A 400 does NOT fall through.  Missing API key skips the route (logged once,
never a crash mid-run).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import psycopg  # psycopg3

log = logging.getLogger("rhobear_neo.usage")


# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS neo_usage (
    id              BIGSERIAL PRIMARY KEY,
    job_id          BIGINT NOT NULL,            -- FK to neo_jobs.id
    attempt_number  INTEGER NOT NULL DEFAULT 1, -- which attempt of the job
    route_idx       INTEGER NOT NULL,           -- position in the route chain (0-based)
    provider        TEXT NOT NULL,              -- e.g. 'deepinfra', 'zai', 'deepseek'
    model           TEXT NOT NULL,              -- e.g. 'glm-5.3-flash'
    uncached_input  BIGINT,                     -- tokens (None = unknown)
    cached_input    BIGINT,
    output_tokens   BIGINT,
    reasoning_tokens BIGINT,
    latency_ms      BIGINT,                     -- wall-clock milliseconds
    outcome         TEXT NOT NULL,              -- success | skipped | fallthrough | error
    error_class     TEXT,                       -- e.g. '429', '402', '5xx', 'timeout', '400'
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS neo_usage_attempt_key
    ON neo_usage (job_id, attempt_number, route_idx);

CREATE INDEX IF NOT EXISTS neo_usage_job_idx
    ON neo_usage (job_id);
"""


# ── Route model ──────────────────────────────────────────────────────────────

class Outcome(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"       # route skipped (missing key)
    FALLTHROUGH = "fallthrough"  # busy/429/402/5xx/timeout-before-first-token
    ERROR = "error"           # 400 or other non-fallthrough error


KNOWN_PROVIDERS = frozenset({"deepinfra", "zai", "deepseek", "openrouter", "openai"})


@dataclass
class Route:
    """A single provider:model entry in the route chain."""
    provider: str
    model: str

    @property
    def env_key_name(self) -> str:
        """The env var name for this provider's API key."""
        return f"{self.provider.upper()}_API_KEY"

    @classmethod
    def parse(cls, s: str) -> Route:
        """Parse 'provider:model' from a route string."""
        parts = s.strip().split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"invalid route {s!r} — expected 'provider:model'")
        provider = parts[0].strip().lower()
        model = parts[1].strip()
        if provider not in KNOWN_PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r} in route {s!r} — "
                f"known: {', '.join(sorted(KNOWN_PROVIDERS))}"
            )
        return cls(provider=provider, model=model)

    @classmethod
    def default_chain(cls) -> list[Route]:
        """The default NEO_ROUTES chain."""
        return [
            cls("deepinfra", "glm-5.3-flash"),
            cls("zai", "glm-5.3-flash"),
            cls("deepinfra", "deepseek-v4-flash"),
            cls("deepinfra", "deepseek-v4-flash-0731"),
            cls("deepseek", "deepseek-v4-flash"),
            cls("deepinfra", "deepseek-v4.1-flash-1017"),
        ]

    def api_key_available(self) -> bool:
        """Check if the API key for this route is set in the environment."""
        return bool(os.environ.get(self.env_key_name, "").strip())


# ── Usage attempt record ─────────────────────────────────────────────────────

@dataclass
class UsageAttempt:
    """One attempt record for storage and retrieval."""
    job_id: int
    attempt_number: int
    route_idx: int
    provider: str
    model: str
    uncached_input: int | None = None
    cached_input: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_ms: int | None = None
    outcome: str = "error"
    error_class: str | None = None

    def total_input(self) -> int:
        return (self.uncached_input or 0) + (self.cached_input or 0)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ── UsageStore ───────────────────────────────────────────────────────────────

@dataclass
class UsageStore:
    """Thread-safe Postgres usage store.

    Every method opens a fresh psycopg.connect(dsn, autocommit=True) — no
    shared connection, cursor, or mutable state.
    """
    dsn: str

    def _conn(self):
        return psycopg.connect(self.dsn, autocommit=True)

    def ensure_schema(self) -> None:
        """Create the usage table and indexes if absent. Idempotent."""
        with self._conn() as c:
            c.execute(SCHEMA)

    def record_attempt(self, attempt: UsageAttempt) -> bool:
        """Record one attempt, idempotent per (job_id, attempt_number, route_idx).

        Returns True if a new row was inserted, False if the key already existed
        (duplicate — the existing record is kept unchanged).
        """
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO neo_usage "
                "(job_id, attempt_number, route_idx, provider, model, "
                " uncached_input, cached_input, output_tokens, reasoning_tokens, "
                " latency_ms, outcome, error_class) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (job_id, attempt_number, route_idx) DO NOTHING "
                "RETURNING id",
                (attempt.job_id, attempt.attempt_number, attempt.route_idx,
                 attempt.provider, attempt.model,
                 attempt.uncached_input, attempt.cached_input,
                 attempt.output_tokens, attempt.reasoning_tokens,
                 attempt.latency_ms, attempt.outcome, attempt.error_class),
            )
            return cur.fetchone() is not None

    def get_attempts(self, job_id: int) -> list[UsageAttempt]:
        """Get all recorded attempts for a job, ordered by route_idx."""
        with self._conn() as c:
            cur = c.execute(
                "SELECT job_id, attempt_number, route_idx, provider, model, "
                "uncached_input, cached_input, output_tokens, reasoning_tokens, "
                "latency_ms, outcome, error_class "
                "FROM neo_usage WHERE job_id = %s "
                "ORDER BY attempt_number, route_idx",
                (job_id,),
            )
            cols = [d[0] for d in cur.description]
            return [UsageAttempt(**dict(zip(cols, r))) for r in cur.fetchall()]

    def sum_usage(self, job_id: int) -> dict[str, int]:
        """Sum all known usage for a job (only successful attempts).

        Returns {uncached_input, cached_input, output_tokens, reasoning_tokens}.
        Unknown fields (None in the DB) are excluded from the sum.
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT "
                "  COALESCE(SUM(uncached_input), 0), "
                "  COALESCE(SUM(cached_input), 0), "
                "  COALESCE(SUM(output_tokens), 0), "
                "  COALESCE(SUM(reasoning_tokens), 0) "
                "FROM neo_usage "
                "WHERE job_id = %s AND outcome = 'success'",
                (job_id,),
            ).fetchone()
        return {
            "uncached_input": int(row[0]),
            "cached_input": int(row[1]),
            "output_tokens": int(row[2]),
            "reasoning_tokens": int(row[3]),
        }

    def clear(self, job_id: int) -> int:
        """Remove all usage records for a job (test helper)."""
        with self._conn() as c:
            cur = c.execute("DELETE FROM neo_usage WHERE job_id = %s", (job_id,))
            return cur.rowcount


# ── Route resolver ───────────────────────────────────────────────────────────

def resolve_route_chain(routes_raw: str) -> list[Route]:
    """Parse the NEO_ROUTES env var value into an ordered list of Routes.

    Empty or whitespace-only input returns the default chain.
    Raises ValueError on unknown provider or malformed entry.
    """
    if not routes_raw or not routes_raw.strip():
        return Route.default_chain()
    parts = [p.strip() for p in routes_raw.split(",") if p.strip()]
    if not parts:
        return Route.default_chain()
    return [Route.parse(p) for p in parts]


def available_routes(routes: list[Route]) -> list[tuple[int, Route]]:
    """Return (index, route) pairs for routes whose API key is set.

    Missing-key routes are logged once and skipped — never a crash mid-run.
    """
    result: list[tuple[int, Route]] = []
    for idx, route in enumerate(routes):
        if route.api_key_available():
            result.append((idx, route))
        else:
            log.info("route %d (%s/%s) skipped — %s not set",
                     idx, route.provider, route.model, route.env_key_name)
    return result


# ── Fallthrough classification ───────────────────────────────────────────────

def classify_http_error(status_code: int | None) -> str | None:
    """Classify an HTTP status code into an error class string.

    Returns the class or None for unknown/success codes.
    """
    if status_code is None:
        return None
    if status_code == 429:
        return "429"
    if status_code == 402:
        return "402"
    if 500 <= status_code < 600:
        return "5xx"
    if status_code == 400:
        return "400"
    return str(status_code)


def is_fallthrough(status_code: int | None, timeout_before_first_token: bool = False) -> bool:
    """True if this error should cause a fallthrough to the next route.

    Fallthrough: busy/429/402/5xx/timeout-before-first-token.
    400 does NOT fall through (bad request — retrying the same route won't help).
    """
    if timeout_before_first_token:
        return True
    if status_code is None:
        return False
    if status_code in (429, 402):
        return True
    if 500 <= status_code < 600:
        return True
    return False


# classify_cli_fallthrough (a divergent dead duplicate of the agents'
# _classify_agent_error, which anchored differently — "cost 1500 tokens"
# classified as 5xx there) is DELETED: it was imported but never used on the
# decision path, and two fallthrough classifiers alive and disagreeing is a
# trap.  The surviving classifiers live in agent_claude / agent_hermes and
# anchor on `api_error_status` / `HTTP <code>` via the helpers above.
