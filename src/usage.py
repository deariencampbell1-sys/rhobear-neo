"""Per-attempt provider usage for Neo and Hermes Builder.

One record per attempt (route, model, uncached/cached input, output, reasoning
tokens, latency, outcome, error class).  Unknown stays None.  Stored in Neo's
DB next to the job (idempotent per attempt — same job_id + attempt_number
upserts rather than duplicating).

W1-06 (Zeus through RouteGate): every attempt is also metered with the
attribution the gate needs — consumer (`reviews:zeus/<install>`, written even
when the caller passes none), the PR / head SHA / builder round it served
(from the attempt's brief or from the owning job row), the RouteGate
capability (`route_name`) when the attempt ran through the gate, and the
cost the response actually carried (`cost_usd` — None when it did not).
Never a guessed consumer, target, or cost.

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
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import psycopg  # psycopg3

log = logging.getLogger("rhobear_neo.usage")

# The one consumer attribution for this service's model calls (Zeus's triage
# and its builder).  Written by the store when an attempt carries none, so the
# gate always sees `reviews:zeus/...` on a row even when the caller is an
# older lane file that predates the consumer field.
DEFAULT_CONSUMER = "reviews:zeus"


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
    consumer        TEXT,                       -- 'reviews:zeus/<install>' attribution
    pr_number       BIGINT,                     -- PR this attempt served (None = unknown)
    head_sha        TEXT,                       -- head SHA the attempt ran against
    builder_round   INTEGER,                    -- builder round (None = unknown / never)
    route_name      TEXT,                       -- RouteGate capability when gated ('peak')
    cost_usd        DOUBLE PRECISION,           -- cost the response carried (None = not carried)
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS neo_usage_attempt_key
    ON neo_usage (job_id, attempt_number, route_idx);

CREATE INDEX IF NOT EXISTS neo_usage_job_idx
    ON neo_usage (job_id);

-- In-place columns for a deployment created before W1-06 (same pattern as
-- src/jobs.py's ALTERs: idempotent, no migrations directory).  No-ops on a
-- fresh table — CREATE above already carries them.
ALTER TABLE neo_usage ADD COLUMN IF NOT EXISTS consumer TEXT;
ALTER TABLE neo_usage ADD COLUMN IF NOT EXISTS pr_number BIGINT;
ALTER TABLE neo_usage ADD COLUMN IF NOT EXISTS head_sha TEXT;
ALTER TABLE neo_usage ADD COLUMN IF NOT EXISTS builder_round INTEGER;
ALTER TABLE neo_usage ADD COLUMN IF NOT EXISTS route_name TEXT;
ALTER TABLE neo_usage ADD COLUMN IF NOT EXISTS cost_usd DOUBLE PRECISION;
"""


# ── Route model ──────────────────────────────────────────────────────────────

class Outcome(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"       # route skipped (missing key)
    FALLTHROUGH = "fallthrough"  # busy/429/402/5xx/timeout-before-first-token
    ERROR = "error"           # 400 or other non-fallthrough error


KNOWN_PROVIDERS = frozenset({"deepinfra", "zai", "deepseek", "openrouter",
                            "openai", "routegate"})
# `routegate` is the gate itself (capability routes, set via
# NEO_ROUTEGATE_ANTHROPIC_URL/NEO_ROUTEGATE_MODEL — never a NEO_ROUTES entry):
# admitting it to KNOWN_PROVIDERS keeps Route() construction consistent for the
# gated attempt rows, whose provider/model is `routegate`/<capability>.


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
    # --- W1-06 metering: what this attempt served and what it cost ---
    consumer: str | None = None        # 'reviews:zeus/<install>' (store default)
    pr_number: int | None = None
    head_sha: str | None = None
    builder_round: int | None = None
    route_name: str | None = None      # RouteGate capability when gated ('peak')
    cost_usd: float | None = None      # only ever what the response carried

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

        Attribution (W1-06):
          - consumer defaults to DEFAULT_CONSUMER (`reviews:zeus`) — every row
            is written with the consumer even when the caller passes none —
            and gains the install id (`reviews:zeus/<install_id>`, L0's
            consumer shape) from the owning job's payload when the caller did
            not already name one.
          - the owning job row is AUTHORITATIVE for PR / head SHA / builder
            round; caller-supplied (brief-parsed) values only fill what the
            row does not know.  When neither side knows, the column stays
            NULL — never guessed.

        One connection for lookup + insert: the job lookup shares the INSERT's
        connection (metering is a hot path; a second connect per attempt is
        churn we do not need).
        """
        consumer = (attempt.consumer or "").strip() or DEFAULT_CONSUMER
        with self._conn() as c:
            job_pr, job_sha, job_round, install_id = self._job_target(
                c, attempt.job_id)
            if install_id and "/" not in consumer:
                # Bare lane default → the L0 consumer shape reviews:zeus/<install>.
                # An already-qualified consumer (a caller that knew the
                # install) is never rewritten.
                consumer = f"{consumer}/{install_id}"
            pr_number = job_pr if job_pr is not None else attempt.pr_number
            head_sha = job_sha if job_sha is not None else attempt.head_sha
            builder_round = (job_round if job_round is not None
                             else attempt.builder_round)
            cur = c.execute(
                "INSERT INTO neo_usage "
                "(job_id, attempt_number, route_idx, provider, model, "
                " uncached_input, cached_input, output_tokens, reasoning_tokens, "
                " latency_ms, outcome, error_class, "
                " consumer, pr_number, head_sha, builder_round, "
                " route_name, cost_usd) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "        %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (job_id, attempt_number, route_idx) DO NOTHING "
                "RETURNING id",
                (attempt.job_id, attempt.attempt_number, attempt.route_idx,
                 attempt.provider, attempt.model,
                 attempt.uncached_input, attempt.cached_input,
                 attempt.output_tokens, attempt.reasoning_tokens,
                 attempt.latency_ms, attempt.outcome, attempt.error_class,
                 consumer, pr_number, head_sha, builder_round,
                 attempt.route_name, attempt.cost_usd),
            )
            return cur.fetchone() is not None

    def _job_target(
        self, c, job_id: int | None,
    ) -> tuple[int | None, str | None, int | None, str | None]:
        """(pr_number, head_sha, builder_round, install_id) of the owning job
        row — the authoritative target and consumer suffix for an attempt,
        read on the caller's open connection.

        Best effort only: usage recording must never fail because the jobs
        table (owned by src/jobs.py, created by its own schema) is absent or
        the row is gone.  Anything unknown comes back as None.
        """
        if job_id is None:
            return None, None, None, None
        try:
            row = c.execute(
                "SELECT pr_number, head_sha, builder_round, "
                "       payload->>'installation_id' "
                "FROM neo_jobs WHERE id = %s",
                (job_id,),
            ).fetchone()
        except Exception:  # noqa: BLE001 — metering must never raise here
            log.debug("job target lookup failed for job %s", job_id,
                      exc_info=True)
            return None, None, None, None
        if row is None:
            return None, None, None, None
        pr, sha, rnd, install_id = row
        return (int(pr) if pr is not None else None,
                str(sha) if sha else None,
                int(rnd) if rnd is not None else None,
                str(install_id) if install_id else None)

    def get_attempts(self, job_id: int) -> list[UsageAttempt]:
        """Get all recorded attempts for a job, ordered by route_idx."""
        with self._conn() as c:
            cur = c.execute(
                "SELECT job_id, attempt_number, route_idx, provider, model, "
                "uncached_input, cached_input, output_tokens, reasoning_tokens, "
                "latency_ms, outcome, error_class, "
                "consumer, pr_number, head_sha, builder_round, "
                "route_name, cost_usd "
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

# Status-code extraction for fallthrough classification (the ONE surviving
# classifier family — usage.classify_cli_fallthrough, a divergent dead
# duplicate, is deleted).  Anchored on `api_error_status` / `HTTP <code>`
# shapes ONLY: a bare 3-digit value ("429") or a code next to an
# HTTP/status/error-code marker ("HTTP/1.1 503 …", "status: 429", "Error
# code: 402 …").  Bare substrings are never codes — "cost 1500 tokens" must
# not classify as 5xx and "read 400 bytes" must not classify as 400.
_STATUS_CODE = re.compile(
    r'(?:\bHTTP(?:/\d(?:\.\d)?)?\s*:?\s*'
    r'|\bstatus(?:\s*code)?\s*:?\s*'
    r'|\berror\s+code\s*:?\s*)'
    r'(\d{3})\b',
    re.IGNORECASE,
)


def status_code_from(err_str: str) -> int | None:
    """Anchored status-code extraction from an api_error_status value or
    provider error line.  Accepted shapes ONLY: a bare/leading code
    ("429", "400 Bad Request", "503 Service Unavailable") or a code next to
    an HTTP/status/error-code marker ("HTTP/1.1 503 …", "Error code: 402 …").
    None when no anchored code is present — bare substrings ("cost 1500
    tokens", "read 400 bytes") are never codes."""
    err_str = (err_str or "").strip()
    m = re.match(r"(\d{3})(?!\d)", err_str)
    if m:
        return int(m.group(1))
    m = _STATUS_CODE.search(err_str)
    return int(m.group(1)) if m else None


def classify_error_text(err_str: str) -> tuple[bool, str | None]:
    """Classify a provider/agent error string by ANCHORED status code only.

    Returns (is_fallthrough, error_class).  One family: the agents' own
    `_classify_error_text` methods are thin delegates over this — never a
    second implementation.
    """
    err_str = err_str or ""
    code = status_code_from(err_str)
    if code is not None:
        return is_fallthrough(code), classify_http_error(code)
    if re.search(r'\btime(?:d\s+)?out\b', err_str, re.IGNORECASE):
        return True, "timeout"
    # Unknown API error — not fallthrough.
    return False, err_str[:32]


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
# trap.  The surviving family anchors on `api_error_status` / `HTTP <code>`
# via the helpers above: classify_error_text() is the text classifier and
# agent_claude/agent_hermes `_classify_error_text` delegate over it.


# ── Brief target (what an attempt served) ──────────────────────────────────────

_BRIEF_TARGET_PR = re.compile(r"TARGET:\s*PR\s*#(\d+)", re.IGNORECASE)
# The full 40-hex head SHA appears in the real brief twice: the canon status
# call (`gh api repos/…/commits/<sha>/status`) and the checkout line
# (`git checkout <sha>`).  Only those two contexts are matched — a bare
# 40-hex elsewhere is not evidence of a target.
_BRIEF_TARGET_SHA = re.compile(
    r"(?:commits/|checkout\s+)([0-9A-Fa-f]{40})")
_BRIEF_BUILDER_ROUND = re.compile(r"spent\s+(\d+)/(\d+)\s+builder rounds",
                                  re.IGNORECASE)


@dataclass(frozen=True)
class BriefTarget:
    """The PR / head SHA / builder round a Neo brief drives at."""
    pr_number: int | None = None
    head_sha: str | None = None
    builder_round: int | None = None


def brief_target(brief: str) -> BriefTarget:
    """Parse the target out of a Neo brief so every metered attempt row can
    say which PR / head SHA / builder round it served.

    Read from the brief's own TARGET line, the canon status call (full 40-hex
    head SHA) and the builder-round sentence.  Anything the brief does not
    carry stays None — never guessed.
    """
    brief = brief or ""
    pr = _BRIEF_TARGET_PR.search(brief)
    sha = _BRIEF_TARGET_SHA.search(brief)
    rnd = _BRIEF_BUILDER_ROUND.search(brief)
    return BriefTarget(
        pr_number=int(pr.group(1)) if pr else None,
        head_sha=sha.group(1) if sha else None,
        builder_round=int(rnd.group(1)) if rnd else None,
    )
