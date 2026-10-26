"""Claude Code CLI agent for Neo — headless, tool-capable, no raw HTTP.

Replaces the old OpenRouterClient (direct HTTP, no tools) with a real agentic
CLI loop: Claude Code runs the Neo brief with full tool access (gh, git, file
edit, test runner) in an isolated per-run work directory.

Architecture:
  - RouteGate first (L2-platform, loopback): when the gate serves the
    Anthropic wire (cfg.routegate_anthropic_url) every model call runs
    through it as consumer `reviews:zeus/<install>` and the gate owns
    failover.  Until L2 exposes that endpoint the CLI keeps DeepSeek Direct
    (https://api.deepseek.com/anthropic) — brief step 2's "if not" branch —
    still metered per attempt with the same consumer header.
  - Claude Code CLI via subprocess
  - Isolated per-run temp work directory (deleted after the run)
  - Single JSON result object parsing (not NDJSON — Claude Code --output-format
    json returns one result object, not a stream)
  - Fail-closed on nonzero exit, timeout, missing result, unknown verdict,
    permission denials, truncation, or malformed JSON

Usage:
    agent = ClaudeAgent.from_config(cfg, gh_token=installation_token)
    usage, verdict = agent.run(brief)
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit

from .config import Config, ROUTEGATE_MODEL, ROUTEGATE_PORT
from .usage import (
    DEFAULT_CONSUMER, BriefTarget, Route, UsageAttempt, brief_target,
    classify_error_text,
)

log = logging.getLogger("rhobear_neo.agent")

# Fallthrough classification lives in src/usage.py (usage.classify_error_text /
# usage.status_code_from) — ONE classifier family for every lane.  The anchors
# are `api_error_status` / `HTTP <code>` shapes ONLY: a bare 3-digit value
# ("429") or a code next to an HTTP/status/error-code marker.  Bare substrings
# are never codes — "cost 1500 tokens" must not classify as 5xx and "read 400
# bytes" must not classify as 400.

# Warn once per process when from_config() gets no gh_token (a missing
# identity is a warning, never a crash — see from_config).  Guarded by a lock
# so concurrent from_config calls cannot double-warn.
_GH_TOKEN_WARNED = False
_GH_TOKEN_WARN_LOCK = threading.Lock()


def _warn_missing_gh_token() -> None:
    global _GH_TOKEN_WARNED
    with _GH_TOKEN_WARN_LOCK:
        if _GH_TOKEN_WARNED:
            return
        _GH_TOKEN_WARNED = True
    log.warning(
        "ClaudeAgent.from_config() called without gh_token — the CLI runs "
        "with no GH_TOKEN; no personal token may become a silent identity"
    )


ROUTEGATE_LOOPBACK_PORT = ROUTEGATE_PORT   # single source: src/config.py

# The consumer may be only what this lane mints: `reviews:zeus` or
# `reviews:zeus/<install-id>`.  Checked AT THE SINK (agent construction) as
# well as at the source (Config.zeus_consumer / UsageStore.job_consumer), so
# a poisoned job payload or any caller can never put header bytes into
# ANTHROPIC_CUSTOM_HEADERS — validation at the source alone is bypassable.
_CONSUMER_RE = re.compile(r"reviews:zeus(/[A-Za-z0-9_-]{1,64})?")


def _scrub_consumer(consumer: str | None) -> str:
    """Validate the consumer where it enters the client (the header sink).

    Non-conforming → the bare lane default, logged as a repr (control
    characters are escaped, so the warning itself cannot be forged)."""
    value = (consumer or "").strip()
    if not value:
        return ""
    if not _CONSUMER_RE.fullmatch(value):
        log.warning("consumer refused at the agent sink: %r — using the "
                    "bare lane default", value)
        return DEFAULT_CONSUMER
    return value


def _scrub_capability(name: str | None) -> str:
    """Validate the RouteGate capability name AT THE SINK — it lands in the
    child's argv (`--model`) and in the metered `route_name`, so a
    post-validate mutation of `cfg.routegate_model` must not put e.g.
    `--evil` into either.  Non-conforming → the real capability (`peak`)."""
    value = (name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,32}", value):
        log.warning("routegate capability refused at the agent sink: %r — "
                    "using %r", value, ROUTEGATE_MODEL)
        return ROUTEGATE_MODEL
    return value


def _is_gate_url(url: str) -> bool:
    """True when `url` is the loopback RouteGate base (parsed by the same
    fail-closed guard that refuses foreign URLs)."""
    try:
        _require_loopback_roulegate(url)
    except AgentError:
        return False
    return True


def _require_loopback_roulegate(url: str) -> str:
    """Fail-closed: only the loopback RouteGate may receive a brief.

    The value is a BASE (`http://127.0.0.1:8645`) — the Claude CLI appends
    `/v1/messages` itself, exactly as it does for `deepseek_base_url`
    (`https://api.deepseek.com/anthropic`).  The guard therefore requires
    scheme http, host exactly 127.0.0.1, no userinfo, RouteGate's port, and
    NO path at all: any path (including `..`, `%2e%2e`, or a doubled
    `/v1/messages`) is refused, so the client always resolves
    `base + /v1/messages` and nothing else.

    Parsed, not prefixed: a prefix check lets
    `http://127.0.0.1:8645@evil.example/...` (userinfo) or an arbitrary
    loopback port through.  validate() whitelists the exact base, but a cfg
    can be mutated after it — so both construction paths (from_config and
    _for_roulegate) re-check here and refuse anything that would route a
    brief (and its repo context) off-box.  Returns the normalised URL.
    """
    clean = (url or "").strip().rstrip("/")
    parts = urlsplit(clean)
    path = parts.path or ""
    try:
        port = parts.port
    except ValueError:  # unparseable port → not the gate
        port = None
    if (parts.scheme != "http"
            or parts.hostname != "127.0.0.1"
            or parts.username is not None
            or parts.password is not None
            or port != ROUTEGATE_LOOPBACK_PORT
            or path not in ("", "/")):
        raise AgentError(
            f"routegate_anthropic_url must be the loopback RouteGate BASE "
            f"(http://127.0.0.1:{ROUTEGATE_LOOPBACK_PORT} — the CLI "
            f"appends /v1/messages), got {url!r}"
        )
    return clean


# Canonical Neo verdicts — anything else is rejected.
CANONICAL_VERDICTS = frozenset({
    "FIXED",  # builder alias for ACCEPT-READY
    "ACCEPT-MERGED",  # backward compat — old agents may still emit this
    "ACCEPT",         # B2: agent no longer merges, returns ACCEPT
    "ACCEPT-READY",
    "FIX-FORWARD",
    "BOUNCE-BUILDER",
    "ESCALATE",
})

# Verdict extraction regex: anchored to start of line (after optional markdown
# prefix like **, *, >, -) followed by VERDICT: and the value.  This prevents
# prose like "Do not output VERDICT: ..." from matching.
_VERDICT_RE = re.compile(
    r'^\s*(?:[*]{1,2}|>`?\s*|-\s+)?VERDICT:\s*(.*?)(?:\s*[*`]+)?\s*$',
    re.IGNORECASE,
)


class AgentError(Exception):
    """Base for all Claude Code agent failures — fail-closed root."""


class NonZeroExit(AgentError):
    """CLI exited with a nonzero status."""


class TimeoutError(AgentError):
    """CLI did not finish within the timeout."""


class TruncatedOutput(AgentError):
    """Output was truncated (stop_reason=length / max_tokens)."""


class MissingResult(AgentError):
    """No output produced at all."""


class MalformedStream(AgentError):
    """JSON output could not be parsed or has unexpected shape."""


class ApiError(AgentError):
    """API returned an error status (api_error_status in JSON)."""

    def __init__(self, status: str, message: str = ""):
        self.status = status
        super().__init__(f"api_error: {status} — {message}")


class UnknownVerdict(AgentError):
    """No VERDICT: line found in the agent's output or verdict not canonical."""


class ClaudeAgent:
    """Run the Neo brief through Claude Code CLI (headless, with tools).

    Each call creates an isolated temp work directory so the agent has a clean
    slate.  The work directory is deleted after the run — no accumulated
    clone/copy graveyard.

    Thread-safe for sequential calls (each call creates its own subprocess).
    """

    def __init__(
        self,
        *,
        claude_bin: str,
        api_key: str,
        base_url: str,
        model: str,
        effort: str = "max",
        max_tokens: int = 32000,
        timeout: int = 1800,
        gh_token: str = "",
        consumer: str = "reviews:zeus",
    ) -> None:
        self.claude_bin = claude_bin
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # Exact direct model ID (deepseek-v4-flash) — no /deepseek prefix, no
        # [1m] context suffix. The same value goes to --model and env vars.
        self.model = model.strip()
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.gh_token = gh_token
        # Consumer attribution for gated calls (brief step 1):
        # `reviews:zeus` or `reviews:zeus/<install_id>` — validated at the
        # SINK (a poisoned value never reaches a header env var) and only
        # attached when the call goes through RouteGate: a third-party
        # provider never receives this lane's internal attribution.
        self.consumer = _scrub_consumer(consumer)
        self.gated = _is_gate_url(self.base_url)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Config, *, gh_token: str = "",
                    install_id: str = "") -> ClaudeAgent:
        """Build from a Config object.

        `gh_token` is optional (W1-06): production passes the repo's GitHub
        App installation token so the agent's GH_TOKEN env is never a personal
        token.  Absent → warn once per process and run without GH_TOKEN, never
        raise.  The cfg.gh_token field is ignored as an identity.

        RouteGate first (brief step 1): when the gate serves this wire the
        base URL is RouteGate's Anthropic endpoint and the attempt is gated
        and metered as consumer `reviews:zeus/<install_id>` — the agent never
        builds a direct provider client on its own initiative.  When the gate
        serves no Anthropic wire yet (the current L2 state: no /v1/messages
        in routegate.py, none documented in L2 CONTRACT.md), the direct
        DeepSeek Anthropic endpoint stands exactly as step 2's "if not"
        branch requires.
        """
        if not gh_token:
            _warn_missing_gh_token()
        routegate_url = str(
            getattr(cfg, "routegate_anthropic_url", "") or "").strip()
        if routegate_url:
            # Gated: RouteGate needs no provider credential, and the model is
            # the gate's capability name (route-policy.yaml), not a provider ID.
            # Loopback is re-checked here, not only in validate(): a cfg
            # mutated after startup must not carry a brief off-box.
            base_url = _require_loopback_roulegate(routegate_url)
            api_key = "routegate-loopback"
            model = _scrub_capability(cfg.routegate_model)
        else:
            api_key = cfg.deepseek_key
            base_url = cfg.deepseek_base_url
            model = cfg.deepseek_model
        return cls(
            claude_bin=cfg.claude_bin,
            api_key=api_key,
            base_url=base_url,
            model=model,
            effort=cfg.deepseek_reasoning_effort,
            max_tokens=cfg.deepseek_max_tokens,
            timeout=cfg.deepseek_timeout,
            gh_token=gh_token,
            consumer=cfg.zeus_consumer(install_id),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, brief: str) -> tuple[dict[str, int | float], str]:
        """Run the Neo brief through Claude Code CLI on the configured route.

        Returns (usage, verdict) where usage is normalised to
        {input_tokens: int, output_tokens: int} plus cached_input (int) and
        cost_usd (float) ONLY when the response carried them — hence the
        widened value type — and verdict is the extracted VERDICT: line value.

        Raises AgentError subclasses on failure — the caller should
        catch and return empty usage + empty verdict.
        """
        # Temp work directory — the agent clones/edits/tests here.
        with tempfile.TemporaryDirectory(prefix="neo-work-") as work_dir:
            # Temp config directory — empty, deleted after run for cleanup.
            # (No settings file — --dangerously-skip-permissions replaces that.)
            with tempfile.TemporaryDirectory(prefix="neo-config-") as config_dir:
                return self._run_in(brief, work_dir, config_dir)

    # ------------------------------------------------------------------
    # Route chain: walk available routes, fall through on transient errors
    # ------------------------------------------------------------------

    @classmethod
    def _for_route(cls, cfg: Config, route: Route, gh_token: str,
                   base_url: str,
                   consumer: str | None = None) -> ClaudeAgent:
        """Create a ClaudeAgent for a specific route in the chain.

        `base_url` is the Anthropic-wire endpoint for this provider.
        The API key is read from the route's env var.
        `consumer` is the run's wire consumer (install-qualified from the
        owning job when known) so the header matches the stored row.
        """
        api_key = cfg.api_key_for(route.provider)
        return cls(
            claude_bin=cfg.claude_bin,
            api_key=api_key,
            base_url=base_url,
            model=route.model,
            effort=cfg.deepseek_reasoning_effort,
            max_tokens=cfg.deepseek_max_tokens,
            timeout=cfg.deepseek_timeout,
            gh_token=gh_token,
            consumer=consumer or cfg.zeus_consumer(),
        )

    def _for_roulegate(self, cfg: Config, gh_token: str,
                       base_url: str,
                       consumer: str | None = None) -> "ClaudeAgent":
        """ClaudeAgent pointed at RouteGate's Anthropic wire (brief steps 1+2).

        Fail-closed: only the loopback gate may receive a brief — a foreign
        URL is refused here even when validate() was bypassed upstream
        (shared guard with from_config).  RouteGate needs no provider
        credential (loopback probe: 200 with no auth); the non-secret
        placeholder satisfies Claude Code's own key check while the real
        attribution travels as the consumer header.
        """
        url = _require_loopback_roulegate(base_url)
        return ClaudeAgent(
            claude_bin=cfg.claude_bin,
            api_key="routegate-loopback",
            base_url=url,
            model=_scrub_capability(cfg.routegate_model),
            effort=cfg.deepseek_reasoning_effort,
            max_tokens=cfg.deepseek_max_tokens,
            timeout=cfg.deepseek_timeout,
            gh_token=gh_token,
            consumer=consumer or self.consumer,
        )

    def _wire_consumer(self, usage_store, job_id: int | None) -> str:
        """The consumer for THIS run's wire header.

        Install-qualified from the owning job's payload when the store knows
        it, so the consumer the gate meters (header) and the consumer in the
        row can never disagree; falls back to the agent's own consumer when
        the store or row has nothing.  A store without the method (test
        doubles, older callers) keeps the agent's own consumer.
        """
        consumer = (self.consumer or "").strip() or DEFAULT_CONSUMER
        if usage_store is None or job_id is None or "/" in consumer:
            return consumer
        getter = getattr(usage_store, "job_consumer", None)
        if getter is None:
            return consumer
        return getter(job_id) or consumer

    @staticmethod
    def _record_attempt(usage_store, *, job_id: int | None, **kwargs) -> None:
        """Record one UsageAttempt; metering must never take the run down."""
        if usage_store is None or job_id is None:
            return
        try:
            usage_store.record_attempt(UsageAttempt(job_id=job_id, **kwargs))
        except Exception:  # noqa: BLE001
            log.exception("failed to record usage attempt (job %s, route %s)",
                          job_id, kwargs.get("route_idx"))

    @staticmethod
    def _attribution(target: BriefTarget, consumer: str) -> dict:
        """Per-attempt metering fields: consumer + the target the attempt
        served (brief step 3).  Unknown stays None."""
        return {
            "consumer": consumer,
            "pr_number": target.pr_number,
            "head_sha": target.head_sha,
            "builder_round": target.builder_round,
        }

    def run_with_chain(
        self,
        brief: str,
        available_routes: list[tuple[int, Route]],
        cfg: Config,
        gh_token: str,
        usage_store=None,
        job_id: int | None = None,
        attempt_number: int = 1,
    ) -> tuple[dict[str, int | float], str, int | None, Route | None]:
        """Walk the available route chain, trying each in order.

        RouteGate first (W1-06): when the gate serves this wire
        (cfg.routegate_anthropic_url) the run is ONE attempt through
        RouteGate, metered with route name `peak` and consumer
        reviews:zeus — the gate owns failover, so there is deliberately no
        second attempt on a direct provider here (a provider fallback
        outside RouteGate is drift).

        Otherwise the direct chain runs unchanged:
          - Skip routes with no Anthropic-wire endpoint (logged once).
          - Skip routes whose API key is not set.
          - Launch on route 1.
          - If the run fails with a fallthrough class (429/402/5xx/timeout
            before any output), record the attempt and try the next route.
          - A non-fallthrough failure stops the chain.
          - A successful run returns with its usage, verdict, route info.

        Each launch records one UsageAttempt with its real route (provider,
        model), the consumer, the PR/head SHA/builder round it served, and
        only the tokens/cost the response actually carried.  Failed attempts
        record the error class.

        Returns (usage, verdict, route_idx, route).
        usage is the final successful attempt's usage.
        verdict is the final successful verdict.
        route_idx/route are the winning route (None if all failed).
        """
        target = brief_target(brief)
        rg_url = str(getattr(cfg, "routegate_anthropic_url", "") or "").strip()
        if rg_url:
            # Resolve the QUALIFIED consumer only on the gated branch — the
            # header exists only there, and the direct chain's rows are
            # upgraded by the store anyway (no wasted connection).
            return self._run_roulegate(
                brief, cfg, rg_url, gh_token, usage_store=usage_store,
                job_id=job_id, attempt_number=attempt_number, target=target,
                consumer=self._wire_consumer(usage_store, job_id))

        start_time = time.monotonic()
        attribution = {
            **self._attribution(target, self.consumer),
            # A run without GitHub identity is visible in METERING, not just
            # in a boot log (Apollo r4): every row says whether the CHILD of
            # THIS run carried one (the run's gh_token argument — r5).
            "had_gh_token": bool(gh_token),
        }

        for idx, route in available_routes:
            # Skip routes Claude Code can't serve (no Anthropic-wire endpoint).
            base_url = cfg.claude_endpoint_for(route.provider)
            if base_url is None:
                log.info("route %d (%s/%s) skipped for Claude agent — "
                         "no Anthropic-wire endpoint",
                         idx, route.provider, route.model)
                self._record_attempt(
                    usage_store, job_id=job_id,
                    attempt_number=attempt_number, route_idx=idx,
                    provider=route.provider, model=route.model,
                    outcome="skipped", error_class="no_anthropic_wire",
                    **attribution)
                continue

            # Build per-route agent.
            api_key = cfg.api_key_for(route.provider)
            if not api_key:
                log.info("route %d (%s/%s) skipped — API key not set",
                         idx, route.provider, route.model)
                self._record_attempt(
                    usage_store, job_id=job_id,
                    attempt_number=attempt_number, route_idx=idx,
                    provider=route.provider, model=route.model,
                    outcome="skipped", error_class="missing_key",
                    **attribution)
                continue

            route_agent = self._for_route(cfg, route, gh_token, base_url,
                                          consumer=self.consumer)

            with tempfile.TemporaryDirectory(prefix="neo-work-") as work_dir:
                with tempfile.TemporaryDirectory(prefix="neo-config-") as config_dir:
                    route_start = time.monotonic()
                    try:
                        usage, verdict = route_agent._run_in(
                            brief, work_dir, config_dir)
                    except AgentError as exc:
                        elapsed_ms = int((time.monotonic() - route_start) * 1000)
                        # Classify the failure for fallthrough decision.
                        fallthrough, err_class = self._classify_agent_error(
                            exc, route_agent,
                            proc=getattr(route_agent, '_last_proc', None))
                        log.info("route %d (%s/%s) failed: %s (fallthrough=%s)",
                                 idx, route.provider, route.model, exc, fallthrough)
                        # Record the failed attempt (metered: 402 etc. lands
                        # one fallthrough row with its route name).
                        self._record_attempt(
                            usage_store, job_id=job_id,
                            attempt_number=attempt_number, route_idx=idx,
                            provider=route.provider, model=route.model,
                            latency_ms=elapsed_ms,
                            outcome="fallthrough" if fallthrough else "error",
                            error_class=err_class, **attribution)
                        if not fallthrough:
                            # Non-fallthrough failure — re-raise so the
                            # caller's except AgentError handler retries
                            # the job rather than treating it as completed.
                            raise exc
                        # Fallthrough — try next route.
                        continue

                    # Success.
                    elapsed_ms = int((time.monotonic() - route_start) * 1000)
                    self._record_attempt(
                        usage_store, job_id=job_id,
                        attempt_number=attempt_number, route_idx=idx,
                        provider=route.provider, model=route.model,
                        uncached_input=usage.get("input_tokens"),
                        cached_input=usage.get("cached_input"),
                        output_tokens=usage.get("output_tokens"),
                        latency_ms=elapsed_ms,
                        outcome="success",
                        cost_usd=usage.get("cost_usd"),
                        **attribution)
                    log.info("route %d (%s/%s) succeeded — verdict=%s",
                             idx, route.provider, route.model, verdict)
                    return usage, verdict, idx, route

        # All routes exhausted.
        return {}, "", None, None

    def _run_roulegate(
        self,
        brief: str,
        cfg: Config,
        base_url: str,
        gh_token: str,
        *,
        usage_store,
        job_id: int | None,
        attempt_number: int,
        target: BriefTarget,
        consumer: str,
    ) -> tuple[dict[str, int | float], str, int | None, Route | None]:
        """One attempt through RouteGate, metered as its capability (`peak`).

        RouteGate owns failover ("same-model fallback is RouteGate's job"),
        so a fallthrough-class failure records the row and ends the run —
        the caller's normal retry is the next attempt.  Anything else fails
        closed and raises.  Every row carries the run's consumer — the same
        value that went out on the wire header — and the
        PR / head SHA / builder round the brief drove at; cost is recorded
        only if the response carried one.
        """
        route_agent = self._for_roulegate(cfg, gh_token, base_url,
                                          consumer=consumer)
        # The capability is scrubbed once at the sink and used for argv,
        # metering and the returned Route alike — one value everywhere.
        capability = _scrub_capability(cfg.routegate_model)
        attribution = {
            **self._attribution(target, consumer),
            "route_name": capability,
            # Truth for metering = the identity this run hands the CHILD (the
            # gh_token argument the chain builds its agents from), never
            # self.gh_token (Apollo r5: the divergent case must read False).
            "had_gh_token": bool(gh_token),
        }
        # route_idx 0: the gate occupies the first chain slot.  The unique
        # key (job_id, attempt_number, route_idx) then provides idempotency:
        # jobs.py advances `attempt` per retry, so a fresh attempt always
        # lands a fresh row, while a replayed attempt_number dedupes instead
        # of double-metering (same semantics as the direct chain's route 0).
        with tempfile.TemporaryDirectory(prefix="neo-work-") as work_dir:
            with tempfile.TemporaryDirectory(prefix="neo-config-") as config_dir:
                route_start = time.monotonic()
                try:
                    usage, verdict = route_agent._run_in(
                        brief, work_dir, config_dir)
                except AgentError as exc:
                    elapsed_ms = int((time.monotonic() - route_start) * 1000)
                    fallthrough, err_class = self._classify_agent_error(
                        exc, route_agent,
                        proc=getattr(route_agent, '_last_proc', None))
                    log.info("routegate (%s) failed: %s (fallthrough=%s)",
                             capability, exc, fallthrough)
                    self._record_attempt(
                        usage_store, job_id=job_id,
                        attempt_number=attempt_number, route_idx=0,
                        provider="routegate", model=capability,
                        latency_ms=elapsed_ms,
                        outcome="fallthrough" if fallthrough else "error",
                        error_class=err_class, **attribution)
                    if not fallthrough:
                        raise
                    # Fallthrough ends the run: no attempt outside RouteGate.
                    return {}, "", None, None
                elapsed_ms = int((time.monotonic() - route_start) * 1000)
                self._record_attempt(
                    usage_store, job_id=job_id,
                    attempt_number=attempt_number, route_idx=0,
                    provider="routegate", model=capability,
                    uncached_input=usage.get("input_tokens"),
                    cached_input=usage.get("cached_input"),
                    output_tokens=usage.get("output_tokens"),
                    latency_ms=elapsed_ms, outcome="success",
                    cost_usd=usage.get("cost_usd"),
                    **attribution)
                log.info("routegate (%s) succeeded — verdict=%s",
                         capability, verdict)
                return usage, verdict, 0, Route("routegate", capability)

    def _classify_agent_error(
        self, exc: AgentError, agent: ClaudeAgent,
        proc: subprocess.CompletedProcess | None = None,
    ) -> tuple[bool, str | None]:
        """Classify an agent error into (is_fallthrough, error_class).

        Anchored on `api_error_status` / `HTTP <code>` (see
        usage.status_code_from),
        never on bare substrings.  Falls back to error type classification.
        """
        # Try to extract the API error status from the JSON envelope first.
        if proc is not None and proc.stdout:
            try:
                result = json.loads(proc.stdout.strip())
                if isinstance(result, dict):
                    api_status = result.get("api_error_status")
                    if api_status is not None and api_status != "":
                        return self._classify_error_text(str(api_status).strip())
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # ApiError — the status string raised from api_error_status.
        if isinstance(exc, ApiError):
            return self._classify_error_text(exc.status)

        # The stream closed before any output — a fallthrough (the next route
        # may well work), never a "successful empty review".  A run that DID
        # produce output but no verdict text stays non-fallthrough below: the
        # model ran, silently retrying another route would hide it.
        if proc is not None and isinstance(exc, MissingResult) \
                and not (proc.stdout or "").strip():
            return True, "stream_closed"

        # Fall back to error type classification.
        if isinstance(exc, TimeoutError):
            return True, "timeout"
        if isinstance(exc, NonZeroExit):
            return True, "exit_nonzero"
        # TruncatedOutput, MissingResult, MalformedStream, UnknownVerdict
        # are all non-fallthrough — the model ran but produced bad output.
        return False, type(exc).__name__.lower()

    @staticmethod
    def _classify_error_text(err_str: str) -> tuple[bool, str | None]:
        """Classify a provider error string by ANCHORED status code only.

        Thin delegate to usage.classify_error_text — one classifier family
        across the lanes, never a second implementation here.
        """
        return classify_error_text(err_str)

    def _run_in(
        self, brief: str, work_dir: str, config_dir: str,
    ) -> tuple[dict[str, int | float], str]:
        """Run inside pre-created temp directories."""
        env = self._build_env(config_dir)
        cmd = self._build_cmd(brief)

        log.debug("running claude in %s: %s", work_dir, " ".join(cmd[:-1]) + " ...")

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=work_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            log.error("claude timed out after %.0fs (timeout=%ds)", elapsed, self.timeout)
            raise TimeoutError(
                f"claude did not finish within {self.timeout}s"
            )

        elapsed = time.monotonic() - start
        log.info("claude exit=%d elapsed=%.0fs stdout=%d stderr=%d",
                 proc.returncode, elapsed, len(proc.stdout or ""),
                 len(proc.stderr or ""))

        self._last_proc = proc
        return self._parse_output(proc)

    # ------------------------------------------------------------------
    # Internal: env / cmd
    # ------------------------------------------------------------------

    def _build_env(self, config_dir: str) -> dict[str, str]:
        """Build the environment for the Claude Code subprocess.

        Uses ANTHROPIC_BASE_URL (the gateway env var that Claude Code respects)
        and sets both ANTHROPIC_AUTH_TOKEN and ANTHROPIC_API_KEY for maximum
        compatibility.  The base environment is an explicit allowlist (PATH,
        HOME, LANG, TMPDIR — same discipline as advisor/server.py::_agent_env
        and HermesAgent._build_env): the agent runs with
        --dangerously-skip-permissions over PR-derived briefs, so DATABASE_URL,
        NEO_ADVISOR_TOKEN, NEO_GITHUB_APP_KEY/_KEY_PATH, provider keys and any
        other service secret must never travel with it.  Never logs or echoes
        the API key.
        """
        defaults = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": os.path.expanduser("~") or "/tmp",
            "LANG": "C.UTF-8",
            "TMPDIR": tempfile.gettempdir(),
        }
        env = {key: (os.environ.get(key) or defaults[key])
               for key in ("PATH", "HOME", "LANG", "TMPDIR")}
        env.update({
            "CLAUDE_CONFIG_DIR": config_dir,
            "ANTHROPIC_BASE_URL": self.base_url,
            "ANTHROPIC_API_KEY": self.api_key,
            "ANTHROPIC_AUTH_TOKEN": self.api_key,
            "ANTHROPIC_MODEL": self.model,
            "CLAUDE_CODE_EFFORT_LEVEL": self.effort,
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self.max_tokens),
            # DeepSeek Direct supports 1M context, but Claude Code does not
            # know this newly named model. Keep the direct model ID bare and
            # provide the real window explicitly; [1m] is an OpenRouter-only
            # CLI hint and must not be sent to DeepSeek Direct.
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1048576",
            "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT": "1",
            "CLAUDE_CODE_OUTPUT_FORMAT": "json",
        })
        # Use only the configured GitHub credential. A stale host token must
        # never leak into Neo when its explicit token is absent.
        if self.gh_token:
            env["GH_TOKEN"] = self.gh_token
        # Consumer attribution (W1-06): the gate's per-lane header rides with
        # GATED calls only (a third-party provider never receives it).
        # ANTHROPIC_CUSTOM_HEADERS is how Claude Code carries a custom header
        # on the wire; X_RHOBEAR_CONSUMER is this service's own auditable env
        # form of the same value.  Neither is a secret.  The child env is a
        # strict allowlist: an operator's own ANTHROPIC_CUSTOM_HEADERS is NOT
        # re-admitted (credentials parked there must never ride our calls).
        if self.gated and self.consumer:
            env["X_RHOBEAR_CONSUMER"] = self.consumer
            env["ANTHROPIC_CUSTOM_HEADERS"] = (
                f"X-RHOBEAR-Consumer: {self.consumer}"
            )
        return env

    def _build_cmd(self, brief: str) -> list[str]:
        """Build the Claude Code CLI command.

        Runs in one-shot prompt mode (-p) with the exact model, max effort,
        dangerously-skip-permissions (pre-approves all tools — Neo is a
        dedicated service agent), no session persistence, and JSON output.
        """
        return [
            self.claude_bin,
            "-p", brief,
            "--model", self.model,
            "--effort", self.effort,
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--output-format", "json",
        ]

    # ------------------------------------------------------------------
    # Internal: output parsing
    # ------------------------------------------------------------------

    def _parse_output(
        self, proc: subprocess.CompletedProcess[str],
    ) -> tuple[dict[str, int | float], str]:
        """Parse the Claude Code CLI output.

        Claude Code --output-format json returns a single JSON result object
        (not NDJSON lines).  The object has type=result, subtype=success,
        is_error=false, a result string, usage, stop_reason, num_turns, and
        permission_denials.

        Fail-closed on:
          - Nonzero exit (even with a verdict — the process must exit clean)
          - Truncation (stop_reason=length or max_tokens)
          - Permission denials (non-empty array)
          - No output / no result text
          - No VERDICT: line in the result text
          - Non-canonical verdict value
          - Malformed JSON

        Returns (usage, verdict).
        """
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        # --- Empty output ---
        if not stdout:
            if stderr:
                log.warning("claude produced no stdout (exit=%d, stderr_len=%d)",
                            proc.returncode, len(stderr))
            raise MissingResult("no stdout from claude")

        # --- Parse JSON result ---
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            log.warning("stdout is not valid JSON: %.200s", stdout[:200])
            raise MalformedStream("stdout is not valid JSON")

        if not isinstance(result, dict):
            raise MalformedStream(f"expected JSON object, got {type(result).__name__}")

        # --- Validate envelope ---
        if result.get("type") != "result":
            raise MalformedStream(f"unexpected json type: {result.get('type')!r}")

        # --- API error status: check BEFORE is_error/subtype guards so that
        # a real provider 429/5xx (which carries is_error=true, subtype=
        # "error_during_execution", and api_error_status=429) is classified as
        # ApiError for fallthrough rather than MalformedStream.
        api_error_status = result.get("api_error_status")
        if api_error_status is not None and api_error_status != "":
            raise ApiError(str(api_error_status).strip(),
                           f"api_error_status present: {api_error_status}")

        if result.get("is_error", False):
            error_info = result.get("error", "unknown error")
            raise MalformedStream(f"claude returned error: {error_info}")
        if result.get("subtype") != "success":
            raise MalformedStream(f"unexpected subtype: {result.get('subtype')!r}")

        # --- Permission denials (fail-closed) ---
        permission_denials = result.get("permission_denials") or []
        if permission_denials:
            raise MalformedStream(
                f"permission denied: {json.dumps(permission_denials, ensure_ascii=False)}"
            )

        # --- Truncation check ---
        stop_reason = result.get("stop_reason", "") or ""
        if stop_reason in ("length", "max_tokens"):
            raise TruncatedOutput(
                f"claude output truncated (stop_reason={stop_reason})"
            )

        # --- Non-stopping terminal reason (fail-closed if not a proper stop) ---
        # Acceptable: end_turn, stop.  Anything else is suspicious.
        terminal_stop_reasons = {"end_turn", "stop"}
        if stop_reason and stop_reason not in terminal_stop_reasons:
            raise MalformedStream(
                f"non-terminal stop_reason: {stop_reason!r}"
            )

        # --- Terminal reason (from the API envelope, not stop_reason) ---
        terminal_reason = result.get("terminal_reason") or ""
        if terminal_reason:
            acceptable_terminal = {"stop", "end_turn", "completed"}
            if terminal_reason not in acceptable_terminal:
                raise MalformedStream(
                    f"non-terminal terminal_reason: {terminal_reason!r}"
                )



        # --- At least one turn taken ---
        num_turns = result.get("num_turns", 0) or 0
        if num_turns < 1:
            raise MissingResult("no turns taken by claude")

        # --- Extract result text ---
        result_text = (result.get("result") or "").strip()
        if not result_text:
            raise MissingResult("no result text from claude")

        # --- Nonzero exit (always fail-closed, even with a verdict) ---
        if proc.returncode != 0:
            log.warning("claude exit=%d but produced output: %.200s",
                        proc.returncode, result_text[:200])
            raise NonZeroExit(
                f"claude exited {proc.returncode}"
            )

        # --- Extract usage from the usage object ---
        usage_raw = result.get("usage") or {}
        if not isinstance(usage_raw, dict):
            usage_raw = {}
        usage = {
            "input_tokens": int(usage_raw.get("input_tokens", 0) or 0),
            "output_tokens": int(usage_raw.get("output_tokens", 0) or 0),
        }
        # Cached tokens and the response's own cost estimate (RouteGate
        # `usage.estimated_cost`): carried ONLY when the response actually
        # had them — a present-but-null field is NOT REPORTED (unknown stays
        # unknown), never 0, never invented or estimated here.
        if usage_raw.get("cache_read_input_tokens") is not None:
            usage["cached_input"] = int(usage_raw["cache_read_input_tokens"])
        cost = usage_raw.get("estimated_cost")
        if cost is not None:
            try:
                usage["cost_usd"] = float(cost)
            except (TypeError, ValueError):
                log.warning("unusable estimated_cost in usage: %r", cost)

        # --- Extract verdict ---
        verdict = self._extract_verdict(result_text)
        if not verdict:
            log.warning("no VERDICT line in result: %.500s", result_text)
            raise UnknownVerdict("no VERDICT: line in claude output")

        # Normalize to canonical uppercase for consistent phase mapping.
        verdict = verdict.upper()
        # Strip trailing sentence punctuation that would break canonical
        # matching (e.g. "ACCEPT-MERGED." -> "ACCEPT-MERGED").  Preserves
        # hyphens inside ACCEPT-READY, BOUNCE-BUILDER, etc.
        verdict = verdict.rstrip(".,;:!?")

        # --- Validate verdict against canonical set ---
        if verdict not in CANONICAL_VERDICTS:
            log.warning("non-canonical verdict: %r", verdict)
            raise UnknownVerdict(f"non-canonical verdict: {verdict}")

        return usage, verdict

    @staticmethod
    def _extract_verdict(text: str) -> str:
        """Extract the VERDICT: line from the agent's output.

        Only matches VERDICT: at the start of a line (after optional markdown
        prefix like **, *, >, -).  This prevents prose like "Do not output
        VERDICT: ..." from being mistaken for the final verdict.

        Returns the verdict value (with enclosing markdown stripped) or empty
        string if no VERDICT: line is found.
        """
        for line in text.splitlines():
            m = _VERDICT_RE.match(line)
            if m:
                value = m.group(1).strip()
                # Strip enclosing markdown the model sometimes wraps around
                # the verdict value (e.g., **ACCEPT-MERGED** or `BOUNCE-BUILDER`).
                value = value.replace("*", "").replace("`", "").strip()
                if value:
                    return value
        return ""