"""Claude Code CLI agent for Neo — headless, tool-capable, no raw HTTP.

Replaces the old OpenRouterClient (direct HTTP, no tools) with a real agentic
CLI loop: Claude Code runs the Neo brief with full tool access (gh, git, file
edit, test runner) in an isolated per-run work directory.

Architecture:
  - Claude Code CLI via subprocess, DeepSeek Direct as Anthropic-compatible
    backend (https://api.deepseek.com/anthropic)
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
import time

from .config import Config
from .usage import (
    Route, UsageAttempt, classify_cli_fallthrough,
)

log = logging.getLogger("rhobear_neo.agent")

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

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Config, *, gh_token: str) -> ClaudeAgent:
        """Build from a Config object.

        `gh_token` is REQUIRED: production passes the repo's GitHub App
        installation token so the agent's GH_TOKEN env is never a personal
        token.  The cfg.gh_token field is ignored as an identity.
        """
        return cls(
            claude_bin=cfg.claude_bin,
            api_key=cfg.deepseek_key,
            base_url=cfg.deepseek_base_url,
            model=cfg.deepseek_model,
            effort=cfg.deepseek_reasoning_effort,
            max_tokens=cfg.deepseek_max_tokens,
            timeout=cfg.deepseek_timeout,
            gh_token=gh_token,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, brief: str) -> tuple[dict[str, int], str]:
        """Run the Neo brief through Claude Code CLI on the configured route.

        Returns (usage, verdict) where usage is normalised to
        {input_tokens, output_tokens} and verdict is the extracted
        VERDICT: line value.

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
                   base_url: str) -> ClaudeAgent:
        """Create a ClaudeAgent for a specific route in the chain.

        `base_url` is the Anthropic-wire endpoint for this provider.
        The API key is read from the route's env var.
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
        )

    def run_with_chain(
        self,
        brief: str,
        available_routes: list[tuple[int, Route]],
        cfg: Config,
        gh_token: str,
        usage_store=None,
        job_id: int | None = None,
    ) -> tuple[dict[str, int], str, int | None, Route | None]:
        """Walk the available route chain, trying each in order.

        For each route:
          - Skip routes with no Anthropic-wire endpoint (logged once).
          - Skip routes whose API key is not set.
          - Launch on route 1.
          - If the run fails with a fallthrough class (429/402/5xx/timeout
            before any output), record the attempt and try the next route.
          - A non-fallthrough failure stops the chain.
          - A successful run returns with its usage, verdict, route info.

        Each launch records one UsageAttempt with its real route (provider,
        model).  Failed attempts record the error class.

        Returns (usage, verdict, route_idx, route).
        usage is the final successful attempt's usage.
        verdict is the final successful verdict.
        route_idx/route are the winning route (None if all failed).
        """
        start_time = time.monotonic()

        for idx, route in available_routes:
            # Skip routes Claude Code can't serve (no Anthropic-wire endpoint).
            base_url = cfg.claude_endpoint_for(route.provider)
            if base_url is None:
                log.info("route %d (%s/%s) skipped for Claude agent — "
                         "no Anthropic-wire endpoint",
                         idx, route.provider, route.model)
                if usage_store is not None and job_id is not None:
                    attempt = UsageAttempt(
                        job_id=job_id, attempt_number=1, route_idx=idx,
                        provider=route.provider, model=route.model,
                        outcome="skipped",
                        error_class="no_anthropic_wire",
                    )
                    try:
                        usage_store.record_attempt(attempt)
                    except Exception:  # noqa: BLE001
                        log.exception("failed to record skipped route %s", idx)
                continue

            # Build per-route agent.
            api_key = cfg.api_key_for(route.provider)
            if not api_key:
                log.info("route %d (%s/%s) skipped — API key not set",
                         idx, route.provider, route.model)
                if usage_store is not None and job_id is not None:
                    attempt = UsageAttempt(
                        job_id=job_id, attempt_number=1, route_idx=idx,
                        provider=route.provider, model=route.model,
                        outcome="skipped",
                        error_class="missing_key",
                    )
                    try:
                        usage_store.record_attempt(attempt)
                    except Exception:  # noqa: BLE001
                        log.exception("failed to record skipped route %s", idx)
                continue

            route_agent = self._for_route(cfg, route, gh_token, base_url)

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
                        # Record the failed attempt.
                        if usage_store is not None and job_id is not None:
                            attempt = UsageAttempt(
                                job_id=job_id, attempt_number=1, route_idx=idx,
                                provider=route.provider, model=route.model,
                                latency_ms=elapsed_ms,
                                outcome="fallthrough" if fallthrough else "error",
                                error_class=err_class,
                            )
                            try:
                                usage_store.record_attempt(attempt)
                            except Exception:  # noqa: BLE001
                                log.exception("failed to record attempt %s", idx)
                        if not fallthrough:
                            # Non-fallthrough failure — stop the chain.
                            return {}, "", idx, route
                        # Fallthrough — try next route.
                        continue

                    # Success.
                    elapsed_ms = int((time.monotonic() - route_start) * 1000)
                    if usage_store is not None and job_id is not None:
                        attempt = UsageAttempt(
                            job_id=job_id, attempt_number=1, route_idx=idx,
                            provider=route.provider, model=route.model,
                            uncached_input=usage.get("input_tokens"),
                            output_tokens=usage.get("output_tokens"),
                            latency_ms=elapsed_ms,
                            outcome="success",
                        )
                        try:
                            usage_store.record_attempt(attempt)
                        except Exception:  # noqa: BLE001
                            log.exception("failed to record success %s", idx)
                    log.info("route %d (%s/%s) succeeded — verdict=%s",
                             idx, route.provider, route.model, verdict)
                    return usage, verdict, idx, route

        # All routes exhausted.
        return {}, "", None, None

    def _classify_agent_error(
        self, exc: AgentError, agent: ClaudeAgent,
        proc: subprocess.CompletedProcess | None = None,
    ) -> tuple[bool, str | None]:
        """Classify an agent error into (is_fallthrough, error_class).

        For Claude Code JSON output, examines the JSON envelope's
        api_error_status to determine the HTTP status code.
        Falls back to error type classification.
        """
        # Try to extract HTTP status from the JSON output first.
        if proc is not None and proc.stdout:
            try:
                result = json.loads(proc.stdout.strip())
                if isinstance(result, dict):
                    api_status = result.get("api_error_status")
                    if api_status is not None and api_status != "":
                        err_str = str(api_status).strip()
                        from .usage import classify_http_error, is_fallthrough
                        # Try to extract numeric status code
                        for code_str in ("429", "402", "400"):
                            if code_str in err_str:
                                return is_fallthrough(int(code_str)), \
                                    classify_http_error(int(code_str))
                        if any(f"{i}0" in err_str for i in range(5, 10)):
                            return True, "5xx"
                        if "timeout" in err_str.lower():
                            return True, "timeout"
                        # Unknown API error — not fallthrough
                        return False, err_str[:32]
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # ApiError — extract status code from the exception.
        if isinstance(exc, ApiError):
            from .usage import classify_http_error, is_fallthrough
            err_str = exc.status
            for code_str in ("429", "402", "400"):
                if code_str in err_str:
                    return is_fallthrough(int(code_str)), \
                        classify_http_error(int(code_str))
            if any(f"{i}0" in err_str for i in range(5, 10)):
                return True, "5xx"
            if "timeout" in err_str.lower():
                return True, "timeout"
            return False, err_str[:32]

        # Fall back to error type classification.
        if isinstance(exc, TimeoutError):
            return True, "timeout"
        if isinstance(exc, NonZeroExit):
            return True, "exit_nonzero"
        # TruncatedOutput, MissingResult, MalformedStream, UnknownVerdict
        # are all non-fallthrough — the model ran but produced bad output.
        return False, type(exc).__name__.lower()

    def _run_in(
        self, brief: str, work_dir: str, config_dir: str,
    ) -> tuple[dict[str, int], str]:
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
        compatibility.  Preserves the host PATH and HOME.  Never logs or echoes
        the API key.
        """
        env = {
            **os.environ,
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
        }
        # Use only the configured GitHub credential. A stale host token must
        # never leak into Neo when its explicit token is absent.
        env.pop("GH_TOKEN", None)
        if self.gh_token:
            env["GH_TOKEN"] = self.gh_token
        # Remove conflicting env vars that might point at a different provider
        # (e.g. a stale OpenRouter gateway URL or the CLI's own base-url var).
        env.pop("CLAUDE_CODE_ANTHROPIC_BASE_URL", None)
        env.pop("CLAUDE_CODE_BASE_URL", None)
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
    ) -> tuple[dict[str, int], str]:
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

        # --- API error status (fail-closed on any non-null status) ---
        api_error_status = result.get("api_error_status")
        if api_error_status is not None and api_error_status != "":
            raise ApiError(str(api_error_status).strip(),
                           f"api_error_status present: {api_error_status}")

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