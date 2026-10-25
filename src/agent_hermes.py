"""Hermes CLI agent for Neo — headless, tool-capable, no raw HTTP.

Replaces the Claude Code CLI engine (agent_claude.py, retired 2026-08-24 —
owner directive: Hermes is the one dispatching harness for this fabric, not
a second CLI). Runs the Neo brief through `hermes -z` (one-shot mode) with
full tool access (gh, git, file edit, test runner) in an isolated per-run
work directory, against an approved Hermes provider profile.

Architecture:
  - Hermes CLI via subprocess: `hermes -z <brief> -m <model> --provider
    <profile> --yolo` (--yolo = approvals auto-bypassed, the Hermes
    equivalent of Claude Code's --dangerously-skip-permissions)
  - Isolated per-run temp work directory (deleted after the run)
  - `-z` prints ONLY the final response text to stdout — no JSON envelope.
    This is a real reduction in fail-closed signal versus the old Claude
    Code engine: there is no stop_reason, permission_denials, or num_turns
    to check. What survives: nonzero exit, empty output, and (as before)
    a canonical VERDICT: line extracted from the text. Truncation detection
    is not implemented — there is no reliable signal for it in this mode,
    and that gap is deliberate rather than faked.
  - Fail-closed on nonzero exit, timeout, missing result, or unknown verdict.

Usage:
    agent = HermesAgent.from_config(cfg, gh_token=installation_token)
    usage, verdict = agent.run(brief)
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import time

from .config import Config
from .neo_protocol import extract_verdict_line
from .usage import Route, UsageAttempt

log = logging.getLogger("rhobear_neo.agent")

CANONICAL_VERDICTS = frozenset({
    "ACCEPT-MERGED",
    "ACCEPT-READY",
    "FIX-FORWARD",
    "BOUNCE-BUILDER",
    "ESCALATE",
})

class AgentError(Exception):
    """Base for all Hermes agent failures — fail-closed root."""


class NonZeroExit(AgentError):
    """CLI exited with a nonzero status."""


class TimeoutError(AgentError):
    """CLI did not finish within the timeout."""


class MissingResult(AgentError):
    """No output produced at all."""


class UnknownVerdict(AgentError):
    """No VERDICT: line found in the agent's output or verdict not canonical."""


class HermesAgent:
    """Run the Neo brief through the Hermes CLI (headless, with tools).

    Each call creates an isolated temp work directory so the agent has a
    clean slate. The directory is deleted after the run — no accumulated
    clone/copy graveyard.

    Thread-safe for sequential calls (each call creates its own subprocess).

    Subclasses may override CANONICAL_VERDICTS with their own output
    contract (BuilderAgent uses {FIXED, NO-CHANGE, BLOCKED}) and/or override
    _parse_output for a different fail-open policy on completed runs.
    """

    CANONICAL_VERDICTS = CANONICAL_VERDICTS

    def __init__(
        self,
        *,
        hermes_bin: str,
        provider: str,
        model: str,
        timeout: int = 1800,
        gh_token: str = "",
    ) -> None:
        self.hermes_bin = hermes_bin
        self.provider = provider.strip()
        self.model = model.strip()
        self.timeout = timeout
        self.gh_token = gh_token
        self._last_proc = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Config, *, gh_token: str) -> HermesAgent:
        """`gh_token` is REQUIRED: production passes the repo's GitHub App
        installation token so the agent env never carries a personal token.
        The cfg.gh_token field is ignored as an identity."""
        return cls(
            hermes_bin=cfg.hermes_bin,
            provider=cfg.hermes_provider,
            model=cfg.hermes_model,
            timeout=cfg.deepseek_timeout,
            gh_token=gh_token,
        )

    # ------------------------------------------------------------------
    # Route chain: subclass hook for _for_route
    # ------------------------------------------------------------------

    @classmethod
    def _for_route_hermes_provider(cls, cfg: Config, route: Route) -> str | None:
        """Resolve the Hermes --provider name for a route.

        Overridden by BuilderAgent to use the builder's own provider config.
        Returns None if the route cannot be served.
        """
        return cfg.hermes_provider_for(route.provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, brief: str) -> tuple[dict[str, int], str]:
        """Run the Neo brief through the Hermes CLI on the configured route.

        Returns (usage, verdict) where usage is {input_tokens, output_tokens}
        (best-effort — see note in _parse_output) and verdict is the
        extracted VERDICT: line value.

        Raises AgentError subclasses on failure — the caller should catch
        and return empty usage + empty verdict.
        """
        with tempfile.TemporaryDirectory(prefix="neo-work-") as work_dir:
            return self._run_in(brief, work_dir)

    # ------------------------------------------------------------------
    # Route chain: walk available routes, fall through on transient errors
    # ------------------------------------------------------------------

    @classmethod
    def _for_route(cls, cfg: Config, route: Route, gh_token: str,
                   hermes_provider: str) -> HermesAgent:
        """Create a HermesAgent for a specific route in the chain.

        `hermes_provider` is the Hermes --provider name for this route.
        """
        return cls(
            hermes_bin=cfg.hermes_bin,
            provider=hermes_provider,
            model=route.model,
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
        attempt_number: int = 1,
    ) -> tuple[dict[str, int], str, int | None, Route | None]:
        """Walk the available route chain, trying each in order.

        For each route:
          - Skip routes Hermes can't serve (no provider mapping).
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
            # Skip routes Hermes can't serve.
            hermes_provider = cfg.hermes_provider_for(route.provider)
            if hermes_provider is None:
                log.info("route %d (%s/%s) skipped for Hermes agent — "
                         "no Hermes provider mapping",
                         idx, route.provider, route.model)
                if usage_store is not None and job_id is not None:
                    attempt = UsageAttempt(
                        job_id=job_id, attempt_number=attempt_number, route_idx=idx,
                        provider=route.provider, model=route.model,
                        outcome="skipped",
                        error_class="no_hermes_provider",
                    )
                    try:
                        usage_store.record_attempt(attempt)
                    except Exception:  # noqa: BLE001
                        log.exception("failed to record skipped route %s", idx)
                continue

            route_agent = self._for_route(cfg, route, gh_token, hermes_provider)

            with tempfile.TemporaryDirectory(prefix="neo-work-") as work_dir:
                route_start = time.monotonic()
                try:
                    usage, verdict = route_agent._run_in(brief, work_dir)
                except AgentError as exc:
                    elapsed_ms = int((time.monotonic() - route_start) * 1000)
                    fallthrough, err_class = self._classify_agent_error(
                        exc, route_agent,
                        proc=getattr(route_agent, '_last_proc', None))
                    log.info("route %d (%s/%s) failed: %s (fallthrough=%s)",
                             idx, route.provider, route.model, exc, fallthrough)
                    # Record the failed attempt.
                    if usage_store is not None and job_id is not None:
                        attempt = UsageAttempt(
                            job_id=job_id, attempt_number=attempt_number, route_idx=idx,
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
                        # Non-fallthrough failure — re-raise so the
                        # caller's except AgentError handler retries
                        # the job rather than treating it as completed.
                        raise exc
                    # Fallthrough — try next route.
                    continue

                # Success.
                elapsed_ms = int((time.monotonic() - route_start) * 1000)
                if usage_store is not None and job_id is not None:
                    attempt = UsageAttempt(
                        job_id=job_id, attempt_number=attempt_number, route_idx=idx,
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
        self, exc: AgentError, agent: HermesAgent,
        proc: subprocess.CompletedProcess | None = None,
    ) -> tuple[bool, str | None]:
        """Classify an agent error into (is_fallthrough, error_class).

        For Hermes output, examines stderr for HTTP status codes.
        Falls back to error type classification.
        """
        # Check stderr for HTTP status codes.
        # Uses anchored patterns to avoid false positives from bare substrings
        # (e.g. "150 files", "Retry-After: 400", "read 400 bytes").
        # Anchors: "HTTP 429", "HTTP: 429", "HTTP/1.1 429 …", "status: 429",
        # "status 429", "Error code: 402 …" (provider SDK error lines).
        # Check timeout FIRST so "timed out after reading 400 bytes" is
        # classified as timeout, not as 400.
        if proc is not None and proc.stderr:
            stderr = proc.stderr
            stderr_lower = stderr.lower()
            status_anchor = (r'(?:\bHTTP(?:/\d(?:\.\d)?)?\s*:?\s*'
                             r'|\bstatus\s*:?\s*|\berror\s+code\s*:?\s*)')

            # Timeout check first (anchored)
            if re.search(r'\btime(?:d\s+)?out\b', stderr_lower):
                return True, "timeout"

            # 429 / 402 (anchored near HTTP/status/error-code)
            if re.search(status_anchor + r'429\b', stderr, re.IGNORECASE):
                return True, "429"
            if re.search(status_anchor + r'402\b', stderr, re.IGNORECASE):
                return True, "402"

            # 5xx (anchored near HTTP/status/error-code)
            if re.search(status_anchor + r'5\d{2}\b', stderr, re.IGNORECASE):
                return True, "5xx"

            # 400 (anchored near HTTP/status/error-code) — does NOT fall through
            if re.search(status_anchor + r'400\b', stderr, re.IGNORECASE):
                return False, "400"
            if re.search(r'\bbad request\b', stderr_lower):
                return False, "400"

        # Fall back to error type classification.
        if isinstance(exc, TimeoutError):
            return True, "timeout"
        if isinstance(exc, NonZeroExit):
            # Nonzero exit without stderr classification — could be a
            # transient provider error; fall through.
            return True, "exit_nonzero"
        # MissingResult, UnknownVerdict — not fallthrough
        return False, type(exc).__name__.lower()

    def _run_in(self, brief: str, work_dir: str) -> tuple[dict[str, int], str]:
        env = self._build_env()
        cmd = self._build_cmd(brief)

        log.debug("running hermes in %s: %s", work_dir, " ".join(cmd[:-1]) + " ...")

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
            log.error("hermes timed out after %.0fs (timeout=%ds)", elapsed, self.timeout)
            raise TimeoutError(f"hermes did not finish within {self.timeout}s")

        elapsed = time.monotonic() - start
        log.info("hermes exit=%d elapsed=%.0fs stdout=%d stderr=%d",
                 proc.returncode, elapsed, len(proc.stdout or ""), len(proc.stderr or ""))

        self._last_proc = proc
        return self._parse_output(proc)

    # ------------------------------------------------------------------
    # Internal: env / cmd
    # ------------------------------------------------------------------

    def _build_env(self) -> dict[str, str]:
        """Build the environment for the Hermes subprocess.

        Preserves the host PATH and HOME. Never logs or echoes credentials —
        Hermes resolves its own provider credentials from its own config, we
        pass nothing sensitive on the command line or here.
        """
        env = {**os.environ}
        env.pop("GH_TOKEN", None)
        if self.gh_token:
            env["GH_TOKEN"] = self.gh_token
        return env

    def _build_cmd(self, brief: str) -> list[str]:
        """Build the Hermes CLI command.

        One-shot mode (-z), the exact approved provider/model, --yolo
        (approvals auto-bypassed — Neo is a dedicated service agent),
        --accept-hooks (headless, no TTY to confirm an unseen shell hook).
        """
        return [
            self.hermes_bin,
            "-z", brief,
            "-m", self.model,
            "--provider", self.provider,
            "--yolo",
            "--accept-hooks",
        ]

    # ------------------------------------------------------------------
    # Internal: output parsing
    # ------------------------------------------------------------------

    def _parse_output(self, proc: subprocess.CompletedProcess[str]) -> tuple[dict[str, int], str]:
        """Parse Hermes -z output.

        -z prints ONLY the final response text — no JSON envelope, no
        stop_reason, no permission_denials, no num_turns. Fail-closed on:
          - Nonzero exit (even with a verdict — the process must exit clean)
          - No output / no result text
          - No VERDICT: line in the result text
          - Non-canonical verdict value

        NOT detected (reduction versus the old Claude Code engine, noted
        rather than faked): output truncation. There is no reliable signal
        for it in one-shot text mode. Token usage is not available from -z
        in the Hermes version currently installed on this box (no
        --usage-file support) — usage is reported as zeros rather than a
        made-up estimate.

        Returns (usage, verdict).
        """
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if not stdout:
            if stderr:
                log.warning("hermes produced no stdout (exit=%d, stderr_len=%d)",
                            proc.returncode, len(stderr))
            raise MissingResult("no stdout from hermes")

        if proc.returncode != 0:
            log.warning("hermes exit=%d but produced output: %.200s",
                        proc.returncode, stdout[:200])
            raise NonZeroExit(f"hermes exited {proc.returncode}")

        verdict = self._extract_verdict(stdout)
        if verdict is None:
            raise UnknownVerdict("no VERDICT: line found in hermes output")
        if verdict not in self.CANONICAL_VERDICTS:
            raise UnknownVerdict(f"non-canonical verdict: {verdict!r}")

        # Usage is not observable from -z on this Hermes version. Reported
        # as zeros — an honest gap, not a fabricated number.
        usage = {"input_tokens": 0, "output_tokens": 0}
        return usage, verdict

    @staticmethod
    def _extract_verdict(text: str) -> str | None:
        """Verdict word from the answer tail — shape-tolerant extraction
        (case-insensitive `VERDICT: <word>` in the last 40 lines, also inside a
        fenced block or after markdown bold) lives in neo_protocol; this lane
        only decides which verdict words it accepts."""
        return extract_verdict_line(text)
