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
    agent = HermesAgent.from_config(cfg)
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

log = logging.getLogger("rhobear_neo.agent")

CANONICAL_VERDICTS = frozenset({
    "ACCEPT-MERGED",
    "ACCEPT-READY",
    "FIX-FORWARD",
    "BOUNCE-BUILDER",
    "ESCALATE",
})

_VERDICT_RE = re.compile(
    r'^\s*(?:[*]{1,2}|>`?\s*|-\s+)?VERDICT:\s*(.*?)(?:\s*[*`]+)?\s*$',
    re.IGNORECASE,
)


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
    """

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

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Config) -> HermesAgent:
        return cls(
            hermes_bin=cfg.hermes_bin,
            provider=cfg.hermes_provider,
            model=cfg.hermes_model,
            timeout=cfg.deepseek_timeout,
            gh_token=cfg.gh_token,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, brief: str) -> tuple[dict[str, int], str]:
        """Run the Neo brief through the Hermes CLI.

        Returns (usage, verdict) where usage is {input_tokens, output_tokens}
        (best-effort — see note in _parse_output) and verdict is the
        extracted VERDICT: line value.

        Raises AgentError subclasses on failure — the caller should catch
        and return empty usage + empty verdict.
        """
        with tempfile.TemporaryDirectory(prefix="neo-work-") as work_dir:
            return self._run_in(brief, work_dir)

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
        if verdict not in CANONICAL_VERDICTS:
            raise UnknownVerdict(f"non-canonical verdict: {verdict!r}")

        # Usage is not observable from -z on this Hermes version. Reported
        # as zeros — an honest gap, not a fabricated number.
        usage = {"input_tokens": 0, "output_tokens": 0}
        return usage, verdict

    @staticmethod
    def _extract_verdict(text: str) -> str | None:
        for line in text.splitlines():
            m = _VERDICT_RE.match(line)
            if m:
                return m.group(1).strip().upper()
        return None
