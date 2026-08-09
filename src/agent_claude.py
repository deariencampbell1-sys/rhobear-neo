"""Claude Code CLI agent for Neo — headless, tool-capable, no raw HTTP.

Replaces the old OpenRouterClient (direct HTTP, no tools) with a real agentic
CLI loop: Claude Code runs the Neo brief with full tool access (gh, git, file
edit, test runner) in an isolated per-run work directory.

Architecture:
  - Claude Code CLI via subprocess, OpenRouter as Anthropic-compatible backend
  - Isolated CLAUDE_CONFIG_DIR (pre-armed with a settings file that allows all
    tools — this is a dedicated service, not a shared login shell)
  - Per-run temp work directory (gh repo clone, git, editor, test runner live
    here; deleted after the run)
  - NDJSON output parsing for deterministic text + usage extraction
  - Fail-closed on nonzero exit, timeout, missing result, unknown verdict,
    or malformed stream

Usage:
    agent = ClaudeAgent.from_config(cfg)
    usage, verdict = agent.run(brief)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time

from .config import Config

log = logging.getLogger("rhobear_neo.agent")

# Claude Code settings file content — pre-authorises all tools since Neo is a
# dedicated service agent (not a shared login shell).  Written into the isolated
# CLAUDE_CONFIG_DIR at each run.
CLAUDE_SETTINGS = json.dumps({
    "permissions": {
        "*": True,
    },
    "allow_all_tools": True,
    "disablePromptCache": True,       # no cache between runs — each brief is unique
    "verbose": False,                  # keep stderr quiet
}, indent=2)


class AgentError(Exception):
    """Base for all Claude Code agent failures — fail-closed root."""


class NonZeroExit(AgentError):
    """CLI exited with a nonzero status."""


class TimeoutError(AgentError):
    """CLI did not finish within the timeout."""


class TruncatedOutput(AgentError):
    """Output was truncated (max_tokens hit)."""


class MissingResult(AgentError):
    """No output produced at all."""


class MalformedStream(AgentError):
    """NDJSON output could not be parsed."""


class UnknownVerdict(AgentError):
    """No VERDICT: line found in the agent's output."""


class ClaudeAgent:
    """Run the Neo brief through Claude Code CLI (headless, with tools).

    Each call creates an isolated temp work directory + temp CLAUDE_CONFIG_DIR
    so the agent has a clean slate.  The work directory is deleted after the
    run — no accumulated clone/copy graveyard.

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
        timeout: int = 300,
        gh_token: str = "",
    ) -> None:
        self.claude_bin = claude_bin
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.gh_token = gh_token

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Config) -> ClaudeAgent:
        """Build from a Config object."""
        return cls(
            claude_bin=cfg.claude_bin,
            api_key=cfg.openrouter_key,
            base_url=cfg.openrouter_base_url,
            model=cfg.openrouter_model,
            effort=cfg.openrouter_reasoning_effort,
            max_tokens=cfg.openrouter_max_tokens,
            timeout=cfg.openrouter_timeout,
            gh_token=cfg.gh_token,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, brief: str) -> tuple[dict[str, int], str]:
        """Run the Neo brief through Claude Code CLI.

        Returns (usage, verdict) where usage is normalised to
        {input_tokens, output_tokens} and verdict is the extracted
        VERDICT: line value.

        Raises AgentError subclasses on failure — the caller should
        catch and return empty usage + empty verdict.
        """
        # Temp work directory — the agent clones/clones/edit/test here.
        with tempfile.TemporaryDirectory(prefix="neo-work-") as work_dir:
            # Temp config directory — isolated Claude Code settings.
            with tempfile.TemporaryDirectory(prefix="neo-config-") as config_dir:
                return self._run_in(brief, work_dir, config_dir)

    def _run_in(
        self, brief: str, work_dir: str, config_dir: str,
    ) -> tuple[dict[str, int], str]:
        """Run inside pre-created temp directories."""
        self._write_settings(config_dir)
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

        return self._parse_output(proc)

    # ------------------------------------------------------------------
    # Internal: env / cmd / settings
    # ------------------------------------------------------------------

    def _write_settings(self, config_dir: str) -> None:
        """Write claude_settings.json into the isolated config dir."""
        cfg_path = os.path.join(config_dir, "claude_settings.json")
        with open(cfg_path, "w") as f:
            f.write(CLAUDE_SETTINGS)
        log.debug("wrote %s", cfg_path)

    def _build_env(self, config_dir: str) -> dict[str, str]:
        """Build the environment for the Claude Code subprocess.

        Preserves the host PATH and HOME but overrides everything Claude Code
        cares about.  Never logs or echoes the API key.
        """
        env = {
            **os.environ,
            "CLAUDE_CONFIG_DIR": config_dir,
            "CLAUDE_CODE_ANTHROPIC_BASE_URL": self.base_url,
            "ANTHROPIC_API_KEY": self.api_key,
            "ANTHROPIC_MODEL": self.model,
            "CLAUDE_CODE_EFFORT_LEVEL": self.effort,
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self.max_tokens),
            "CLAUDE_CODE_OUTPUT_FORMAT": "json",
        }
        # Propagate GH_TOKEN for the agent's gh/git operations.
        if self.gh_token:
            env["GH_TOKEN"] = self.gh_token
        # Remove conflicting env vars that might point at a different provider.
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("CLAUDE_CODE_BASE_URL", None)
        return env

    def _build_cmd(self, brief: str) -> list[str]:
        """Build the Claude Code CLI command.

        Runs in one-shot prompt mode (-p) with the exact model and max effort.
        No runtime probe/fallback — the provider already accepted max effort.
        """
        return [
            self.claude_bin,
            "-p", brief,
            "--model", self.model,
            "--effort", self.effort,
            "--output-format", "json",
        ]

    # ------------------------------------------------------------------
    # Internal: output parsing
    # ------------------------------------------------------------------

    def _parse_output(
        self, proc: subprocess.CompletedProcess[str],
    ) -> tuple[dict[str, int], str]:
        """Parse the Claude Code CLI output.

        Handles NDJSON output format (each line is a JSON event with a 'type'
        field).  Falls back to treating the entire stdout as response text if
        NDJSON parsing fails.

        Fail-closed on:
          - Nonzero exit (unless we got a valid verdict)
          - Truncation (finish_reason=length or max_tokens edge)
          - No output (stdout empty)
          - No VERDICT: line in the response
          - Malformed NDJSON

        Returns (usage, verdict).
        """
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        # --- Empty output ---
        if not stdout:
            # If stderr has content, log it for debugging.
            if stderr:
                log.warning("claude produced no stdout; stderr=%.500s", stderr)
            raise MissingResult("no stdout from claude")

        # --- Parse NDJSON ---
        text_parts: list[str] = []
        usage: dict[str, int] = {}
        truncated = False
        ndjson_ok = False

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Not NDJSON — fall back to treating stdout as raw text.
                log.debug("stdout is not NDJSON (line not JSON: %.120s)", line)
                text_parts = []
                break

            ndjson_ok = True
            event_type = event.get("type", "")

            if event_type == "text":
                text_parts.append(event.get("text", ""))
            elif event_type == "usage":
                usage = {
                    "input_tokens": int(event.get("input_tokens", 0) or 0),
                    "output_tokens": int(event.get("output_tokens", 0) or 0),
                }
            elif event_type == "error":
                log.error("claude error event: %s", event.get("error", "unknown"))
            elif event_type == "finish":
                reason = event.get("reason", "")
                if reason == "length" or reason == "max_tokens":
                    truncated = True

        # --- Fallback: treat stdout as raw text ---
        if not ndjson_ok and not text_parts:
            text_parts = [stdout]

        full_text = "\n".join(text_parts).strip()

        # --- Missing result ---
        if not full_text:
            raise MissingResult("no response text from claude")

        # --- Truncation ---
        if truncated:
            raise TruncatedOutput(
                "claude output truncated (finish_reason=length / max_tokens)"
            )

        # --- Nonzero exit (even with output, fail closed unless we have a verdict) ---
        if proc.returncode != 0:
            log.warning("claude exit=%d but produced output: %.200s",
                        proc.returncode, full_text[:200])
            # If we have output AND a verdict, let it through — the model may
            # have finished its work before the CLI hit a tool-call edge.
            verdict = self._extract_verdict(full_text)
            if not verdict:
                raise NonZeroExit(
                    f"claude exited {proc.returncode} with no verdict"
                )
            # Use the usage we collected (or empty).
            return usage, verdict

        # --- Extract verdict ---
        verdict = self._extract_verdict(full_text)
        if not verdict:
            # Log the first 500 chars of output for debugging.
            log.warning("no VERDICT line in output: %.500s", full_text)
            raise UnknownVerdict("no VERDICT: line in claude output")

        return usage, verdict

    @staticmethod
    def _extract_verdict(text: str) -> str:
        """Extract the VERDICT: line from the agent's output.

        Returns the verdict value (with markdown stripped) or empty string
        if no VERDICT: line is found.
        """
        for line in text.splitlines():
            stripped = line.strip()
            # Case-insensitive match for "VERDICT:".
            upper = stripped.upper()
            idx = upper.find("VERDICT:")
            if idx < 0:
                continue
            after = stripped[idx + 8:]  # after "VERDICT:"
            verdict = after.strip()
            # Strip markdown formatting the model sometimes wraps around the
            # verdict value.
            verdict = verdict.replace("*", "").replace("`", "").strip()
            if verdict:
                return verdict
        return ""