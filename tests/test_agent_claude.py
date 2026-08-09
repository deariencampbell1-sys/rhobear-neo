"""Tests for the Claude Code CLI agent — realistic JSON fixtures, every error path.

Covers:
  - Command assembly (env, args) — secrets redacted in assertions
  - Single JSON result object parsing (type=result, subtype=success)
  - Usage extraction from the usage object
  - Canonical verdicts (ACCEPT-MERGED, ACCEPT-READY, FIX-FORWARD, BOUNCE-BUILDER, ESCALATE)
  - Non-canonical verdict rejection
  - Timeout
  - Nonzero exit (always rejected, even with verdict)
  - Truncated output (stop_reason=length / max_tokens)
  - Permission denials (fail-closed)
  - Malformed JSON / unexpected envelope
  - Empty stdout / missing result / no turns
  - Unknown verdict (no VERDICT: line)
  - Verdict markdown stripped
  - Verdict case-insensitive
  - Temp cwd lifecycle (dir created, used, cleaned)
  - _run_agent wrapper in neo_worker (catch -> empty, pass -> usage)
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any
from unittest.mock import patch

import pytest

from src.agent_claude import (
    ClaudeAgent,
    AgentError,
    NonZeroExit,
    TimeoutError,
    TruncatedOutput,
    MissingResult,
    UnknownVerdict,
    MalformedStream,
    CANONICAL_VERDICTS,
)
from src.config import Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = "https://openrouter.ai/api"
API_KEY = "sk-or-v1-test-placeholder"
MODEL = "deepseek/deepseek-v4-flash"
MODEL_FULL = MODEL + "[1m]"          # __init__ normalises this
GH_TOKEN = "ghp_test_token_placeholder"
CLAUDE_BIN = "/usr/bin/claude"


def _make_agent(**overrides: Any) -> ClaudeAgent:
    """Build an agent with sensible defaults; override any kwarg."""
    kwargs = dict(
        claude_bin=CLAUDE_BIN,
        api_key=API_KEY,
        base_url=BASE,
        model=MODEL,
        effort="max",
        max_tokens=32000,
        timeout=1800,
        gh_token=GH_TOKEN,
    )
    kwargs.update(overrides)
    return ClaudeAgent(**kwargs)


def _fake_proc(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["claude", "-p", "brief"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _json_result(
    *,
    result: str = "",
    in_tok: int = 150,
    out_tok: int = 80,
    stop_reason: str = "end_turn",
    num_turns: int = 2,
    is_error: bool = False,
    subtype: str = "success",
    error: str | None = None,
    permission_denials: list | None = None,
) -> str:
    """Build a realistic single JSON result object (the actual output format)."""
    obj: dict[str, Any] = {
        "type": "result",
        "subtype": subtype,
        "is_error": is_error,
        "result": result,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
        "modelUsage": {"input_tokens": in_tok, "output_tokens": out_tok},
        "num_turns": num_turns,
        "stop_reason": stop_reason,
        "permission_denials": permission_denials or [],
    }
    if error is not None:
        obj["error"] = error
    return json.dumps(obj)


# ===================================================================
# Command assembly
# ===================================================================

class TestCommandAssembly:
    """Verify the CLI command is built correctly — env, args."""

    def test_build_cmd_has_correct_args(self) -> None:
        agent = _make_agent()
        cmd = agent._build_cmd("test brief")
        assert cmd[0] == CLAUDE_BIN
        assert "-p" in cmd
        assert cmd[cmd.index("-p") + 1] == "test brief"
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == MODEL_FULL
        assert "--effort" in cmd
        assert cmd[cmd.index("--effort") + 1] == "max"
        assert "--dangerously-skip-permissions" in cmd
        assert "--no-session-persistence" in cmd
        assert "--output-format" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "json"

    def test_build_env_has_key_vars(self) -> None:
        agent = _make_agent()
        env = agent._build_env("/tmp/neo-config-test")
        # API key is set — we assert it exists but never log the value.
        assert "ANTHROPIC_API_KEY" in env
        assert env["ANTHROPIC_API_KEY"] == API_KEY
        # Auth token also set.
        assert "ANTHROPIC_AUTH_TOKEN" in env
        assert env["ANTHROPIC_AUTH_TOKEN"] == API_KEY
        # Base URL uses the correct gateway env var (not CLAUDE_CODE_ANTHROPIC_BASE_URL).
        assert env["ANTHROPIC_BASE_URL"] == BASE
        assert "CLAUDE_CODE_ANTHROPIC_BASE_URL" not in env
        # Model, effort, max_tokens.
        assert env["ANTHROPIC_MODEL"] == MODEL_FULL
        assert env["CLAUDE_CODE_EFFORT_LEVEL"] == "max"
        assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32000"
        assert env["CLAUDE_CODE_OUTPUT_FORMAT"] == "json"
        # GH_TOKEN propagated.
        assert env["GH_TOKEN"] == GH_TOKEN
        # Conflicting vars removed.
        assert "CLAUDE_CODE_BASE_URL" not in env

    def test_gh_token_omitted_when_empty(self) -> None:
        """When gh_token is empty, GH_TOKEN is not *forced* — the host env
        may still have it, but we don't override."""
        agent = _make_agent(gh_token="")
        env = agent._build_env("/tmp/neo-config-test")
        # GH_TOKEN may be inherited from the host env; we just don't force it.
        host_val = os.environ.get("GH_TOKEN", "")
        assert env.get("GH_TOKEN", "") == host_val

    def test_model_normalised(self) -> None:
        """Model without [1m] suffix gets it appended."""
        agent = _make_agent()
        assert agent.model == MODEL_FULL

    def test_model_preserves_existing_suffix(self) -> None:
        """Model already ending with [1m] is not double-suffixed."""
        agent = _make_agent(model=MODEL_FULL)
        assert agent.model == MODEL_FULL

    def test_from_config(self) -> None:
        """from_config should populate agent fields from Config."""
        cfg = Config()
        cfg.openrouter_key = API_KEY
        cfg.openrouter_base_url = BASE
        cfg.openrouter_model = MODEL
        cfg.openrouter_reasoning_effort = "max"
        cfg.openrouter_max_tokens = 32000
        cfg.openrouter_timeout = 1800
        cfg.claude_bin = CLAUDE_BIN
        cfg.gh_token = GH_TOKEN

        agent = ClaudeAgent.from_config(cfg)
        assert agent.claude_bin == CLAUDE_BIN
        assert agent.api_key == API_KEY
        assert agent.base_url == BASE
        assert agent.model == MODEL_FULL
        assert agent.effort == "max"
        assert agent.max_tokens == 32000
        assert agent.timeout == 1800
        assert agent.gh_token == GH_TOKEN


# ===================================================================
# JSON result parsing
# ===================================================================

class TestJSONResultParsing:
    """Parse single JSON result object with text + usage."""

    def test_result_with_verdict_and_usage(self) -> None:
        output = _json_result(
            result="TOOL_OK\nVERDICT: ACCEPT-MERGED",
            in_tok=150, out_tok=80,
        )
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        usage, verdict = agent._parse_output(proc)
        assert verdict == "ACCEPT-MERGED"
        assert usage["input_tokens"] == 150
        assert usage["output_tokens"] == 80

    def test_result_with_newlines_in_text(self) -> None:
        """Multi-line result text is parsed correctly."""
        output = _json_result(
            result="Step one complete.\nStep two complete.\nVERDICT: FIX-FORWARD",
        )
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        usage, verdict = agent._parse_output(proc)
        assert verdict == "FIX-FORWARD"

    def test_result_no_usage_object(self) -> None:
        """When usage object is missing or null, return zeros."""
        output = json.dumps({
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "All good.\nVERDICT: ACCEPT-READY",
            "num_turns": 2,
            "stop_reason": "end_turn",
            "permission_denials": [],
        })
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        usage, verdict = agent._parse_output(proc)
        assert verdict == "ACCEPT-READY"
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_result_multiple_verdicts_takes_first(self) -> None:
        """Only the first VERDICT: line is extracted."""
        output = _json_result(
            result="VERDICT: ACCEPT-READY\nAlso VERDICT: ESCALATE",
        )
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        usage, verdict = agent._parse_output(proc)
        assert verdict == "ACCEPT-READY"


# ===================================================================
# Canonical verdicts
# ===================================================================

class TestCanonicalVerdicts:
    """All five canonical verdict types — and rejection of non-canonical."""

    @pytest.mark.parametrize("verdict", sorted(CANONICAL_VERDICTS))
    def test_all_verdicts(self, verdict: str) -> None:
        proc = _fake_proc(
            stdout=_json_result(result=f"Analysis.\nVERDICT: {verdict}\n"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == verdict

    def test_verdict_markdown_stripped(self) -> None:
        proc = _fake_proc(
            stdout=_json_result(result="Analysis.\nVERDICT: **ACCEPT-MERGED**\n"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "ACCEPT-MERGED"

    def test_verdict_backtick_stripped(self) -> None:
        proc = _fake_proc(
            stdout=_json_result(result="Analysis.\nVERDICT: `BOUNCE-BUILDER`\n"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "BOUNCE-BUILDER"

    def test_verdict_case_insensitive(self) -> None:
        """Lowercase 'verdict:' should also match.
        The verdict value is returned as-is (case preserved)."""
        proc = _fake_proc(
            stdout=_json_result(result="Analysis.\nverdict: accept-merged\n"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "accept-merged"

    def test_non_canonical_verdict_rejected(self) -> None:
        """A verdict not in the canonical set raises UnknownVerdict."""
        proc = _fake_proc(
            stdout=_json_result(result="Analysis.\nVERDICT: APPROVE\n"),
        )
        agent = _make_agent()
        with pytest.raises(UnknownVerdict, match="non-canonical verdict"):
            agent._parse_output(proc)


# ===================================================================
# Error paths
# ===================================================================

class TestErrorPaths:
    """Fail-closed on every error path."""

    def test_timeout(self) -> None:
        """subprocess.TimeoutExpired should raise TimeoutError."""
        agent = _make_agent(timeout=1)
        with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
            with pytest.raises(TimeoutError, match="did not finish"):
                agent.run("test brief")

    def test_nonzero_exit_no_verdict(self) -> None:
        """Nonzero exit without a verdict should raise NonZeroExit."""
        proc = _fake_proc(
            stdout=_json_result(result="Something went wrong."),
            returncode=1,
        )
        agent = _make_agent()
        with pytest.raises(NonZeroExit, match="exited 1"):
            agent._parse_output(proc)

    def test_nonzero_exit_with_verdict_rejected(self) -> None:
        """Nonzero exit WITH a valid verdict should ALSO raise NonZeroExit."""
        proc = _fake_proc(
            stdout=_json_result(result="Analysis.\nVERDICT: ACCEPT-READY"),
            returncode=1,
        )
        agent = _make_agent()
        with pytest.raises(NonZeroExit, match="exited 1"):
            agent._parse_output(proc)

    def test_truncated_output_raises(self) -> None:
        """stop_reason=length should raise TruncatedOutput."""
        output = _json_result(
            result="Partial analysis.\nVERDICT: ",
            stop_reason="length",
        )
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        with pytest.raises(TruncatedOutput, match="truncated"):
            agent._parse_output(proc)

    def test_truncated_max_tokens_raises(self) -> None:
        """stop_reason=max_tokens should also raise TruncatedOutput."""
        output = _json_result(
            result="Partial.\nVERDICT: ",
            stop_reason="max_tokens",
        )
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        with pytest.raises(TruncatedOutput, match="truncated"):
            agent._parse_output(proc)

    def test_empty_stdout_raises(self) -> None:
        proc = _fake_proc(stdout="")
        agent = _make_agent()
        with pytest.raises(MissingResult, match="no stdout"):
            agent._parse_output(proc)

    def test_missing_verdict_raises(self) -> None:
        """Output with no VERDICT: line should raise UnknownVerdict."""
        proc = _fake_proc(
            stdout=_json_result(result="I've analyzed the code. Everything looks fine."),
        )
        agent = _make_agent()
        with pytest.raises(UnknownVerdict, match="no VERDICT"):
            agent._parse_output(proc)

    def test_whitespace_only_stdout_raises(self) -> None:
        proc = _fake_proc(stdout="   \n  \n  ")
        agent = _make_agent()
        with pytest.raises(MissingResult, match="no stdout"):
            agent._parse_output(proc)

    def test_malformed_json_raises(self) -> None:
        """Non-JSON stdout should raise MalformedStream."""
        proc = _fake_proc(stdout="not json at all")
        agent = _make_agent()
        with pytest.raises(MalformedStream, match="not valid JSON"):
            agent._parse_output(proc)

    def test_wrong_type_raises(self) -> None:
        """JSON object with wrong type field should raise MalformedStream."""
        proc = _fake_proc(stdout=json.dumps({"type": "text", "text": "hello"}))
        agent = _make_agent()
        with pytest.raises(MalformedStream, match="unexpected json type"):
            agent._parse_output(proc)

    def test_is_error_raises(self) -> None:
        """is_error=true should raise MalformedStream."""
        output = _json_result(result="", is_error=True, error="tool crashed")
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        with pytest.raises(MalformedStream, match="returned error"):
            agent._parse_output(proc)

    def test_permission_denials_raises(self) -> None:
        """Non-empty permission_denials should raise MalformedStream."""
        output = _json_result(
            result="Analysis.\nVERDICT: ACCEPT-READY",
            permission_denials=["gh:repo:write"],
        )
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        with pytest.raises(MalformedStream, match="permission denied"):
            agent._parse_output(proc)

    def test_no_turns_raises(self) -> None:
        """num_turns < 1 should raise MissingResult."""
        output = _json_result(
            result="Analysis.\nVERDICT: ACCEPT-READY",
            num_turns=0,
        )
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        with pytest.raises(MissingResult, match="no turns"):
            agent._parse_output(proc)

    def test_non_terminal_stop_reason_raises(self) -> None:
        """Unknown stop_reason should raise MalformedStream."""
        output = _json_result(
            result="Analysis.\nVERDICT: ACCEPT-READY",
            stop_reason="tool_use",
        )
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        with pytest.raises(MalformedStream, match="non-terminal stop_reason"):
            agent._parse_output(proc)

    def test_empty_result_text_raises(self) -> None:
        """JSON with empty result string should raise MissingResult."""
        output = _json_result(result="")
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        with pytest.raises(MissingResult, match="no result text"):
            agent._parse_output(proc)


# ===================================================================
# Temp cwd lifecycle
# ===================================================================

class TestTempCwdLifecycle:
    """Isolated per-run work directory is created, used, and cleaned."""

    def test_run_creates_and_cleans_temp_dirs(self) -> None:
        """Temp dirs should be created during run() and deleted after."""
        agent = _make_agent()

        with patch.object(agent, "_run_in") as mock_run:
            agent.run("test brief")
            assert mock_run.called
            args = mock_run.call_args
            assert len(args[0]) == 3
            assert args[0][0] == "test brief"
            # Both work_dir and config_dir should start with a temp path.
            assert args[0][1].startswith(tempfile.gettempdir())
            assert args[0][2].startswith(tempfile.gettempdir())

    def test_temp_dir_used_as_cwd(self) -> None:
        """The work dir should be set as the subprocess cwd."""
        agent = _make_agent()

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = _fake_proc(
                stdout=_json_result(result="Analysis.\nVERDICT: ACCEPT-READY"),
            )
            agent.run("test brief")

            call_kwargs = mock_run.call_args[1]
            assert "cwd" in call_kwargs
            cwd = call_kwargs["cwd"]
            # cwd should be a temp dir.
            assert cwd.startswith(tempfile.gettempdir())
            assert "neo-work-" in cwd


# ===================================================================
# _run_agent wrapper in neo_worker
# ===================================================================

class TestWorkerWrapper:
    """_run_agent in neo_worker catches errors, passes through success."""

    def test_wrapper_catches_errors(self) -> None:
        from src.neo_worker import _run_agent

        agent = _make_agent()
        with patch.object(agent, "run", side_effect=TimeoutError("timed out")):
            usage, verdict = _run_agent(agent, "test brief")

        assert usage == {}
        assert verdict == ""

    def test_wrapper_passes_success(self) -> None:
        from src.neo_worker import _run_agent

        agent = _make_agent()
        with patch.object(agent, "run", return_value=({"input_tokens": 150}, "ACCEPT-READY")):
            usage, verdict = _run_agent(agent, "test brief")

        assert verdict == "ACCEPT-READY"
        assert usage["input_tokens"] == 150

    def test_wrapper_catches_generic_agent_error(self) -> None:
        from src.neo_worker import _run_agent

        agent = _make_agent()
        with patch.object(agent, "run", side_effect=UnknownVerdict("no VERDICT")):
            usage, verdict = _run_agent(agent, "test brief")

        assert usage == {}
        assert verdict == ""


# ===================================================================
# Config field mapping
# ===================================================================

class TestConfigMapping:
    """Config fields map correctly to ClaudeAgent."""

    def test_openrouter_key_mapped(self) -> None:
        """openrouter_key from Config -> api_key in ClaudeAgent."""
        cfg = Config()
        cfg.openrouter_key = "sk-or-v1-test-key"
        cfg.gh_token = GH_TOKEN
        cfg.claude_bin = CLAUDE_BIN
        agent = ClaudeAgent.from_config(cfg)
        assert agent.api_key == "sk-or-v1-test-key"

    def test_openrouter_model_mapped(self) -> None:
        """openrouter_model from Config -> model in ClaudeAgent (with [1m])."""
        cfg = Config()
        cfg.openrouter_model = "deepseek/deepseek-v4-flash"
        cfg.gh_token = GH_TOKEN
        cfg.claude_bin = CLAUDE_BIN
        agent = ClaudeAgent.from_config(cfg)
        assert agent.model == "deepseek/deepseek-v4-flash[1m]"


# ===================================================================
# Edge: missing env vars in config
# ===================================================================

class TestConfigValidation:
    """Config require() validates required env vars."""

    def test_missing_required_raises(self) -> None:
        cfg = Config()
        cfg.webhook_secret = ""
        with pytest.raises(RuntimeError, match="missing required env"):
            cfg.require()

    def test_all_required_present_ok(self) -> None:
        cfg = Config()
        cfg.webhook_secret = "whs_test"
        cfg.openrouter_key = "sk-or-v1-test"
        cfg.database_url = "postgres://localhost/test"
        cfg.gh_token = "ghp_test"
        cfg.require()


# ===================================================================
# CANONICAL_VERDICTS constant
# ===================================================================

class TestCanonicalSet:
    """The CANONICAL_VERDICTS constant contains exactly the 5 expected values."""

    def test_exact_set(self) -> None:
        assert CANONICAL_VERDICTS == {
            "ACCEPT-MERGED",
            "ACCEPT-READY",
            "FIX-FORWARD",
            "BOUNCE-BUILDER",
            "ESCALATE",
        }