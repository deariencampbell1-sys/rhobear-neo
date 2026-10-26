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
import sys
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
    ApiError,
    CANONICAL_VERDICTS,
)
from src.config import Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = "https://api.deepseek.com/anthropic"
API_KEY = "sk-test-placeholder"
MODEL = "deepseek-v4-flash"          # exact direct ID — no prefix, no [1m] suffix
GH_TOKEN = "test_token_placeholder"
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
        assert cmd[cmd.index("--model") + 1] == MODEL
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
        # CLAUDE_CONFIG_DIR is set to the temp config dir.
        assert env["CLAUDE_CONFIG_DIR"] == "/tmp/neo-config-test"
        # Model in env vars is the exact direct ID (no [1m] suffix).
        assert env["ANTHROPIC_MODEL"] == MODEL
        assert env["CLAUDE_CODE_EFFORT_LEVEL"] == "max"
        assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32000"
        assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1048576"
        assert env["CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT"] == "1"
        assert env["CLAUDE_CODE_OUTPUT_FORMAT"] == "json"
        # GH_TOKEN propagated.
        assert env["GH_TOKEN"] == GH_TOKEN
        # Conflicting vars removed.
        assert "CLAUDE_CODE_BASE_URL" not in env

    def test_gh_token_omitted_when_empty(self) -> None:
        """An empty configured token must not inherit a stale host token."""
        with patch.dict(os.environ, {"GH_TOKEN": "stale-host-token"}):
            agent = _make_agent(gh_token="")
            env = agent._build_env("/tmp/neo-config-test")
        assert "GH_TOKEN" not in env

    def test_model_passed_exactly(self) -> None:
        """The direct model ID is used verbatim in every active path — CLI
        --model, ANTHROPIC_MODEL env var, and the agent's own model field.
        No /deepseek prefix is added and no [1m] context suffix is appended
        (that was an OpenRouter CLI hint; DeepSeek Direct takes the bare ID)."""
        agent = _make_agent()
        assert agent.model == MODEL
        env = agent._build_env("/tmp/neo-config-test")
        assert env["ANTHROPIC_MODEL"] == MODEL
        cmd = agent._build_cmd("brief")
        assert cmd[cmd.index("--model") + 1] == MODEL
        assert "[1m]" not in env["ANTHROPIC_MODEL"]
        assert "[1m]" not in agent.model

    def test_from_config(self) -> None:
        """from_config should populate agent fields from Config."""
        cfg = Config()
        cfg.deepseek_key = API_KEY
        cfg.deepseek_base_url = BASE
        cfg.deepseek_model = MODEL
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.deepseek_timeout = 1800
        cfg.claude_bin = CLAUDE_BIN
        cfg.gh_token = GH_TOKEN

        agent = ClaudeAgent.from_config(cfg, gh_token=GH_TOKEN)
        assert agent.claude_bin == CLAUDE_BIN
        assert agent.api_key == API_KEY
        assert agent.base_url == BASE
        assert agent.model == MODEL
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
        The verdict value is normalized to uppercase for canonical matching."""
        proc = _fake_proc(
            stdout=_json_result(result="Analysis.\nverdict: accept-merged\n"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "ACCEPT-MERGED"

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

    def test_empty_stdout_logs_safe_metadata_not_secret(self) -> None:
        """Empty stdout with stderr containing a secret must log only safe metadata,
        never the raw stderr content."""
        import logging
        from io import StringIO

        secret = "sk-or-v1-secret-key-1234567890abcdef"
        proc = _fake_proc(stdout="", stderr=f"Error: {secret}\nTraceback ...")
        agent = _make_agent()

        # Capture log output at WARNING level.
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("rhobear_neo.agent")
        logger.addHandler(handler)
        try:
            with pytest.raises(MissingResult, match="no stdout"):
                agent._parse_output(proc)
        finally:
            logger.removeHandler(handler)

        log_text = buf.getvalue()
        # The raw stderr (including the planted secret) must NOT appear in logs.
        assert secret not in log_text, "secret leaked into log output"
        # Safe metadata should be present: exit code and stderr length.
        assert "exit=" in log_text, "exit code not in log"
        assert "stderr_len=" in log_text, "stderr length not in log"

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
    """_run_agent in neo_worker catches errors, passes through success.

    _run_agent now returns 4 values: (usage, verdict, route_idx, route).
    """

    def test_wrapper_catches_errors(self) -> None:
        from src.neo_worker import _run_agent

        agent = _make_agent()
        with patch.object(agent, "run", side_effect=TimeoutError("timed out")):
            usage, verdict, ridx, route = _run_agent(agent, "test brief")

        assert usage == {}
        assert verdict == ""
        assert ridx is None
        assert route is None

    def test_wrapper_passes_success(self) -> None:
        from src.neo_worker import _run_agent

        agent = _make_agent()
        with patch.object(agent, "run", return_value=({"input_tokens": 150}, "ACCEPT-READY")):
            usage, verdict, ridx, route = _run_agent(agent, "test brief")

        assert verdict == "ACCEPT-READY"
        assert usage["input_tokens"] == 150
        assert ridx is None
        assert route is None

    def test_wrapper_catches_generic_agent_error(self) -> None:
        from src.neo_worker import _run_agent

        agent = _make_agent()
        with patch.object(agent, "run", side_effect=UnknownVerdict("no VERDICT")):
            usage, verdict, ridx, route = _run_agent(agent, "test brief")

        assert usage == {}
        assert verdict == ""
        assert ridx is None
        assert route is None


# ===================================================================
# Config field mapping
# ===================================================================

class TestConfigMapping:
    """Config fields map correctly to ClaudeAgent."""

    def test_deepseek_key_mapped(self) -> None:
        """deepseek_key from Config -> api_key in ClaudeAgent."""
        cfg = Config()
        cfg.deepseek_key = "sk-test-key"
        cfg.gh_token = GH_TOKEN
        cfg.claude_bin = CLAUDE_BIN
        agent = ClaudeAgent.from_config(cfg, gh_token=GH_TOKEN)
        assert agent.api_key == "sk-test-key"

    def test_deepseek_model_mapped(self) -> None:
        """deepseek_model from Config -> model in ClaudeAgent, exact and
        unsuffixed (no [1m])."""
        cfg = Config()
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.gh_token = GH_TOKEN
        cfg.claude_bin = CLAUDE_BIN
        agent = ClaudeAgent.from_config(cfg, gh_token=GH_TOKEN)
        assert agent.model == "deepseek-v4-flash"
        assert agent.model == cfg.deepseek_model


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
        cfg.deepseek_key = "sk-test"
        cfg.database_url = "postgres://localhost/test"
        cfg.gh_token = "test-token"
        cfg.require()


# ===================================================================
# Stale OpenRouter env vars — must be invisible to Config
# ===================================================================

class TestStaleOpenRouterEnv:
    """Env vars from the disabled OpenRouter route must never be read by
    Config. A stale OPENROUTER_API_KEY / NEO_OPENROUTER_* cannot silently
    select a route or model."""

    def test_stale_openrouter_env_ignored(self) -> None:
        # DEEPSEEK_API_KEY is pinned to "" so the test is hermetic on any
        # machine (a real key may be present in the shell environment).
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "sk-or-v1-stale-secret",
            "NEO_OPENROUTER_BASE_URL": "https://openrouter.ai/api",
            "NEO_OPENROUTER_MODEL": "deepseek/deepseek-v4-flash",
            "DEEPSEEK_API_KEY": "",
        }, clear=False):
            cfg = Config()
        # Config holds only the DeepSeek Direct defaults — no stale values leak.
        assert cfg.deepseek_key == ""
        assert cfg.deepseek_base_url == "https://api.deepseek.com/anthropic"
        assert cfg.deepseek_model == "deepseek-v4-flash"

    def test_stale_key_does_not_satisfy_require(self) -> None:
        """require() demands DEEPSEEK_API_KEY — a stale OPENROUTER_API_KEY in
        the environment must not satisfy it."""
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "sk-or-v1-stale-secret",
            "DEEPSEEK_API_KEY": "",
        }, clear=False):
            cfg = Config()
            cfg.webhook_secret = "whs_test"
            cfg.database_url = "postgres://localhost/test"
            cfg.gh_token = "test-token"
            with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
                cfg.require()


# ===================================================================
# CANONICAL_VERDICTS constant
# ===================================================================

class TestCanonicalSet:
    """The CANONICAL_VERDICTS constant contains exactly the 6 expected values (production adds FIXED)."""

    def test_exact_set(self) -> None:
        # Production adds FIXED as a builder alias for ACCEPT-READY.
        # B2 adds ACCEPT (agent no longer merges, returns ACCEPT).
        assert CANONICAL_VERDICTS == {
            "FIXED",
            "ACCEPT",
            "ACCEPT-MERGED",
            "ACCEPT-READY",
            "FIX-FORWARD",
            "BOUNCE-BUILDER",
            "ESCALATE",
        }


# ===================================================================
# Verdict extraction — anchored regex
# ===================================================================

class TestVerdictExtraction:
    """Verdict extraction anchored to start of line — rejects misleading prose."""

    def test_anchored_rejects_misleading_prose(self) -> None:
        """A line like 'Do not output VERDICT: ...' should NOT be treated as a verdict."""
        proc = _fake_proc(
            stdout=_json_result(
                result="Do not output VERDICT: ACCEPT-MERGED\nLet me reconsider.\nVERDICT: ESCALATE",
            ),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        # Should pick up the second (real) verdict, not the prose one.
        assert v == "ESCALATE"

    def test_anchored_rejects_misleading_only_line(self) -> None:
        """When the only match is misleading prose, raise UnknownVerdict."""
        proc = _fake_proc(
            stdout=_json_result(
                result="Do not output VERDICT: ACCEPT-MERGED\nNothing else here.",
            ),
        )
        agent = _make_agent()
        with pytest.raises(UnknownVerdict, match="no VERDICT"):
            agent._parse_output(proc)

    def test_anchored_markdown_prefix_bold(self) -> None:
        """**VERDICT: ...** with bold markdown prefix should match."""
        proc = _fake_proc(
            stdout=_json_result(result="**VERDICT: FIX-FORWARD**"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "FIX-FORWARD"

    def test_anchored_markdown_prefix_blockquote(self) -> None:
        """> VERDICT: ... with blockquote prefix should match."""
        proc = _fake_proc(
            stdout=_json_result(result="> VERDICT: BOUNCE-BUILDER"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "BOUNCE-BUILDER"

    def test_anchored_markdown_prefix_list(self) -> None:
        """- VERDICT: ... with list prefix should match."""
        proc = _fake_proc(
            stdout=_json_result(result="- VERDICT: ACCEPT-READY"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "ACCEPT-READY"


# ===================================================================
# Verdict normalization
# ===================================================================

class TestVerdictNormalization:
    """Verdict normalized to uppercase after extraction."""

    def test_lowercase_verdict_normalized(self) -> None:
        """Lowercase verdict value is normalized to uppercase."""
        proc = _fake_proc(
            stdout=_json_result(result="VERDICT: accept-ready"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "ACCEPT-READY"

    def test_mixed_case_verdict_normalized(self) -> None:
        """Mixed case verdict value is normalized to uppercase."""
        proc = _fake_proc(
            stdout=_json_result(result="VERDICT: Fix-Forward"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "FIX-FORWARD"

    def test_uppercase_verdict_stays_uppercase(self) -> None:
        """Already uppercase verdict stays uppercase."""
        proc = _fake_proc(
            stdout=_json_result(result="VERDICT: ESCALATE"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "ESCALATE"


# ===================================================================
# Trailing punctuation tolerance
# ===================================================================

class TestVerdictTrailingPunctuation:
    """Verdict values with trailing sentence punctuation must still match
    the canonical set.  Hyphens inside ACCEPT-READY, BOUNCE-BUILDER, etc.
    must be preserved."""

    @pytest.mark.parametrize("trailing", [".", ",", ";", ":", "!", "?"])
    def test_trailing_punctuation_stripped(self, trailing: str) -> None:
        """A verdict followed by trailing punctuation must parse correctly."""
        proc = _fake_proc(
            stdout=_json_result(
                result=f"Analysis.\nVERDICT: ACCEPT-MERGED{trailing}\n",
            ),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "ACCEPT-MERGED"

    @pytest.mark.parametrize("verdict", ["ACCEPT-READY", "FIX-FORWARD", "BOUNCE-BUILDER", "ESCALATE"])
    def test_all_verdicts_with_trailing_period(self, verdict: str) -> None:
        """All canonical verdicts with trailing period must parse correctly."""
        proc = _fake_proc(
            stdout=_json_result(result=f"Analysis.\nVERDICT: {verdict}.\n"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == verdict

    def test_hyphens_preserved_inside_verdict(self) -> None:
        """Hyphens in ACCEPT-READY must not be stripped by punctuation removal."""
        proc = _fake_proc(
            stdout=_json_result(result="Analysis.\nVERDICT: ACCEPT-READY.\n"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "ACCEPT-READY"
        # Verify hyphens are intact.
        assert "-" in v

    def test_multiple_trailing_punctuation_stripped(self) -> None:
        """Multiple trailing punctuation marks (e.g. '!!') must be stripped."""
        proc = _fake_proc(
            stdout=_json_result(result="Analysis.\nVERDICT: ACCEPT-READY!!\n"),
        )
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "ACCEPT-READY"

    def test_non_canonical_still_rejected(self) -> None:
        """A non-canonical value like 'APPROVE' must still be rejected
        even after punctuation stripping."""
        proc = _fake_proc(
            stdout=_json_result(result="Analysis.\nVERDICT: APPROVE.\n"),
        )
        agent = _make_agent()
        with pytest.raises(UnknownVerdict, match="non-canonical verdict"):
            agent._parse_output(proc)

    def test_punctuation_only_still_rejected(self) -> None:
        """A verdict value that is only punctuation after stripping
        (e.g. 'VERDICT: .') should not match and raise UnknownVerdict."""
        proc = _fake_proc(
            stdout=_json_result(result="Analysis.\nVERDICT: .\n"),
        )
        agent = _make_agent()
        with pytest.raises(UnknownVerdict, match="non-canonical verdict"):
            agent._parse_output(proc)


# ===================================================================
# Terminal reason and API error status
# ===================================================================

class TestResultEnvelope:
    """Validate terminal_reason and api_error_status in the result envelope."""

    def test_terminal_reason_error_fails(self) -> None:
        """terminal_reason='error' should raise MalformedStream."""
        output = json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "Analysis.\nVERDICT: ACCEPT-READY",
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "num_turns": 1, "stop_reason": "end_turn",
            "terminal_reason": "error", "permission_denials": [],
        })
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        with pytest.raises(MalformedStream, match="terminal_reason"):
            agent._parse_output(proc)

    def test_terminal_reason_end_turn_ok(self) -> None:
        """terminal_reason='end_turn' should be accepted."""
        output = json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "Analysis.\nVERDICT: ACCEPT-READY",
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "num_turns": 1, "stop_reason": "end_turn",
            "terminal_reason": "end_turn", "permission_denials": [],
        })
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "ACCEPT-READY"

    def test_terminal_reason_null_ok(self) -> None:
        """No terminal_reason (absent or null) should be accepted."""
        output = json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "Analysis.\nVERDICT: ACCEPT-READY",
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "num_turns": 1, "stop_reason": "end_turn",
            "permission_denials": [],
        })
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "ACCEPT-READY"

    def test_api_error_status_present_fails(self) -> None:
        """Non-null api_error_status should raise ApiError."""
        output = json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "Analysis.\nVERDICT: ACCEPT-READY",
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "num_turns": 1, "stop_reason": "end_turn",
            "api_error_status": "rate_limited", "permission_denials": [],
        })
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        with pytest.raises(ApiError, match="api_error"):
            agent._parse_output(proc)

    def test_api_error_status_null_ok(self) -> None:
        """Null/absent api_error_status should be accepted."""
        output = json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "Analysis.\nVERDICT: ACCEPT-READY",
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "num_turns": 1, "stop_reason": "end_turn",
            "permission_denials": [],
        })
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        usage, v = agent._parse_output(proc)
        assert v == "ACCEPT-READY"


# ===================================================================
# Config.validate() — strict startup validation
# ===================================================================

class TestConfigValidate:
    """Config.validate() rejects any non-direct DeepSeek config."""

    def test_validate_rejects_stale_openrouter_url(self) -> None:
        """The disabled OpenRouter route must be rejected — a stale
        NEO_OPENROUTER_BASE_URL cannot silently select a route."""
        cfg = Config()
        cfg.deepseek_base_url = "https://openrouter.ai/api"
        with pytest.raises(ValueError, match="deepseek_base_url"):
            cfg.validate()

    def test_validate_rejects_bare_deepseek_url(self) -> None:
        """https://api.deepseek.com without the /anthropic path is rejected."""
        cfg = Config()
        cfg.deepseek_base_url = "https://api.deepseek.com"
        with pytest.raises(ValueError, match="deepseek_base_url"):
            cfg.validate()

    def test_validate_base_url_trailing_slash_ok(self) -> None:
        """Trailing slash on the base URL should be tolerated."""
        cfg = Config()
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic/"
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.validate()  # should not raise

    def test_validate_rejects_stale_prefixed_model(self) -> None:
        """The old OpenRouter model slug (deepseek/... prefix) must be
        rejected — the direct ID is unprefixed."""
        cfg = Config()
        cfg.deepseek_model = "deepseek/deepseek-v4-flash"
        with pytest.raises(ValueError, match="deepseek_model"):
            cfg.validate()

    def test_validate_rejects_context_suffixed_model(self) -> None:
        """A [1m]-suffixed model must be rejected — that suffix was an
        OpenRouter CLI hint and has no meaning on the direct route."""
        cfg = Config()
        cfg.deepseek_model = "deepseek-v4-flash[1m]"
        with pytest.raises(ValueError, match="deepseek_model"):
            cfg.validate()

    def test_validate_wrong_model(self) -> None:
        cfg = Config()
        cfg.deepseek_model = "deepseek-chat"
        with pytest.raises(ValueError, match="deepseek_model"):
            cfg.validate()

    def test_validate_wrong_effort(self) -> None:
        cfg = Config()
        cfg.deepseek_reasoning_effort = "high"
        with pytest.raises(ValueError, match="deepseek_reasoning_effort"):
            cfg.validate()

    def test_validate_undersized_tokens(self) -> None:
        cfg = Config()
        cfg.deepseek_max_tokens = 16000
        with pytest.raises(ValueError, match="deepseek_max_tokens"):
            cfg.validate()

    def test_validate_ok(self) -> None:
        """Valid config should pass validate() without error."""
        cfg = Config()
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.validate()  # should not raise

    def test_validate_missing_claude_bin_posix(self) -> None:
        """On POSIX, a missing claude_bin must raise ValueError.

        Uses mocking to safely test the POSIX branch without affecting
        Windows development."""
        cfg = Config()
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.claude_bin = "/usr/bin/claude"

        with patch("src.config.os.name", "posix"), \
             patch("src.config.shutil.which", return_value=None), \
             patch("src.config.os.path.isfile", return_value=False):
            with pytest.raises(ValueError, match="claude_bin not found"):
                cfg.validate()

    def test_validate_claude_bin_ok_on_windows(self) -> None:
        """On Windows, claude_bin validation is skipped (the binary is on
        the remote VPS, not the local dev box)."""
        cfg = Config()
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.claude_bin = "/usr/bin/claude"

        with patch("src.config.os.name", "nt"):
            cfg.validate()  # should not raise despite missing binary

    def test_validate_claude_bin_found_in_path(self) -> None:
        """When claude_bin resolves via shutil.which on POSIX, validation
        should pass."""
        cfg = Config()
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.claude_bin = "claude"

        with patch("src.config.os.name", "posix"), \
             patch("src.config.shutil.which", return_value="/usr/local/bin/claude"):
            cfg.validate()  # should not raise

    def test_validate_claude_bin_found_exact_path(self) -> None:
        """When shutil.which returns None but the exact path exists,
        validation should pass."""
        cfg = Config()
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.claude_bin = "/usr/bin/claude"

        with patch("src.config.os.name", "posix"), \
             patch("src.config.shutil.which", return_value=None), \
             patch("src.config.os.path.isfile", return_value=True):
            cfg.validate()  # should not raise


# ===================================================================
# Startup validation — Config.validate() must fire before side effects
# ===================================================================

class TestStartupValidation:
    """Startup with invalid config must raise before any side effect
    (state connection, server start)."""

    def _make_invalid_cfg(self, **overrides: Any) -> Config:
        cfg = Config()
        cfg.webhook_secret = "whs_test"
        cfg.deepseek_key = "sk-test"
        cfg.database_url = "postgres://localhost/test"
        cfg.gh_token = "test-token"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_startup_wrong_base_url_raises(self) -> None:
        """Invalid base_url in startup must raise ValueError before any
        side effect — NeoState/serve never reached. The disabled OpenRouter
        route is the canonical invalid value."""
        cfg = self._make_invalid_cfg(deepseek_base_url="https://openrouter.ai/api")
        with pytest.raises(ValueError, match="deepseek_base_url"):
            cfg.require().validate()
        # If we got here, validate() caught it before any DB/server call.

    def test_startup_wrong_model_raises(self) -> None:
        cfg = self._make_invalid_cfg(deepseek_model="deepseek/deepseek-v4-flash")
        with pytest.raises(ValueError, match="deepseek_model"):
            cfg.require().validate()

    def test_startup_wrong_effort_raises(self) -> None:
        cfg = self._make_invalid_cfg(deepseek_reasoning_effort="high")
        with pytest.raises(ValueError, match="deepseek_reasoning_effort"):
            cfg.require().validate()

    def test_startup_undersized_budget_raises(self) -> None:
        cfg = self._make_invalid_cfg(deepseek_max_tokens=16000)
        with pytest.raises(ValueError, match="deepseek_max_tokens"):
            cfg.require().validate()

    def test_startup_validate_after_require_ok(self) -> None:
        """A valid config passes require().validate() chain."""
        cfg = self._make_invalid_cfg()  # all defaults are valid
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.require().validate()  # should not raise

    def test_main_uses_validate_chain(self) -> None:
        """main() must call .validate() — verify the chain is wired."""
        from src import __main__ as entrypoint
        import inspect
        source = inspect.getsource(entrypoint.main)
        assert "require().validate()" in source, \
            "main() must call require().validate() before any side effect"


# ===================================================================
# Protocol census — no stale Pro/direct/Pi/Windows-home text
# ===================================================================

class TestProtocolCensus:
    """neo_protocol.py must not contain stale DeepSeek-Pro, Pi, direct-HTTP,
    Windows-home-path, OpenRouter-prefixed-model, or [1m]-suffix references.
    The exact direct model ID `deepseek-v4-flash` (lowercase) is allowed —
    only the stale/prefix/suffix forms are banned."""

    @staticmethod
    def _protocol_source() -> str:
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        with open(os.path.join(src_dir, "neo_protocol.py")) as f:
            return f.read()

    def test_no_deepseek_pro_reference(self) -> None:
        """'DeepSeek-Pro' (capital D, capital S, capital P) is stale."""
        src = self._protocol_source()
        assert "DeepSeek-Pro" not in src, (
            "neo_protocol.py must not mention DeepSeek-Pro"
        )

    def test_no_headless_deepseek_reference(self) -> None:
        """'headless DeepSeek' with capital letters is the old agent description."""
        src = self._protocol_source()
        assert "headless DeepSeek" not in src, (
            "neo_protocol.py must not mention 'headless DeepSeek'"
        )

    def test_no_pi_reference(self) -> None:
        """'Pi' (the old Pi protocol) is stale in the protocol text."""
        src = self._protocol_source()
        assert " Pi " not in src and "Pi\n" not in src, (
            "neo_protocol.py must not reference Pi protocol"
        )

    def test_no_direct_http_reference(self) -> None:
        """'direct' as in 'direct HTTP' or 'direct DeepSeek' is stale."""
        src = self._protocol_source()
        assert "direct " not in src, (
            "neo_protocol.py must not describe direct-HTTP or direct-DeepSeek paths"
        )

    def test_no_prefixed_model_reference(self) -> None:
        """'deepseek/deepseek-v4-flash' (the OpenRouter slug) is stale — the
        protocol must name the exact unprefixed direct ID."""
        src = self._protocol_source()
        assert "deepseek/deepseek-v4-flash" not in src, (
            "neo_protocol.py must not mention the prefixed OpenRouter model slug"
        )

    def test_no_context_suffix_reference(self) -> None:
        """'[1m]' (the OpenRouter CLI context hint) is stale — the direct
        route takes the bare model ID."""
        src = self._protocol_source()
        assert "[1m]" not in src, (
            "neo_protocol.py must not mention the [1m] context suffix"
        )

    def test_no_windows_home_path(self) -> None:
        """Hardcoded Windows home paths (C:/Users/) must not appear."""
        src = self._protocol_source()
        assert "C:/Users/" not in src, (
            "neo_protocol.py must not contain hardcoded Windows home paths"
        )

    def test_no_windows_path_reference(self) -> None:
        """'Windows' as a path reference is stale (VPS runtime)."""
        src = self._protocol_source()
        assert "Windows" not in src, (
            "neo_protocol.py must not reference Windows paths"
        )


# ===================================================================
# Real smoke test — full ClaudeAgent.run() lifecycle
# ===================================================================

class TestRealSmoke:
    """Smoke test using a real subprocess (fake Claude binary).

    Verifies the full run() lifecycle: temp dirs created and cleaned,
    subprocess invoked, output parsed, verdict returned.
    """

    def test_smoke_run_with_temp_config(self) -> None:
        """Full run() lifecycle with a real subprocess and temp config dir.

        Creates a temp batch file that acts as the Claude CLI, outputs
        valid JSON, and exits cleanly.  Verifies temp dirs are created
        during run() and cleaned after.
        """
        import tempfile as _tf
        import os as _os

        # Temp Python script that outputs valid Claude JSON result.
        _script = '''import json, sys
data = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "VERDICT: ACCEPT-READY",
    "usage": {"input_tokens": 100, "output_tokens": 50},
    "num_turns": 1,
    "stop_reason": "end_turn",
    "permission_denials": [],
}
print(json.dumps(data))
sys.exit(0)
'''
        _cleanup = []
        if _os.name == "nt":
            # Windows: temp batch file that calls the Python script.
            with _tf.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as _f:
                _f.write(_script)
                _py_path = _f.name
            _cleanup.append(_py_path)
            _bat_content = '@echo off\npython "' + _py_path + '"\nexit /b 0\n'
            with _tf.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as _f:
                _f.write(_bat_content)
                _claude_bin = _f.name
        else:
            # POSIX (W1-02: made hermetic on the build box): the script itself
            # is the CLI stub — shebang + exec bit. A .bat has no interpreter
            # here and temp files are not executable by default.
            with _tf.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as _f:
                _f.write("#!" + sys.executable + "\n" + _script)
                _claude_bin = _f.name
            _os.chmod(_claude_bin, 0o755)
        _cleanup.append(_claude_bin)

        try:
            agent = ClaudeAgent(
                claude_bin=_claude_bin,
                api_key="sk-smoke",
                base_url="https://api.deepseek.com/anthropic",
                model="deepseek-v4-flash",
                effort="max",
                max_tokens=32000,
                timeout=30,
                gh_token="test-token-smoke",
            )
            usage, verdict = agent.run("test brief")
            assert verdict == "ACCEPT-READY"
            assert usage["input_tokens"] == 100
            assert usage["output_tokens"] == 50
        finally:
            for _p in _cleanup:
                if _os.path.exists(_p):
                    _os.unlink(_p)

# ===================================================================
# Round 2 — env allowlist for the triage agent + anchored fallthrough
# classification (the divergent usage.classify_cli_fallthrough is deleted)
# ===================================================================

class TestClaudeEnvAllowlist:
    """The triage agent runs --dangerously-skip-permissions over PR-derived
    briefs — its child env is an explicit allowlist plus the route's own
    Anthropic credentials, never a copy of the service environment."""

    def test_service_secrets_never_reach_the_subprocess(self, monkeypatch) -> None:
        for name, value in {
            "DATABASE_URL": "postgresql://app:hunter2@db.internal/prod",
            "NEO_ADVISOR_TOKEN": "advisor-bearer-secret",
            "NEO_GITHUB_APP_KEY": "-----BEGIN PRIVATE KEY-----",
            "NEO_GITHUB_APP_KEY_PATH": "/opt/rhobear/rhobear-neo/app-key.pem",
            "OPENROUTER_API_KEY": "sk-or-v1-secret",
        }.items():
            monkeypatch.setenv(name, value)
        agent = _make_agent()
        env = agent._build_env("/tmp/neo-config-test")
        for name in ("DATABASE_URL", "NEO_ADVISOR_TOKEN", "NEO_GITHUB_APP_KEY",
                     "NEO_GITHUB_APP_KEY_PATH", "OPENROUTER_API_KEY"):
            assert name not in env, f"secret env var {name} reached the subprocess"
        # Only the route's own API key may be a *_KEY — everything else is gone.
        assert [k for k in env if k.endswith("_KEY")] == ["ANTHROPIC_API_KEY"]
        assert env["ANTHROPIC_API_KEY"] == API_KEY


class TestAnchoredFallthroughClassification:
    """The surviving fallthrough classifier anchors on `api_error_status` /
    `HTTP <code>` — never on bare substrings."""

    def _agent(self) -> ClaudeAgent:
        return ClaudeAgent.__new__(ClaudeAgent)

    def test_cost_1500_tokens_is_not_5xx(self) -> None:
        """Round-2 finding: "cost 1500 tokens" classified as 5xx via the bare
        "50" substring. It is an unknown API error — NOT a fallthrough."""
        got = self._agent()._classify_agent_error(
            ApiError("cost 1500 tokens"), self._agent())
        assert got[0] is False
        assert got[1] != "5xx"

    def test_read_400_bytes_is_not_400(self) -> None:
        got = self._agent()._classify_agent_error(
            ApiError("read 400 bytes then 150 files"), self._agent())
        assert got != (False, "400")

    @pytest.mark.parametrize("err_str,expected", [
        ("HTTP/1.1 503 Service Unavailable", (True, "5xx")),
        ("HTTP 429 Too Many Requests", (True, "429")),
        ("Error code: 402 — payment required", (True, "402")),
        ("status: 400 bad request", (False, "400")),
        ("429", (True, "429")),
        ("503", (True, "5xx")),
        ("request timed out", (True, "timeout")),
    ])
    def test_anchored_status_codes(self, err_str, expected) -> None:
        got = self._agent()._classify_agent_error(ApiError(err_str), self._agent())
        assert got == expected

    def test_json_envelope_api_error_status_is_anchored(self) -> None:
        """The JSON envelope path uses api_error_status with the same anchor."""
        agent = self._agent()
        proc = subprocess.CompletedProcess(
            args=["claude"], returncode=1,
            stdout=json.dumps({"type": "result",
                               "api_error_status": "HTTP/1.1 503 Service Unavailable"}),
            stderr="")
        assert agent._classify_agent_error(
            NonZeroExit("x"), agent, proc=proc) == (True, "5xx")

        proc = subprocess.CompletedProcess(
            args=["claude"], returncode=1,
            stdout=json.dumps({"type": "result",
                               "api_error_status": "cost 1500 tokens"}),
            stderr="")
        got = agent._classify_agent_error(NonZeroExit("x"), agent, proc=proc)
        assert got[0] is False and got[1] != "5xx"



# ===================================================================
# W1-06 — RouteGate wiring, optional gh_token, config defaults
# ===================================================================

class _FakeUsageStore:
    """In-memory UsageStore stand-in (no Postgres): same idempotency rule
    (unique on job_id + attempt_number + route_idx)."""

    def __init__(self) -> None:
        self.rows: list[Any] = []
        self._keys: set[tuple] = set()

    def record_attempt(self, attempt: Any) -> bool:
        key = (attempt.job_id, attempt.attempt_number, attempt.route_idx)
        if key in self._keys:
            return False
        self._keys.add(key)
        self.rows.append(attempt)
        return True


def _zeus_brief() -> str:
    """A real-shaped triage brief: TARGET line, canon status call with the
    full 40-hex head SHA, builder-round sentence."""
    head = "ab" * 20
    return (
        "TARGET: PR #44 in acme/widget at head " + head[:12] + "\n"
        "gh api repos/acme/widget/commits/" + head + "/status\n"
        "You have already spent 1/2 builder rounds on this PR.\n"
    )


def test_zeus_calls_go_through_roulegate(tmp_path: Any) -> None:
    """Zeus's model calls resolve through RouteGate — never a direct provider
    client built on the agent's own initiative.

    from_config points the CLI at RouteGate's Anthropic wire (brief step 2's
    conditional endpoint) with the consumer header, the attempt executes
    against a live loopback RouteGate, and the row lands metered with the
    consumer, PR, head SHA, builder round, tokens in/out/cached and only the
    cost the response actually carried.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    captured: dict[str, Any] = {}

    class RouteGateHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            captured["path"] = self.path
            captured["consumer"] = self.headers.get("X-RHOBEAR-Consumer")
            body = json.dumps({
                "id": "msg_1", "type": "message", "role": "assistant",
                "model": "peak",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 7, "output_tokens": 3,
                          "estimated_cost": 0.0042},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RouteGateHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    # Fake Claude CLI: talks to whatever base URL the agent configured,
    # carrying the consumer header the way Claude Code carries
    # ANTHROPIC_CUSTOM_HEADERS, then emits a normal result envelope (with the
    # cost RouteGate reported — nothing invented).
    script = (
        "import json, os, sys, urllib.request\n"
        "base = os.environ['ANTHROPIC_BASE_URL']\n"
        "headers = {'Content-Type': 'application/json'}\n"
        "custom = os.environ.get('ANTHROPIC_CUSTOM_HEADERS', '')\n"
        "if ':' in custom:\n"
        "    name, value = custom.split(':', 1)\n"
        "    headers[name.strip()] = value.strip()\n"
        "if os.environ.get('X_RHOBEAR_CONSUMER'):\n"
        "    headers['X-RHOBEAR-Consumer'] = os.environ['X_RHOBEAR_CONSUMER']\n"
        "req = urllib.request.Request(base, data=b'{}', headers=headers)\n"
        "try:\n"
        "    urllib.request.urlopen(req, timeout=5).read()\n"
        "except Exception as exc:\n"
        "    print('route gate probe failed: %s' % exc, file=sys.stderr)\n"
        "    sys.exit(3)\n"
        "print(json.dumps({\n"
        "    'type': 'result', 'subtype': 'success', 'is_error': False,\n"
        "    'result': 'VERDICT: ACCEPT-READY',\n"
        "    'usage': {'input_tokens': 11, 'output_tokens': 5,\n"
        "              'cache_read_input_tokens': 4,\n"
        "              'estimated_cost': 0.0042},\n"
        "    'num_turns': 1, 'stop_reason': 'end_turn',\n"
        "    'permission_denials': [],\n"
        "}))\n"
        "sys.exit(0)\n"
    )
    script_path = tmp_path / "claude-fake"
    script_path.write_text("#!" + sys.executable + "\n" + script)
    script_path.chmod(0o755)

    cfg = Config()
    cfg.claude_bin = str(script_path)
    cfg.deepseek_base_url = BASE
    cfg.deepseek_model = MODEL
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.deepseek_timeout = 60
    cfg.routegate_anthropic_url = (
        "http://127.0.0.1:%d/v1/messages" % server.server_address[1]
    )
    cfg.routegate_model = "peak"

    # The guard demands RouteGate's loopback PORT; this test's fake gate
    # binds an ephemeral one, so pin the constant to the bound port for the
    # run (production keeps 8645 — see ROUTEGATE_LOOPBACK_PORT).
    port_patch = patch("src.agent_claude.ROUTEGATE_LOOPBACK_PORT",
                       server.server_address[1])
    port_patch.start()

    try:
        # RouteGate first: the agent must NOT build a direct provider client.
        agent = ClaudeAgent.from_config(cfg, gh_token=GH_TOKEN)
        assert agent.base_url.rstrip("/") == cfg.routegate_anthropic_url
        env = agent._build_env(str(tmp_path / "neo-config"))
        assert env["ANTHROPIC_BASE_URL"] == cfg.routegate_anthropic_url
        # The consumer header travels with the call.
        assert env["ANTHROPIC_CUSTOM_HEADERS"].startswith(
            "X-RHOBEAR-Consumer: reviews:zeus"
        )
        assert env["X_RHOBEAR_CONSUMER"] == env["ANTHROPIC_CUSTOM_HEADERS"].split(
            "X-RHOBEAR-Consumer: ", 1
        )[1]

        store = _FakeUsageStore()
        usage, verdict, ridx, route = agent.run_with_chain(
            _zeus_brief(), [], cfg, GH_TOKEN,
            usage_store=store, job_id=11,
        )

        # The call actually reached RouteGate, with the header.
        assert captured.get("path", "").endswith("/v1/messages")
        assert captured.get("consumer", "").startswith("reviews:zeus")
        assert captured["consumer"] == env["X_RHOBEAR_CONSUMER"]

        assert verdict == "ACCEPT-READY"
        assert usage["input_tokens"] == 11
        assert route is not None and route.provider == "routegate"

        # The attempt is metered: consumer, PR, head SHA, builder round,
        # tokens in/out/cached, and only the cost the response carried.
        assert len(store.rows) == 1
        row = store.rows[0]
        assert row.outcome == "success"
        assert row.route_name == "peak"
        assert row.consumer == captured["consumer"]
        assert row.pr_number == 44
        assert row.head_sha == "ab" * 20
        assert row.builder_round == 1
        assert row.uncached_input == 11
        assert row.output_tokens == 5
        assert row.cached_input == 4
        assert row.cost_usd == 0.0042
    finally:
        server.shutdown()
        server.server_close()
        port_patch.stop()


def test_from_config_without_gh_token(caplog: Any) -> None:
    """from_config() must construct WITHOUT a gh_token keyword, warn once per
    process, and never raise (edge 5: a caller that does pass a token still
    gets it — see test_from_config above)."""
    import logging as _logging

    import src.agent_claude as agent_mod

    cfg = Config()
    cfg.deepseek_base_url = BASE
    cfg.deepseek_model = MODEL
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.deepseek_timeout = 1800
    cfg.claude_bin = CLAUDE_BIN

    with patch.object(agent_mod, "_GH_TOKEN_WARNED", False):
        with caplog.at_level(_logging.WARNING):
            agent = ClaudeAgent.from_config(cfg)
            ClaudeAgent.from_config(cfg)  # second call must not re-warn

    assert agent is not None
    assert agent.gh_token == ""
    warnings = [r.getMessage() for r in caplog.records
                if "GH_TOKEN" in r.getMessage()]
    assert len(warnings) == 1, warnings


def test_neo_max_builder_rounds_default_is_2() -> None:
    """Behavioural, not source-text: the env reader's default for
    NEO_MAX_BUILDER_ROUNDS is the owner's rule (2), proved by constructing a
    Config whose clamp floor allows it — a source default of 5 would produce
    5 here.  The base's clamp is documented alongside: with the shipped
    floor (5) the EFFECTIVE default stays 5, and tests on main pin that
    clamp (this lane does not redesign it — see the PR's QUESTION)."""
    import src.config as config_mod

    assert config_mod.MAX_BUILDER_ROUNDS_DEFAULT == 2

    with patch.dict(os.environ, {"NEO_BOUNCES_BEFORE_ESCALATION": "2"},
                    clear=True):
        unclamped = config_mod.Config()
    assert unclamped.max_builder_rounds == 2, (
        "the env default must be 2 when the clamp floor permits it"
    )

    with patch.dict(os.environ, {}, clear=True):
        clamped = config_mod.Config()
    assert clamped.max_builder_rounds == clamped.bounces_before_escalation == 5, (
        "the base's clamp (not this lane's to change) still lifts the "
        "effective default to bounces_before_escalation"
    )


def test_validate_rejects_foreign_routegate_url() -> None:
    """Fail-closed: only the loopback RouteGate may hold Zeus's briefs — a
    foreign routegate_base_url / routegate_anthropic_url is refused at
    startup, same discipline as deepseek_base_url."""
    cfg = Config()
    cfg.deepseek_base_url = BASE
    cfg.deepseek_model = MODEL
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.claude_bin = sys.executable  # hermetic: no production binary needed

    cfg.validate()  # loopback defaults are valid

    foreign = Config()
    foreign.routegate_base_url = "https://elsewhere.example/v1"
    with pytest.raises(ValueError, match="routegate_base_url"):
        foreign.validate()

    offbox = Config()
    offbox.routegate_anthropic_url = "https://elsewhere.example/v1/messages"
    with pytest.raises(ValueError, match="routegate_anthropic_url"):
        offbox.validate()


def test_roulegate_refuses_foreign_url() -> None:
    """[Apollo HIGH/MEDIUM] The loopback guard is a security path: a
    routegate_anthropic_url that is not the loopback gate must be refused
    BOTH by from_config and by run_with_chain, so a cfg mutated after
    validate() can never carry a brief off-box."""
    cfg = Config()
    cfg.deepseek_base_url = BASE
    cfg.deepseek_model = MODEL
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.claude_bin = CLAUDE_BIN
    cfg.routegate_anthropic_url = "https://evil.example/v1/messages"

    with pytest.raises(AgentError, match="loopback"):
        ClaudeAgent.from_config(cfg, gh_token=GH_TOKEN)

    agent = _make_agent()
    with pytest.raises(AgentError, match="loopback"):
        agent.run_with_chain(
            _zeus_brief(), [], cfg, GH_TOKEN,
            usage_store=_FakeUsageStore(), job_id=1,
        )


def test_consumer_header_rejects_bad_install_id(tmp_path: Any) -> None:
    """[Apollo MEDIUM] The consumer embeds an install id that reaches a
    header env var: a value with newlines/CRLF must never get there."""
    cfg = Config()
    cfg.deepseek_base_url = BASE
    cfg.deepseek_model = MODEL
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.claude_bin = CLAUDE_BIN

    bad = ClaudeAgent.from_config(
        cfg, gh_token=GH_TOKEN, install_id="123\r\nX-Injected: 1")
    assert bad.consumer == "reviews:zeus"
    env = bad._build_env(str(tmp_path / "neo-config"))
    assert "\n" not in env["ANTHROPIC_CUSTOM_HEADERS"]
    assert "\r" not in env["ANTHROPIC_CUSTOM_HEADERS"]

    # A real install id still qualifies the consumer (L0's consumer shape).
    good = ClaudeAgent.from_config(cfg, gh_token=GH_TOKEN, install_id="987654321")
    assert good.consumer == "reviews:zeus/987654321"
    assert good._build_env(str(tmp_path / "neo-config"))["X_RHOBEAR_CONSUMER"] == \
        "reviews:zeus/987654321"


def test_roulegate_guard_parses_the_url() -> None:
    """[Apollo r2, HIGH] The loopback guard must PARSE the URL: scheme,
    host, no userinfo, RouteGate's port — a prefix check lets
    `http://127.0.0.1:8645@evil.example/...` (userinfo) and non-RouteGate
    loopback ports straight through."""
    cfg = Config()
    cfg.deepseek_base_url = BASE
    cfg.deepseek_model = MODEL
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.claude_bin = CLAUDE_BIN

    for evil in (
        "http://127.0.0.1:8645@evil.example/v1/messages",   # userinfo bypass
        "http://127.0.0.1:1/v1/messages",                   # not RouteGate's port
        "https://127.0.0.1:8645/v1/messages",               # not the gate's scheme
        "http://[::1]:8645/v1/messages",                    # not the gate's host
        "http://127.0.0.1:8645/v1/messages/../exfil",       # traversal out of /v1
    ):
        cfg.routegate_anthropic_url = evil
        with pytest.raises(AgentError, match="loopback") as excinfo:
            ClaudeAgent.from_config(cfg, gh_token=GH_TOKEN)
        assert evil in str(excinfo.value), evil


def test_roulegate_failure_is_metered_and_ends_the_run() -> None:
    """[Apollo r2, HIGH] The gated failure path: a 402 at the gate is ONE
    fallthrough row (route name `peak`, consumer, target) and the run ends —
    no second attempt outside RouteGate."""
    cfg = Config()
    cfg.deepseek_base_url = BASE
    cfg.deepseek_model = MODEL
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.claude_bin = CLAUDE_BIN
    cfg.routegate_anthropic_url = "http://127.0.0.1:8645/v1/messages"
    cfg.routegate_model = "peak"

    agent = _make_agent()
    store = _FakeUsageStore()
    calls: list[Any] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            args=["claude"], returncode=0, stderr="",
            stdout=json.dumps({"type": "result", "is_error": True,
                               "subtype": "error_during_execution",
                               "api_error_status": "402"}),
        )

    with patch("src.agent_claude.subprocess.run", side_effect=fake_run):
        result = agent.run_with_chain(
            _zeus_brief(), [], cfg, GH_TOKEN,
            usage_store=store, job_id=3)

    assert result == ({}, "", None, None)      # gate owns failover: run ends
    assert len(calls) == 1                     # exactly one attempt, ever
    row, = store.rows
    assert row.outcome == "fallthrough"
    assert row.error_class == "402"
    assert row.route_name == "peak"
    assert row.provider == "routegate"
    assert row.consumer.startswith("reviews:zeus")
    assert row.pr_number == 44 and row.head_sha == "ab" * 20


def test_roulegate_non_fallthrough_reraises_after_metering() -> None:
    """[Apollo r2, HIGH] A non-fallthrough failure at the gate is recorded
    (outcome `error`) and then re-raised — never silently swallowed."""
    cfg = Config()
    cfg.deepseek_base_url = BASE
    cfg.deepseek_model = MODEL
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.claude_bin = CLAUDE_BIN
    cfg.routegate_anthropic_url = "http://127.0.0.1:8645/v1/messages"
    cfg.routegate_model = "peak"

    agent = _make_agent()
    store = _FakeUsageStore()

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            args=["claude"], returncode=0, stderr="", stdout="not json at all")

    with patch("src.agent_claude.subprocess.run", side_effect=fake_run):
        with pytest.raises(AgentError):
            agent.run_with_chain(
                _zeus_brief(), [], cfg, GH_TOKEN,
                usage_store=store, job_id=4)

    row, = store.rows
    assert row.outcome == "error"
    assert row.route_name == "peak"
    assert row.consumer.startswith("reviews:zeus")
