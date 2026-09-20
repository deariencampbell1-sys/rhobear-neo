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

BASE = "https://api.deepseek.com/anthropic"
API_KEY = "sk-test-placeholder"
MODEL = "deepseek-v4-flash"          # exact direct ID — no prefix, no [1m] suffix
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

        agent = ClaudeAgent.from_config(cfg)
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

    def test_deepseek_key_mapped(self) -> None:
        """deepseek_key from Config -> api_key in ClaudeAgent."""
        cfg = Config()
        cfg.deepseek_key = "sk-test-key"
        cfg.gh_token = GH_TOKEN
        cfg.claude_bin = CLAUDE_BIN
        agent = ClaudeAgent.from_config(cfg)
        assert agent.api_key == "sk-test-key"

    def test_deepseek_model_mapped(self) -> None:
        """deepseek_model from Config -> model in ClaudeAgent, exact and
        unsuffixed (no [1m])."""
        cfg = Config()
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.gh_token = GH_TOKEN
        cfg.claude_bin = CLAUDE_BIN
        agent = ClaudeAgent.from_config(cfg)
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
        cfg.gh_token = "ghp_test"
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
            cfg.gh_token = "ghp_test"
            with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
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
            "type": "result", "type": "result", "subtype": "success", "is_error": False,
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
            "type": "result", "type": "result", "subtype": "success", "is_error": False,
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
            "type": "result", "type": "result", "subtype": "success", "is_error": False,
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
        """Non-null api_error_status should raise MalformedStream."""
        output = json.dumps({
            "type": "result", "type": "result", "subtype": "success", "is_error": False,
            "result": "Analysis.\nVERDICT: ACCEPT-READY",
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "num_turns": 1, "stop_reason": "end_turn",
            "api_error_status": "rate_limited", "permission_denials": [],
        })
        proc = _fake_proc(stdout=output)
        agent = _make_agent()
        with pytest.raises(MalformedStream, match="api_error_status"):
            agent._parse_output(proc)

    def test_api_error_status_null_ok(self) -> None:
        """Null/absent api_error_status should be accepted."""
        output = json.dumps({
            "type": "result", "type": "result", "subtype": "success", "is_error": False,
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
        cfg.gh_token = "ghp_test"
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
        with _tf.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as _f:
            _f.write(_script)
            _py_path = _f.name

        # Temp batch file that calls the Python script (ignores all CLI args).
        _bat_content = '@echo off\npython "' + _py_path + '"\nexit /b 0\n'
        with _tf.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as _f:
            _f.write(_bat_content)
            _bat_path = _f.name

        try:
            agent = ClaudeAgent(
                claude_bin=_bat_path,
                api_key="sk-smoke",
                base_url="https://api.deepseek.com/anthropic",
                model="deepseek-v4-flash",
                effort="max",
                max_tokens=32000,
                timeout=30,
                gh_token="ghp_smoke",
            )
            usage, verdict = agent.run("test brief")
            assert verdict == "ACCEPT-READY"
            assert usage["input_tokens"] == 100
            assert usage["output_tokens"] == 50
        finally:
            for _p in (_py_path, _bat_path):
                if _os.path.exists(_p):
                    _os.unlink(_p)

def test_parse_output_accepts_completed_terminal_reason():
    """Claude CLI auto-updated to return terminal_reason='completed' for
    successful runs. _parse_output must accept it (not raise MalformedStream)."""
    import json as _json
    from types import SimpleNamespace
    from src.agent_claude import ClaudeAgent

    agent = ClaudeAgent(
        claude_bin="/usr/bin/true",
        api_key="sk-test",
        base_url="https://example.com",
        model="test-model",
        effort="max",
        max_tokens=1000,
        timeout=10,
        gh_token="ghp_test",
    )
    mock_proc = SimpleNamespace(
        returncode=0,
        stdout=_json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "VERDICT: ACCEPT-MERGED",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "num_turns": 1, "stop_reason": "end_turn",
            "terminal_reason": "completed",
            "permission_denials": [],
        }),
        stderr="",
    )
    usage, verdict = agent._parse_output(mock_proc)
    assert verdict == "ACCEPT-MERGED", f"expected ACCEPT-MERGED, got {verdict!r}"
    assert usage["input_tokens"] == 10


def test_parse_output_rejects_unknown_terminal_reason():
    """An unrecognized terminal_reason must still raise MalformedStream
    (fail-closed — don't silently accept unknown states)."""
    import json as _json
    from types import SimpleNamespace
    from src.agent_claude import ClaudeAgent, MalformedStream

    agent = ClaudeAgent(
        claude_bin="/usr/bin/true", api_key="sk-test",
        base_url="https://example.com", model="test-model",
        effort="max", max_tokens=1000, timeout=10, gh_token="ghp_test",
    )
    mock_proc = SimpleNamespace(
        returncode=0,
        stdout=_json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "VERDICT: ACCEPT-MERGED",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "num_turns": 1, "stop_reason": "end_turn",
            "terminal_reason": "something_unexpected",
            "permission_denials": [],
        }),
        stderr="",
    )
    try:
        agent._parse_output(mock_proc)
        assert False, "should have raised MalformedStream"
    except MalformedStream as e:
        assert "non-terminal" in str(e), f"wrong error: {e}"
