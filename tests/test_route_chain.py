"""Tests for B4 Neo route chain — agents walk the chain, fall through on
transient errors, record per-attempt usage, and report the winning route.

Tests use fake CLIs (as the existing agent tests do):
  - Route 1 returns 429 → route 2 runs and is recorded
  - 400 → no fall-through
  - All routes busy → failure recorded for each
  - Zero-token usage stays 0
  - Claude agent skips a route with no Anthropic endpoint
  - Hermes skips a route it can't serve
  - credits_for unchanged for the same usage

Run:  timeout 900 /work/repos/rb-B4-neo-durable-jobs/.venv/bin/python -m pytest -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest.mock import patch, MagicMock

import pytest

from src.config import Config, CLAUDE_WIRE_ENDPOINTS, HERMES_PROVIDER_MAP
from src.usage import Route, UsageAttempt, UsageStore, available_routes, resolve_route_chain
from src.agent_claude import ClaudeAgent, ApiError
from src.agent_hermes import HermesAgent, AgentError as HermesAgentError

# ---------------------------------------------------------------------------
# Test database
# ---------------------------------------------------------------------------

DSN = os.environ.get("NEO_TEST_DSN", "")
requires_pg = pytest.mark.skipif(not DSN, reason="NEO_TEST_DSN not set")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claude_agent(**kw):
    defaults = dict(
        claude_bin="/usr/bin/claude",
        api_key="sk-test",
        base_url="https://api.deepseek.com/anthropic",
        model="deepseek-v4-flash",
        effort="max",
        max_tokens=32000,
        timeout=30,
        gh_token="test-token",
    )
    defaults.update(kw)
    return ClaudeAgent(**defaults)


def _make_hermes_agent(**kw):
    defaults = dict(
        hermes_bin="/usr/bin/hermes",
        provider="zai",
        model="glm-5.3-flash",
        timeout=30,
        gh_token="test-token",
    )
    defaults.update(kw)
    return HermesAgent(**defaults)


def _cfg() -> Config:
    cfg = Config()
    cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
    cfg.deepseek_model = "deepseek-v4-flash"
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.deepseek_timeout = 30
    cfg.neo_routes = "deepinfra:glm-5.3-flash,zai:glm-5.3-flash,deepseek:deepseek-v4-flash"
    cfg.claude_bin = sys.executable  # hermetic: no production binary needed
    cfg.gh_token = "test-token"
    cfg.webhook_secret = "whs_test"
    cfg.database_url = DSN
    cfg.validate()
    return cfg


def _fake_proc(*, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr,
    )


def _claude_json(result_text="VERDICT: ACCEPT-READY", in_tok=100, out_tok=50,
                  stop_reason="end_turn", api_error_status=None):
    obj = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": result_text,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
        "num_turns": 1,
        "stop_reason": stop_reason,
        "permission_denials": [],
    }
    if api_error_status is not None:
        obj["api_error_status"] = api_error_status
    return json.dumps(obj)


def _wipe(store):
    import psycopg
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("DELETE FROM neo_usage")
        c.execute("DELETE FROM neo_jobs")
        c.execute("DELETE FROM neo_actions")
        c.execute("DELETE FROM neo_installs")


@pytest.fixture()
def store():
    s = UsageStore(DSN)
    s.ensure_schema()
    _wipe(s)
    return s


@pytest.fixture()
def cfg():
    return _cfg()


# ===================================================================
# Route 1 returns 429 → route 2 runs and is recorded
# ===================================================================

@requires_pg
class TestFallthrough429:
    """Route 1 returns 429 → route 2 runs and is recorded."""

    def test_claude_429_fallthrough(self, cfg, store):
        """Claude agent: deepinfra skipped (no Anthropic endpoint),
        zai returns 429, deepseek succeeds."""
        agent = _make_claude_agent()

        calls = []

        def fake_subprocess_run(cmd, **kw):
            calls.append(cmd)
            if len(calls) == 1:
                # Route zai (idx 1) — 429 error
                return _fake_proc(
                    stdout=_claude_json(api_error_status="429"),
                    returncode=0,
                )
            # Route deepseek (idx 4) — success
            return _fake_proc(
                stdout=_claude_json(result_text="VERDICT: ACCEPT-READY", in_tok=50, out_tok=20),
                returncode=0,
            )

        with patch.dict(os.environ, {
            "DEEPINFRA_API_KEY": "key1",
            "ZAI_API_KEY": "key2",
            "DEEPSEEK_API_KEY": "key3",
        }, clear=True):
            avail = available_routes(cfg.parsed_routes)
            with patch("src.agent_claude.subprocess.run", side_effect=fake_subprocess_run):
                usage, verdict, ridx, route = agent.run_with_chain(
                    "test brief", avail, cfg, "test-token",
                    usage_store=store, job_id=42,
                )

        assert verdict == "ACCEPT-READY"
        # cfg has 3 routes: deepinfra(0), zai(1), deepseek(2)
        # deepinfra skipped for Claude, zai(1) 429 -> fallthrough, deepseek(2) wins
        assert ridx == 2  # deepseek (idx 2) won after zai (idx 1) fell through
        assert route is not None
        assert route.provider == "deepseek"
        assert len(calls) == 2

        # Check usage store: zai (1) fallthrough, deepseek (2) success
        attempts = store.get_attempts(42)
        # deepinfra (0) skipped + zai (1) fallthrough + deepseek (2) success
        assert len(attempts) == 3
        fallthroughs = [a for a in attempts if a.outcome == "fallthrough"]
        successes = [a for a in attempts if a.outcome == "success"]
        skipped = [a for a in attempts if a.outcome == "skipped"]
        assert len(fallthroughs) == 1
        assert fallthroughs[0].route_idx == 1  # zai
        assert fallthroughs[0].error_class == "429"
        assert len(successes) == 1
        assert successes[0].route_idx == 2  # deepseek
        assert successes[0].output_tokens == 20
        assert len(skipped) == 1  # deepinfra (0)

    def test_hermes_429_fallthrough(self, cfg, store):
        """Hermes agent: deepinfra returns 429, zai succeeds."""
        agent = _make_hermes_agent()
        calls = []

        def fake_subprocess_run(cmd, **kw):
            calls.append(cmd)
            if len(calls) == 1:
                return _fake_proc(stderr="HTTP 429 Too Many Requests", returncode=1)
            return _fake_proc(stdout="VERDICT: ACCEPT-READY\n", returncode=0)

        with patch.dict(os.environ, {
            "DEEPINFRA_API_KEY": "key1",
            "ZAI_API_KEY": "key2",
            "DEEPSEEK_API_KEY": "key3",
        }, clear=True):
            avail = available_routes(cfg.parsed_routes)
            with patch("src.agent_hermes.subprocess.run", side_effect=fake_subprocess_run):
                usage, verdict, ridx, route = agent.run_with_chain(
                    "test brief", avail, cfg, "test-token",
                    usage_store=store, job_id=43,
                )

        assert verdict == "ACCEPT-READY"
        assert ridx == 1  # zai (idx 1) won after deepinfra (idx 0) fell through
        assert len(calls) == 2

        attempts = store.get_attempts(43)
        assert len(attempts) == 2
        assert attempts[0].route_idx == 0
        assert attempts[0].outcome == "fallthrough"
        assert attempts[1].route_idx == 1
        assert attempts[1].outcome == "success"


# ===================================================================
# 402 fallthrough
# ===================================================================

@requires_pg
class TestFallthrough402:
    """402 (payment required) fall through."""

    def test_claude_402_fallthrough(self, cfg, store):
        """Claude: deepinfra skipped, zai 402 falls through to deepseek."""
        agent = _make_claude_agent()
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if len(calls) == 1:
                return _fake_proc(stdout=_claude_json(api_error_status="402 Payment Required"))
            return _fake_proc(stdout=_claude_json(result_text="VERDICT: ACCEPT-READY"), returncode=0)

        with patch.dict(os.environ, {
            "DEEPINFRA_API_KEY": "k1", "ZAI_API_KEY": "k2", "DEEPSEEK_API_KEY": "k3",
        }, clear=True):
            avail = available_routes(cfg.parsed_routes)
            with patch("src.agent_claude.subprocess.run", side_effect=fake_run):
                usage, verdict, ridx, route = agent.run_with_chain(
                    "brief", avail, cfg, "t", usage_store=store, job_id=44)

        # cfg has 3 routes: deepinfra(0), zai(1), deepseek(2)
        # deepinfra skipped for Claude, zai(1) 402 -> fallthrough, deepseek(2) wins
        assert verdict == "ACCEPT-READY"
        assert ridx == 2  # deepseek (idx 2) won after zai (idx 1) fell through
        assert len(calls) == 2

        attempts = store.get_attempts(44)
        fallthroughs = [a for a in attempts if a.outcome == "fallthrough"]
        assert len(fallthroughs) == 1
        assert fallthroughs[0].error_class == "402"


# ===================================================================
# 5xx fallthrough
# ===================================================================

@requires_pg
class TestFallthrough5xx:
    """5xx errors fall through."""

    def test_claude_5xx_fallthrough(self, cfg, store):
        """Claude: deepinfra skipped, zai 503 falls through to deepseek."""
        agent = _make_claude_agent()
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if len(calls) == 1:
                return _fake_proc(stdout=_claude_json(api_error_status="503 Service Unavailable"))
            return _fake_proc(stdout=_claude_json(result_text="VERDICT: ACCEPT-READY"), returncode=0)

        with patch.dict(os.environ, {
            "DEEPINFRA_API_KEY": "k1", "ZAI_API_KEY": "k2", "DEEPSEEK_API_KEY": "k3",
        }, clear=True):
            avail = available_routes(cfg.parsed_routes)
            with patch("src.agent_claude.subprocess.run", side_effect=fake_run):
                usage, verdict, ridx, route = agent.run_with_chain(
                    "brief", avail, cfg, "t", usage_store=store, job_id=45)

        # cfg has 3 routes: deepinfra(0), zai(1), deepseek(2)
        # deepinfra skipped for Claude, zai(1) 503 -> fallthrough, deepseek(2) wins
        assert verdict == "ACCEPT-READY"
        assert ridx == 2  # deepseek (idx 2) won after zai (idx 1) fell through
        attempts = store.get_attempts(45)
        fallthroughs = [a for a in attempts if a.outcome == "fallthrough"]
        assert len(fallthroughs) == 1
        assert fallthroughs[0].error_class == "5xx"


# ===================================================================
# 400 does NOT fall through
# ===================================================================

@requires_pg
class TestNoFallthrough400:
    """400 (bad request) does NOT fall through — stops the chain."""

    def test_claude_400_stops_chain(self, cfg, store):
        """Claude: deepinfra skipped, zai 400 raises ApiError (re-raised)."""
        agent = _make_claude_agent()
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _fake_proc(stdout=_claude_json(api_error_status="400 Bad Request"))

        with patch.dict(os.environ, {
            "DEEPINFRA_API_KEY": "k1", "ZAI_API_KEY": "k2", "DEEPSEEK_API_KEY": "k3",
        }, clear=True):
            avail = available_routes(cfg.parsed_routes)
            with patch("src.agent_claude.subprocess.run", side_effect=fake_run):
                with pytest.raises(ApiError):
                    agent.run_with_chain(
                        "brief", avail, cfg, "t", usage_store=store, job_id=46)

        # Only one route attempted (zai, idx 1 — deepinfra idx 0 skipped)
        assert len(calls) == 1

        attempts = store.get_attempts(46)
        # deepinfra (0) skipped + zai (1) error
        assert len(attempts) == 2
        assert attempts[0].outcome == "skipped"
        assert attempts[1].outcome == "error"
        assert attempts[1].error_class == "400"

    def test_hermes_400_stops_chain(self, cfg, store):
        """Hermes: deepinfra 400 raises AgentError (re-raised)."""
        agent = _make_hermes_agent()
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _fake_proc(stderr="HTTP 400 Bad Request", returncode=1)

        with patch.dict(os.environ, {
            "DEEPINFRA_API_KEY": "k1", "ZAI_API_KEY": "k2", "DEEPSEEK_API_KEY": "k3",
        }, clear=True):
            avail = available_routes(cfg.parsed_routes)
            with patch("src.agent_hermes.subprocess.run", side_effect=fake_run):
                with pytest.raises(HermesAgentError):
                    agent.run_with_chain(
                        "brief", avail, cfg, "t", usage_store=store, job_id=47)

        assert len(calls) == 1

        attempts = store.get_attempts(47)
        assert len(attempts) == 1
        assert attempts[0].outcome == "error"


# ===================================================================
# All routes busy → failure recorded for each
# ===================================================================

@requires_pg
class TestAllRoutesExhausted:
    """All routes return fallthrough errors — all recorded, no winner."""

    def test_all_routes_429(self, cfg, store):
        """Claude: deepinfra skipped, zai and deepseek both 429 — all exhausted."""
        agent = _make_claude_agent()
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _fake_proc(stdout=_claude_json(api_error_status="429"))

        with patch.dict(os.environ, {
            "DEEPINFRA_API_KEY": "k1", "ZAI_API_KEY": "k2", "DEEPSEEK_API_KEY": "k3",
        }, clear=True):
            avail = available_routes(cfg.parsed_routes)
            with patch("src.agent_claude.subprocess.run", side_effect=fake_run):
                usage, verdict, ridx, route = agent.run_with_chain(
                    "brief", avail, cfg, "t", usage_store=store, job_id=48)

        assert verdict == ""
        assert ridx is None
        # cfg has 3 routes: deepinfra(0), zai(1), deepseek(2)
        # deepinfra skipped, zai(1) 429, deepseek(2) 429 = 2 attempts
        assert len(calls) == 2

        attempts = store.get_attempts(48)
        fallthroughs = [a for a in attempts if a.outcome == "fallthrough"]
        assert len(fallthroughs) == 2  # zai and deepseek


# ===================================================================
# Zero-token usage stays 0
# ===================================================================

@requires_pg
class TestZeroTokenUsage:
    """Zero-token usage stays 0 (not None)."""

    def test_zero_tokens_recorded(self, cfg, store):
        """Claude: deepinfra skipped, zai succeeds with zero tokens."""
        agent = _make_claude_agent()
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _fake_proc(
                stdout=_claude_json(result_text="VERDICT: ACCEPT-READY", in_tok=0, out_tok=0),
            )

        with patch.dict(os.environ, {
            "DEEPINFRA_API_KEY": "k1", "ZAI_API_KEY": "k2", "DEEPSEEK_API_KEY": "k3",
        }, clear=True):
            avail = available_routes(cfg.parsed_routes)
            with patch("src.agent_claude.subprocess.run", side_effect=fake_run):
                usage, verdict, ridx, route = agent.run_with_chain(
                    "brief", avail, cfg, "t", usage_store=store, job_id=49)

        assert verdict == "ACCEPT-READY"
        assert ridx == 1  # zai (idx 1) is first non-skipped route
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0

        attempts = store.get_attempts(49)
        # deepinfra (0) skipped + zai (1) success
        successes = [a for a in attempts if a.outcome == "success"]
        assert len(successes) == 1
        assert successes[0].uncached_input == 0
        assert successes[0].output_tokens == 0


# ===================================================================
# Claude agent skips route with no Anthropic endpoint
# ===================================================================

@requires_pg
class TestClaudeSkipsNoEndpoint:
    """Claude agent skips a route with no Anthropic-wire endpoint."""

    def test_skip_unknown_provider(self, cfg, store):
        """A route with an unknown provider (not in CLAUDE_WIRE_ENDPOINTS)
        is skipped for the Claude agent."""
        agent = _make_claude_agent()
        # Add a route with a provider that has no Anthropic endpoint
        routes = [
            (0, Route("unknown_provider", "some-model")),
            (1, Route("deepseek", "deepseek-v4-flash")),
        ]

        def fake_run(cmd, **kw):
            return _fake_proc(stdout=_claude_json(result_text="VERDICT: ACCEPT-READY"), returncode=0)

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k1"}, clear=True):
            with patch("src.agent_claude.subprocess.run", side_effect=fake_run):
                usage, verdict, ridx, route = agent.run_with_chain(
                    "brief", routes, cfg, "t", usage_store=store, job_id=50)

        assert verdict == "ACCEPT-READY"
        assert ridx == 1  # second route won (first was skipped)

        attempts = store.get_attempts(50)
        assert len(attempts) == 2
        assert attempts[0].outcome == "skipped"
        assert attempts[0].error_class == "no_anthropic_wire"
        assert attempts[1].outcome == "success"


# ===================================================================
# Hermes skips route it can't serve
# ===================================================================

@requires_pg
class TestHermesSkipsNoProvider:
    """Hermes agent skips a route it can't serve."""

    def test_skip_unknown_provider(self, cfg, store):
        agent = _make_hermes_agent()
        # Use a provider not in HERMES_PROVIDER_MAP
        routes = [
            (0, Route("nonexistent", "some-model")),
            (1, Route("zai", "glm-5.3-flash")),
        ]

        def fake_run(cmd, **kw):
            return _fake_proc(stdout="VERDICT: ACCEPT-READY\n", returncode=0)

        with patch.dict(os.environ, {"ZAI_API_KEY": "k1"}, clear=True):
            with patch("src.agent_hermes.subprocess.run", side_effect=fake_run):
                usage, verdict, ridx, route = agent.run_with_chain(
                    "brief", routes, cfg, "t", usage_store=store, job_id=51)

        assert verdict == "ACCEPT-READY"
        assert ridx == 1  # second route won (first was skipped)

        attempts = store.get_attempts(51)
        assert len(attempts) == 2  # skip + success
        assert attempts[0].outcome == "skipped"
        assert attempts[0].error_class == "no_hermes_provider"
        assert attempts[1].outcome == "success"


# ===================================================================
# credits_for unchanged for the same usage
# ===================================================================

class TestCreditsUnchanged:
    """credits_for unchanged for the same usage (backward compat)."""

    def test_credits_same_for_same_usage(self):
        from src.neo_worker import credits_for
        usage = {"input_tokens": 1000, "output_tokens": 100}
        # Flash (non-builder)
        c1 = credits_for(usage, builder=False)
        # Pro (builder)
        c2 = credits_for(usage, builder=True)
        assert c1 == 69  # known value
        assert c2 == 403  # known value


@requires_pg
class TestCreditsUnchangedRouteChain:
    """Backward-compat credits through the PG-backed route chain."""

    def test_credits_unchanged_with_route_chain(self, cfg, store):
        """Same usage via route chain produces same credits as before."""
        from src.neo_worker import credits_for
        usage = {"input_tokens": 500, "output_tokens": 50}
        c = credits_for(usage, builder=False)
        assert c == 34  # 500*0.30/1e6 + 50*2.50/1e6 = 0.000275 * 1.26 * 100000 = 34.65 -> 34


# ===================================================================
# Missing key skips the route
# ===================================================================

@requires_pg
class TestMissingKeySkips:
    """Missing API key skips the route (logged once, never a crash)."""

    def test_missing_deepinfra_key_skips(self, cfg, store):
        agent = _make_claude_agent()
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _fake_proc(stdout=_claude_json(result_text="VERDICT: ACCEPT-READY"), returncode=0)

        # Only set ZAI_API_KEY, not DEEPINFRA_API_KEY
        with patch.dict(os.environ, {
            "ZAI_API_KEY": "k2",
            "DEEPSEEK_API_KEY": "k3",
        }, clear=True):
            avail = available_routes(cfg.parsed_routes)
            # avail should skip deepinfra (no key) and start with zai
            with patch("src.agent_claude.subprocess.run", side_effect=fake_run):
                usage, verdict, ridx, route = agent.run_with_chain(
                    "brief", avail, cfg, "t", usage_store=store, job_id=52)

        assert verdict == "ACCEPT-READY"
        # zai is the first available route (deepinfra skipped due to missing key)
        assert route is not None
        assert route.provider == "zai"
        assert len(calls) == 1  # only one call made


# ===================================================================
# Startup validation
# ===================================================================

class TestStartupValidation:
    """Startup refuses a bad NEO_ROUTES (unknown provider/model)."""

    def test_unknown_provider_refused(self):
        cfg = Config()
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.neo_routes = "unknown_provider:some-model"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        with pytest.raises(ValueError, match="unknown provider"):
            cfg.validate()

    def test_malformed_route_refused(self):
        cfg = Config()
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.neo_routes = "just-a-string"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        with pytest.raises(ValueError, match="expected 'provider:model'"):
            cfg.validate()

    def test_valid_routes_accepted(self):
        cfg = Config()
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.neo_routes = "deepinfra:glm-5.3-flash,zai:glm-5.3-flash"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.validate()
        assert cfg.parsed_routes is not None
        assert len(cfg.parsed_routes) == 2


# ===================================================================
# Route order matches Reviews B4
# ===================================================================

class TestRouteOrder:
    """Route order matches Reviews B4."""

    def test_default_chain_order(self):
        chain = resolve_route_chain("")
        expected = [
            ("deepinfra", "glm-5.3-flash"),
            ("zai", "glm-5.3-flash"),
            ("deepinfra", "deepseek-v4-flash"),
            ("deepinfra", "deepseek-v4-flash-0731"),
            ("deepseek", "deepseek-v4-flash"),
            ("deepinfra", "deepseek-v4.1-flash-1017"),
        ]
        assert [(r.provider, r.model) for r in chain] == expected


# ===================================================================
# Claude agent builds correct per-route agent
# ===================================================================

class TestClaudePerRouteAgent:
    """Claude agent builds correct per-route agent with Anthropic wire endpoints."""

    def test_claude_endpoint_for_deepseek(self):
        assert CLAUDE_WIRE_ENDPOINTS.get("deepseek") == "https://api.deepseek.com/anthropic"

    def test_claude_endpoint_for_zai(self):
        assert CLAUDE_WIRE_ENDPOINTS.get("zai") == "https://api.z.ai/api/anthropic"

    def test_claude_endpoint_for_deepinfra(self):
        # Defect 3: deepinfra has no Anthropic-wire endpoint — the local
        # Claude-wire bridge serves DeepSeek/Bedrock only and does NOT
        # accept forwarded provider credentials for GLM.
        assert CLAUDE_WIRE_ENDPOINTS.get("deepinfra") is None

    def test_claude_skips_unknown_provider(self):
        assert CLAUDE_WIRE_ENDPOINTS.get("openrouter") is None
        assert CLAUDE_WIRE_ENDPOINTS.get("openai") is None

    def test_hermes_provider_mapping(self):
        assert HERMES_PROVIDER_MAP.get("deepinfra") == "deepinfra"
        assert HERMES_PROVIDER_MAP.get("zai") == "zai"
        assert HERMES_PROVIDER_MAP.get("deepseek") == "deepseek"


# ===================================================================
# Integrator round: status-line formats, builder stderr, no reset over a push
# ===================================================================

class TestStatusLineFormats:
    @pytest.mark.parametrize("stderr,expected", [
        ("HTTP/1.1 400 invalid model id", (False, "400")),
        ("HTTP: 400", (False, "400")),
        ("http status 400", (False, "400")),
        ("HTTP/2 429 slow down", (True, "429")),
        ("status 503 upstream", (True, "5xx")),
        ("read 400 bytes then 150 files", (True, "exit_nonzero")),
    ])
    def test_classification(self, stderr, expected):
        from src.agent_hermes import HermesAgent, NonZeroExit
        agent = HermesAgent.__new__(HermesAgent)
        got = agent._classify_agent_error(NonZeroExit("x"), agent, proc=_fake_proc(stderr=stderr, returncode=1))
        assert got == expected

    @pytest.mark.parametrize("stderr,expected", [
        ("hermes timed out after 30s", (True, "timeout")),
        ("HTTP 429 Too Many Requests", (True, "429")),
        ("HTTP/1.1 402 Payment Required", (True, "402")),
        ("Error code: 402", (True, "402")),
        ("HTTP/1.1 503 Service Unavailable", (True, "5xx")),
        ("HTTP/1.1 400 Bad Request", (False, "400")),
        ("upstream said: bad request", (False, "400")),
    ])
    def test_error_classes(self, stderr, expected):
        """HIGH-1: stderr classification needs `import re` in agent_hermes.

        This is the live-repro path: NonZeroExit plus a CompletedProcess
        carrying the HTTP error text.
        """
        from src.agent_hermes import HermesAgent, NonZeroExit
        agent = HermesAgent.__new__(HermesAgent)
        got = agent._classify_agent_error(
            NonZeroExit("x"), agent,
            proc=_fake_proc(stderr=stderr, returncode=1),
        )
        assert got == expected


class TestBuilderChainNeverResetsOverCommittedWork:
    def test_failed_route_that_committed_stops_the_chain(self, cfg, tmp_path):
        from src import neo_builder as nb
        from src.agent_hermes import NonZeroExit
        repo = tmp_path / "wt"
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
        subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "base"], check=True)
        runs = []

        def fake_run(self, brief):
            runs.append(self.provider)
            subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "route fix"], check=True)
            self._last_proc = _fake_proc(stderr="HTTP 503 upstream", returncode=1)
            raise NonZeroExit("exit 1")

        agent = nb.BuilderAgent(work_dir=str(repo), hermes_bin="/bin/false", provider="zai", model="glm-5.3-flash")
        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "k1", "ZAI_API_KEY": "k2", "DEEPSEEK_API_KEY": "k3"}, clear=True), \
             patch.object(nb.BuilderAgent, "run", fake_run):
            avail = available_routes(cfg.parsed_routes)
            usage, verdict, idx, route = agent.run_with_chain("brief", avail, cfg, "t")
        assert len(runs) == 1, "the next route never runs over committed work"
        assert verdict == "" and route is not None
        log = subprocess.run(git + ["log", "--format=%s"], capture_output=True, text=True).stdout
        assert "route fix" in log, "the committed fix was not reset away"


def test_builder_main_ensures_usage_schema(monkeypatch):
    """Behavioural (Defect 6 — no source text): drive nb.main() and prove
    it creates the usage store and calls ensure_schema() ITSELF before the
    poll loop — the builder process writes usage rows, so it must not
    depend on the webhook process having created the table first."""
    from src import neo_builder as nb
    from src import usage as usage_mod

    calls: list[str] = []

    class StopAfterSchemas(RuntimeError):
        pass

    class _FakeJobs:
        def __init__(self, dsn):
            self.dsn = dsn

        def ensure_schema(self):
            calls.append("jobs.ensure_schema")

    class _FakeUsageStore:
        def __init__(self, dsn):
            self.dsn = dsn

        def ensure_schema(self):
            calls.append("usage.ensure_schema")

    def _boom_pool(*args, **kwargs):
        raise StopAfterSchemas()   # stop right after both schemas exist

    monkeypatch.setattr(nb, "Jobs", _FakeJobs)
    monkeypatch.setattr(nb, "github_app_for", lambda cfg: object())
    monkeypatch.setattr(usage_mod, "UsageStore", _FakeUsageStore)
    monkeypatch.setattr(nb.concurrent.futures, "ThreadPoolExecutor",
                        _boom_pool)
    # env for a config that passes require().validate() hermetically
    monkeypatch.setenv("RHOBEAR_NEO_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("NEO_DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/neo_test")
    monkeypatch.setenv("NEO_CLAUDE_BIN", sys.executable)

    with pytest.raises(StopAfterSchemas):
        nb.main()
    assert calls == ["jobs.ensure_schema", "usage.ensure_schema"], calls
