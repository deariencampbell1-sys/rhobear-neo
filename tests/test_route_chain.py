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
from unittest.mock import patch, MagicMock

import pytest

from src.config import Config, CLAUDE_WIRE_ENDPOINTS, HERMES_PROVIDER_MAP
from src.usage import Route, UsageAttempt, UsageStore, available_routes, resolve_route_chain
from src.agent_claude import ClaudeAgent
from src.agent_hermes import HermesAgent

# ---------------------------------------------------------------------------
# Test database
# ---------------------------------------------------------------------------

DSN = os.environ.get("NEO_TEST_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="NEO_TEST_DSN not set")


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
        gh_token="ghp_test",
    )
    defaults.update(kw)
    return ClaudeAgent(**defaults)


def _make_hermes_agent(**kw):
    defaults = dict(
        hermes_bin="/usr/bin/hermes",
        provider="zai",
        model="glm-5.3-flash",
        timeout=30,
        gh_token="ghp_test",
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
    cfg.claude_bin = "/usr/bin/claude"
    cfg.gh_token = "ghp_test"
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

class TestFallthrough429:
    """Route 1 returns 429 → route 2 runs and is recorded."""

    def test_claude_429_fallthrough(self, cfg, store):
        """Claude agent: route 1 (deepinfra) returns 429, route 2 (zai) succeeds."""
        agent = _make_claude_agent()

        # Mock subprocess.run to simulate 429 on first call, success on second.
        calls = []

        def fake_subprocess_run(cmd, **kw):
            calls.append(cmd)
            if len(calls) == 1:
                # Route 1 (deepinfra) — 429 error
                return _fake_proc(
                    stdout=_claude_json(api_error_status="429"),
                    returncode=0,
                )
            # Route 2 (zai) — success
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
                    "test brief", avail, cfg, "ghp_test",
                    usage_store=store, job_id=42,
                )

        assert verdict == "ACCEPT-READY"
        assert ridx == 1  # second route (zai) won
        assert route is not None
        assert route.provider == "zai"
        assert len(calls) == 2

        # Check usage store: route 0 recorded as fallthrough, route 1 as success
        attempts = store.get_attempts(42)
        assert len(attempts) == 2
        assert attempts[0].route_idx == 0
        assert attempts[0].outcome == "fallthrough"
        assert attempts[0].error_class == "429"
        assert attempts[1].route_idx == 1
        assert attempts[1].outcome == "success"
        assert attempts[1].output_tokens == 20

    def test_hermes_429_fallthrough(self, cfg, store):
        """Hermes agent: route 1 returns 429, route 2 succeeds."""
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
                    "test brief", avail, cfg, "ghp_test",
                    usage_store=store, job_id=43,
                )

        assert verdict == "ACCEPT-READY"
        assert ridx == 1
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

class TestFallthrough402:
    """402 (payment required) fall through."""

    def test_claude_402_fallthrough(self, cfg, store):
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

        assert verdict == "ACCEPT-READY"
        assert ridx == 1
        assert len(calls) == 2

        attempts = store.get_attempts(44)
        assert attempts[0].outcome == "fallthrough"
        assert attempts[0].error_class == "402"


# ===================================================================
# 5xx fallthrough
# ===================================================================

class TestFallthrough5xx:
    """5xx errors fall through."""

    def test_claude_5xx_fallthrough(self, cfg, store):
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

        assert verdict == "ACCEPT-READY"
        assert ridx == 1
        attempts = store.get_attempts(45)
        assert attempts[0].outcome == "fallthrough"
        assert attempts[0].error_class == "5xx"


# ===================================================================
# 400 does NOT fall through
# ===================================================================

class TestNoFallthrough400:
    """400 (bad request) does NOT fall through — stops the chain."""

    def test_claude_400_stops_chain(self, cfg, store):
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
                usage, verdict, ridx, route = agent.run_with_chain(
                    "brief", avail, cfg, "t", usage_store=store, job_id=46)

        # Should have stopped after route 0 — no fallthrough
        assert verdict == ""
        assert ridx == 0  # route 0 is where it stopped
        assert len(calls) == 1  # only one route attempted

        attempts = store.get_attempts(46)
        assert len(attempts) == 1
        assert attempts[0].outcome == "error"
        assert attempts[0].error_class == "400"

    def test_hermes_400_stops_chain(self, cfg, store):
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
                usage, verdict, ridx, route = agent.run_with_chain(
                    "brief", avail, cfg, "t", usage_store=store, job_id=47)

        assert verdict == ""
        assert ridx == 0
        assert len(calls) == 1

        attempts = store.get_attempts(47)
        assert attempts[0].outcome == "error"


# ===================================================================
# All routes busy → failure recorded for each
# ===================================================================

class TestAllRoutesExhausted:
    """All routes return fallthrough errors — all recorded, no winner."""

    def test_all_routes_429(self, cfg, store):
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
        assert len(calls) == 3  # all 3 routes tried

        attempts = store.get_attempts(48)
        assert len(attempts) == 3
        for a in attempts:
            assert a.outcome == "fallthrough"


# ===================================================================
# Zero-token usage stays 0
# ===================================================================

class TestZeroTokenUsage:
    """Zero-token usage stays 0 (not None)."""

    def test_zero_tokens_recorded(self, cfg, store):
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
        assert ridx == 0
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0

        attempts = store.get_attempts(49)
        assert len(attempts) == 1
        assert attempts[0].uncached_input == 0
        assert attempts[0].output_tokens == 0


# ===================================================================
# Claude agent skips route with no Anthropic endpoint
# ===================================================================

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

    def test_credits_unchanged_with_route_chain(self, cfg, store):
        """Same usage via route chain produces same credits as before."""
        from src.neo_worker import credits_for
        usage = {"input_tokens": 500, "output_tokens": 50}
        c = credits_for(usage, builder=False)
        assert c == 34  # 500*0.30/1e6 + 50*2.50/1e6 = 0.000275 * 1.26 * 100000 = 34.65 -> 34


# ===================================================================
# Missing key skips the route
# ===================================================================

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
        cfg.neo_routes = "unknown_provider:some-model"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        with pytest.raises(ValueError, match="unknown provider"):
            cfg.validate()

    def test_malformed_route_refused(self):
        cfg = Config()
        cfg.neo_routes = "just-a-string"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        with pytest.raises(ValueError, match="expected 'provider:model'"):
            cfg.validate()

    def test_valid_routes_accepted(self):
        cfg = Config()
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
        assert CLAUDE_WIRE_ENDPOINTS.get("deepinfra") == "http://127.0.0.1:8790"

    def test_claude_skips_unknown_provider(self):
        assert CLAUDE_WIRE_ENDPOINTS.get("openrouter") is None
        assert CLAUDE_WIRE_ENDPOINTS.get("openai") is None

    def test_hermes_provider_mapping(self):
        assert HERMES_PROVIDER_MAP.get("deepinfra") == "deepinfra"
        assert HERMES_PROVIDER_MAP.get("zai") == "zai"
        assert HERMES_PROVIDER_MAP.get("deepseek") == "deepseek"
