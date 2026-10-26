"""Tests for B4 Neo provider metering — route chain, per-attempt usage, credits.

Covers:
  - Route parsing (default chain, custom, malformed, unknown provider)
  - Route ordering (same order as Reviews B4)
  - 429/402/5xx fall through; 400 does not
  - Missing API key skips the route (logged, never a crash)
  - Startup refuses a bad NEO_ROUTES (unknown provider/model)
  - Per-attempt usage recorded for success and failure (real Postgres)
  - Unknown usage stays None (never zero)
  - Credits identical to before for the same usage (backward compat)
  - credits_for works with both raw agent and summed usage dicts
  - Real Postgres via NEO_TEST_DSN for persistence

Run:  NEO_TEST_DSN='<dsn>' timeout 900 /work/repos/rb-B4-neo-durable-jobs/.venv/bin/python -m pytest tests/test_usage.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any
from unittest.mock import patch, MagicMock

import psycopg
import pytest

from src.config import Config
from src.usage import (
    Route, Outcome, UsageAttempt, UsageStore,
    resolve_route_chain, available_routes,
    classify_http_error, is_fallthrough,
    KNOWN_PROVIDERS, SCHEMA,
)
from src.neo_worker import credits_for

# ---------------------------------------------------------------------------
# Test database
# ---------------------------------------------------------------------------

DSN = os.environ.get("NEO_TEST_DSN", "")
requires_pg = pytest.mark.skipif(not DSN, reason="NEO_TEST_DSN not set")


def _wipe() -> None:
    with psycopg.connect(DSN, autocommit=True) as c:
        for table in ("neo_usage", "neo_jobs", "neo_actions", "neo_installs"):
            exists = c.execute(
                "SELECT to_regclass(%s)", (table,)
            ).fetchone()[0]
            if exists is not None:
                c.execute(f"DELETE FROM {table}")


@pytest.fixture()
def store():
    s = UsageStore(DSN)
    s.ensure_schema()
    _wipe()
    return s


# ===================================================================
# Route parsing
# ===================================================================

class TestRouteParsing:
    """Route.parse and resolve_route_chain."""

    def test_default_chain_has_six_routes(self):
        chain = Route.default_chain()
        assert len(chain) == 6
        assert chain[0] == Route("deepinfra", "glm-5.3-flash")
        assert chain[1] == Route("zai", "glm-5.3-flash")
        assert chain[2] == Route("deepinfra", "deepseek-v4-flash")
        assert chain[3] == Route("deepinfra", "deepseek-v4-flash-0731")
        assert chain[4] == Route("deepseek", "deepseek-v4-flash")
        assert chain[5] == Route("deepinfra", "deepseek-v4.1-flash-1017")

    def test_default_order_matches_reviews_b4(self):
        """Same order as Reviews B4: GLM 5.3 Flash on DeepInfra -> Z.ai
        direct GLM 5.3 Flash -> DeepInfra V4 Flash -> V4 Flash 0731 ->
        DeepSeek direct Flash -> DeepInfra V4.1 Flash 1017."""
        chain = Route.default_chain()
        expected = [
            ("deepinfra", "glm-5.3-flash"),
            ("zai", "glm-5.3-flash"),
            ("deepinfra", "deepseek-v4-flash"),
            ("deepinfra", "deepseek-v4-flash-0731"),
            ("deepseek", "deepseek-v4-flash"),
            ("deepinfra", "deepseek-v4.1-flash-1017"),
        ]
        assert [(r.provider, r.model) for r in chain] == expected

    def test_parse_valid_route(self):
        r = Route.parse("deepinfra:glm-5.3-flash")
        assert r.provider == "deepinfra"
        assert r.model == "glm-5.3-flash"

    def test_parse_case_insensitive_provider(self):
        r = Route.parse("DeepInfra:glm-5.3-flash")
        assert r.provider == "deepinfra"

    def test_parse_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown provider"):
            Route.parse("unknown:some-model")

    def test_parse_missing_model_raises(self):
        with pytest.raises(ValueError, match="expected 'provider:model'"):
            Route.parse("deepinfra:")

    def test_parse_missing_provider_raises(self):
        with pytest.raises(ValueError, match="expected 'provider:model'"):
            Route.parse(":some-model")

    def test_parse_empty_raises(self):
        with pytest.raises(ValueError, match="expected 'provider:model'"):
            Route.parse("")

    def test_parse_whitespace_stripped(self):
        r = Route.parse("  deepinfra : glm-5.3-flash  ")
        assert r.provider == "deepinfra"
        assert r.model == "glm-5.3-flash"

    def test_resolve_chain_default(self):
        chain = resolve_route_chain("")
        assert len(chain) == 6

    def test_resolve_chain_custom(self):
        chain = resolve_route_chain("deepinfra:model-a,zai:model-b")
        assert len(chain) == 2
        assert chain[0] == Route("deepinfra", "model-a")
        assert chain[1] == Route("zai", "model-b")

    def test_resolve_chain_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown provider"):
            resolve_route_chain("unknown:model")

    def test_env_key_name(self):
        assert Route("deepinfra", "m").env_key_name == "DEEPINFRA_API_KEY"
        assert Route("zai", "m").env_key_name == "ZAI_API_KEY"
        assert Route("deepseek", "m").env_key_name == "DEEPSEEK_API_KEY"
        assert Route("openrouter", "m").env_key_name == "OPENROUTER_API_KEY"


# ===================================================================
# Available routes (API key check)
# ===================================================================

class TestAvailableRoutes:
    """Missing API key skips the route (logged once, never a crash mid-run)."""

    def test_all_routes_available_with_keys(self):
        chain = [Route("deepinfra", "m1"), Route("zai", "m2")]
        with patch.dict(os.environ, {
            "DEEPINFRA_API_KEY": "key1",
            "ZAI_API_KEY": "key2",
        }, clear=True):
            avail = available_routes(chain)
        assert len(avail) == 2
        assert avail[0] == (0, Route("deepinfra", "m1"))
        assert avail[1] == (1, Route("zai", "m2"))

    def test_missing_key_skips_route(self):
        """A route without a key is skipped — never a crash mid-run."""
        chain = [Route("deepinfra", "m1"), Route("zai", "m2")]
        with patch.dict(os.environ, {
            "DEEPINFRA_API_KEY": "key1",
            # ZAI_API_KEY is missing
        }, clear=True):
            avail = available_routes(chain)
        assert len(avail) == 1
        assert avail[0] == (0, Route("deepinfra", "m1"))

    def test_all_missing_returns_empty(self):
        """When no API keys are set, the chain is empty — no routes available."""
        chain = Route.default_chain()
        with patch.dict(os.environ, {}, clear=True):
            avail = available_routes(chain)
        assert len(avail) == 0

    def test_route_api_key_available(self):
        r = Route("deepinfra", "m")
        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "key"}, clear=True):
            assert r.api_key_available() is True

    def test_route_api_key_missing(self):
        r = Route("deepinfra", "m")
        with patch.dict(os.environ, {}, clear=True):
            assert r.api_key_available() is False

    def test_route_api_key_empty_string(self):
        r = Route("deepinfra", "m")
        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": ""}, clear=True):
            assert r.api_key_available() is False


# ===================================================================
# Fallthrough classification
# ===================================================================

class TestFallthrough:
    """429/402/5xx/timeout-before-first-token fall through; 400 does not."""

    def test_429_fallthrough(self):
        assert is_fallthrough(429) is True

    def test_402_fallthrough(self):
        assert is_fallthrough(402) is True

    def test_500_fallthrough(self):
        assert is_fallthrough(500) is True
        assert is_fallthrough(503) is True
        assert is_fallthrough(599) is True

    def test_400_does_not_fallthrough(self):
        assert is_fallthrough(400) is False

    def test_200_does_not_fallthrough(self):
        assert is_fallthrough(200) is False

    def test_none_does_not_fallthrough(self):
        assert is_fallthrough(None) is False

    def test_timeout_before_first_token_fallthrough(self):
        assert is_fallthrough(None, timeout_before_first_token=True) is True

    def test_timeout_with_status_still_fallthrough(self):
        assert is_fallthrough(429, timeout_before_first_token=True) is True

    def test_classify_429(self):
        assert classify_http_error(429) == "429"

    def test_classify_402(self):
        assert classify_http_error(402) == "402"

    def test_classify_5xx(self):
        assert classify_http_error(502) == "5xx"

    def test_classify_400(self):
        assert classify_http_error(400) == "400"

    def test_classify_200(self):
        assert classify_http_error(200) == "200"

    def test_classify_none(self):
        assert classify_http_error(None) is None


# ===================================================================
# UsageAttempt dataclass
# ===================================================================

class TestUsageAttempt:
    """UsageAttempt construction and helpers."""

    def test_total_input_with_both(self):
        a = UsageAttempt(1, 1, 0, "deepinfra", "m", uncached_input=100, cached_input=50)
        assert a.total_input() == 150

    def test_total_input_with_uncached_only(self):
        a = UsageAttempt(1, 1, 0, "deepinfra", "m", uncached_input=100)
        assert a.total_input() == 100

    def test_total_input_with_none(self):
        a = UsageAttempt(1, 1, 0, "deepinfra", "m")
        assert a.total_input() == 0

    def test_to_dict_excludes_none(self):
        a = UsageAttempt(1, 1, 0, "deepinfra", "m",
                         uncached_input=100, output_tokens=50)
        d = a.to_dict()
        assert d["uncached_input"] == 100
        assert d["output_tokens"] == 50
        # None values should be excluded
        assert "cached_input" not in d
        assert "reasoning_tokens" not in d


# ===================================================================
# Real Postgres: usage persistence
# ===================================================================

@requires_pg
class TestUsageStorePG:
    """Per-attempt usage recorded for success and failure, idempotent per
    attempt, unknown usage stays None."""

    def test_record_and_retrieve_success(self, store):
        attempt = UsageAttempt(
            job_id=1, attempt_number=1, route_idx=0,
            provider="deepinfra", model="glm-5.3-flash",
            uncached_input=150, cached_input=50,
            output_tokens=80, latency_ms=2500,
            outcome="success",
        )
        assert store.record_attempt(attempt) is True

        attempts = store.get_attempts(1)
        assert len(attempts) == 1
        assert attempts[0].job_id == 1
        assert attempts[0].provider == "deepinfra"
        assert attempts[0].model == "glm-5.3-flash"
        assert attempts[0].uncached_input == 150
        assert attempts[0].cached_input == 50
        assert attempts[0].output_tokens == 80
        assert attempts[0].latency_ms == 2500
        assert attempts[0].outcome == "success"

    def test_record_failure(self, store):
        attempt = UsageAttempt(
            job_id=1, attempt_number=1, route_idx=0,
            provider="deepinfra", model="glm-5.3-flash",
            outcome="error", error_class="429",
        )
        assert store.record_attempt(attempt) is True

        attempts = store.get_attempts(1)
        assert len(attempts) == 1
        assert attempts[0].outcome == "error"
        assert attempts[0].error_class == "429"
        # Unknown tokens stay None
        assert attempts[0].uncached_input is None

    def test_unknown_usage_stays_none(self, store):
        """Unknown usage fields in the DB stay None (never silently zero)."""
        attempt = UsageAttempt(
            job_id=1, attempt_number=1, route_idx=0,
            provider="deepinfra", model="m",
            outcome="error", error_class="timeout",
            # All token fields omitted -> None
        )
        store.record_attempt(attempt)

        attempts = store.get_attempts(1)
        assert attempts[0].uncached_input is None
        assert attempts[0].cached_input is None
        assert attempts[0].output_tokens is None
        assert attempts[0].reasoning_tokens is None

    def test_idempotent_per_attempt(self, store):
        """Same (job_id, attempt_number, route_idx) upserts — no duplicate."""
        a1 = UsageAttempt(1, 1, 0, "deepinfra", "m", output_tokens=80, outcome="success")
        a2 = UsageAttempt(1, 1, 0, "deepinfra", "m", output_tokens=999, outcome="success")

        assert store.record_attempt(a1) is True  # inserted
        assert store.record_attempt(a2) is False  # conflict — NOTHING (keeps original)

        attempts = store.get_attempts(1)
        assert len(attempts) == 1
        # Original values preserved (first insert wins)
        assert attempts[0].output_tokens == 80

    def test_multiple_routes_per_job(self, store):
        """Multiple route attempts for the same job are all recorded."""
        store.record_attempt(UsageAttempt(1, 1, 0, "deepinfra", "m1", outcome="error", error_class="429"))
        store.record_attempt(UsageAttempt(1, 1, 1, "zai", "m2", outcome="success", output_tokens=50))
        store.record_attempt(UsageAttempt(1, 1, 2, "deepinfra", "m3", outcome="success", output_tokens=30))

        attempts = store.get_attempts(1)
        assert len(attempts) == 3
        assert [a.route_idx for a in attempts] == [0, 1, 2]

    def test_sum_usage_only_success(self, store):
        """sum_usage sums only successful attempts."""
        store.record_attempt(UsageAttempt(1, 1, 0, "d", "m", uncached_input=100, output_tokens=50, outcome="success"))
        store.record_attempt(UsageAttempt(1, 1, 1, "d", "m", uncached_input=50, output_tokens=20, outcome="error"))
        store.record_attempt(UsageAttempt(1, 1, 2, "d", "m", uncached_input=200, output_tokens=80, outcome="success"))

        summed = store.sum_usage(1)
        assert summed["uncached_input"] == 300  # 100 + 200
        assert summed["output_tokens"] == 130  # 50 + 80
        assert summed["cached_input"] == 0
        assert summed["reasoning_tokens"] == 0

    def test_sum_usage_empty(self, store):
        """No attempts returns all zeros."""
        summed = store.sum_usage(999)
        assert summed == {"uncached_input": 0, "cached_input": 0,
                          "output_tokens": 0, "reasoning_tokens": 0}

    def test_sum_usage_with_nulls(self, store):
        """NULL fields are excluded from the sum (not counted as 0)."""
        store.record_attempt(UsageAttempt(1, 1, 0, "d", "m",
                                          uncached_input=100,  # cached_input is None
                                          output_tokens=50,
                                          outcome="success"))
        summed = store.sum_usage(1)
        assert summed["uncached_input"] == 100
        assert summed["cached_input"] == 0  # NULL sums to 0 via COALESCE
        assert summed["output_tokens"] == 50

    def test_clear(self, store):
        store.record_attempt(UsageAttempt(1, 1, 0, "d", "m", outcome="success"))
        assert store.clear(1) == 1
        assert store.get_attempts(1) == []


# ===================================================================
# credits_for backward compatibility
# ===================================================================

class TestCreditsForBackwardCompat:
    """Credits identical to before for the same usage (no silent tariff change)."""

    def test_flash_same_as_before(self):
        """Same input_tokens/output_tokens as the old credits_for produces
        the same credit count."""
        usage = {"input_tokens": 1000, "output_tokens": 100}
        credits = credits_for(usage, builder=False)
        assert isinstance(credits, int)
        assert credits > 0

    def test_pro_same_as_before(self):
        usage = {"input_tokens": 1000, "output_tokens": 100}
        credits = credits_for(usage, builder=True)
        assert isinstance(credits, int)
        assert credits > 0

    def test_flash_specific_value(self):
        """Known value check: 1000 input + 100 output Flash."""
        usage = {"input_tokens": 1000, "output_tokens": 100}
        credits = credits_for(usage, builder=False)
        # Gemini Flash: 1000 * 0.30/1e6 + 100 * 2.50/1e6 = 0.0003 + 0.00025 = 0.00055
        # * 1.26 * 100000 = 69.3 -> 69
        assert credits == 69, credits

    def test_pro_specific_value(self):
        """Known value check: 1000 input + 100 output Pro."""
        usage = {"input_tokens": 1000, "output_tokens": 100}
        credits = credits_for(usage, builder=True)
        # Gemini Pro: 1000 * 2.00/1e6 + 100 * 12.00/1e6 = 0.002 + 0.0012 = 0.0032
        # * 1.26 * 100000 = 403.2 -> 403
        assert credits == 403, credits

    def test_cache_read_used_for_pro(self):
        """Cache read tokens are included in Pro (builder) pricing."""
        usage = {"input_tokens": 1000, "output_tokens": 100,
                 "cache_read_input_tokens": 500}
        credits = credits_for(usage, builder=True)
        # Gemini Pro: 1000 * 2.00/1e6 + 500 * 0.50/1e6 + 100 * 12.00/1e6
        # = 0.002 + 0.00025 + 0.0012 = 0.00345 * 1.26 * 100000 = 434.7 -> 434
        assert credits == 434, credits

    def test_cache_read_ignored_for_flash(self):
        """Cache read tokens are NOT included in Flash (non-builder) pricing."""
        usage = {"input_tokens": 1000, "output_tokens": 100,
                 "cache_read_input_tokens": 500}
        credits = credits_for(usage, builder=False)
        # Flash: 1000 * 0.30/1e6 + 100 * 2.50/1e6 = 0.00055 * 1.26 * 100000 = 69.3 -> 69
        assert credits == 69, credits

    def test_credits_with_summed_dict(self):
        """credits_for also accepts summed usage dict (uncached_input/cached_input)."""
        usage = {"uncached_input": 1000, "cached_input": 200, "output_tokens": 100}
        credits = credits_for(usage, builder=True)
        # Same as: input_tokens=1000, cache_read=200, output=100
        # Pro: 1000 * 2.00/1e6 + 200 * 0.50/1e6 + 100 * 12.00/1e6
        # = 0.002 + 0.0001 + 0.0012 = 0.0033 * 1.26 * 100000 = 415.8 -> 415
        assert credits == 415, credits

    def test_credits_zero_usage(self):
        """Zero usage yields zero credits."""
        assert credits_for({"input_tokens": 0, "output_tokens": 0}, builder=False) == 0
        assert credits_for({}, builder=False) == 0

    def test_credits_none_usage_is_zero(self):
        """usage=None is the explicit no-usage contract: charge 0, never raise."""
        assert credits_for(None, builder=False) == 0
        assert credits_for(None, builder=True) == 0

    def test_zero_token_count_is_real_not_missing(self):
        """Defect 9: 0 is a real token count — never coalesce through `or` to
        the other usage format's key."""
        # input_tokens=0 must not fall through to uncached_input=1000.
        assert credits_for(
            {"input_tokens": 0, "uncached_input": 1000, "output_tokens": 0},
            builder=False,
        ) == 0
        # cache_read_input_tokens=0 must not fall through to cached_input=500:
        # Pro = 1000*2.00/1e6 + 100*12.00/1e6 = 0.0032 * 1.26 * 100000 = 403.
        assert credits_for(
            {"input_tokens": 1000, "output_tokens": 100,
             "cache_read_input_tokens": 0, "cached_input": 500},
            builder=True,
        ) == 403

    def test_missing_keys_default_to_zero(self):
        """Missing keys in usage dict default to 0."""
        assert credits_for({}, builder=False) == 0


# ===================================================================
# Config validation for NEO_ROUTES
# ===================================================================

class TestConfigRoutesValidation:
    """Startup refuses a bad NEO_ROUTES (unknown provider/model)."""

    def test_default_routes_valid(self):
        cfg = Config()
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.validate()
        assert cfg.parsed_routes is not None
        assert len(cfg.parsed_routes) == 6

    def test_custom_routes_valid(self):
        cfg = Config()
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.neo_routes = "deepinfra:model-a,zai:model-b"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.validate()
        assert cfg.parsed_routes is not None
        assert len(cfg.parsed_routes) == 2
        assert cfg.parsed_routes[0].provider == "deepinfra"
        assert cfg.parsed_routes[1].provider == "zai"

    def test_unknown_provider_raises_on_validate(self):
        cfg = Config()
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.neo_routes = "unknown_provider:model"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        with pytest.raises(ValueError, match="unknown provider"):
            cfg.validate()

    def test_malformed_route_raises_on_validate(self):
        cfg = Config()
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.neo_routes = "just-a-string"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        with pytest.raises(ValueError, match="expected 'provider:model'"):
            cfg.validate()

    def test_empty_routes_uses_default(self):
        cfg = Config()
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.neo_routes = ""
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.validate()
        assert cfg.parsed_routes is not None
        assert len(cfg.parsed_routes) == 6

    def test_require_validate_chain(self):
        """require().validate() chain works with default routes."""
        cfg = Config()
        cfg.claude_bin = sys.executable  # hermetic: no production binary needed
        cfg.webhook_secret = "whs_test"
        cfg.deepseek_key = "sk-test"
        cfg.database_url = "postgres://localhost/test"
        cfg.gh_token = "test-token"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.require().validate()

    def test_neo_routes_env_var_default(self):
        """Default NEO_ROUTES env value matches the brief."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config()
        assert "deepinfra:glm-5.3-flash" in cfg.neo_routes
        assert "zai:glm-5.3-flash" in cfg.neo_routes
        assert "deepseek:deepseek-v4-flash" in cfg.neo_routes
        assert "deepinfra:deepseek-v4.1-flash-1017" in cfg.neo_routes


# ===================================================================
# KNOWN_PROVIDERS set
# ===================================================================

class TestKnownProviders:
    """The KNOWN_PROVIDERS set contains all expected providers."""

    def test_contains_deepinfra(self):
        assert "deepinfra" in KNOWN_PROVIDERS

    def test_contains_zai(self):
        assert "zai" in KNOWN_PROVIDERS

    def test_contains_deepseek(self):
        assert "deepseek" in KNOWN_PROVIDERS


# ===================================================================
# End-to-end: usage store wired through run_neo
# ===================================================================

@requires_pg
class TestE2EUsageRecording:
    """Per-attempt usage recorded for success and failure through run_neo."""

    @staticmethod
    def _cfg() -> Config:
        cfg = Config()
        cfg.database_url = DSN
        cfg.gh_token = "test-token"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.max_builder_rounds = 2
        cfg.auto_merge_default = False
        cfg.deepseek_timeout = 60
        cfg.neo_routes = "deepinfra:glm-5.3-flash,zai:glm-5.3-flash"
        setattr(cfg, "_rhobear_github_app", _FakeGitHubApp("test-ghs-token"))
        return cfg

    def test_usage_recorded_on_successful_run(self, store):
        """A successful run records usage in neo_usage."""
        from src.jobs import Jobs
        from src.neo_state import NeoState
        from src.neo_worker import run_neo
        from src.jev_review import JevAssessment

        jobs = Jobs(DSN)
        jobs.ensure_schema()
        state = NeoState(DSN)
        state.ensure_schema()
        cfg = self._cfg()

        SHA = "c" * 40
        wake = {"repo": "o/r", "sha": SHA, "green": True,
                "context": "rhobear-reviews", "pr_number": 5}

        with patch("src.neo_worker.resolve_pr", return_value=5), \
             patch("src.neo_worker._review_evidence",
                   return_value=_evidence(SHA)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent",
                   return_value=({"input_tokens": 1000, "output_tokens": 100},
                                 "ACCEPT-READY", 0, None)):
            run_neo(cfg, state, jobs, wake, usage_store=store)

        # Usage should be recorded — look up the job from the DB
        with psycopg.connect(DSN, autocommit=True) as c:
            row = c.execute(
                "SELECT id FROM neo_jobs WHERE kind='wake' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row is not None, "no wake job found"
        job_id = int(row[0])
        attempts = store.get_attempts(job_id)
        assert len(attempts) >= 1
        success_attempts = [a for a in attempts if a.outcome == "success"]
        assert len(success_attempts) >= 1

    def test_usage_not_recorded_when_no_store(self, store):
        """When usage_store is None, no crash — backward compat."""
        from src.jobs import Jobs
        from src.neo_state import NeoState
        from src.neo_worker import run_neo
        from src.jev_review import JevAssessment

        jobs = Jobs(DSN)
        jobs.ensure_schema()
        state = NeoState(DSN)
        state.ensure_schema()
        cfg = self._cfg()

        SHA = "d" * 40
        wake = {"repo": "o/r", "sha": SHA, "green": True,
                "context": "rhobear-reviews", "pr_number": 6}

        with patch("src.neo_worker.resolve_pr", return_value=6), \
             patch("src.neo_worker._review_evidence",
                   return_value=_evidence(SHA)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent",
                   return_value=({"input_tokens": 1, "output_tokens": 1},
                                 "ACCEPT-READY", 0, None)):
            # Should not crash despite no usage_store
            run_neo(cfg, state, jobs, wake, usage_store=None)

    def test_phase_row_has_route_info(self, store):
        """The phase row records which route actually ran."""
        from src.jobs import Jobs
        from src.neo_state import NeoState
        from src.neo_worker import run_neo
        from src.jev_review import JevAssessment

        jobs = Jobs(DSN)
        jobs.ensure_schema()
        state = NeoState(DSN)
        state.ensure_schema()
        cfg = self._cfg()

        SHA = "e" * 40
        wake = {"repo": "o/r", "sha": SHA, "green": True,
                "context": "rhobear-reviews", "pr_number": 7}

        with patch("src.neo_worker.resolve_pr", return_value=7), \
             patch("src.neo_worker._review_evidence",
                   return_value=_evidence(SHA)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent",
                   return_value=({"input_tokens": 100, "output_tokens": 50},
                                 "ACCEPT-READY", 0, None)):
            run_neo(cfg, state, jobs, wake, usage_store=store)

        # Check neo_actions for route info
        with psycopg.connect(DSN, autocommit=True) as c:
            rows = c.execute(
                "SELECT phase, detail FROM neo_actions "
                "WHERE repo='o/r' AND pr_number=7 ORDER BY id"
            ).fetchall()

        route_details = [r for r in rows if "route_provider" in (r[1] if isinstance(r[1], dict) else {})]
        # At minimum the triage row should have route info
        triage_rows = [r for r in rows if r[0] == "triage"]
        assert len(triage_rows) >= 1
        detail = triage_rows[-1][1]
        if isinstance(detail, str):
            detail = json.loads(detail)
        assert "route_provider" in detail
        assert "route_model" in detail


# ===================================================================
# Helpers
# ===================================================================

class _FakeGitHubApp:
    """Fake GitHubApp that returns a fixed token for any repo."""
    def __init__(self, token="test-ghs-token"):
        self._token = token

    def token_for_repo(self, repo, installation_id=None, permissions=None):
        return self._token


def _evidence(sha: str) -> dict:
    return {"ok": True, "head_sha": sha, "decided_sha": sha, "review_sha": sha,
            "review_bound_ok": True, "trusted_states": ["success"],
            "trusted_green": True, "failing_checks": 0, "rollup_green": True,
            "trusted_producer_ok": True, "review_sha_match": True,
            "unresolved_critical_high": 0, "coverage_complete": True}


# ===================================================================
# The allowed baseline failure must still fail
# ===================================================================

@requires_pg
class TestRealSmoke:
    """Smoke test that must pass (the only allowed failure is
    TestRealSmoke::test_smoke_run_with_temp_config from test_agent_claude.py)."""

    def test_smoke_usage_basic(self, store):
        """Minimal smoke: record and retrieve a usage attempt."""
        attempt = UsageAttempt(
            job_id=42, attempt_number=1, route_idx=0,
            provider="deepinfra", model="glm-5.3-flash",
            uncached_input=200, output_tokens=100,
            latency_ms=1500, outcome="success",
        )
        assert store.record_attempt(attempt) is True
        attempts = store.get_attempts(42)
        assert len(attempts) == 1
        assert attempts[0].uncached_input == 200
        assert attempts[0].output_tokens == 100
        assert attempts[0].latency_ms == 1500


# ---------------------------------------------------------------------------
# Round 2 (MEDIUM) — the dead divergent classifier is deleted
# ---------------------------------------------------------------------------

def test_divergent_classify_cli_fallthrough_is_deleted():
    """usage.classify_cli_fallthrough re-implemented fallthrough
    classification differently from the agents' _classify_agent_error ("cost
    1500 tokens" classified as 5xx there) and was imported but never used on
    the decision path.  One classifier family survives — anchored on
    api_error_status / HTTP <code> in agent_claude / agent_hermes — the
    divergent duplicate must be GONE."""
    import src.usage as usage
    assert not hasattr(usage, "classify_cli_fallthrough")


# ===================================================================
# W1-06 — RouteGate metering: per-attempt rows, anchored classification
# ===================================================================

class _FakeUsageStore:
    """In-memory UsageStore stand-in (no Postgres): same idempotency rule
    (unique on job_id + attempt_number + route_idx)."""

    def __init__(self):
        self.rows = []
        self._keys = set()

    def record_attempt(self, attempt):
        key = (attempt.job_id, attempt.attempt_number, attempt.route_idx)
        if key in self._keys:
            return False
        self._keys.add(key)
        self.rows.append(attempt)
        return True


def test_attempt_is_metered():
    """Edge 1 + 4: a 402 mid-attempt is ONE fallthrough row recorded with its
    route name and an incremented attempt counter; two attempts in one run are
    two rows; replaying a row never double counts."""
    from src.agent_claude import ClaudeAgent

    cfg = Config()
    cfg.claude_bin = sys.executable
    cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
    cfg.deepseek_model = "deepseek-v4-flash"
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.neo_routes = "zai:glm-5.3-flash,deepseek:deepseek-v4-flash"

    agent = ClaudeAgent(
        claude_bin=sys.executable, api_key="k",
        base_url="https://api.deepseek.com/anthropic",
        model="deepseek-v4-flash", timeout=30, gh_token="t",
    )
    store = _FakeUsageStore()
    head = "cd" * 20
    brief = (
        "TARGET: PR #44 in acme/widget at head " + head[:12] + "\n"
        "gh api repos/acme/widget/commits/" + head + "/status\n"
        "You have already spent 1/2 builder rounds on this PR.\n"
    )

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            # Route 1 (zai) — a provider 402 mid-attempt.
            return subprocess.CompletedProcess(
                args=["claude"], returncode=0, stderr="",
                stdout=json.dumps({
                    "type": "result", "is_error": True,
                    "subtype": "error_during_execution",
                    "api_error_status": "402",
                }),
            )
        # Route 2 (deepseek) — success, no cost carried by the response.
        return subprocess.CompletedProcess(
            args=["claude"], returncode=0, stderr="",
            stdout=json.dumps({
                "type": "result", "subtype": "success", "is_error": False,
                "result": "VERDICT: ACCEPT-READY",
                "usage": {"input_tokens": 50, "output_tokens": 20},
                "num_turns": 1, "stop_reason": "end_turn",
                "permission_denials": [],
            }),
        )

    with patch.dict(os.environ, {"ZAI_API_KEY": "k1",
                                 "DEEPSEEK_API_KEY": "k2"}, clear=True):
        avail = available_routes(resolve_route_chain(cfg.neo_routes))
        with patch("src.agent_claude.subprocess.run", side_effect=fake_run):
            usage, verdict, ridx, route = agent.run_with_chain(
                brief, avail, cfg, "t",
                usage_store=store, job_id=7,
            )

    assert verdict == "ACCEPT-READY"
    assert ridx == 1
    assert len(calls) == 2
    # Attempt counter incremented: one row per attempt, in order.
    assert [r.route_idx for r in store.rows] == [0, 1]

    first, second = store.rows
    # Edge 1: 402 → exactly one fallthrough, recorded with its route name.
    assert first.outcome == "fallthrough"
    assert first.error_class == "402"
    assert (first.provider, first.model) == ("zai", "glm-5.3-flash")
    assert first.route_name is None          # direct route — no RouteGate cap
    assert first.consumer == "reviews:zeus"  # written with consumer reviews:zeus
    assert first.pr_number == 44
    assert first.head_sha == head
    assert first.builder_round == 1
    assert first.cost_usd is None            # unknown stays None
    assert first.latency_ms is not None

    # Edge 4: two attempts → two rows, the winner metered with tokens.
    assert second.outcome == "success"
    assert (second.provider, second.model) == ("deepseek", "deepseek-v4-flash")
    assert second.uncached_input == 50
    assert second.output_tokens == 20
    assert second.cost_usd is None
    assert second.pr_number == 44 and second.head_sha == head

    # Idempotent per attempt: replaying the same row never double counts.
    assert store.record_attempt(first) is False
    assert len(store.rows) == 2


def test_fallthrough_classification_not_substring():
    """usage.classify_error_text is the text classifier for attempts and it
    anchors on `api_error_status` / `HTTP <code>` shapes only: "cost 1500
    tokens" contains "500" but must never classify as a 5xx."""
    from src.usage import classify_error_text

    fall, cls = classify_error_text("run burned cost 1500 tokens")
    assert fall is False
    assert cls != "5xx"

    assert classify_error_text("HTTP 503 Service Unavailable") == (True, "5xx")
    assert classify_error_text("402") == (True, "402")
    assert classify_error_text("Error code: 402 — payment required") == (True, "402")
    assert classify_error_text("status: 400 bad request") == (False, "400")
    assert classify_error_text("read 400 bytes then 150 files") != (False, "400")

    # Edge 3: a log line containing the word "error" in a normal message is
    # not a status at all — no anchor, no fallthrough.
    fall, _cls = classify_error_text("retrying after error, all tools available")
    assert fall is False


def test_stream_close_is_fallthrough():
    """Edge 2: a stream that closes before any output is a fallthrough —
    never a successful empty review."""
    from src.agent_claude import ClaudeAgent, MissingResult

    agent = ClaudeAgent.__new__(ClaudeAgent)
    closed = subprocess.CompletedProcess(
        args=["claude"], returncode=0, stdout="", stderr="")
    fall, cls = agent._classify_agent_error(
        MissingResult("no stdout from claude"), agent, proc=closed)
    assert fall is True
    assert cls == "stream_closed"

    # A run that DID take turns but produced no verdict text still fails
    # closed and is NOT a fallthrough — the model ran, don't silently retry.
    ran = subprocess.CompletedProcess(
        args=["claude"], returncode=0, stderr="",
        stdout=json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "", "num_turns": 2, "stop_reason": "end_turn",
        }),
    )
    fall2, cls2 = agent._classify_agent_error(
        MissingResult("no result text from claude"), agent, proc=ran)
    assert fall2 is False


@requires_pg
def test_attempt_consumer_carries_install_id(store):
    """L0's consumer shape: a row for a job that knows its GitHub App
    installation is written `reviews:zeus/<install_id>`; a job payload without
    one keeps the bare lane default — never an invented id."""
    from src.jobs import Jobs

    Jobs(DSN).ensure_schema()
    with psycopg.connect(DSN, autocommit=True) as c:
        job_id = c.execute(
            "INSERT INTO neo_jobs(kind, repo, head_sha, payload) "
            "VALUES ('wake', 'acme/widget', %s, %s) RETURNING id",
            ("ee" * 40, json.dumps({"installation_id": "987654321"})),
        ).fetchone()[0]
        bare_job_id = c.execute(
            "INSERT INTO neo_jobs(kind, repo, head_sha, payload) "
            "VALUES ('wake', 'acme/other', %s, '{}') RETURNING id",
            ("ef" * 40,),
        ).fetchone()[0]

    store.record_attempt(UsageAttempt(
        job_id=job_id, attempt_number=1, route_idx=0,
        provider="deepseek", model="deepseek-v4-flash",
        uncached_input=10, output_tokens=5, outcome="success",
    ))
    store.record_attempt(UsageAttempt(
        job_id=bare_job_id, attempt_number=1, route_idx=0,
        provider="deepseek", model="deepseek-v4-flash",
        outcome="success", consumer="reviews:zeus/42",
    ))

    with_install, bare = store.get_attempts(job_id)[0], \
        store.get_attempts(bare_job_id)[0]
    assert with_install.consumer == "reviews:zeus/987654321"
    # An already-qualified consumer is explicit — never rewritten.
    assert bare.consumer == "reviews:zeus/42"


def test_brief_target_reads_the_real_brief():
    """[Apollo MEDIUM] The target must come from a REAL brief, not a shape
    the test invented: parse neo_protocol.build_brief's own output."""
    from src.neo_protocol import build_brief
    from src.usage import brief_target

    head = "3c9e1aa" + "b" * 33
    brief = build_brief(
        repo="acme/widget", pr=44, head_sha=head,
        reviewer_context="rhobear-reviews", reviewer_green=True,
        auto_merge=False, builder_round=2, max_builder_rounds=5,
        builder_model="glm-5.3-flash",
    )
    target = brief_target(brief)
    assert target.pr_number == 44
    assert target.head_sha == head          # full 40-hex, from the canon
    assert target.builder_round == 2


@requires_pg
def test_attempt_target_prefers_job_row(store):
    """[Apollo MEDIUM] The job row is authoritative: a brief-parsed target
    never overrides the structured PR / head SHA / round of the job."""
    import psycopg as _psycopg

    with _psycopg.connect(DSN, autocommit=True) as c:
        job_id = c.execute(
            "INSERT INTO neo_jobs(kind, repo, pr_number, head_sha, "
            "builder_round, payload) VALUES ('builder', 'acme/widget', 7, %s, "
            "3, '{}') RETURNING id",
            ("cd" * 40,),
        ).fetchone()[0]

    store.record_attempt(UsageAttempt(
        job_id=job_id, attempt_number=1, route_idx=0,
        provider="deepseek", model="deepseek-v4-flash",
        outcome="success",
        # brief-parse disagrees with the job row — the job row must win
        pr_number=99, head_sha="ee" * 20, builder_round=0,
    ))
    row = store.get_attempts(job_id)[0]
    assert row.pr_number == 7
    assert row.head_sha == "cd" * 40
    assert row.builder_round == 3


def test_one_classifier_family_import_smoke():
    """[Apollo LOW] The classifier moved to src.usage: every lane module must
    still import, and no agent module may keep a second copy of the anchors."""
    import importlib
    import pkgutil

    import src
    import src.agent_claude as agent_mod

    for mod in pkgutil.iter_modules(src.__path__):
        importlib.import_module(f"src.{mod.name}")

    assert not hasattr(agent_mod, "_status_code_from")
    assert not hasattr(agent_mod, "_STATUS_CODE")
    from src.usage import classify_error_text, status_code_from
    assert callable(classify_error_text)
    assert status_code_from("HTTP/1.1 503 down") == 503


# ---------------------------------------------------------------------------
# W1-06 — Apollo round 2: wire header == stored row; job lookup can never
# kill metering.
# ---------------------------------------------------------------------------

_GUESS_BRIEF = (
    "TARGET: PR #44 in deariencampbell1-sys/rhobear-neo at head ababababab.\n"
    "gh api repos/deariencampbell1-sys/rhobear-neo/commits/"
    + "ab" * 20 + "/status\n"
    "git fetch origin && git checkout " + "ab" * 20 + "\n"
    "You have already spent 1/2 builder rounds on this PR.\n"
)


@requires_pg
def test_header_and_row_consumer_match(store: UsageStore) -> None:
    """[Apollo r2, MEDIUM] The consumer ON THE WIRE and the consumer IN THE
    ROW must be the same value: install-qualified from the owning job's
    payload, captured from the env the gated subprocess really runs with."""
    from src.agent_claude import ClaudeAgent

    assert DSN
    with psycopg.connect(DSN, autocommit=True) as c:
        job_id = c.execute(
            "INSERT INTO neo_jobs (kind, repo, head_sha, payload) "
            "VALUES ('wake', 'x/y', %s, %s) RETURNING id",
            ("ab" * 20, json.dumps({"installation_id": "777"})),
        ).fetchone()[0]

    agent = ClaudeAgent(
        claude_bin=sys.executable, api_key="k",
        base_url="https://api.deepseek.com/anthropic", model="m",
        timeout=30.0, gh_token="t")
    cfg = Config()
    cfg.claude_bin = sys.executable
    cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
    cfg.deepseek_model = "deepseek-v4-flash"
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.routegate_anthropic_url = "http://127.0.0.1:8645"
    cfg.routegate_model = "peak"

    captured: dict[str, str] = {}

    def fake_run(cmd, **kw):
        captured.update({k: v for k, v in (kw.get("env") or {}).items()
                         if "CONSUMER" in k or "CUSTOM_HEADERS" in k})
        return subprocess.CompletedProcess(
            args=["claude"], returncode=0, stderr="",
            stdout=json.dumps({"type": "result",
                               "result": "VERDICT: ACCEPT",
                               "is_error": False, "subtype": "success", "num_turns": 1,
                               "usage": {"input_tokens": 1, "output_tokens": 1,
                                         "cache_read_input_tokens": 0,
                                         "cache_creation_input_tokens": 0}}),
        )

    try:
        from src.agent_claude import Route as AgentRoute

        routes = [(4, AgentRoute(provider="deepseek",
                                 model="deepseek-v4-flash"))]
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}), \
                patch("src.agent_claude.subprocess.run", side_effect=fake_run):
            agent.run_with_chain(_GUESS_BRIEF, routes, cfg, "t",
                                 usage_store=store, job_id=job_id)

        wire = captured["X_RHOBEAR_CONSUMER"]
        assert wire == "reviews:zeus/777", wire
        attempts = store.get_attempts(job_id)
        assert len(attempts) == 1
        assert attempts[0].consumer == wire, (
            f"wire {wire!r} != row {attempts[0].consumer!r}")
    finally:
        store.clear(job_id)
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute("DELETE FROM neo_jobs WHERE id = %s", (job_id,))


@requires_pg
def test_bad_install_id_never_reaches_the_wire(store: UsageStore) -> None:
    """[Apollo r3, HIGH] A forged/poisoned `installation_id` in the job
    payload must not ride into the child env: refused at the source
    (job_consumer) AND at the sink (agent __init__), so neither the wire nor
    the row can carry header-injection bytes."""
    from src.agent_claude import ClaudeAgent, Route as AgentRoute

    assert DSN
    with psycopg.connect(DSN, autocommit=True) as c:
        job_id = c.execute(
            "INSERT INTO neo_jobs (kind, repo, head_sha, payload) "
            "VALUES ('wake', 'x/y', %s, %s) RETURNING id",
            ("ab" * 20, json.dumps({"installation_id": "777\r\nX-Injected: 1"})),
        ).fetchone()[0]

    agent = ClaudeAgent(claude_bin=sys.executable, api_key="k",
                        base_url="https://api.deepseek.com/anthropic",
                        model="m", timeout=30.0, gh_token="t")
    cfg = Config()
    cfg.claude_bin = sys.executable
    cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
    cfg.deepseek_model = "deepseek-v4-flash"
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.routegate_anthropic_url = "http://127.0.0.1:8645"
    cfg.routegate_model = "peak"

    captured: dict[str, str] = {}

    def fake_run(cmd, **kw):
        captured.update({k: v for k, v in (kw.get("env") or {}).items()
                         if "CONSUMER" in k or "CUSTOM_HEADERS" in k})
        return subprocess.CompletedProcess(
            args=["claude"], returncode=0, stderr="",
            stdout=json.dumps({"type": "result", "result": "VERDICT: ACCEPT",
                               "is_error": False, "subtype": "success",
                               "num_turns": 1,
                               "usage": {"input_tokens": 1, "output_tokens": 1}}),
        )

    try:
        routes = [(4, AgentRoute(provider="deepseek",
                                 model="deepseek-v4-flash"))]
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}), \
                patch("src.agent_claude.subprocess.run", side_effect=fake_run):
            agent.run_with_chain(_GUESS_BRIEF, routes, cfg, "t",
                                 usage_store=store, job_id=job_id)

        wire = captured["X_RHOBEAR_CONSUMER"]
        assert wire == "reviews:zeus", wire          # bare, not poisoned
        assert "\r" not in wire and "\n" not in wire
        attempts = store.get_attempts(job_id)
        assert attempts[0].consumer == wire
    finally:
        store.clear(job_id)
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute("DELETE FROM neo_jobs WHERE id = %s", (job_id,))


@requires_pg
def test_job_target_failure_warns_once(
        store: UsageStore, caplog: Any, monkeypatch: Any) -> None:
    """[Apollo r3, MEDIUM] Permanent attribution loss must be OBSERVABLE:
    the first lookup failure logs WARNING (once, never the poisoned value),
    later ones stay quiet (rate-limited) — not DEBUG-silence forever."""
    import logging as _logging

    import src.usage as usage_mod

    monkeypatch.setattr(usage_mod, "_JOB_TARGET_WARNED", False)  # restored

    class BoomConn:
        def execute(self, *args, **kwargs):
            raise RuntimeError("neo_jobs is gone")

    with caplog.at_level(_logging.WARNING, logger="rhobear_neo.usage"):
        store._job_target(BoomConn(), 4242)
        first = [r for r in caplog.records
                 if r.levelno == _logging.WARNING and "job target" in r.message]
        store._job_target(BoomConn(), 4242)
        store._job_target(BoomConn(), 4242)
        again = [r for r in caplog.records
                 if r.levelno == _logging.WARNING and "job target" in r.message]
    assert len(first) == 1, "first attribution failure must warn"
    assert len(again) == 1, "subsequent failures must be rate-limited"


@requires_pg
def test_job_lookup_failure_cannot_poison_metering(
        store: UsageStore, caplog: Any) -> None:
    """[Apollo r3, MEDIUM] The metering connection is autocommit: a failed
    job-lookup statement cannot leave an aborted transaction that kills the
    INSERT (the failure mode that would silently drop ALL metering)."""
    import logging as _logging

    assert DSN
    with store._conn() as c:
        assert c.autocommit is True, "_conn must be autocommit"
        try:
            c.execute("SELECT * FROM no_such_table_w106")
        except Exception:
            pass                                   # the lookup fails …
        cur = c.execute(
            "INSERT INTO neo_usage (job_id, attempt_number, route_idx, "
            " provider, model, outcome, uncached_input, output_tokens) "
            "VALUES (555002, 1, 0, 'deepseek', 'm', 'success', 1, 1) "
            "RETURNING id")
        assert cur.fetchone() is not None, "metering must survive the failure"
    store.clear(555002)


@requires_pg
def test_route_name_and_cost_round_trip_postgres(store: UsageStore) -> None:
    """[Apollo r4, MEDIUM] The new TEXT/DOUBLE columns must round-trip REAL
    values through Postgres — not just the NULLs the rest of the suite
    happens to write."""
    assert DSN
    ok = store.record_attempt(UsageAttempt(
        job_id=555003, attempt_number=1, route_idx=0,
        provider="routegate", model="peak", route_name="peak",
        cost_usd=0.0042, outcome="success",
        uncached_input=11, cached_input=4, output_tokens=5,
        consumer="reviews:zeus/987654321", pr_number=44,
        head_sha="ab" * 20, builder_round=1,
    ))
    assert ok is True
    try:
        row, = store.get_attempts(555003)
        assert row.route_name == "peak"
        assert row.cost_usd == 0.0042
        assert row.consumer == "reviews:zeus/987654321"
        assert row.head_sha == "ab" * 20
    finally:
        store.clear(555003)


def test_brief_target_never_guesses():
    """[Apollo r5, LOW] Anything the brief does not carry stays None — a bare
    40-hex token without the commits/checkout context, a PR mention without
    the TARGET line, and a round ratio outside the builder sentence must all
    be ignored (the anchors may never loosen into guessing)."""
    from src.usage import brief_target

    target = brief_target(
        "Please look at PR 44, sha 1234567890abcdef1234567890abcdef12345678, "
        "and we are at 7/9 rounds of patience. No structure here.")
    assert target.pr_number is None
    assert target.head_sha is None
    assert target.builder_round is None

    empty = brief_target("")
    assert (empty.pr_number, empty.head_sha, empty.builder_round) == (None, None, None)
