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
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        cfg.validate()
        assert cfg.parsed_routes is not None
        assert len(cfg.parsed_routes) == 6

    def test_custom_routes_valid(self):
        cfg = Config()
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
        cfg.neo_routes = "unknown_provider:model"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        with pytest.raises(ValueError, match="unknown provider"):
            cfg.validate()

    def test_malformed_route_raises_on_validate(self):
        cfg = Config()
        cfg.neo_routes = "just-a-string"
        cfg.deepseek_base_url = "https://api.deepseek.com/anthropic"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.deepseek_reasoning_effort = "max"
        cfg.deepseek_max_tokens = 32000
        with pytest.raises(ValueError, match="expected 'provider:model'"):
            cfg.validate()

    def test_empty_routes_uses_default(self):
        cfg = Config()
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
