"""Config-level tests for `src/config.py` (W1-06 r2).

Behavioural only: construct the object, drive the code, assert the effect —
no test here reads or slices source text
(`tests/test_no_source_text_assertions.py` guards that tree-wide).
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

from src.agent_claude import AgentError, ClaudeAgent, _require_loopback_roulegate
from src.config import Config, ROUTEGATE_ANTHROPIC_URL

BASE = "https://api.deepseek.com/anthropic"
MODEL = "deepseek-v4-flash"


def _valid_cfg(routegate_url: str = "") -> Config:
    """A Config that passes validate() once routegate_url is accepted."""
    cfg = Config()
    cfg.deepseek_base_url = BASE
    cfg.deepseek_model = MODEL
    cfg.deepseek_reasoning_effort = "max"
    cfg.deepseek_max_tokens = 32000
    cfg.claude_bin = sys.executable
    cfg.routegate_anthropic_url = routegate_url
    return cfg


# ---------------------------------------------------------------------------
# Defect 31 (Apollo, sixth review) — the reader default must be main's flat
# value, not a number derived from the bounce floor
# ---------------------------------------------------------------------------

def test_env_reader_default_is_flat_not_derived_from_the_bounce_floor() -> None:
    """The builder-rounds reader default is the SAME flat number main ships,
    independent of `NEO_BOUNCES_BEFORE_ESCALATION`: the clamp only lifts an
    explicit value up to the escalation floor at construction — it never
    becomes the default.

    Red on this head: the branch's reader default is a smaller number that
    the clamp then lifts to whatever floor the box sets, so a config-only
    deployment with a floor of 3 or 4 would silently see its builder budget
    change from main's 5."""
    with patch.dict(os.environ, {"NEO_BOUNCES_BEFORE_ESCALATION": "3"},
                    clear=True):
        assert Config().max_builder_rounds == 5, (
            "the reader default must not follow the bounce floor"
        )
    with patch.dict(os.environ, {"NEO_BOUNCES_BEFORE_ESCALATION": "4"},
                    clear=True):
        assert Config().max_builder_rounds == 5, (
            "the reader default must not follow the bounce floor"
        )
    with patch.dict(os.environ, {}, clear=True):
        stock = Config()
        assert stock.max_builder_rounds == stock.bounces_before_escalation == 5


# ---------------------------------------------------------------------------
# Defect 32 (Apollo, seventh review) — round 7's guard must clear the same
# bar: its ACCEPT branch (the branches production takes when the operator
# flips the switch) is asserted by result, and the parsed-part rebuild is
# exercised for a path, a trailing slash and a non-8645 port.
# ---------------------------------------------------------------------------

def test_gate_url_guard_accept_branch_rebuilds_the_base() -> None:
    """Every spelling that means "the loopback gate" is ACCEPTED and
    normalised to the exact base the Claude CLI completes with
    `/v1/messages`; a URL that parses to any other local port is REFUSED —
    including before any rebuild can happen."""
    for accepted in (
        ROUTEGATE_ANTHROPIC_URL,                    # bare base
        ROUTEGATE_ANTHROPIC_URL + "/",              # trailing slash
        ROUTEGATE_ANTHROPIC_URL + "/v1/messages",   # documented endpoint
        ROUTEGATE_ANTHROPIC_URL + "/v1/messages/",  # endpoint + slash
    ):
        assert _require_loopback_roulegate(accepted) == ROUTEGATE_ANTHROPIC_URL, \
            accepted

    # a port that is not RouteGate's is another local service — refused,
    # both as a bare base and with an endpoint path.
    for off_port in ("http://127.0.0.1:9999",
                     "http://127.0.0.1:9999/v1/messages"):
        with pytest.raises(AgentError, match="loopback"):
            _require_loopback_roulegate(off_port)


def test_validate_accept_branches_store_the_canonical_base() -> None:
    """The flip-switch itself: every documented spelling passes validate()
    WITHOUT raising, the gate stays on, and the field is left holding the
    ONE canonical base (trailing slash / whitespace normalised in the
    field, not just in the comparison — red on this head for the slash
    form: validate() compared rstripped but stored the raw string, so the
    post-validate field disagreed with the value from_config would use).
    from_config on that config yields a live gated agent."""
    for accepted in (
        ROUTEGATE_ANTHROPIC_URL,
        ROUTEGATE_ANTHROPIC_URL + "/",
        ROUTEGATE_ANTHROPIC_URL + "/v1/messages",
        ROUTEGATE_ANTHROPIC_URL + "/v1/messages/",
    ):
        cfg = _valid_cfg(accepted)
        cfg.validate()                     # accept branch: must not raise
        assert cfg.routegate_anthropic_url == ROUTEGATE_ANTHROPIC_URL, accepted

        agent = ClaudeAgent.from_config(cfg, gh_token="t")
        assert agent.gated, accepted
        assert agent.base_url == ROUTEGATE_ANTHROPIC_URL, accepted
        assert agent.base_url + "/v1/messages" == ROUTEGATE_ANTHROPIC_URL + \
            "/v1/messages", "the CLI completes the base — never a double path"

    # and a well-formed off-port gate URL keeps the gate OFF (discarded,
    # direct chain stands) rather than raising at boot.
    cfg = _valid_cfg("http://127.0.0.1:9999")
    cfg.validate()
    assert cfg.routegate_anthropic_url == ""
    assert not ClaudeAgent.from_config(cfg, gh_token="t").gated