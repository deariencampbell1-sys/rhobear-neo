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