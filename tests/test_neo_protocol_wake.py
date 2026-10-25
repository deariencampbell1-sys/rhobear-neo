"""W1-02 wake verdict colour — strict green parsing (fix/verdict-flip-redelivery).

A wake payload's colour is normalized to True / False / None; None means
"unknown" and is never collapsed into red. `bool(raw)` is wrong twice over:
a missing key becomes red, and the string "false" becomes True.
"""
from __future__ import annotations

import pytest

from src.neo_state import verdict_color


class TestVerdictColor:
    @pytest.mark.parametrize("raw,expected", [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("True", True),
        ("FALSE", False),
        ("false", False),
        ("success", True),
        ("green", True),
        ("red", False),
        ("failure", False),
        ("  yes  ", True),
        ("no", False),
    ])
    def test_known_values(self, raw, expected):
        assert verdict_color(raw) is expected

    @pytest.mark.parametrize("raw", [None, "", "maybe", "unknown", 2, 3.5, [], {}])
    def test_unknown_is_none_not_red(self, raw):
        assert verdict_color(raw) is None

    def test_string_false_is_not_truthy(self):
        """The production hazard: bool("false") == True inverts red verdicts."""
        assert verdict_color("false") is False
        assert bool("false") is True  # documents the bug this replaces

    def test_missing_key_is_not_red(self):
        assert verdict_color({}.get("green")) is None
