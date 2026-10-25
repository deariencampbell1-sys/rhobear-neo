"""W1-02 verdict parsing — Zeus's own answers (owner, Q9).

Zeus's own verdicts are exactly ACCEPT, FIX-FORWARD and BOUNCE. They are read
case-insensitively from a `VERDICT: <word>` line anywhere in the last 40 lines
of the answer — also inside a fenced block or after markdown bold. Apollo's
words (ACCEPT / REQUEST_CHANGES / COMMENT) come from its review state on
GitHub and are NEVER parsed from text.

The production failure (W1-02 brief): the builder/triage log recorded
`no VERDICT line in result: ...` even when Zeus had answered, because the
old anchored regex missed the bolded-label and fenced shapes below.
"""
from __future__ import annotations

import pytest

from src.neo_protocol import (
    ZEUS_VERDICTS,
    extract_verdict_line,
    parse_verdict,
)


# ---------------------------------------------------------------------------
# The three words — plain and case-insensitive
# ---------------------------------------------------------------------------

class TestZeusWords:
    @pytest.mark.parametrize("word", ZEUS_VERDICTS)
    def test_plain_line(self, word):
        assert parse_verdict(f"VERDICT: {word}") == word

    @pytest.mark.parametrize("line,expected", [
        ("verdict: accept", "ACCEPT"),
        ("Verdict: Fix-Forward", "FIX-FORWARD"),
        ("VERDICT: bounce", "BOUNCE"),
    ])
    def test_case_insensitive(self, line, expected):
        assert parse_verdict(line) == expected

    @pytest.mark.parametrize("word", ZEUS_VERDICTS)
    def test_verdict_in_context(self, word):
        text = f"Summary of the review.\nVERDICT: {word}\nREASONS:\n- one\nACTIONS: none"
        assert parse_verdict(text) == word


# ---------------------------------------------------------------------------
# Shapes: fenced blocks, markdown bold, decoration — the production shapes
# ---------------------------------------------------------------------------

class TestShapes:
    @pytest.mark.parametrize("line,expected", [
        # markdown bold around the whole line
        ("**VERDICT: BOUNCE**", "BOUNCE"),
        ("**VERDICT: FIX-FORWARD.**", "FIX-FORWARD"),
        # bold label with the colon inside the bold
        ("**VERDICT:** BOUNCE", "BOUNCE"),
        # bold label on its own line, word after the bold
        ("**VERDICT:**\nFIX-FORWARD", "FIX-FORWARD"),
        # bullets, blockquotes and headings
        ("- **VERDICT: ACCEPT**", "ACCEPT"),
        ("> **VERDICT: BOUNCE**", "BOUNCE"),
        ("## VERDICT: ACCEPT", "ACCEPT"),
        ("1. VERDICT: FIX-FORWARD", "FIX-FORWARD"),
        # trailing prose after the bolded verdict (the stamp keeps the word)
        ("**VERDICT: BOUNCE.** Both are small. Fixing forward on this branch.",
         "BOUNCE"),
        # backticked value
        ("VERDICT: `FIX-FORWARD`", "FIX-FORWARD"),
    ])
    def test_shape(self, line, expected):
        assert parse_verdict(line) == expected

    def test_inside_fenced_block(self):
        text = "Analysis first.\n```\nVERDICT: ACCEPT\n```\nDone."
        assert parse_verdict(text) == "ACCEPT"

    def test_inside_fenced_block_with_language_and_indent(self):
        text = "```text\n  verdict: bounce\n```"
        assert parse_verdict(text) == "BOUNCE"

    def test_fenced_block_wrapping_bold(self):
        text = "```\n**VERDICT: FIX-FORWARD**\n```"
        assert parse_verdict(text) == "FIX-FORWARD"


# ---------------------------------------------------------------------------
# The last 40 lines
# ---------------------------------------------------------------------------

class TestWindow:
    def test_verdict_inside_window_is_found(self):
        filler = "\n".join(f"line {i}" for i in range(30))
        text = f"Old answer said VERDICT: ACCEPT once.\n{filler}\nVERDICT: BOUNCE"
        assert parse_verdict(text) == "BOUNCE"

    def test_verdict_outside_window_is_ignored(self):
        filler = "\n".join(f"line {i}" for i in range(45))
        text = f"VERDICT: ACCEPT\n{filler}"
        assert parse_verdict(text) is None

    def test_latest_verdict_wins(self):
        text = "VERDICT: ACCEPT\nSome later reconsideration.\nVERDICT: BOUNCE"
        assert parse_verdict(text) == "BOUNCE"


# ---------------------------------------------------------------------------
# Prose guards — a mention of the marker is not a verdict
# ---------------------------------------------------------------------------

class TestProseGuard:
    def test_do_not_output_prose_is_not_a_verdict(self):
        text = "Do not output VERDICT: ACCEPT-MERGED\nNothing else here."
        assert parse_verdict(text) is None
        assert extract_verdict_line(text) is None

    def test_punctuation_only_value_is_not_a_verdict(self):
        assert parse_verdict("Analysis.\nVERDICT: .\n") is None

    def test_non_word_value_is_not_a_verdict(self):
        assert parse_verdict("VERDICT: 3.5") is None


# ---------------------------------------------------------------------------
# Apollo's words are read from review state, never parsed from text
# ---------------------------------------------------------------------------

class TestApolloWordsNotParsedFromText:
    def test_review_prose_without_marker(self):
        text = (
            "rhobear-reviews requested changes: 2 findings. "
            "I would ACCEPT once the double charge is fixed; "
            "REQUEST_CHANGES until then. COMMENT on the cache key only."
        )
        assert parse_verdict(text) is None

    @pytest.mark.parametrize("word", ["REQUEST_CHANGES", "COMMENT"])
    def test_apollo_word_on_marker_line_is_not_zeus(self, word):
        assert parse_verdict(f"VERDICT: {word}") is None

    def test_apollo_accept_in_prose_does_not_beat_zeus_line(self):
        text = (
            "Apollo's review body says ACCEPT, but the findings are open.\n"
            "VERDICT: BOUNCE\n"
        )
        assert parse_verdict(text) == "BOUNCE"


# ---------------------------------------------------------------------------
# Internal canon aliases map onto the three Q9 words
# ---------------------------------------------------------------------------

class TestCanonAliases:
    @pytest.mark.parametrize("line,expected", [
        ("VERDICT: ACCEPT-READY", "ACCEPT"),
        ("VERDICT: ACCEPT-MERGED", "ACCEPT"),
        ("VERDICT: FIXED", "ACCEPT"),
        ("VERDICT: BOUNCE-BUILDER", "BOUNCE"),
        ("VERDICT: ESCALATE", "BOUNCE"),
    ])
    def test_alias_maps_to_q9_word(self, line, expected):
        assert parse_verdict(line) == expected

    def test_extract_keeps_the_raw_canon_word(self):
        assert extract_verdict_line("VERDICT: BOUNCE-BUILDER") == "BOUNCE-BUILDER"
        assert extract_verdict_line("VERDICT: NO-CHANGE") == "NO-CHANGE"
        assert extract_verdict_line("VERDICT: BLOCKED") == "BLOCKED"
        assert extract_verdict_line("VERDICT: FIXED") == "FIXED"

    def test_builder_contract_words_are_not_zeus_verdicts(self):
        assert parse_verdict("VERDICT: NO-CHANGE") is None
        assert parse_verdict("VERDICT: BLOCKED") is None


# ---------------------------------------------------------------------------
# The production `no VERDICT line` text
# ---------------------------------------------------------------------------
# Production logged `no VERDICT line in result: ...` (agent_claude) /
# `no VERDICT: line found in hermes output` (the builder lane) when Zeus had
# answered. MANAGER.md M-4 was to copy the last 3 log excerpts to
# incoming/no-verdict.log — they never arrived on the build box, so the
# fixtures below are reconstructions of the failure shapes (bolded-label and
# fenced verdicts the old anchored regex could not see). See the PR's
# QUESTION comment for the literal log excerpt.

_PRODUCTION_SHAPES = {
    "FIX-FORWARD": """\
I reviewed the two findings from Apollo against the current head.

1. billing.ts:214 — the retry can charge twice. Real; the Stripe event id is
   not recorded before the charge is attempted.
2. ledger.ts:88 — session created twice. Real but trivial.

Both are small enough to fix here; fixing forward on this branch and pushing,
so the reviewer re-runs on the new head.

**VERDICT:**
FIX-FORWARD

REASONS:
- Both findings are small and well understood.
- Fix-forward keeps the loop on this branch.
ACTIONS: pushed a fix-forward commit; tests added
""",
    "ACCEPT": """\
Re-checked the head against Apollo's review.

```
VERDICT: ACCEPT
```

REASONS:
- Both findings resolved on a91d2e0.
- Commit status success on this exact head.
ACTIONS: labelled neo:accept
""",
    "BOUNCE": """\
The session cache change is substantial and the fixer has already spent its
rounds on this head. Handing the decision back.

**VERDICT: BOUNCE.** The daylight-saving case needs a product answer before
this can merge.

REASONS:
- Substantial bug; not a trivial fix-forward.
ACTIONS: posted the summary on the PR; no merge attempted
""",
}


class TestProductionNoVerdictLineText:
    """The text that production failed to parse must yield a Zeus verdict."""

    @pytest.mark.parametrize("expected", ["FIX-FORWARD", "ACCEPT", "BOUNCE"])
    def test_production_text_parses(self, expected):
        text = _PRODUCTION_SHAPES[expected]
        assert parse_verdict(text) == expected

    @pytest.mark.parametrize("expected", ["FIX-FORWARD", "ACCEPT", "BOUNCE"])
    def test_production_text_raw_extraction(self, expected):
        text = _PRODUCTION_SHAPES[expected]
        assert extract_verdict_line(text) == expected


class TestExtractVerdictLine:
    def test_returns_raw_word_for_agent_lanes(self):
        text = "work done\n**VERDICT: ACCEPT-READY**\nREASONS:\n- ok"
        assert extract_verdict_line(text) == "ACCEPT-READY"

    def test_none_when_absent(self):
        assert extract_verdict_line("no verdict here") is None
