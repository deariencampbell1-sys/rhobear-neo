"""W1-08-r1 red-first — an unbounded report would paste whole journals.

Edge case 4: a journal (or findings blob) with 200k lines must be truncated
to a bounded size, and the report must SAY what it dropped — never silently.
"""
from __future__ import annotations

from src.kronos.judge import REPORT_BUDGET_CHARS, build_report


def _huge_finding_text(chars: int) -> str:
    return ("boom " * (chars // 5)).strip()


def _round(round_no: int, findings: str) -> dict:
    return {
        "round": round_no,
        "head_sha": ("ab" * 20)[:40],
        "verdict": "BOUNCE",
        "findings": findings,
    }


def test_report_is_bounded():
    """A 200k-line journal pasted into findings is cut to the budget, and the
    report carries an explicit dropped-note instead of the whole blob."""
    journal_200k = "\n".join(f"Nov 01 10:{m:02d}:00 line {i}"
                             for i, m in enumerate(
                                 [i % 60 for i in range(200_000)]))
    history = [_round(1, _huge_finding_text(5000)),
               _round(2, journal_200k)]
    report = build_report(history, budget=REPORT_BUDGET_CHARS)

    assert len(report) <= REPORT_BUDGET_CHARS, (
        f"report is {len(report)} chars, budget is {REPORT_BUDGET_CHARS}"
    )
    assert "dropped" in report.lower(), (
        "a truncated report must say what it dropped"
    )
    assert "200" in report, "the dropped size should be stated"


def test_small_report_is_verbatim():
    """Under the budget nothing is dropped and the round facts survive."""
    history = [_round(1, "tests fail on timezone")]
    report = build_report(history, budget=REPORT_BUDGET_CHARS)
    assert "tests fail on timezone" in report
    assert "dropped" not in report.lower()


def test_budget_is_env_independent():
    """The budget is a module constant with a sane default (>= 4k, <= 32k)."""
    assert 4000 <= REPORT_BUDGET_CHARS <= 32000
