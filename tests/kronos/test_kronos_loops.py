"""W1-08-r1 red-first — a fix loop that repeats a commit is invisible today.

Edge case 3: a fix loop committing the same tree twice is detected by tree
hash and reported with BOTH SHAs.

The round history comes from neo_actions (read-only): each round row carries
head_sha and, when the worker recorded it, a tree hash in detail. The loop
detector is pure so the test needs no database.
"""
from __future__ import annotations

from src.kronos.judge import same_commit_loop


def _round(round_no: int, sha: str, tree: str = "") -> dict:
    d = {"round": round_no, "head_sha": sha, "verdict": "BOUNCE"}
    if tree:
        d["tree_sha"] = tree
    return d


def test_same_commit_twice_is_a_loop():
    """Two rounds with the SAME head SHA are a loop: both SHAs reported."""
    rounds = [
        _round(1, "a" * 40),
        _round(2, "b" * 40),
        _round(3, "b" * 40),
    ]
    loops = same_commit_loop(rounds)
    assert loops, "a repeated commit went undetected"
    assert loops[0] == (2, 3, "b" * 40), loops


def test_same_tree_different_sha_is_a_loop():
    """Two different commits with the SAME tree (an empty 'fix') are a loop."""
    tree = "c" * 40
    rounds = [
        _round(1, "a" * 40, tree="d" * 40),
        _round(2, "b" * 40, tree=tree),
        _round(3, "e" * 40, tree=tree),
    ]
    loops = same_commit_loop(rounds)
    assert loops, "an empty-fix loop (same tree) went undetected"
    assert loops[0] == (2, 3, tree), loops


def test_distinct_rounds_are_not_a_loop():
    rounds = [
        _round(1, "a" * 40, tree="d" * 40),
        _round(2, "b" * 40, tree="e" * 40),
        _round(3, "c" * 40, tree="f" * 40),
    ]
    assert same_commit_loop(rounds) == []


def test_loop_is_reported_with_both_shas():
    """The report builder names both offending SHAs in the loop note."""
    from src.kronos.judge import loop_note
    rounds = [_round(2, "b" * 40, tree="c" * 40),
              _round(3, "e" * 40, tree="c" * 40)]
    note = loop_note(same_commit_loop(rounds), rounds)
    assert "b" * 40 in note and "e" * 40 in note
    assert "2" in note and "3" in note
