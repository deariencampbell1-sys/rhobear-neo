"""W1-13-r1 Defect 6 — the merge path must run the SAME authority gate.

merge_executor.merge() is the only merge path in the Neo service.  When the
worker's gate and merge() disagree (gate stricter than merge), work the gate
denied is merged anyway.  merge() must therefore run
review_evidence.merge_authority() itself and surface any denial as
MergeDenied carrying the gate's reason.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.merge_executor import MergeDenied, merge
from src.review_evidence import merge_authority

SHA = "a" * 40


def _evidence(**overrides) -> dict:
    ev = {
        "ok": True,
        "head_sha": SHA,
        "decided_sha": SHA,
        "review_sha": SHA,
        "trusted_producer_ok": True,
        "review_sha_match": True,
        "review_bound_ok": True,
        "review_state": "APPROVED",
        "unresolved_critical_high": 0,
        "coverage_complete": True,
        "comments_addressed": True,
        "required_checks_pass": True,
        "auto_merge": True,
    }
    ev.update(overrides)
    return ev


def _fake_gh(calls=None):
    """Fake merge_executor._gh: PR view + merge API + best-effort branch ops."""
    calls = calls if calls is not None else []

    def fake(token, *args, **kw):
        calls.append(args)
        if args[:3] == ("pr", "view", "5"):
            return 0, json.dumps({"headRefOid": SHA, "state": "OPEN",
                                  "mergeable": "MERGEABLE",
                                  "headRefName": "topic",
                                  "isCrossRepository": False})
        if args[:2] == ("api", "-X") and "PUT" in args:
            return 0, json.dumps({"merged": True, "sha": "M" * 40})
        if args[:2] == ("api", "-X") and "DELETE" in args:
            return 0, ""
        return 0, "main"
    return fake


class TestMergeRunsTheAuthorityGate:
    def test_merge_denied_before_the_merge_api(self):
        """An evidence bundle the gate denies (2 unresolved critical/high)
        must never reach the merge API — even though its `ok` is True."""
        ev = _evidence(unresolved_critical_high=2)
        assert merge_authority(ev)[0] is False
        calls = []
        with patch("src.merge_executor._gh", side_effect=_fake_gh(calls)):
            with pytest.raises(MergeDenied) as exc:
                merge("o/r", 5, SHA, ev, "tok")
        assert "unresolved critical/high" in str(exc.value).lower()
        assert not [c for c in calls if "PUT" in c], \
            "the merge API must never run on denied evidence"

    def test_merge_denied_carries_the_reason(self):
        """The denial message carries WHY the gate denied."""
        ev = _evidence(required_checks_pass=False)
        with patch("src.merge_executor._gh", side_effect=_fake_gh()):
            with pytest.raises(MergeDenied) as exc:
                merge("o/r", 5, SHA, ev, "tok")
        assert "required check" in str(exc.value).lower()

    def test_merge_and_the_gate_never_disagree(self):
        """Parity — for every evidence bundle that carries the gate's decision
        fields (which every production bundle does — build_evidence always
        emits them), merge() either merges (and the gate grants) or raises
        MergeDenied (and the gate denies).  This is exactly the disagreement
        the defect allowed."""
        cases = [
            _evidence(),
            _evidence(unresolved_critical_high=2),
            _evidence(required_checks_pass=None),      # unknown -> denied
            _evidence(required_checks_pass=False),
            _evidence(coverage_complete=False),
            _evidence(review_sha_match=False),
            _evidence(auto_merge=False),
            _evidence(trusted_producer_ok=False),
            _evidence(comments_addressed=None),
            {},                              # no evidence at all
            {"ok": False, "reason": "pr view unavailable"},
        ]
        for ev in cases:
            granted, reason = merge_authority(ev)
            with patch("src.merge_executor._gh", side_effect=_fake_gh()):
                if granted:
                    assert merge("o/r", 5, SHA, ev, "tok")["merged"] is True
                else:
                    with pytest.raises(MergeDenied):
                        merge("o/r", 5, SHA, ev, "tok")

    def test_legacy_minimal_shape_stays_accepted(self):
        """Contract (tests/test_merge_authority.py pins it): evidence carrying
        NONE of the gate's decision fields — the legacy {"ok", shas} shape —
        is this mechanics layer's accepted input: with no findings on the
        bundle there is nothing for the gate to consult here, so the original
        ok check applies.  Production bundles always carry the fields and are
        gated in full (parity test above)."""
        ev = {"ok": True, "head_sha": SHA, "decided_sha": SHA}
        with patch("src.merge_executor._gh", side_effect=_fake_gh()):
            assert merge("o/r", 5, SHA, ev, "tok")["merged"] is True

    def test_valid_evidence_still_merges(self):
        calls = []
        with patch("src.merge_executor._gh", side_effect=_fake_gh(calls)):
            out = merge("o/r", 5, SHA, _evidence(), "tok")
        assert out["merged"] is True and out["sha"] == SHA


class TestOkAndGateMatrix:
    """W1-13 defect 1 (HIGH) — the exact `ok` / gate cells merge() must pass.

    Every ambiguous cell fails closed: `{}` (no evidence at all), `ok=False`
    alone, `ok=False` with ALL SIX gate fields green (the HIGH's parity cell),
    `ok=True` with a red gate field, and all-green (the only cell that merges).
    """

    @pytest.mark.parametrize("ev, merges", [
        ({}, False),
        ({"ok": False, "reason": "evidence read failed"}, False),
        (_evidence(ok=False, reason="evidence retrieval partially failed"), False),
        (_evidence(required_checks_pass=False), False),
        (_evidence(), True),
    ], ids=["empty", "ok-false-alone", "ok-false-green-fields",
            "ok-true-red-field", "all-green"])
    def test_every_cell_fails_closed(self, ev, merges):
        calls = []
        with patch("src.merge_executor._gh", side_effect=_fake_gh(calls)):
            if merges:
                assert merge("o/r", 5, SHA, ev, "tok")["merged"] is True
            else:
                with pytest.raises(MergeDenied):
                    merge("o/r", 5, SHA, ev, "tok")
                assert not [c for c in calls if "PUT" in c], \
                    "denied evidence must never reach the merge API"


class TestOkBeltIsLocal:
    """The HIGH's mechanism, driven directly: `ok=False` with GREEN gate
    fields must deny even when the gate itself GRANTS.

    merge_authority subsumes `ok` today (its first condition denies an
    ok=False bundle), so the live matrix above closes through the gate — but
    the belt belongs HERE: `ok` encodes evidence completeness the six gate
    fields do not, and merge() must fail closed on it even if the gate's
    answer is wrong.  A gate that grants must never override `ok=False`.
    """

    def test_ok_false_denies_even_when_the_gate_grants(self):
        ev = _evidence(ok=False, reason="evidence retrieval partially failed")
        calls = []
        with patch("src.review_evidence.merge_authority",
                   return_value=(True, "granted")), \
             patch("src.merge_executor._gh", side_effect=_fake_gh(calls)):
            with pytest.raises(MergeDenied):
                merge("o/r", 5, SHA, ev, "tok")
        assert not [c for c in calls if "PUT" in c], \
            "the merge API must never run for ok=False evidence"

    def test_ok_true_and_gate_grant_still_merges(self):
        ev = _evidence()
        with patch("src.review_evidence.merge_authority",
                   return_value=(True, "granted")), \
             patch("src.merge_executor._gh", side_effect=_fake_gh()):
            assert merge("o/r", 5, SHA, ev, "tok")["merged"] is True
