"""Phase A/B regression tests: exact-SHA authority, structured Jev findings.

These reproduce the two incident classes recorded in
docs/AGENT-CONTROL-PLANE-ARCHITECTURE.md:

  * a review produced for commit A authorized a PR that had moved to commit B;
  * `risk=high` (true of every auth/security patch, defect-free or not) created a
    permanent Jev veto through `needs_attention`.
"""
import io
import json

import src.neo_worker as worker
from src.jev_review import JevAssessment, JevFinding, assess


def _answers(**over):
    base = {
        "risk": {"choice": "high"},
        "security": {"noul": 0.95},
        "correctness": {"noul": 0.05},
        "review_quality": {"choice": "adequate", "confidence": 0.92},
        "escalate": {"noul": 0.10},
    }
    base.update(over)
    return {"answers": base}


def _patch_urlopen(monkeypatch, payload):
    def fake_open(request, timeout):
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("src.jev_review.urllib.request.urlopen", fake_open)


# --- Phase B: risk is a surface, not a verdict ------------------------------

def test_high_risk_without_concrete_finding_does_not_block(monkeypatch):
    """risk=high + findings=[] ⇒ deeper review asked for, authority retained."""
    _patch_urlopen(monkeypatch, _answers())
    result = assess("+if not user: 401", "Reviewed the auth change with evidence.", "k")
    assert result.available
    assert result.risk == "high" and result.security == 0.95
    assert result.requires_full_review is True     # Neo must inspect
    assert result.blockers == ()                   # but no veto
    assert result.needs_attention is True          # deprecated alias unchanged


def test_high_risk_patch_passes_authority_together_with_sha_evidence(monkeypatch):
    _patch_urlopen(monkeypatch, _answers())
    jev = assess("+auth", "review", "k", head_sha="a" * 40)
    evidence = {"ok": True, "decided_sha": "a" * 40, "head_sha": "a" * 40,
                "review_sha": "a" * 40, "review_bound_ok": True, "trusted_green": True}
    granted, reason = worker.merge_authority(True, evidence, jev)
    assert granted, reason
    assert jev.head_sha == "a" * 40


def test_poor_review_coverage_is_a_concrete_blocking_finding(monkeypatch):
    _patch_urlopen(monkeypatch, _answers(
        review_quality={"choice": "poor", "confidence": 0.9}))
    jev = assess("+auth", "looks fine", "k")
    rules = [f.rule for f in jev.blockers]
    assert rules == ["REVIEW_NOT_ADEQUATE"]
    granted, reason = worker.merge_authority(
        True, {"ok": True, "decided_sha": "a", "head_sha": "a",
               "review_sha": "a", "review_bound_ok": True, "trusted_green": True}, jev)
    assert not granted and "REVIEW_NOT_ADEQUATE" in reason


def test_truncated_jev_input_is_review_incomplete(monkeypatch):
    """Oversize evidence cannot authorize a merge (nothing was fully inspected)."""
    _patch_urlopen(monkeypatch, _answers())
    jev = assess("x" * 200_000, "review", "k")
    assert jev.partial and [f.rule for f in jev.blockers] == ["REVIEW_INCOMPLETE"]


def test_provider_findings_are_respected_and_filtered(monkeypatch):
    payload = _answers()
    payload["answers"]["findings"] = [
        {"rule": "REVIEW_NOT_ADEQUATE", "severity": "high", "blocking": True,
         "evidence": "review never looks at the new 401 path"},
        {"rule": "SOMETHING_ELSE", "severity": "high", "blocking": True,
         "evidence": "not a configured blocking rule"},
    ]
    _patch_urlopen(monkeypatch, payload)
    jev = assess("+auth", "review", "k")
    assert [f.rule for f in jev.blockers] == ["REVIEW_NOT_ADEQUATE"]


def test_unavailable_jev_removes_authority_but_never_blocks_forever(monkeypatch):
    jev = JevAssessment(False, reason="no OpenRouter key")
    granted, reason = worker.merge_authority(
        True, {"ok": True, "decided_sha": "a", "head_sha": "a",
               "review_sha": "a", "review_bound_ok": True, "trusted_green": True}, jev)
    assert granted, reason  # no key is not evidence of a defect
    assert jev.blockers == ()


# --- Phase A: reviews are SHA-bound, never merely PR-bound ------------------

class _Cfg:
    trusted_review_contexts = ("rhobear-reviews",)


def _gh_level(monkeypatch, payload, status=None):
    """Mock gh: pr view returns payload, the commit status returns `status`."""
    status = status if status is not None else {
        "statuses": [{"context": "rhobear-reviews", "state": "success"}]}

    def fake_gh(cfg, *args, timeout=30):
        if args and args[0] == "api":
            return 0, json.dumps(status)
        return 0, json.dumps(payload)

    monkeypatch.setattr(worker, "_gh", fake_gh)
    return payload


def test_stale_review_cannot_authorize_a_newer_head(monkeypatch):
    """review success for A, push B ⇒ B is NOT mergeable from A's review."""
    _gh_level(monkeypatch, {
        "headRefOid": "b" * 40,
        "reviews": [{"commit_id": "a" * 40, "body": "Reviewed A: clean."}],
        "statusCheckRollup": [],
    })
    ev = worker._review_evidence(_Cfg(), "r/r", 233, "a" * 40)
    assert ev["head_sha"] == "b" * 40
    assert ev["trusted_green"] is True    # the status we were woken for is real
    granted, reason = worker.merge_authority(True, ev, JevAssessment(True, "low", 0, 0, "adequate", 0.9, 0))
    assert not granted and "stale" in reason


def test_review_bound_to_a_different_sha_never_authorizes(monkeypatch):
    """A review explicitly attached to another commit is not evidence for this one."""
    _gh_level(monkeypatch, {
        "headRefOid": "b" * 40,
        "reviews": [{"commit_id": "c" * 40, "body": "Reviewed C."}],
        "statusCheckRollup": [],
    })
    ev = worker._review_evidence(_Cfg(), "r/r", 233, "b" * 40)
    assert ev["review_bound_ok"] is False
    granted, reason = worker.merge_authority(
        True, ev, JevAssessment(True, "low", 0.01, 0.01, "adequate", 0.95, 0.01))
    assert not granted and "reviewed commit" in reason


def test_nil_commit_id_reviews_are_the_normal_shape(monkeypatch):
    """rhobear-reviews posts COMMENTED reviews with commit_id=null (live PR #233)."""
    _gh_level(monkeypatch, {
        "headRefOid": "b" * 40,
        "reviews": [{"commit_id": None, "body": "x" * 6000}],
        "statusCheckRollup": [],
    })
    ev = worker._review_evidence(_Cfg(), "r/r", 233, "b" * 40)
    assert ev["review_bound_ok"] is True and ev["trusted_green"] is True
    granted, _ = worker.merge_authority(
        True, ev, JevAssessment(True, "low", 0.01, 0.01, "adequate", 0.95, 0.01))
    assert granted


def test_current_head_with_trusted_status_grants_authority(monkeypatch):
    _gh_level(monkeypatch, {
        "headRefOid": "b" * 40,
        "reviews": [{"commit_id": "b" * 40, "body": "Reviewed B: clean."}],
        "statusCheckRollup": [{"name": "Lint", "status": "COMPLETED", "conclusion": "FAILURE"}],
    })
    ev = worker._review_evidence(_Cfg(), "r/r", 233, "b" * 40)
    assert ev["trusted_green"] and ev["failing_checks"] == 1  # informational only
    granted, reason = worker.merge_authority(
        True, ev, JevAssessment(True, "low", 0.01, 0.01, "adequate", 0.95, 0.01))
    assert granted, reason


def test_untrusted_or_failed_reviewer_status_revokes_authority(monkeypatch):
    """Somebody else's success, or the reviewer's failure, is not green."""
    payload = {"headRefOid": "b" * 40,
               "reviews": [{"commit_id": None, "body": "Reviewed B."}],
               "statusCheckRollup": []}
    for status in (
        {"statuses": [{"context": "some-other-bot", "state": "success"}]},
        {"statuses": [{"context": "rhobear-reviews", "state": "failure"}]},
    ):
        _gh_level(monkeypatch, payload, status)
        ev = worker._review_evidence(_Cfg(), "r/r", 233, "b" * 40)
        assert ev["trusted_green"] is False
        granted, reason = worker.merge_authority(
            True, ev, JevAssessment(True, "low", 0.01, 0.01, "adequate", 0.95, 0.01))
        assert not granted and "trusted review status" in reason


def test_evidence_brief_states_authority_for_the_agent():
    text = worker.evidence_brief(
        {"ok": True, "decided_sha": "a" * 40, "head_sha": "a" * 40,
         "review_sha": "", "review_bound_ok": True,
         "trusted_states": ["success"], "failing_checks": 3},
        False, "the trusted review status is not success on this exact head SHA")
    assert "MERGE_AUTHORITY: NOT_GRANTED" in text
    assert "Merge only when MERGE_AUTHORITY is GRANTED" in text
