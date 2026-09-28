"""Jev may tighten Neo's merge gate, never create a green verdict."""

import io
import json

from src.jev_review import JevAssessment, _bounded_diff, assess


def test_low_risk_requires_every_signal_to_be_clear(monkeypatch):
    response = {
        "answers": {
            "risk": {"choice": "low"},
            "security": {"noul": 0.02},
            "correctness": {"noul": 0.04},
            "review_quality": {"choice": "adequate", "confidence": 0.91},
            "escalate": {"noul": 0.03},
        }
    }
    seen = {}

    def fake_open(request, timeout):
        seen["body"] = json.loads(request.data)
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("src.jev_review.urllib.request.urlopen", fake_open)
    result = assess("+color: blue", "Reviewed color change.", "test-key")
    assert result.available and not result.needs_attention
    assert seen["body"]["model"] == "typesafe/jev-1.13"
    assert len(seen["body"]["questions"]) == 5


def test_missing_or_oversize_evidence_cannot_auto_merge():
    assert assess("", "review", "test-key").needs_attention
    compact, partial = _bounded_diff(
        "diff --git a/package-lock.json b/package-lock.json\n" + "+x\n" * 20_000
        + "diff --git a/src/auth.py b/src/auth.py\n-    require_admin()\n"
    )
    assert partial and "require_admin" in compact
    assert "generated dependency body omitted" in compact
    assert JevAssessment(True, "low", 0.01, 0.01, "poor", 0.95, 0.01).needs_attention
    assert JevAssessment(True, "low", 0.21, 0.01, "adequate", 0.95, 0.01).needs_attention
    assert JevAssessment(True, "low", 0.01, 0.01, "adequate", 0.95, 0.01,
                         partial=True).needs_attention
