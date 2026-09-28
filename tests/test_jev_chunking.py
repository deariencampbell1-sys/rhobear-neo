"""Phase D: a fully-examined large diff is not "incomplete"; a cut one still is.

Reproduces the live case: the ~10k-line Expo app PR was truncated to 56k chars,
so Jev reported REVIEW_INCOMPLETE and a fully-reviewed PR could never merge.
"""
import io
import json

from src.jev_review import (JevAssessment, MAX_DIFF_CHARS, _chunk_diff, _merge,
                            assess)


def _answers(risk="low", quality="adequate", confidence=0.9, security=0.02,
             correctness=0.02, escalate=0.02, findings=None):
    answers = {
        "risk": {"choice": risk},
        "security": {"noul": security},
        "correctness": {"noul": correctness},
        "review_quality": {"choice": quality, "confidence": confidence},
        "escalate": {"noul": escalate},
    }
    if findings is not None:
        answers["findings"] = findings
    return {"answers": answers}


def _mock(monkeypatch, calls, responder=None):
    """Record every request; reply with `responder(index)` or a default answer."""
    def fake_open(request, timeout):
        body = json.loads(request.data)
        calls.append(body)
        if responder is not None:
            payload = responder(len(calls) - 1, body)
            if payload is None:
                raise OSError("provider down")
            return io.BytesIO(json.dumps(payload).encode())
        return io.BytesIO(json.dumps(_answers()).encode())

    monkeypatch.setattr("src.jev_review.urllib.request.urlopen", fake_open)


def _big_diff(files=4, body_lines=1400):
    out = []
    for i in range(files):
        out.append(f"diff --git a/src/mod{i}.ts b/src/mod{i}.ts\n"
                   + "".join(f"+line {n} of module {i}\n" for n in range(body_lines)))
    return "".join(out)


def test_small_diff_is_still_one_call(monkeypatch):
    calls = []
    _mock(monkeypatch, calls)
    result = assess("+color: blue", "Reviewed color change.", "k")
    assert len(calls) == 1 and result.available and not result.partial
    assert result.blockers == ()


def test_large_diff_is_chunked_and_complete(monkeypatch):
    diff = _big_diff()
    assert len(diff) > MAX_DIFF_CHARS
    calls = []
    _mock(monkeypatch, calls)
    result = assess(diff, "Reviewed every module; no findings.", "k")
    assert len(calls) > 1, "large diff must be split, not truncated"
    # every chunk carries the manifest of ALL changed files, not just its own
    for payload in calls:
        text = payload["state"]["diff"]
        assert text.startswith("[file manifest")
        for i in range(4):
            assert f"a/src/mod{i}.ts" in text
    assert result.partial is False, "every surface was inspected"
    assert result.blockers == ()
    assert result.requires_full_review is False


def test_a_cut_surface_still_blocks(monkeypatch):
    """One file larger than a chunk budget cannot be hidden by chunking."""
    diff = ("diff --git a/package-lock.json b/package-lock.json\n"
            + "+x\n" * 90_000
            + "diff --git a/src/auth.ts b/src/auth.ts\n+requireAdmin()\n")
    calls = []
    _mock(monkeypatch, calls)
    result = assess(diff, "Reviewed.", "k")
    assert result.partial is True
    assert [f.rule for f in result.blockers] == ["REVIEW_INCOMPLETE"]


def test_provider_failure_on_any_chunk_blocks(monkeypatch):
    diff = _big_diff()
    calls = []
    _mock(monkeypatch, calls, responder=lambda i, body: None if i == 1 else _answers())
    result = assess(diff, "Reviewed.", "k")
    assert result.partial is True, "a failed chunk is an uninspected surface"
    assert [f.rule for f in result.blockers] == ["REVIEW_INCOMPLETE"]


def test_aggregate_takes_the_worst_chunk(monkeypatch):
    calls = []
    _mock(monkeypatch, calls, responder=lambda i, body: [
        _answers(risk="low", quality="adequate", confidence=0.95),
        _answers(risk="high", quality="poor", confidence=0.4, security=0.9),
    ][min(i, 1)])
    result = assess(_big_diff(), "Reviewed.", "k")
    assert result.risk == "high" and result.review_quality == "poor"
    assert result.review_confidence == 0.4 and result.security == 0.9
    assert [f.rule for f in result.blockers] == ["REVIEW_NOT_ADEQUATE"]


def test_merge_never_upgrades_and_dedupes_findings(monkeypatch):
    merged = _merge([
        {"risk": "medium", "quality": "unclear", "confidence": 0.8, "security": 0.1,
         "correctness": 0.4, "escalate": 0.3,
         "findings": (type("F", (), {"rule": "REVIEW_NOT_ADEQUATE", "evidence": "x" * 100})(),)},
        {"risk": "low", "quality": "adequate", "confidence": 0.95, "security": 0.0,
         "correctness": 0.0, "escalate": 0.0, "findings": ()},
    ])
    assert merged["risk"] == "medium" and merged["quality"] == "unclear"
    assert merged["confidence"] == 0.8
    assert len(merged["findings"]) == 1


def test_chunking_keeps_every_file_in_the_manifest():
    diff = _big_diff(files=3, body_lines=2000)
    chunks, complete = _chunk_diff(diff)
    assert complete and len(chunks) > 1
    for chunk in chunks:
        assert "file manifest" in chunk
        for i in range(3):
            assert f"a/src/mod{i}.ts" in chunk
    joined = "".join(chunks)
    for i in range(3):
        assert joined.count(f"a/src/mod{i}.ts") >= 2  # manifest + section
