"""ESCALATE must be a handoff, not a dead end.

Neo escalated 67 times against 12 merges in 24h. Two causes, both in the
prompt: "brand" was listed as a standalone owner-gated fork, and running out
of builder rounds was defined AS an escalation. Meanwhile phase='escalated'
has no consumer anywhere in the codebase -- neo_state declares it, neo_worker
writes it, nothing reads it -- so every escalation simply stopped.
"""
from src import neo_protocol


def _prompt(**kw):
    kw.setdefault("repo", "o/r"); kw.setdefault("pr", 1)
    kw.setdefault("head_sha", "a" * 40); kw.setdefault("reviewer_green", True)
    kw.setdefault("reviewer_context", "rhobear-reviews")
    kw.setdefault("auto_merge", True)
    kw.setdefault("builder_round", 2); kw.setdefault("max_builder_rounds", 2)
    kw.setdefault("builder_model", "builder_top")
    return neo_protocol.build_brief(**kw)


def test_out_of_rounds_is_not_an_escalation():
    p = _prompt(builder_round=2, max_builder_rounds=2)
    assert "OUT of builder rounds" in p
    assert "Running out of rounds is NOT an escalation" in p
    assert "ESCALATE with the outstanding findings" not in p


def test_escalate_is_a_handoff_to_the_owner_proxy():
    p = _prompt()
    assert "HANDOFF to the owner-proxy agent" in p
    assert "not a stop" in p


def test_brand_alone_does_not_justify_escalation():
    p = _prompt()
    assert '"Brand" alone is NOT sufficient' in p
    assert "If the repo states a rule, APPLY it" in p


def test_repo_contradiction_is_a_judgement_call():
    p = _prompt()
    assert "a contradiction is a judgement call" in p


def test_escalation_must_enumerate_answerable_options():
    p = _prompt()
    assert "A1/A2" in p
    assert "NEVER auto-merge an escalation" in p


def test_irreversible_actions_still_escalate():
    p = _prompt()
    for must in ("spending money", "signing certs", "public launch", "deleting production data"):
        assert must in p, must


# ===================================================================
# GREEN definition — COMMENT + success is green (non-blocking notes)
# ===================================================================

def test_brief_defines_green_as_commit_status_not_verdict_word():
    brief = _prompt(reviewer_green=True)
    assert "COMMIT STATUS is `success`" in brief
    assert "`COMMENT` delivered with that success status is GREEN" in brief
    assert "do not escalate on the verdict word `COMMENT` alone" in brief
    assert "`REQUEST_CHANGES` is never green" in brief


def test_brief_comment_notes_are_advisory_but_high_still_blocks():
    brief = _prompt(reviewer_green=True)
    assert "COMMENT-with-success notes are advisory" in brief
    assert "a note that is CRITICAL or HIGH is an open finding and still blocks" in brief


def test_brief_not_green_path_unchanged():
    brief = _prompt(reviewer_green=False)
    assert "NOT green (changes requested / failing)" in brief
    assert "commit status failure/pending" in brief
