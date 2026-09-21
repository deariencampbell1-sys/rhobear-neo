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
