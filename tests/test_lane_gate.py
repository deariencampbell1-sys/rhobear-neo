"""North Star lane branches: Neo never fixes them and merges only an architect-accepted exact head.

Ported from the lane-gate commits on zeus/live-app-identity (0241eaf lane gate,
2aaf1ee switch) onto main's worker shape: `_gh`/`_review_evidence` take a token
first here, and the head branch is read by the evidence collector's existing
`pr view` call (one read, one source) instead of a second `gh pr view`.
"""
import dataclasses
import json
from unittest import mock

from src import neo_worker
from src.config import Config
from src.webhook_server import NeoDispatcher

HEAD = "a" * 40
OWNER = "deariencampbell1-sys"
REPO = f"{OWNER}/rhobear-north-star"


class FakeState:
    def __init__(self):
        self.rows = []

    def already_acted(self, repo, pr, sha, phase, green=None):
        return any(r[:4] == (repo, pr, sha, phase) for r in self.rows)

    def builder_rounds(self, repo, pr):
        return 0

    def install(self, *a):
        return {"auto_merge": True}

    def record(self, repo, pr, sha, phase, **kw):
        self.rows.append((repo, pr, sha, phase, kw.get("verdict", ""), kw.get("detail") or {}))


def evidence(**over):
    e = {"ok": True, "decided_sha": HEAD, "head_sha": HEAD, "trusted_green": True,
         "required_checks_pass": True, "head_ref": "l3-kit/package-tokens", "reviews": [],
         "green1_ok": True, "green1_reason": "", "trusted_red": [],
         "trusted_producer_ok": True, "review_sha_match": True,
         "unresolved_critical_high": 0, "coverage_complete": True, "auto_merge": True}
    e.update(over)
    return e


def architect(state="success", login=OWNER, lane="l3-kit", created=""):
    return {"context": f"architect/{lane}", "state": state,
            "creator": {"login": login}, "created_at": created}


def test_lane_of():
    assert neo_worker.lane_of("l3-kit/package-tokens") == "l3-kit"
    assert neo_worker.lane_of("L8d-engine-mcps/engine-jobs") == "l8d-engine-mcps"
    assert neo_worker.lane_of("l2-platform/worker-ingress-followup") == "l2-platform"
    assert neo_worker.lane_of("lane/reviewer-deepinfra-peak") == ""
    assert neo_worker.lane_of("fix/neo-deepseek-bedrock-pair") == ""
    assert neo_worker.lane_of("main") == ""
    # l8b/…-prefixed branches are always named as lane branches on main
    assert neo_worker.lane_of("l8b/fix-usb-macros") == "l8b"
    assert neo_worker.lane_of("l8b-short-pov-video/builder") == "l8b-short-pov-video"


def test_authority_needs_apollo_and_the_architect_on_this_head():
    ok, _ = neo_worker.lane_merge_authority(True, evidence(), "l3-kit", [architect()], [], OWNER)
    assert ok
    for args, why in (
        ((True, evidence(trusted_green=False), "l3-kit", [architect()], []), "Apollo"),
        ((True, evidence(required_checks_pass=False), "l3-kit", [architect()], []), "required checks"),
        ((True, evidence(), "l3-kit", [], []), "waiting"),
        ((True, evidence(), "l3-kit", [architect(lane="l9-front-door")], []), "waiting"),
        ((True, evidence(), "l3-kit", [architect(login="rhobear-neo[bot]")], []), "owner"),
        ((True, evidence(head_sha="b" * 40), "l3-kit", [architect()], []), "head moved"),
        ((False, evidence(), "l3-kit", [architect()], []), "disabled"),
        ((True, evidence(green1_ok=False, green1_reason="CHANGES_REQUESTED"),
          "l3-kit", [architect()], []), "green 1"),
    ):
        ok, reason = neo_worker.lane_merge_authority(*args, OWNER)
        assert not ok and why in reason, reason


def test_lane_gate_requires_apollo_accept_not_only_a_status():
    ok, reason = neo_worker.lane_merge_authority(
        True, evidence(green1_ok=False, green1_reason="COMMENTED"),
        "l3-kit", [architect()], [], OWNER)
    assert ok is False
    assert "green 1" in reason and "Apollo" in reason, reason


def test_lane_gate_refuses_a_red_trusted_reviewer():
    ok, reason = neo_worker.lane_merge_authority(
        True, evidence(trusted_red=["CodeAnt (failure)"]),
        "l3-kit", [architect()], [], OWNER)
    assert ok is False and "CodeAnt" in reason, reason


def test_only_the_newest_architect_status_counts():
    newest_first = [architect(state="failure"), architect()]
    ok, _ = neo_worker.lane_merge_authority(True, evidence(), "l3-kit", newest_first, [], OWNER)
    assert not ok
    oldest_first = [architect(created="2026-10-26T11:50:00Z"),
                    architect(state="failure", created="2026-10-26T12:00:00Z")]
    ok, _ = neo_worker.lane_merge_authority(True, evidence(), "l3-kit", oldest_first, [], OWNER)
    assert not ok


def test_lane_pr_empty_or_unavailable_head_ref_does_not_raise():
    assert neo_worker._lane_pr({"ok": True, "head_ref": ""}) is None
    assert neo_worker._lane_pr({"ok": False, "head_ref": ""}) is None


def test_old_changes_requested_on_another_head_does_not_block():
    reviews = [{"state": "CHANGES_REQUESTED", "commit": {"oid": "c" * 40}}]
    ok, _ = neo_worker.lane_merge_authority(True, evidence(), "l3-kit", [architect()], reviews, OWNER)
    assert ok


def fake_gh(statuses, merge_rc=0):
    calls = []

    def gh(token, *args, timeout=30):
        calls.append(args)
        if args[:2] == ("pr", "view") and args[-1] == "headRefName":
            return 0, json.dumps({"headRefName": "feature/not-a-lane"})
        if args[:2] == ("pr", "view"):
            return 0, json.dumps({"headRefOid": HEAD, "state": "OPEN",
                                  "mergeable": "MERGEABLE", "isCrossRepository": False,
                                  "headRefName": "l3-kit/package-tokens", "reviews": []})
        if args[:2] == ("api", "-X") and "PUT" in args:
            return merge_rc, (json.dumps({"merged": True, "sha": HEAD}) if merge_rc == 0
                              else "GitHub refused merge: required checks")
        if args[:2] == ("pr", "comment"):
            return 0, "commented"
        if args[0] == "api" and args[1].endswith("/statuses?per_page=100"):
            return 0, json.dumps(statuses)
        if args[0] == "api" and args[1] == f"repos/{REPO}":
            return 0, "main"
        if args[:2] == ("pr", "list"):
            return 0, ""
        if args[:2] == ("api", "-X") and "DELETE" in args:
            return 0, ""
        raise AssertionError(f"unexpected gh call {args}")
    return gh, calls


def run(statuses, merge_rc=0, evidence_dict=None, cfg=None):
    state = FakeState()
    gh, calls = fake_gh(statuses, merge_rc)
    cfg = dataclasses.replace(Config(), lane_merge=True) if cfg is None else cfg
    ev = evidence(statuses=statuses) if evidence_dict is None else {
        **evidence_dict, "statuses": evidence_dict.get("statuses", statuses)}
    with mock.patch.object(neo_worker, "_gh", gh), \
         mock.patch("src.merge_executor._gh", gh), \
         mock.patch.object(neo_worker, "repo_token", return_value="tok"), \
         mock.patch.object(neo_worker, "resolve_pr", return_value=51), \
         mock.patch.object(neo_worker, "_review_evidence", return_value=ev), \
         mock.patch.object(neo_worker.ClaudeAgent, "from_config") as agent:
        neo_worker.run_neo(cfg, state, mock.MagicMock(),
                           {"job_id": 1, "repo": REPO, "sha": HEAD, "green": True,
                            "context": "rhobear-reviews"})
    agent.assert_not_called()          # never fix-forward or bounce a lane branch
    return state, calls


def test_lane_pr_merges_the_exact_head_without_the_agent():
    state, calls = run([architect()])
    merge = next(c for c in calls if c[:2] == ("api", "-X") and "PUT" in c)
    assert f"sha={HEAD}" in merge
    assert state.rows[-1][3] == "merged"


def test_lane_pr_without_architect_accept_holds():
    state, calls = run([])
    assert not any(c[:2] == ("api", "-X") and "PUT" in c for c in calls)
    assert state.rows[-1][3] == "lane_hold" and "architect" in state.rows[-1][5]["reason"]


def test_refused_merge_is_a_hold_not_a_merge():
    state, _ = run([architect()], merge_rc=1)
    assert state.rows[-1][3] == "lane_hold" and "refused" in state.rows[-1][5]["reason"]


def test_lane_gate_reads_evidence_through_the_real_collector():
    """Defect 2 (arity): run_lane_gate's calls must match THIS line's shapes —
    `_gh(token, …)` and `_review_evidence(token, cfg, …)`.  This test patches
    only `_gh` (the network) and runs the real evidence collector through the
    lane gate: a `_gh(cfg, …)`-shaped call would TypeError here, not merge.
    The fixture carries the green producer run for the head so the gate's
    review_bound_ok condition (defect 1) is satisfied by this evidence."""
    state = FakeState()
    statuses = [architect(lane="l8b"),
                {"context": "rhobear-reviews", "state": "success"}]

    def gh(token, *args, timeout=30):
        if args[:2] == ("pr", "view"):
            return 0, json.dumps({"headRefOid": HEAD, "headRefName": "l8b/fix-usb-macros",
                                  "state": "OPEN", "mergeable": "MERGEABLE",
                                  "isCrossRepository": False, "installation_auto_merge": True,
                                  "reviews": [{"state": "APPROVED", "commit_id": HEAD, "body": "",
                                               "author": {"login": "rhobear-reviews[bot]",
                                                          "type": "Bot"}}],
                                  "statusCheckRollup": [
                                      {"context": "rhobear-reviews", "state": "SUCCESS"}]})
        if args[0] == "api" and "/check-runs" in args[1]:
            return 0, json.dumps({"total_count": 1, "check_runs": [
                {"id": 1, "name": "rhobear-reviews", "app": {"id": 17265050},
                 "head_sha": HEAD, "status": "completed", "conclusion": "success",
                 "completed_at": "2026-10-26T11:50:00Z",
                 "output": {"summary":
                            '{"unresolved_critical_high": 0, "coverage": "complete"}'}}]})
        if args[0] == "api" and args[1].endswith("/statuses?per_page=100"):
            return 0, json.dumps(statuses)
        if args[0] == "api" and args[1].endswith("/status"):
            return 0, json.dumps({"total_count": len(statuses), "statuses": statuses})
        if args[:2] == ("api", "-X") and "PUT" in args:
            return 0, json.dumps({"merged": True, "sha": HEAD})
        if args[:2] == ("pr", "comment"):
            return 0, "commented"
        if args[0] == "api" and any("/pulls/51/reviews" in a for a in args):
            return 0, json.dumps([[
                {"user": {"login": "rhobear-reviews[bot]", "type": "Bot"}}]])
        if args[0] == "api" and args[1] == f"repos/{REPO}":
            return 0, "main"
        if args[:2] == ("pr", "list"):
            return 0, ""
        if args[:2] == ("api", "-X") and "DELETE" in args:
            return 0, ""
        raise AssertionError(f"unexpected gh call {args}")

    cfg = dataclasses.replace(Config(), lane_merge=True,
                              trusted_reviewer_app_id="17265050")
    with mock.patch.object(neo_worker, "_gh", gh), \
         mock.patch("src.merge_executor._gh", gh), \
         mock.patch.object(neo_worker, "repo_token", return_value="tok"), \
         mock.patch.object(neo_worker, "resolve_pr", return_value=51), \
         mock.patch.object(neo_worker.ClaudeAgent, "from_config") as agent:
        neo_worker.run_neo(cfg, state, mock.MagicMock(),
                           {"job_id": 1, "repo": REPO, "sha": HEAD, "green": True,
                            "context": "rhobear-reviews"})
    agent.assert_not_called()
    assert state.rows[-1][3] == "merged", state.rows


def test_lane_gate_uses_fresh_reviews_not_the_stale_outer_list():
    fresh_review = {"state": "CHANGES_REQUESTED", "commit_id": HEAD,
                    "author": {"login": "rhobear-reviews[bot]", "type": "Bot"}}
    fresh = evidence(reviews=[fresh_review], statuses=[architect()],
                     green1_ok=False, green1_reason="CHANGES_REQUESTED")
    state = FakeState()
    gh, calls = fake_gh([architect()])
    cfg = dataclasses.replace(Config(), lane_merge=True)
    with mock.patch.object(neo_worker, "_gh", gh), \
         mock.patch("src.merge_executor._gh", gh), \
         mock.patch.object(neo_worker, "_review_evidence", return_value=fresh), \
         mock.patch.object(neo_worker, "_enabled_reviewers", return_value=()):
        neo_worker.run_lane_gate("tok", cfg, state, REPO, 51, HEAD, "l3-kit")
    assert not any(c[:2] == ("api", "-X") and "PUT" in c for c in calls), calls
    assert not any(c[0] == "api" and c[1].endswith("/statuses?per_page=100") for c in calls), calls
    assert state.rows[-1][3] == "lane_hold", state.rows
    assert "green 1" in state.rows[-1][5]["reason"], state.rows


def test_architect_status_wakes_neo_but_other_untrusted_contexts_do_not():
    wakes = []
    cfg = mock.Mock(repo_allowlist=(REPO,), trusted_review_contexts=("rhobear-reviews",))
    d = NeoDispatcher(cfg, wakes.append)
    payload = {"repository": {"full_name": REPO}, "sha": HEAD, "state": "success"}
    assert d.dispatch("status", {**payload, "context": "architect/l3-kit"}).code == 200
    assert d.dispatch("status", {**payload, "context": "ci/random"}).code == 204
    assert [w["context"] for w in wakes] == ["architect/l3-kit"]


def test_drift_gate_needs_a_green_l0_drift_on_this_head():
    """SPRINT section 9: with require_drift, a failing or missing l0/drift holds the merge."""
    drift_ok = {"context": "l0/drift", "state": "success", "creator": {"login": OWNER}}
    drift_bad = {"context": "l0/drift", "state": "failure", "description": "2 hard", "creator": {"login": OWNER}}
    ok, _ = neo_worker.lane_merge_authority(True, evidence(), "l3-kit", [drift_ok, architect()], [], OWNER,
                                            require_drift=True)
    assert ok
    ok, reason = neo_worker.lane_merge_authority(True, evidence(), "l3-kit", [architect()], [], OWNER,
                                                 require_drift=True)
    assert not ok and "waiting for L0" in reason
    ok, reason = neo_worker.lane_merge_authority(True, evidence(), "l3-kit", [drift_bad, architect()], [], OWNER,
                                                 require_drift=True)
    assert not ok and "fails" in reason
    ok, _ = neo_worker.lane_merge_authority(True, evidence(), "l3-kit", [architect()], [], OWNER)
    assert ok   # switch off: unchanged behaviour


def test_drift_success_requires_the_owner_creator():
    for creator in ({"login": "mallory"}, None):
        drift = {"context": "l0/drift", "state": "success", "creator": creator}
        ok, reason = neo_worker.lane_merge_authority(
            True, evidence(), "l3-kit", [drift, architect()], [], OWNER,
            require_drift=True)
        assert ok is False and "owner's account" in reason, reason


def test_lane_require_drift_is_off_by_default_and_on_with_the_switch(monkeypatch):
    """NEO_LANE_REQUIRE_DRIFT is opt-in: default off, and the run-time check
    (`… is True`) refuses a non-True value — an unset switch is never green."""
    monkeypatch.delenv("NEO_LANE_REQUIRE_DRIFT", raising=False)
    assert Config().lane_require_drift is False
    monkeypatch.setenv("NEO_LANE_REQUIRE_DRIFT", "true")
    assert Config().lane_require_drift is True


def test_drift_status_wakes_the_lane_gate_but_never_the_agent():
    """An l0/drift event wakes the lane gate like an architect status, and never
    starts the agent on a non-lane PR (dc317b0)."""
    wakes = []
    cfg = mock.Mock(repo_allowlist=(REPO,), trusted_review_contexts=("rhobear-reviews",))
    d = NeoDispatcher(cfg, wakes.append)
    payload = {"repository": {"full_name": REPO}, "sha": HEAD, "state": "success"}
    assert d.dispatch("status", {**payload, "context": "l0/drift"}).code == 200
    assert [w["context"] for w in wakes] == ["l0/drift"]

    state = FakeState()
    gh, calls = fake_gh([])
    with mock.patch.object(neo_worker, "_gh", gh), \
         mock.patch("src.merge_executor._gh", gh), \
         mock.patch.object(neo_worker, "repo_token", return_value="tok"), \
         mock.patch.object(neo_worker, "resolve_pr", return_value=51), \
         mock.patch.object(neo_worker, "_review_evidence",
                           return_value=evidence(head_ref="feature/not-a-lane")), \
         mock.patch.object(neo_worker.ClaudeAgent, "from_config") as agent:
        neo_worker.run_neo(dataclasses.replace(Config(), lane_merge=True),
                           state, mock.MagicMock(),
                           {"job_id": 1, "repo": REPO, "sha": HEAD, "green": True,
                            "context": "l0/drift"})
    agent.assert_not_called()
    assert not any(c[:2] == ("api", "-X") and "PUT" in c for c in calls)
    assert state.rows == []


def test_review_bound_is_required_by_the_lane_gate():
    """Defect 1: the gate's conditions dropped review_bound_ok — an ACCEPT on
    an older commit passed the lane gate and merged. The reviewed commit must
    be this head SHA, and the hold names it."""
    state, calls = run([architect()], evidence_dict=evidence(review_bound_ok=False))
    assert not any(c[:2] == ("api", "-X") and "PUT" in c for c in calls)
    assert state.rows[-1][3] == "lane_hold"
    assert "reviewed commit" in state.rows[-1][5]["reason"]
    # The same evidence, bound to the head, merges.
    state, calls = run([architect()], evidence_dict=evidence(review_bound_ok=True))
    assert any(c[:2] == ("api", "-X") and "PUT" in c for c in calls)
    assert state.rows[-1][3] == "merged"


def test_architect_prefix_match_is_case_insensitive():
    """Defect 4: the webhook dispatch matched the architect/ prefix
    case-sensitively — "Architect/L3-Kit" was dropped as untrusted and the
    lane gate never woke. One case-insensitive normaliser, shared by the
    dispatch and the worker's wake routing."""
    from src import webhook_server

    wakes = []
    cfg = mock.Mock(repo_allowlist=(REPO,), trusted_review_contexts=("rhobear-reviews",))
    d = NeoDispatcher(cfg, wakes.append)
    payload = {"repository": {"full_name": REPO}, "sha": HEAD, "state": "success"}
    assert d.dispatch("status", {**payload, "context": "Architect/L3-Kit"}).code == 200
    assert d.dispatch("status", {**payload, "context": "L0/Drift"}).code == 200
    assert [w["context"] for w in wakes] == ["Architect/L3-Kit", "L0/Drift"]
    assert d.dispatch("status", {**payload, "context": "ci/random"}).code == 204
    assert webhook_server.is_architect_context("Architect/L3-Kit")
    assert webhook_server.is_drift_context("L0/DRIFT")


def test_lane_gate_fails_closed_with_a_lane_hold_when_the_evidence_read_breaks():
    """Edge 5: if the evidence read itself raises (broken gh, an arity slip),
    the lane gate must fail closed — a lane_hold naming the break — never
    merge, and never let the error escape and kill the lane wake."""
    state = FakeState()
    gh, calls = fake_gh([architect()])
    cfg = dataclasses.replace(Config(), lane_merge=True)
    ev = evidence()
    with mock.patch.object(neo_worker, "_gh", gh), \
         mock.patch("src.merge_executor._gh", gh), \
         mock.patch.object(neo_worker, "repo_token", return_value="tok"), \
         mock.patch.object(neo_worker, "resolve_pr", return_value=51), \
         mock.patch.object(neo_worker, "_review_evidence",
                           side_effect=[ev, TypeError("got an unexpected keyword")]), \
         mock.patch.object(neo_worker.ClaudeAgent, "from_config") as agent:
        neo_worker.run_neo(cfg, state, mock.MagicMock(),
                           {"job_id": 1, "repo": REPO, "sha": HEAD, "green": True,
                            "context": "rhobear-reviews"})
    agent.assert_not_called()
    assert not any(c[:2] == ("api", "-X") and "PUT" in c for c in calls)
    assert state.rows[-1][3] == "lane_hold"
    assert "evidence" in state.rows[-1][5]["reason"]


def test_transient_evidence_failure_does_not_skip_agent_on_non_lane_pr():
    state = FakeState()
    gh, _ = fake_gh([])
    transient = {"ok": False, "reason": "pr view unavailable",
                 "decided_sha": HEAD, "head_ref": ""}
    with mock.patch.object(neo_worker, "_gh", gh), \
         mock.patch("src.merge_executor._gh", gh), \
         mock.patch.object(neo_worker, "repo_token", return_value="tok"), \
         mock.patch.object(neo_worker, "resolve_pr", return_value=51), \
         mock.patch.object(neo_worker, "_review_evidence", return_value=transient), \
         mock.patch.object(neo_worker, "_run_agent", return_value=(None, "", 0, "")), \
         mock.patch.object(neo_worker.ClaudeAgent, "from_config") as agent:
        neo_worker.run_neo(Config(), state, mock.MagicMock(),
                           {"job_id": 1, "repo": REPO, "sha": HEAD, "green": True,
                            "context": "rhobear-reviews"})
    agent.assert_called_once()

def test_lane_merge_disabled_by_default_records_a_lane_hold():
    state, calls = run([architect()], cfg=Config())
    assert Config().lane_merge is False
    assert not any(c[:2] == ("api", "-X") and "PUT" in c for c in calls)
    assert state.rows[-1][3] == "lane_hold"
    assert "lane merges disabled" in state.rows[-1][5]["reason"]

def test_lane_gate_runs_the_same_belts_then_posts_the_comment():
    state, calls = run([architect()])
    comment = next((c for c in calls if c[:2] == ("pr", "comment")), None)
    assert comment is not None
    body = comment[comment.index("--body") + 1]
    assert body.startswith("Two greens on " + HEAD[:7])
    assert "CI:" in body
    assert state.rows[-1][3] == "merged"
    state, calls = run([architect()], evidence_dict=evidence(trusted_red=["CodeAnt (failure)"]))
    assert not any(c[:2] == ("api", "-X") and "PUT" in c for c in calls)
    assert not any(c[:2] == ("pr", "comment") for c in calls)
    assert state.rows[-1][3] == "lane_hold"

def test_lane_of_honours_the_operator_lane_exclusion():
    from src import neo_builder
    cfg = dataclasses.replace(Config(), lane_exclude=("l1-hotfix",))
    assert neo_worker.lane_of("l1-hotfix/x", cfg) == ""
    assert neo_worker.lane_of("l3-kit/x", cfg) == "l3-kit"
    assert neo_worker.lane_of("l1-hotfix/x") == "l1-hotfix"
    assert neo_builder.is_lane_branch("l1-hotfix/x", cfg) is False
    assert neo_builder.is_lane_branch("l3-kit/x", cfg) is True


def test_failed_evidence_still_reads_lane_head_independently_and_never_runs_agent():
    state = FakeState()
    transient = {"ok": False, "reason": "pr view unavailable",
                 "decided_sha": HEAD, "head_ref": ""}

    def gh(token, *args, timeout=30):
        if args[:2] == ("pr", "view") and args[-1] == "headRefName":
            return 0, json.dumps({"headRefName": "l3-kit/package-tokens"})
        raise AssertionError(f"unexpected gh call {args}")

    with mock.patch.object(neo_worker, "_gh", gh), \
         mock.patch.object(neo_worker, "repo_token", return_value="tok"), \
         mock.patch.object(neo_worker, "resolve_pr", return_value=51), \
         mock.patch.object(neo_worker, "_review_evidence", side_effect=[transient, transient]), \
         mock.patch.object(neo_worker, "_run_agent", return_value=(None, "", 0, "")), \
         mock.patch.object(neo_worker.ClaudeAgent, "from_config") as agent:
        neo_worker.run_neo(Config(), state, mock.MagicMock(),
                           {"job_id": 1, "repo": REPO, "sha": HEAD, "green": True,
                            "context": "rhobear-reviews"})
    agent.assert_not_called()
    assert state.rows[-1][3] == "lane_hold"


def test_unknown_head_ref_holds_without_agent():
    state = FakeState()
    transient = {"ok": False, "reason": "pr view unavailable",
                 "decided_sha": HEAD, "head_ref": ""}
    with mock.patch.object(neo_worker, "_gh", return_value=(1, "unavailable")), \
         mock.patch.object(neo_worker, "repo_token", return_value="tok"), \
         mock.patch.object(neo_worker, "resolve_pr", return_value=51), \
         mock.patch.object(neo_worker, "_review_evidence", return_value=transient), \
         mock.patch.object(neo_worker, "_run_agent", return_value=(None, "", 0, "")), \
         mock.patch.object(neo_worker.ClaudeAgent, "from_config") as agent:
        neo_worker.run_neo(Config(), state, mock.MagicMock(),
                           {"job_id": 1, "repo": REPO, "sha": HEAD, "green": True,
                            "context": "rhobear-reviews"})
    agent.assert_not_called()
    assert len(state.rows) == 1
    assert state.rows[0][3] == "lane_hold"
    assert state.rows[0][5]["lane"] == "unknown"
    assert "head branch unavailable" in state.rows[0][5]["reason"]


def test_lane_merge_uses_shared_executor():
    state = FakeState()
    ev = evidence(statuses=[architect()])
    with mock.patch.object(neo_worker, "_enabled_reviewers", return_value=()), \
         mock.patch.object(neo_worker, "_review_evidence", return_value=ev), \
         mock.patch.object(neo_worker, "_gh") as direct_gh, \
         mock.patch("src.merge_executor.merge", return_value={"merged": True, "sha": HEAD}) as shared:
        neo_worker.run_lane_gate(
            "tok", dataclasses.replace(Config(), lane_merge=True),
            state, REPO, 51, HEAD, "l3-kit")
    shared.assert_called_once()
    args = shared.call_args.args
    assert args[:3] == (REPO, 51, HEAD)
    assert args[3] is ev and args[3]["green2_kind"] == "architect"
    assert args[4] == "tok"
    assert not any(c.args[:2] == ("pr", "merge") for c in direct_gh.call_args_list)
    assert state.rows[-1][3] == "merged"


def test_lane_merge_denies_unresolved_critical_high_through_shared_executor():
    state = FakeState()
    ev = evidence(statuses=[architect()], trusted_producer_ok=True,
                  review_sha_match=True, unresolved_critical_high=2,
                  coverage_complete=True, auto_merge=True)
    calls = []

    def gh(token, *args, timeout=30):
        calls.append(args)
        if args[:2] == ("pr", "view"):
            return 0, json.dumps({"headRefOid": HEAD, "state": "OPEN",
                                  "mergeable": "MERGEABLE", "headRefName": "l3-kit/package-tokens",
                                  "isCrossRepository": False})
        if args[:2] == ("api", "-X") and "PUT" in args:
            return 0, json.dumps({"merged": True, "sha": HEAD})
        raise AssertionError(f"unexpected gh call {args}")

    with mock.patch.object(neo_worker, "_enabled_reviewers", return_value=()), \
         mock.patch.object(neo_worker, "_review_evidence", return_value=ev), \
         mock.patch.object(neo_worker, "_gh", gh), \
         mock.patch("src.merge_executor._gh", gh):
        neo_worker.run_lane_gate(
            "tok", dataclasses.replace(Config(), lane_merge=True),
            state, REPO, 51, HEAD, "l3-kit")
    assert not any("PUT" in c for c in calls)
    assert state.rows[-1][3] == "lane_hold"
    assert "critical" in state.rows[-1][5]["reason"].lower()
