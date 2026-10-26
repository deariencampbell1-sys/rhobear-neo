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
         "head_ref": "l3-kit/package-tokens", "reviews": []}
    e.update(over)
    return e


def architect(state="success", login=OWNER, lane="l3-kit"):
    return {"context": f"architect/{lane}", "state": state, "creator": {"login": login}}


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
        ((True, evidence(), "l3-kit", [], []), "waiting"),
        ((True, evidence(), "l3-kit", [architect(lane="l9-front-door")], []), "waiting"),
        ((True, evidence(), "l3-kit", [architect(login="rhobear-neo[bot]")], []), "owner"),
        ((True, evidence(head_sha="b" * 40), "l3-kit", [architect()], []), "head moved"),
        ((False, evidence(), "l3-kit", [architect()], []), "disabled"),
        ((True, evidence(), "l3-kit", [architect()],
          [{"state": "CHANGES_REQUESTED", "commit": {"oid": HEAD}}]), "requested changes"),
    ):
        ok, reason = neo_worker.lane_merge_authority(*args, OWNER)
        assert not ok and why in reason, reason


def test_only_the_newest_architect_status_counts():
    newest_first = [architect(state="failure"), architect()]
    ok, _ = neo_worker.lane_merge_authority(True, evidence(), "l3-kit", newest_first, [], OWNER)
    assert not ok


def test_old_changes_requested_on_another_head_does_not_block():
    reviews = [{"state": "CHANGES_REQUESTED", "commit": {"oid": "c" * 40}}]
    ok, _ = neo_worker.lane_merge_authority(True, evidence(), "l3-kit", [architect()], reviews, OWNER)
    assert ok


def fake_gh(statuses, merge_rc=0):
    calls = []

    def gh(token, *args, timeout=30):
        calls.append(args)
        if args[:2] == ("pr", "view") and "headRefName" in args:
            return 0, json.dumps({"headRefOid": HEAD, "headRefName": "l3-kit/package-tokens",
                                  "reviews": []})
        if args[:2] == ("pr", "merge"):
            return merge_rc, "merged" if merge_rc == 0 else "Head branch was modified"
        if args[0] == "api" and args[1].endswith("/statuses?per_page=100"):
            return 0, json.dumps(statuses)
        raise AssertionError(f"unexpected gh call {args}")
    return gh, calls


def run(statuses, merge_rc=0, evidence_dict=None):
    state = FakeState()
    gh, calls = fake_gh(statuses, merge_rc)
    cfg = dataclasses.replace(Config(), lane_merge=True)
    ev = evidence() if evidence_dict is None else evidence_dict
    with mock.patch.object(neo_worker, "_gh", gh), \
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
    merge = next(c for c in calls if c[:2] == ("pr", "merge"))
    assert "--match-head-commit" in merge and HEAD in merge
    assert state.rows[-1][3] == "merged"


def test_lane_pr_without_architect_accept_holds():
    state, calls = run([])
    assert not any(c[:2] == ("pr", "merge") for c in calls)
    assert state.rows[-1][3] == "lane_hold" and "architect" in state.rows[-1][5]["reason"]


def test_refused_merge_is_a_hold_not_a_merge():
    state, _ = run([architect()], merge_rc=1)
    assert state.rows[-1][3] == "lane_hold" and "refused" in state.rows[-1][5]["reason"]


def test_lane_gate_reads_evidence_through_the_real_collector():
    """Defect 2 (arity): run_lane_gate's calls must match THIS line's shapes —
    `_gh(token, …)` and `_review_evidence(token, cfg, …)`.  This test patches
    only `_gh` (the network) and runs the real evidence collector through the
    lane gate: a `_gh(cfg, …)`-shaped call would TypeError here, not merge."""
    state = FakeState()
    statuses = [architect(lane="l8b"),
                {"context": "rhobear-reviews", "state": "success"}]

    def gh(token, *args, timeout=30):
        if args[:2] == ("pr", "view"):
            return 0, json.dumps({"headRefOid": HEAD, "headRefName": "l8b/fix-usb-macros",
                                  "reviews": [], "statusCheckRollup": []})
        if args[0] == "api" and "/check-runs" in args[1]:
            return 0, json.dumps({"total_count": 0, "check_runs": []})
        if args[0] == "api" and args[1].endswith("/statuses?per_page=100"):
            return 0, json.dumps(statuses)
        if args[0] == "api" and args[1].endswith("/status"):
            return 0, json.dumps({"total_count": len(statuses), "statuses": statuses})
        if args[:2] == ("pr", "merge"):
            return 0, "merged"
        raise AssertionError(f"unexpected gh call {args}")

    cfg = dataclasses.replace(Config(), lane_merge=True,
                              trusted_reviewer_app_id="17265050")
    with mock.patch.object(neo_worker, "_gh", gh), \
         mock.patch.object(neo_worker, "repo_token", return_value="tok"), \
         mock.patch.object(neo_worker, "resolve_pr", return_value=51), \
         mock.patch.object(neo_worker.ClaudeAgent, "from_config") as agent:
        neo_worker.run_neo(cfg, state, mock.MagicMock(),
                           {"job_id": 1, "repo": REPO, "sha": HEAD, "green": True,
                            "context": "rhobear-reviews"})
    agent.assert_not_called()
    assert state.rows[-1][3] == "merged", state.rows


def test_architect_status_wakes_neo_but_other_untrusted_contexts_do_not():
    wakes = []
    cfg = mock.Mock(repo_allowlist=(REPO,), trusted_review_contexts=("rhobear-reviews",))
    d = NeoDispatcher(cfg, wakes.append)
    payload = {"repository": {"full_name": REPO}, "sha": HEAD, "state": "success"}
    assert d.dispatch("status", {**payload, "context": "architect/l3-kit"}).code == 200
    assert d.dispatch("status", {**payload, "context": "ci/random"}).code == 204
    assert [w["context"] for w in wakes] == ["architect/l3-kit"]
