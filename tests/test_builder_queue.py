"""Builder lane-branch exclusion (port of 8dc58c8): one writer per lane branch.

A lane branch belongs to its lane's briefed worker — Builder never clones,
fixes, or pushes to one, and the builder row records it (skipped_lane /
brief_dispatched) BEFORE any clone so the row cannot re-queue in a loop.
"""
import logging
from unittest import mock

from src import neo_builder
from src.config import Config

SHA = "f" * 40


def test_builder_never_takes_a_lane_branch():
    """One writer per lane branch (owner, 2026-10-25): full and short lane prefixes are skipped, others are not."""
    from src.neo_builder import is_lane_branch
    for lane in ("l8d-engine-mcps/engine-store", "l8b/mask-fix", "L1-reviews/apollo", "l4b-builds-app/a1"):
        assert is_lane_branch(lane), lane
    for other in ("feat/loop-watchdog", "main", "lane/reviewer-deepinfra-peak", "dependabot/npm/x", "fix-pr-12", ""):
        assert not is_lane_branch(other), other


def test_dispatch_one_skips_a_lane_branch_and_marks_the_row_first():
    """Test the row state, not just the log line: skipped_lane/brief_dispatched
    are written before any clone, and no worktree is ever prepared."""
    jobs = mock.MagicMock()
    row = {"id": 7, "repo": "acme/a", "pr_number": 5, "head_sha": SHA,
           "builder_round": 1, "attempt": 1, "worker_token": "w1", "payload": {}}
    with mock.patch.object(neo_builder, "repo_token", return_value="tok"), \
         mock.patch.object(neo_builder, "pr_state", return_value="OPEN\tl8b/mask-fix"), \
         mock.patch.object(neo_builder, "prepare_worktree") as prep:
        neo_builder.dispatch_one(Config(), jobs, row)
    prep.assert_not_called()
    jobs.resolve.assert_called_once()
    args, kw = jobs.resolve.call_args
    assert args[:2] == (7, "no_change")
    assert kw["worker_token"] == "w1"
    assert kw["payload"] == {"skipped_lane": "l8b/mask-fix", "brief_dispatched": True}


def test_dispatch_one_still_takes_an_ordinary_branch():
    """A non-lane branch keeps its normal path (the skip must be lane-only)."""
    jobs = mock.MagicMock()
    row = {"id": 8, "repo": "acme/a", "pr_number": 5, "head_sha": SHA,
           "builder_round": 1, "attempt": 1, "worker_token": "w1", "payload": {}}
    seen = {}

    def prep(token, repo, pr):
        seen["prepared"] = True
        raise RuntimeError("stop here — this test only proves the lane skip is lane-only")

    with mock.patch.object(neo_builder, "repo_token", return_value="tok"), \
         mock.patch.object(neo_builder, "pr_state", return_value="OPEN\tfeature/not-a-lane"), \
         mock.patch.object(neo_builder, "prepare_worktree", side_effect=prep):
        try:
            neo_builder.dispatch_one(Config(), jobs, row)
        except RuntimeError:
            pass
    assert seen.get("prepared"), "an ordinary branch must reach prepare_worktree"
    jobs.resolve.assert_not_called()


def test_dry_run_names_the_lane_skip(monkeypatch, caplog):
    """The dry run must name the skip, not just omit the row (8dc58c8)."""
    monkeypatch.setattr(Config, "require", lambda self: self)
    monkeypatch.setattr(Config, "validate", lambda self: self)
    monkeypatch.setattr(neo_builder, "github_app_for", lambda cfg: object())
    monkeypatch.setattr(neo_builder, "pr_state",
                        lambda token, repo, pr: "OPEN\tl8b/mask-fix" if pr == 5 else "OPEN\tother/topic")
    monkeypatch.setattr(neo_builder, "repo_token", lambda cfg, repo, iid: "tok")
    rows = [{"id": 1, "repo": "acme/a", "pr_number": 5, "head_sha": SHA,
             "builder_round": 1, "attempt": 1, "payload": {}},
            {"id": 2, "repo": "acme/a", "pr_number": 6, "head_sha": SHA,
             "builder_round": 1, "attempt": 1, "payload": {}}]
    jobs = mock.MagicMock()
    jobs.peek_queued.return_value = rows
    monkeypatch.setattr(neo_builder, "Jobs", lambda dsn: jobs)
    import sys
    monkeypatch.setattr(sys, "argv", ["neo_builder", "--dry-run", "--once"])
    with caplog.at_level(logging.INFO):
        assert neo_builder.main() == 0
    dry = [r.getMessage() for r in caplog.records if r.getMessage().startswith("dry-run:")]
    assert any("would SKIP (lane branch)" in m and "l8b/mask-fix" in m for m in dry), dry
    assert any("would dispatch" in m and "other/topic" in m for m in dry), dry

def test_lane_detection_parses_the_real_pr_state_shape():
    """Defect 8 (MEDIUM): the lane skip parsed pr_state's LAST tab field — a
    contract only the mocks asserted, so a format change silently disabled
    the skip and Builder started pushing to lane branches.  Parse with the
    documented delimiter + a length check, on the REAL return shape pr_state
    produces (captured through its own gh call — not a hand-rolled string)."""
    from src.neo_builder import parse_pr_state, is_lane_branch

    def fake_gh(token, *args, **kw):
        # Exactly what GitHub returns for pr_state's jq "[.state, .head.ref] | @tsv".
        return 0, "OPEN\tl8b/mask-fix\n"

    with mock.patch.object(neo_builder, "gh", side_effect=fake_gh):
        captured = neo_builder.pr_state("tok", "acme/a", 5)
    parsed = parse_pr_state(captured)
    assert parsed is not None, captured
    state, head_ref = parsed
    assert (state, head_ref) == ("OPEN", "l8b/mask-fix"), parsed
    assert is_lane_branch(head_ref)


def test_a_changed_pr_state_shape_never_silently_disables_the_lane_skip():
    """A pr_state line whose shape is no longer "<state> TAB <head ref>" is
    UNPARSEABLE: the job stays running (fail closed) and no worktree is ever
    prepared — Builder must never take a lane branch because the delimiter or
    the field count changed."""
    jobs = mock.MagicMock()
    row = {"id": 9, "repo": "acme/a", "pr_number": 5, "head_sha": SHA,
           "builder_round": 1, "attempt": 1, "worker_token": "w1", "payload": {}}
    for drifted in ("state=OPEN head=l8b/mask-fix",
                    "OPEN\tl8b/mask-fix\textra"):
        with mock.patch.object(neo_builder, "repo_token", return_value="tok"), \
             mock.patch.object(neo_builder, "pr_state", return_value=drifted), \
             mock.patch.object(neo_builder, "prepare_worktree") as prep:
            neo_builder.dispatch_one(Config(), jobs, row)
        prep.assert_not_called()
        jobs.resolve.assert_not_called()   # stays running — never a silent dispatch


def test_builder_and_worker_share_the_lane_classifier():
    from src import neo_worker
    assert neo_builder.lane_of is neo_worker.lane_of
    assert neo_builder.LANE_BRANCH is neo_worker.LANE_BRANCH


def test_main_claim_loop_skips_lane_before_submit(monkeypatch):
    """The main claim loop applies the same skip before dispatch_one reaches a pool."""
    monkeypatch.setattr(Config, "require", lambda self: self)
    monkeypatch.setattr(Config, "validate", lambda self: self)
    monkeypatch.setattr(neo_builder, "github_app_for", lambda cfg: object())
    monkeypatch.setattr(neo_builder, "repo_token", lambda cfg, repo, iid: "tok")
    monkeypatch.setattr(neo_builder, "pr_state", lambda token, repo, pr: "OPEN\tl8b/mask-fix")
    row = {"id": 10, "repo": "acme/a", "pr_number": 5, "head_sha": SHA,
           "builder_round": 1, "attempt": 1, "worker_token": "w1", "payload": {}}
    jobs = mock.MagicMock()
    jobs.claim_queued.side_effect = [row, None]
    jobs.recover_expired.return_value = []
    jobs.supersede_older.return_value = []
    monkeypatch.setattr(neo_builder, "Jobs", lambda dsn: jobs)
    submit = mock.Mock(side_effect=AssertionError("lane row reached pool submit"))
    monkeypatch.setattr(neo_builder.concurrent.futures.ThreadPoolExecutor, "submit", submit)
    import sys
    monkeypatch.setattr(sys, "argv", ["neo_builder", "--once"])
    assert neo_builder.main() == 0
    submit.assert_not_called()
    jobs.resolve.assert_called_once_with(
        10, "no_change", worker_token="w1",
        payload={"skipped_lane": "l8b/mask-fix", "lane_skip": True,
                 "brief_dispatched": False})
