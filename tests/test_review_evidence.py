"""W1-13-r1 — review_evidence defects from Apollo's review of #10.

Defect 3: _trusted_check_run must not trust a run just because it matches
app.id + head_sha — the run must be COMPLETED and successful/neutral, and
when several runs match, the NEWEST decides (never "first in the list").
Defect 4: build_evidence must fail closed on an empty rollup.
"""
from __future__ import annotations

import json
from unittest import mock

import pytest

from src.config import Config
from src.review_evidence import build_evidence, merge_authority

TRUSTED_APP = "12345"
SHA = "a" * 40


@pytest.fixture(autouse=True)
def _owner_install_id_env_cleared(monkeypatch):
    """W1-13-r2 defect 9: `_cfg()` builds a bare Config() — nothing in this
    module may depend on deployment env.  NEO_OWNER_INSTALL_ID is deployment
    state, not test input."""
    monkeypatch.delenv("NEO_OWNER_INSTALL_ID", raising=False)


def _cfg() -> Config:
    c = Config()
    c.trusted_reviewer_app_id = TRUSTED_APP
    return c


def _run(sha=SHA, *, app_id=TRUSTED_APP, name="rhobear-reviews",
         status="completed", conclusion="success",
         summary='{"unresolved_critical_high": 0, "coverage": "complete"}',
         completed_at=None, run_id=None) -> dict:
    """One check run as returned by GET /repos/{repo}/commits/{sha}/check-runs."""
    cr = {
        "name": name,
        "head_sha": sha,
        "app": {"id": app_id, "slug": "rhobear-reviews"},
        "status": status,
        "conclusion": conclusion,
        "output": {"summary": summary},
    }
    if completed_at is not None:
        cr["completed_at"] = completed_at
    if run_id is not None:
        cr["id"] = run_id
    return cr


def _green_rollup():
    return [{"context": "ci/test", "conclusion": "SUCCESS"}]


def _pr_data(check_runs, rollup=None) -> dict:
    return {
        "headRefOid": SHA,
        "reviews": [],
        "statusCheckRollup": _green_rollup() if rollup is None else rollup,
        "check_runs": check_runs,
        "installation_auto_merge": True,
    }


def _authority(check_runs, rollup=None):
    ev = build_evidence(_cfg(), _pr_data(check_runs, rollup=rollup), SHA)
    return ev, merge_authority(ev)


# ---------------------------------------------------------------------------
# Defect 3 — the trusted check run is trusted too cheaply
# ---------------------------------------------------------------------------

class TestTrustedCheckRun:
    def test_failing_trusted_run_is_not_evidence(self):
        """A COMPLETED run with conclusion=failure is not merge evidence."""
        ev, (granted, reason) = _authority([
            _run(status="completed", conclusion="failure"),
        ])
        assert ev["trusted_producer_ok"] is False
        assert granted is False

    def test_pending_trusted_run_is_not_evidence(self):
        """A run that has not completed (no conclusion yet) is not evidence."""
        ev, (granted, reason) = _authority([
            _run(status="in_progress", conclusion=None),
        ])
        assert ev["trusted_producer_ok"] is False
        assert granted is False

    def test_queued_trusted_run_is_not_evidence(self):
        ev, (granted, reason) = _authority([
            _run(status="queued", conclusion=None),
        ])
        assert granted is False

    def test_timed_out_and_cancelled_runs_are_not_evidence(self):
        for conclusion in ("timed_out", "cancelled", "action_required"):
            ev, (granted, reason) = _authority([
                _run(status="completed", conclusion=conclusion),
            ])
            assert granted is False, conclusion

    def test_two_matching_runs_the_newest_decides_and_it_is_failing(self):
        """Older green, newer failing on the same head: the NEWER one decides —
        a denial.  Never fall back to the older run."""
        ev, (granted, reason) = _authority([
            _run(status="completed", conclusion="success",
                 completed_at="2026-10-25T20:00:00Z", run_id=1),
            _run(status="completed", conclusion="failure",
                 completed_at="2026-10-25T22:00:00Z", run_id=2),
        ])
        assert granted is False

    def test_two_matching_green_runs_the_newest_summary_decides(self):
        """Two completed green runs: the NEWEST by completed_at (then id) is
        the evidence — never the first in the list."""
        old = _run(summary='{"unresolved_critical_high": 0, "coverage": "complete"}',
                   completed_at="2026-10-25T20:00:00Z", run_id=1)
        new = _run(summary='{"unresolved_critical_high": 2, "coverage": "complete"}',
                   completed_at="2026-10-25T22:00:00Z", run_id=2)
        # listed OLDEST FIRST: the old "first match" rule would accept this
        ev, (granted, reason) = _authority([old, new])
        assert ev["unresolved_critical_high"] == 2
        assert granted is False
        # and reversed listing order changes nothing
        ev, (granted, reason) = _authority([new, old])
        assert ev["unresolved_critical_high"] == 2
        assert granted is False

    def test_completed_neutral_is_evidence(self):
        ev, (granted, reason) = _authority([
            _run(status="completed", conclusion="neutral"),
        ])
        assert ev["trusted_producer_ok"] is True
        assert granted is True

    def test_completed_success_still_grants(self):
        ev, (granted, reason) = _authority([_run()])
        assert ev["trusted_producer_ok"] is True
        assert granted is True

    def test_run_without_status_but_with_conclusion_is_not_completed(self):
        """W1-13-r2 defect 4: the fixtures were wrong, not the predicate.  A
        check-run payload with a `conclusion` but NO `status` is NOT completed
        evidence — GitHub's check-run schema always carries `status`, and the
        production predicate is `status == "completed"`, full stop.  (r1 bent
        the predicate to keep pinned fixtures green; the fixtures now carry
        the real shape instead.)"""
        ev, (granted, reason) = _authority([
            {"name": "rhobear-reviews", "head_sha": SHA,
             "app": {"id": TRUSTED_APP}, "conclusion": "success",
             "output": {"summary": '{"unresolved_critical_high": 0, '
                                   '"coverage": "complete"}'}},
        ])
        assert ev["trusted_producer_ok"] is False
        assert granted is False
        # no status AND no conclusion: nothing says it completed -> refused
        ev, (granted, reason) = _authority([
            {"name": "rhobear-reviews", "head_sha": SHA,
             "app": {"id": TRUSTED_APP}, "conclusion": None,
             "output": {"summary": '{"unresolved_critical_high": 0, '
                                   '"coverage": "complete"}'}},
        ])
        assert granted is False
        # the real shape (status present) keeps deciding normally
        ev, (granted, reason) = _authority([_run(status="completed")])
        assert ev["trusted_producer_ok"] is True
        assert granted is True


# ---------------------------------------------------------------------------
# Defect 4 — build_evidence fails open on an empty rollup
# ---------------------------------------------------------------------------

class TestEmptyRollupFailsClosed:
    def test_empty_rollup_is_unknown_not_pass(self):
        """No rollup data at all is UNKNOWN (None — matching merge_authority's
        treatment of unknown findings), never a pass.  An evidence bundle with
        an empty rollup is denied even when everything else is green."""
        ev, (granted, reason) = _authority([_run()], rollup=[])
        assert ev["required_checks_pass"] is None
        assert granted is False
        assert "check" in reason.lower()

    def test_one_failing_check_is_false(self):
        ev, (granted, reason) = _authority(
            [_run()],
            rollup=[{"context": "ci/test", "conclusion": "FAILURE"}],
        )
        assert ev["required_checks_pass"] is False
        assert granted is False

    def test_all_green_is_true(self):
        ev, (granted, reason) = _authority([_run()])
        assert ev["required_checks_pass"] is True
        assert granted is True

    def test_missing_output_summary_is_a_denial_not_a_pass(self):
        cr = _run(summary="")
        cr["output"] = {}
        ev, (granted, reason) = _authority([cr])
        assert ev["unresolved_critical_high"] is None
        assert granted is False

    def test_summary_that_is_not_valid_json_is_a_denial_not_a_pass(self):
        ev, (granted, reason) = _authority([
            _run(summary="not json {"),
        ])
        assert ev["unresolved_critical_high"] is None
        assert ev["coverage_complete"] is False
        assert granted is False

    def test_summary_that_is_not_a_json_object_is_a_denial(self):
        ev, (granted, reason) = _authority([
            _run(summary='["unresolved_critical_high", 0]'),
        ])
        assert ev["unresolved_critical_high"] is None
        assert granted is False


# ---------------------------------------------------------------------------
# W1-13-r2 defect 10 — run selection must deny, never raise, and order real
# instants
# ---------------------------------------------------------------------------

class TestTrustedRunSelectionIsRobust:
    def test_malformed_id_denies_rather_than_raising(self):
        """A non-numeric run `id` must never turn the evidence path into a
        ValueError: the newest (failing) run still decides and the answer is a
        denial."""
        ev, (granted, reason) = _authority([
            _run(status="completed", conclusion="success",
                 completed_at="2026-10-25T20:00:00Z", run_id=1),
            _run(status="completed", conclusion="failure",
                 completed_at="2026-10-25T22:00:00Z", run_id="not-a-number"),
        ])
        assert granted is False

    def test_z_and_offset_timestamps_order_by_instant(self):
        """Mixed `Z` / `+00:00` completed_at values order by INSTANT (then
        id) — string comparison sees '+' < 'Z' and calls the +00:00 run older
        than a Z run at the same instant, picking the wrong newest."""
        old = _run(summary='{"unresolved_critical_high": 0, "coverage": "complete"}',
                   completed_at="2026-10-25T22:00:00Z", run_id=1)
        new = _run(summary='{"unresolved_critical_high": 2, "coverage": "complete"}',
                   completed_at="2026-10-25T22:00:00+00:00", run_id=2)
        for listing in ([old, new], [new, old]):
            ev, (granted, reason) = _authority(listing)
            assert ev["unresolved_critical_high"] == 2, listing
            assert granted is False


def test_trusted_reviewer_outside_org_counts():
    """W1-07-r1: a trusted reviewer OUTSIDE the org counts as green 2 (board
    `trusted`: "Zeus ACCEPT, or a trusted reviewer below — either counts").
    Before the two-green rule a green from outside the org did not count at
    all. A green from an account on no trusted list still counts for nothing."""
    import dataclasses
    import json
    from unittest import mock as _mock
    from src import neo_worker

    sha = SHA
    statuses = [{"context": "CodeAnt", "state": "success",
                 "creator": {"login": "codeant-bot", "type": "Bot"},
                 "created_at": "2026-10-26T11:50:00Z",
                 "target_url": "https://ci.example/CodeAnt"}]

    def fake_gh(token, *args, timeout=30):
        if args[:2] == ("pr", "view"):
            return 0, json.dumps({
                "headRefOid": sha, "headRefName": "topic",
                "reviews": [{"state": "APPROVED", "commit": {"oid": sha},
                             "author": {"login": "rhobear-reviews[bot]", "type": "Bot"},
                             "submitted_at": "2026-10-26T11:52:00Z",
                             "body": "verdict: ACCEPT"}],
                "statusCheckRollup": []})
        if args[0] == "api" and "/check-runs" in args[1]:
            return 0, json.dumps({"total_count": 0, "check_runs": []})
        if args[0] == "api" and args[1].endswith("/status"):
            return 0, json.dumps({"total_count": len(statuses), "statuses": statuses})
        if args[0] == "api" and any("/pulls/5/reviews" in a for a in args):
            return 0, json.dumps([[{"user": {"login": "rhobear-reviews[bot]", "type": "Bot"}}]])
        raise AssertionError(f"unexpected gh call {args}")

    cfg = dataclasses.replace(_cfg(), trusted_reviewer_app_id=TRUSTED_APP)
    with _mock.patch.object(neo_worker, "_gh", side_effect=fake_gh):
        ev = neo_worker._review_evidence("tok", cfg, "acme/checkout", 5, sha,
                                         enabled_sources=("CodeAnt",))
    assert ev["ok"] is True
    assert ev["green1_ok"] is True, ev
    assert ev["green2_ok"] is True and ev["green2_kind"] == "trusted", ev

    with _mock.patch.object(neo_worker, "_gh", side_effect=fake_gh):
        ev = neo_worker._review_evidence("tok", cfg, "acme/checkout", 5, sha,
                                         enabled_sources=())
    assert ev["green2_ok"] is False, ev

# --- W1-07-r5: green 1 Bot identity comes from real GitHub output --------
CAPTURED_PR_VIEW_REVIEW = {
    "id": "PRR_kwDOTtWIb88AAAABUYbtcA",
    "author": {"login": "rhobear-reviews"},
    "authorAssociation": "NONE",
    "submittedAt": "2026-10-27T05:47:36Z",
    "state": "CHANGES_REQUESTED",
    "commit": {"oid": "938f3ddfa9c5a1025bbd9ee4c7e3c455e47a411d"},
}
CAPTURED_REST_REVIEW_ROW = {
    "id": 5662764400,
    "node_id": "PRR_kwDOTtWIb88AAAABUYbtcA",
    "user": {"login": "rhobear-reviews[bot]", "type": "Bot"},
    "state": "CHANGES_REQUESTED",
    "commit_id": "938f3ddfa9c5a1025bbd9ee4c7e3c455e47a411d",
    "submitted_at": "2026-10-27T05:47:36Z",
    "html_url": ("https://github.com/deariencampbell1-sys/rhobear-neo/pull/13"
                 "#pullrequestreview-5662764400"),
}
SHA1 = "a" * 40

def _captured_accept():
    rec = json.loads(json.dumps(CAPTURED_PR_VIEW_REVIEW))
    rec["state"] = "APPROVED"
    rec["commit"]["oid"] = SHA1
    return rec

def _ev_fake_gh(pr_view_reviews, rest_rows, rest_rc=0):
    def fake_gh(token, *args, timeout=30):
        if args[:2] == ("pr", "view"):
            return 0, json.dumps({"headRefOid": SHA1, "headRefName": "topic",
                                  "reviews": pr_view_reviews, "statusCheckRollup": []})
        if args[0] == "api" and "/check-runs" in args[1]:
            return 0, json.dumps({"total_count": 0, "check_runs": []})
        if args[0] == "api" and args[1].endswith("/status"):
            return 0, json.dumps({"total_count": 0, "statuses": []})
        if args[0] == "api" and any("/pulls/5/reviews" in a for a in args):
            if rest_rc != 0:
                return rest_rc, "gh: HTTP 503"
            return 0, json.dumps([rest_rows])
        raise AssertionError(f"unexpected gh call {args}")
    return fake_gh

def test_captured_real_gh_review_json_has_no_author_type():
    assert CAPTURED_PR_VIEW_REVIEW["author"] == {"login": "rhobear-reviews"}
    assert "type" not in CAPTURED_PR_VIEW_REVIEW["author"]
    assert CAPTURED_REST_REVIEW_ROW["user"]["type"] == "Bot"

def test_rest_reviews_annotate_the_bot_type_for_green_one():
    from src import neo_worker
    with mock.patch.object(neo_worker, "_gh", _ev_fake_gh([_captured_accept()], [CAPTURED_REST_REVIEW_ROW])):
        ev = neo_worker._review_evidence("tok", Config(), "acme/a", 5, SHA1)
    assert ev["reviews"][0]["author"]["type"] == "Bot"
    assert ev["green1_ok"] is True
    assert ev["green1_reason"] == ""

def test_a_failed_rest_reviews_read_fails_green_one_closed():
    from src import neo_worker
    from src.jev_review import JevAssessment
    with mock.patch.object(neo_worker, "_gh", _ev_fake_gh([_captured_accept()], [CAPTURED_REST_REVIEW_ROW], rest_rc=1)):
        ev = neo_worker._review_evidence("tok", _cfg(), "acme/a", 5, SHA1)
    assert ev["reviews"][0]["author"]["type"] == "User"
    assert ev["green1_ok"] is False
    bundle = {**ev, "trusted_producer_ok": True, "review_sha_match": True,
              "required_checks_pass": True, "unresolved_critical_high": 0,
              "coverage_complete": True, "auto_merge": True,
              "review_bound_ok": True, "trusted_green": True,
              "trusted_states": ["success"], "failing_checks": 0}
    ok, reason = neo_worker.merge_authority(True, bundle, JevAssessment(False, reason="no key"))
    assert ok is False and "green 1" in reason, reason

def test_a_user_account_reviewing_as_apollo_is_not_green_one():
    from src import neo_worker
    row = dict(CAPTURED_REST_REVIEW_ROW, user={"login": "rhobear-reviews", "type": "User"})
    with mock.patch.object(neo_worker, "_gh", _ev_fake_gh([_captured_accept()], [row])):
        ev = neo_worker._review_evidence("tok", Config(), "acme/a", 5, SHA1)
    assert ev["reviews"][0]["author"]["type"] == "User"
    assert ev["green1_ok"] is False


def test_captured_graphql_review_uses_rest_link_and_timestamp():
    from src import neo_worker
    with mock.patch.object(
            neo_worker, "_gh",
            _ev_fake_gh([_captured_accept()], [CAPTURED_REST_REVIEW_ROW])):
        ev = neo_worker._review_evidence("tok", Config(), "acme/a", 5, SHA1)
    assert ev["green1_ok"] is True
    assert ev["green1_url"] == CAPTURED_REST_REVIEW_ROW["html_url"]
    assert ev["green1_at"] == CAPTURED_PR_VIEW_REVIEW["submittedAt"]


@pytest.mark.parametrize("reverse", [False, True])
def test_bare_user_collision_cannot_overwrite_exact_bot_identity(reverse, caplog):
    from src import neo_worker
    bot = CAPTURED_REST_REVIEW_ROW
    user = dict(CAPTURED_REST_REVIEW_ROW,
                node_id="PRR_user_collision",
                user={"login": "rhobear-reviews", "type": "User"})
    rows = [user, bot] if reverse else [bot, user]
    with caplog.at_level("WARNING", logger="rhobear_neo.worker"):
        with mock.patch.object(neo_worker, "_gh", _ev_fake_gh([_captured_accept()], rows)):
            ev = neo_worker._review_evidence("tok", Config(), "acme/a", 5, SHA1)
    assert ev["reviews"][0]["author"]["type"] == "Bot"
    assert ev["green1_ok"] is True
    assert "ambiguous" in caplog.text.lower()


def test_malformed_review_author_returns_failed_evidence():
    from src import neo_worker
    malformed = _captured_accept()
    malformed["author"] = "not-a-dict"
    with mock.patch.object(
            neo_worker, "_gh",
            _ev_fake_gh([malformed], [CAPTURED_REST_REVIEW_ROW])):
        ev = neo_worker._review_evidence("tok", Config(), "acme/a", 5, SHA1)
    assert ev["ok"] is False
    assert ev["reason"] == "green evaluation failed"
    assert ev["decided_sha"] == SHA1 and ev["head_ref"] == "topic"


def test_check_run_green_comes_from_unfiltered_check_runs_api():
    from src import neo_worker
    calls = []
    check = {
        "name": "CodeAnt", "head_sha": SHA1, "status": "completed",
        "conclusion": "success", "app": {"id": 151821869, "slug": "codeant-ai"},
        "html_url": "https://github.com/acme/a/runs/1",
        "completed_at": "2026-10-27T05:48:00Z",
    }

    def fake_gh(token, *args, timeout=30):
        calls.append(args)
        if args[:2] == ("pr", "view"):
            return 0, json.dumps({
                "headRefOid": SHA1, "headRefName": "topic",
                "reviews": [_captured_accept()],
                "statusCheckRollup": [{"__typename": "CheckRun", "name": "CodeAnt",
                                       "status": "COMPLETED", "conclusion": "SUCCESS"}],
            })
        if args[0] == "api" and "/check-runs" in args[1]:
            return 0, json.dumps({"total_count": 1, "check_runs": [check]})
        if args[0] == "api" and args[1].endswith("/status"):
            return 0, json.dumps({"total_count": 0, "statuses": []})
        if args[0] == "api" and any("/pulls/5/reviews" in a for a in args):
            return 0, json.dumps([[CAPTURED_REST_REVIEW_ROW]])
        raise AssertionError(f"unexpected gh call {args}")

    with mock.patch.object(neo_worker, "_gh", fake_gh):
        ev = neo_worker._review_evidence(
            "tok", _cfg(), "acme/a", 5, SHA1, enabled_sources=("CodeAnt",))
    assert ev["green2_ok"] is True and ev["green2_kind"] == "trusted"
    check_calls = [c for c in calls if c[0] == "api" and "/check-runs" in c[1]]
    assert check_calls and all("app_id=" not in c[1] for c in check_calls)
