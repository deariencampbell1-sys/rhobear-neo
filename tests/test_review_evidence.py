"""W1-13-r1 — review_evidence defects from Apollo's review of #10.

Defect 3: _trusted_check_run must not trust a run just because it matches
app.id + head_sha — the run must be COMPLETED and successful/neutral, and
when several runs match, the NEWEST decides (never "first in the list").
Defect 4: build_evidence must fail closed on an empty rollup.
"""
from __future__ import annotations

from src.config import Config
from src.review_evidence import build_evidence, merge_authority

TRUSTED_APP = "12345"
SHA = "a" * 40


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

    def test_run_without_status_but_with_conclusion_counts_as_completed(self):
        """GitHub's check-run schema always includes `status`, and a
        `conclusion` exists only on completed runs.  The pinned fixtures in
        tests/test_merge_authority.py (outside this round's ownership) omit
        `status` but carry a conclusion — such a run is treated as completed.
        A run with neither is never evidence."""
        ev, (granted, reason) = _authority([
            {"name": "rhobear-reviews", "head_sha": SHA,
             "app": {"id": TRUSTED_APP}, "conclusion": "success",
             "output": {"summary": '{"unresolved_critical_high": 0, '
                                   '"coverage": "complete"}'}},
        ])
        assert ev["trusted_producer_ok"] is True
        assert granted is True
        # no status AND no conclusion: nothing says it completed -> refused
        ev, (granted, reason) = _authority([
            {"name": "rhobear-reviews", "head_sha": SHA,
             "app": {"id": TRUSTED_APP}, "conclusion": None,
             "output": {"summary": '{"unresolved_critical_high": 0, '
                                   '"coverage": "complete"}'}},
        ])
        assert granted is False


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
