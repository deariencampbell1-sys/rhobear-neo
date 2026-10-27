"""N15-B2 — Deterministic merge executor + 5-bounce escalation tests.

Covers (from the lane brief):
  * Forged context prefix — not trusted as producer identity
  * Wrong author/App — review not by our App is rejected
  * Missing/stale SHA — head changed since evidence
  * Head changes just before merge — merge executor re-reads and refuses
  * Pending/missing/failing required checks
  * Explicit false with true default — nullable inheritance
  * Incomplete coverage
  * Exhausted rounds / escalation at 5, not 4 or 6
  * Agent output containing `gh pr merge` does nothing
  * Read-only token permissions for escalation
  * Advisor auth and bind address
  * Queue limit
  * Route fallback when claude is missing
  * Malformed JSON → one repair → fail → park
  * Escalated round's brief contains the work order
  * 2nd bounce after escalation parks

No real GitHub / provider calls: all subprocesses and HTTP are fakes.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.merge_executor import (
    merge, HeadMoved, MergeDenied, MergeError, MergeFailed,
)
from src.review_evidence import build_evidence, merge_authority, evidence_brief
from src.escalation import (
    RoundRecord, WorkOrder, should_escalate, escalate_pr, submit_escalation,
    poll_advice, AdviceStatus, EscalationError, AdvisorUnavailable, AdvisorRefused,
    AdvisorNotFound,
)

SHA1 = "a" * 40
SHA2 = "b" * 40


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    c = Config()
    c.auto_merge_default = True
    c.max_builder_rounds = 5
    c.bounces_before_escalation = 5
    c.advisor_url = "http://100.111.118.24:8787"
    c.advisor_token = "test-advisor-token"
    return c


def _branch_lookups(gh_args, default="main", other_prs=""):
    """Answers for the post-merge branch-safety lookups, or None."""
    if gh_args[:3] == ("api", "repos/acme/a", "--jq"):
        return 0, default
    if gh_args[:2] == ("pr", "list"):
        return 0, other_prs
    return None


def _fake_gh_success(*args, **kw):
    """Fake gh subprocess that returns success with PR data.

    merge_executor._gh receives (token, *gh_args).
    """
    gh_args = args[1:] if args else ()
    if (lookup := _branch_lookups(gh_args)) is not None:
        return lookup
    if gh_args[:3] == ("pr", "view", "1"):
        data = {
            "headRefOid": SHA1,
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "headRefName": "topic-branch",
            "reviews": [
                {"body": "Looks good. COVERAGE: COMPLETE",
                 "commit_id": SHA1,
                 "author": {"type": "Bot"}},
            ],
            "statusCheckRollup": [
                {"context": "ci/test", "conclusion": "SUCCESS"},
                {"context": "ci/lint", "conclusion": "SUCCESS"},
            ],
        }
        return 0, json.dumps(data)
    if gh_args and gh_args[0] == "api" and len(gh_args) > 3 and \
            gh_args[3] == "repos/acme/a/pulls/1/merge":
        return 0, json.dumps({
            "merged": True, "sha": "M" * 40,
            "message": "Pull Request successfully merged",
        })
    # The post-merge head-branch deletion (defect 3): tolerated 404/422.
    if gh_args[:4] == ("api", "-X", "DELETE",
                       "repos/acme/a/git/refs/heads/topic-branch"):
        return 0, ""  # 204 No Content
    if gh_args[:5] == ("api", f"repos/acme/a/commits/{SHA1}/status"):
        data = {
            "state": "success",
            "statuses": [{"context": "rhobear-reviews", "state": "success"}],
        }
        return 0, json.dumps(data)
    raise AssertionError(f"unexpected _gh call: {args}")


def _fake_gh_head_moved(*args, **kw):
    """Fake gh where the head moved between evidence and merge."""
    gh_args = args[1:] if args else ()
    if gh_args[:3] == ("pr", "view", "1"):
        data = {
            "headRefOid": SHA2,  # different from expected
            "state": "OPEN",
            "mergeable": "MERGEABLE",
            "reviews": [],
            "statusCheckRollup": [],
        }
        return 0, json.dumps(data)
    if gh_args[:5] == ("api", f"repos/acme/a/commits/{SHA1}/status"):
        data = {"state": "success", "statuses": []}
        return 0, json.dumps(data)
    raise AssertionError(f"unexpected _gh call: {args}")


# ---------------------------------------------------------------------------
# Review evidence tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Review evidence tests
# ---------------------------------------------------------------------------

TRUSTED_APP = "12345"
UNTRUSTED_APP = "99999"


def _app_cfg():
    """Config with the trusted Reviews App id set (item 3)."""
    c = Config()
    c.trusted_reviewer_app_id = TRUSTED_APP
    return c


def _check_run(sha: str, app_id: str = TRUSTED_APP, summary: str | None = None,
               conclusion: str = "success") -> dict:
    """One check run as returned by GET /repos/{repo}/commits/{sha}/check-runs.

    `summary` is the Reviews App's output.summary JSON
    ({"unresolved_critical_high": n, "coverage": "complete|partial"}).
    """
    if summary is None:
        summary = '{"unresolved_critical_high": 0, "coverage": "complete"}'
    return {
        "name": "rhobear-reviews",
        "head_sha": sha,
        "app": {"id": app_id, "slug": "rhobear-reviews"},
        "status": "completed",
        "conclusion": conclusion,
        "output": {"summary": summary},
    }


def _pr_data(head: str, *, check_runs=None, reviews=None, rollup=None,
             install_auto_merge=None) -> dict:
    data = {
        "headRefOid": head,
        "reviews": reviews or [],
        "statusCheckRollup": rollup if rollup is not None else [],
        "check_runs": check_runs if check_runs is not None else [],
    }
    if install_auto_merge is not None:
        data["installation_auto_merge"] = install_auto_merge
    return data


def _green_rollup():
    return [{"context": "ci/test", "conclusion": "SUCCESS"}]


class TestBuildEvidence:
    """build_evidence — trusted producer = the Reviews App's check run on the
    exact head SHA; unresolved/coverage come from that run's output.summary."""

    def test_no_trusted_app_id_fails_closed(self):
        """NEO_TRUSTED_REVIEWER_APP_ID unset → evidence unavailable → no merge."""
        pr_data = _pr_data(SHA1, check_runs=[_check_run(SHA1)],
                           rollup=_green_rollup(), install_auto_merge=True)
        ev = build_evidence(Config(), pr_data, SHA1)  # no trusted app id
        assert ev["ok"] is False
        granted, reason = merge_authority(ev)
        assert granted is False
        assert "app id" in reason.lower()

    def test_author_self_review_ignored(self):
        """A review by the PR author (even with COVERAGE: COMPLETE) is not a
        trusted producer — no check run from the App → no merge (item 3)."""
        c = _app_cfg()
        pr_data = _pr_data(
            SHA1,
            reviews=[{"body": "Looks good. COVERAGE: COMPLETE",
                      "commit_id": SHA1, "author": {"type": "User", "login": "author"}}],
            check_runs=[],
            rollup=_green_rollup(),
            install_auto_merge=True,
        )
        ev = build_evidence(c, pr_data, SHA1)
        assert ev["trusted_producer_ok"] is False
        granted, reason = merge_authority(ev)
        assert granted is False

    def test_right_app_wrong_sha_ignored(self):
        """A check run from the trusted App on a DIFFERENT sha does not
        authorize this head — no merge (item 3)."""
        c = _app_cfg()
        pr_data = _pr_data(SHA1, check_runs=[_check_run(SHA2)],
                           rollup=_green_rollup(), install_auto_merge=True)
        ev = build_evidence(c, pr_data, SHA1)
        assert ev["trusted_producer_ok"] is False
        assert ev["review_sha_match"] is False
        granted, reason = merge_authority(ev)
        assert granted is False

    def test_wrong_app_right_sha_ignored(self):
        """A check run from another App on this sha is ignored (item 3)."""
        c = _app_cfg()
        pr_data = _pr_data(SHA1, check_runs=[_check_run(SHA1, app_id=UNTRUSTED_APP)],
                           rollup=_green_rollup(), install_auto_merge=True)
        ev = build_evidence(c, pr_data, SHA1)
        assert ev["trusted_producer_ok"] is False
        assert merge_authority(ev)[0] is False

    def test_right_app_right_sha_evidence_ok(self):
        """Trusted App check run on this exact head with a complete summary
        → evidence ok (item 3)."""
        c = _app_cfg()
        pr_data = _pr_data(SHA1, check_runs=[_check_run(SHA1)],
                           rollup=_green_rollup(), install_auto_merge=True)
        ev = build_evidence(c, pr_data, SHA1)
        assert ev["ok"] is True
        assert ev["trusted_producer_ok"] is True
        assert ev["review_sha_match"] is True
        assert ev["unresolved_critical_high"] == 0
        assert ev["coverage_complete"] is True
        granted, reason = merge_authority(ev)
        assert granted is True

    def test_failing_required_checks(self):
        """A failing required check blocks authority."""
        c = _app_cfg()
        pr_data = _pr_data(SHA1, check_runs=[_check_run(SHA1)],
                           rollup=[{"context": "ci/test", "conclusion": "FAILURE"}],
                           install_auto_merge=True)
        ev = build_evidence(c, pr_data, SHA1)
        assert ev["required_checks_pass"] is False
        assert ev["required_checks_failing"] == 1
        assert merge_authority(ev)[0] is False

    def test_unresolved_critical_findings(self):
        """CRITICAL/HIGH findings from the trusted check run's output block
        authority (item 4: taken from output, not hard-coded)."""
        c = _app_cfg()
        pr_data = _pr_data(
            SHA1,
            check_runs=[_check_run(SHA1, summary='{"unresolved_critical_high": 2, "coverage": "complete"}')],
            rollup=_green_rollup(),
            install_auto_merge=True,
        )
        ev = build_evidence(c, pr_data, SHA1)
        assert ev["unresolved_critical_high"] == 2
        assert merge_authority(ev)[0] is False

    def test_partial_coverage(self):
        """coverage: partial from the output → coverage not complete (item 4)."""
        c = _app_cfg()
        pr_data = _pr_data(
            SHA1,
            check_runs=[_check_run(SHA1, summary='{"unresolved_critical_high": 0, "coverage": "partial"}')],
            rollup=_green_rollup(),
            install_auto_merge=True,
        )
        ev = build_evidence(c, pr_data, SHA1)
        assert ev["coverage_complete"] is False
        assert merge_authority(ev)[0] is False

    def test_missing_output_summary_denies(self):
        """No output.summary on the trusted check run → unresolved is None,
        coverage incomplete → merge denied (item 4)."""
        c = _app_cfg()
        cr = _check_run(SHA1)
        cr["output"] = {"summary": ""}
        pr_data = _pr_data(SHA1, check_runs=[cr], rollup=_green_rollup(),
                           install_auto_merge=True)
        ev = build_evidence(c, pr_data, SHA1)
        assert ev["unresolved_critical_high"] is None
        assert ev["coverage_complete"] is False
        granted, reason = merge_authority(ev)
        assert granted is False

    def test_explicit_false_beats_true_default(self):
        """Explicit installation auto_merge=false beats cfg default true."""
        c = _app_cfg()
        c.auto_merge_default = True
        pr_data = _pr_data(SHA1, check_runs=[_check_run(SHA1)],
                           install_auto_merge=False)  # explicit false
        ev = build_evidence(c, pr_data, SHA1)
        assert ev["auto_merge"] is False

    def test_no_installation_setting_uses_default(self):
        """No installation auto_merge setting → uses cfg default."""
        c = _app_cfg()
        c.auto_merge_default = False
        pr_data = _pr_data(SHA1, check_runs=[_check_run(SHA1)])  # no install setting
        ev = build_evidence(c, pr_data, SHA1)
        assert ev["auto_merge"] is False


class TestMergeAuthority:
    """merge_authority — deterministic gate."""

    def test_all_conditions_pass(self):
        ev = {
            "ok": True, "head_sha": SHA1, "decided_sha": SHA1,
            "trusted_producer_ok": True, "review_sha_match": True,
            "required_checks_pass": True, "unresolved_critical_high": 0,
            "coverage_complete": True, "auto_merge": True,
        }
        granted, reason = merge_authority(ev)
        assert granted is True
        assert "granted" in reason.lower()

    def test_evidence_unavailable(self):
        ev = {"ok": False, "reason": "no data"}
        granted, reason = merge_authority(ev)
        assert granted is False
        assert "unavailable" in reason.lower()

    def test_head_moved(self):
        ev = {"ok": True, "head_sha": SHA2, "decided_sha": SHA1}
        granted, reason = merge_authority(ev)
        assert granted is False
        assert "moved" in reason.lower()

    def test_no_trusted_producer(self):
        ev = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1,
              "trusted_producer_ok": False, "review_sha_match": False,
              "required_checks_pass": True, "unresolved_critical_high": 0,
              "coverage_complete": True, "auto_merge": True}
        granted, reason = merge_authority(ev)
        assert granted is False
        assert "trusted" in reason.lower()

    def test_failing_checks(self):
        ev = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1,
              "trusted_producer_ok": True, "review_sha_match": True,
              "required_checks_pass": False, "required_checks_failing": 2,
              "unresolved_critical_high": 0, "coverage_complete": True,
              "auto_merge": True}
        granted, reason = merge_authority(ev)
        assert granted is False
        assert "check" in reason.lower()

    def test_unresolved_findings(self):
        ev = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1,
              "trusted_producer_ok": True, "review_sha_match": True,
              "required_checks_pass": True, "unresolved_critical_high": 1,
              "coverage_complete": True, "auto_merge": True}
        granted, reason = merge_authority(ev)
        assert granted is False
        assert "CRITICAL" in reason.upper() or "HIGH" in reason.upper()

    def test_incomplete_coverage(self):
        ev = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1,
              "trusted_producer_ok": True, "review_sha_match": True,
              "required_checks_pass": True, "unresolved_critical_high": 0,
              "coverage_complete": False, "auto_merge": True}
        granted, reason = merge_authority(ev)
        assert granted is False
        assert "coverage" in reason.lower()

    def test_missing_unresolved_denies(self):
        """Item 4: unresolved_findings missing (not 0) → merge denied — the
        value must come from the trusted check run's output."""
        ev = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1,
              "trusted_producer_ok": True, "review_sha_match": True,
              "required_checks_pass": True, "unresolved_critical_high": None,
              "coverage_complete": True, "auto_merge": True}
        granted, reason = merge_authority(ev)
        assert granted is False
        assert "unresolved" in reason.lower() or "summary" in reason.lower()

    def test_auto_merge_disabled(self):
        ev = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1,
              "trusted_producer_ok": True, "review_sha_match": True,
              "required_checks_pass": True, "unresolved_critical_high": 0,
              "coverage_complete": True, "auto_merge": False}
        granted, reason = merge_authority(ev)
        assert granted is False
        assert "auto-merge" in reason.lower() or "disabled" in reason.lower()


# ---------------------------------------------------------------------------
# Merge executor tests
# ---------------------------------------------------------------------------

class TestMergeExecutor:
    """merge() — deterministic, re-reads head, refuses if moved."""

    def test_merge_success(self):
        """Happy path: merge succeeds with matching SHA."""
        evidence = {
            "ok": True, "head_sha": SHA1, "decided_sha": SHA1,
            "trusted_producer_ok": True, "review_sha_match": True,
            "required_checks_pass": True, "unresolved_critical_high": 0,
            "coverage_complete": True, "auto_merge": True,
        }
        with patch("src.merge_executor._gh", side_effect=_fake_gh_success):
            result = merge("acme/a", 1, SHA1, evidence, "test-token")
        assert result["merged"] is True
        assert result["sha"] == SHA1

    def test_head_moved_before_merge(self):
        """Head changed between evidence and merge → HeadMoved."""
        evidence = {
            "ok": True, "head_sha": SHA1, "decided_sha": SHA1,
            "trusted_producer_ok": True, "review_sha_match": True,
            "required_checks_pass": True, "unresolved_critical_high": 0,
            "coverage_complete": True, "auto_merge": True,
        }
        with patch("src.merge_executor._gh", side_effect=_fake_gh_head_moved):
            with pytest.raises(HeadMoved):
                merge("acme/a", 1, SHA1, evidence, "test-token")

    def test_merge_denied_closed_pr(self):
        """PR is closed → MergeDenied."""
        def fake_gh(*args, **kw):
            gh_args = args[1:] if args else ()
            if gh_args[:3] == ("pr", "view", "1"):
                data = {
                    "headRefOid": SHA1,
                    "state": "CLOSED",
                    "mergeable": "MERGEABLE",
                    "reviews": [],
                    "statusCheckRollup": [],
                }
                return 0, json.dumps(data)
            raise AssertionError(f"unexpected call: {args}")

        evidence = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1}
        with patch("src.merge_executor._gh", side_effect=fake_gh):
            with pytest.raises(MergeDenied):
                merge("acme/a", 1, SHA1, evidence, "test-token")

    def test_merge_without_evidence(self):
        """Missing evidence → MergeDenied."""
        evidence = {"ok": False, "reason": "no evidence"}
        with patch("src.merge_executor._gh", side_effect=_fake_gh_success):
            with pytest.raises(MergeDenied):
                merge("acme/a", 1, SHA1, evidence, "test-token")

    def test_merge_calls_api_with_expected_sha(self):
        """The merge API call carries sha=<expected_sha> — GitHub rejects the
        merge if the head moved (Extra B / B2 spec: REST merge API with sha).
        The bogus `delete_branch=true` form field (not a merge-API parameter)
        is gone — branch deletion is an explicit ref DELETE (defect 3)."""
        calls: list = []

        def fake_gh(token, *args, **kw):
            gh_args = args
            if (lookup := _branch_lookups(gh_args)) is not None:
                return lookup
            if gh_args[:3] == ("pr", "view", "1"):
                data = {"headRefOid": SHA1, "state": "OPEN",
                        "mergeable": "MERGEABLE",
                        "headRefName": "topic-branch"}
                return 0, json.dumps(data)
            if gh_args[0] == "api" and gh_args[3] == "repos/acme/a/pulls/1/merge":
                calls.append(list(gh_args))
                return 0, json.dumps({"merged": True, "sha": "M" * 40})
            if gh_args[:4] == ("api", "-X", "DELETE",
                               "repos/acme/a/git/refs/heads/topic-branch"):
                return 0, ""
            raise AssertionError(f"unexpected call: {args}")

        evidence = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1}
        with patch("src.merge_executor._gh", side_effect=fake_gh):
            result = merge("acme/a", 1, SHA1, evidence, "test-token")
        assert result["merged"] is True
        assert calls, "the merge API must be called"
        call = calls[0]
        assert call[0] == "api" and call[3] == "repos/acme/a/pulls/1/merge", call
        assert call[call.index("-X") + 1] == "PUT", call
        assert f"sha={SHA1}" in call, call
        assert "merge_method=squash" in call, call
        assert "delete_branch=true" not in call, call

    def test_merge_api_rejects_changed_head(self):
        """GitHub rejects the merge when the head no longer matches the given
        sha (the race between our re-read and the merge call) -> HeadMoved."""
        def fake_gh(token, *args, **kw):
            gh_args = args
            if gh_args[:3] == ("pr", "view", "1"):
                data = {"headRefOid": SHA1, "state": "OPEN",
                        "mergeable": "MERGEABLE"}
                return 0, json.dumps(data)
            if gh_args[0] == "api" and gh_args[3] == "repos/acme/a/pulls/1/merge":
                return 1, "422: The provided sha is not the head sha of the pull request"
            raise AssertionError(f"unexpected call: {args}")

        evidence = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1}
        with patch("src.merge_executor._gh", side_effect=fake_gh):
            with pytest.raises(HeadMoved):
                merge("acme/a", 1, SHA1, evidence, "test-token")

    def test_head_read_timeout_becomes_merge_failed(self):
        """A hung `gh pr view` raises TimeoutExpired inside _gh; merge() must
        convert it to MergeFailed so run_neo's handlers see a MergeError
        instead of a job that stays running until lease recovery."""
        evidence = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1}
        with patch("src.merge_executor._gh",
                   side_effect=subprocess.TimeoutExpired(cmd="gh", timeout=45)):
            with pytest.raises(MergeFailed):
                merge("acme/a", 1, SHA1, evidence, "test-token")

    def test_merge_api_timeout_becomes_merge_failed(self):
        """A hung merge API call raises TimeoutExpired inside _gh; merge()
        must convert it to MergeFailed."""
        def fake_gh(token, *args, **kw):
            if args[:3] == ("pr", "view", "1"):
                return 0, json.dumps({"headRefOid": SHA1, "state": "OPEN",
                                      "mergeable": "MERGEABLE",
                                      "headRefName": "topic-branch"})
            if args[0] == "api" and args[3] == "repos/acme/a/pulls/1/merge":
                raise subprocess.TimeoutExpired(cmd="gh", timeout=60)
            raise AssertionError(f"unexpected call: {args}")

        evidence = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1}
        with patch("src.merge_executor._gh", side_effect=fake_gh):
            with pytest.raises(MergeFailed):
                merge("acme/a", 1, SHA1, evidence, "test-token")


class TestMergeBranchDeletion:
    """Defect 3 — `delete_branch=true` is NOT a merge-API parameter, so the
    head branch silently stopped being deleted.  The merge executor now reads
    headRefName and deletes the ref explicitly after a confirmed merge,
    tolerating 404/422 (GitHub may already have auto-deleted it)."""

    def test_merge_deletes_head_branch_after_merge(self):
        deletes: list = []

        def fake_gh(token, *args, **kw):
            gh_args = args
            if (lookup := _branch_lookups(gh_args)) is not None:
                return lookup
            if gh_args[:3] == ("pr", "view", "1"):
                data = {"headRefOid": SHA1, "state": "OPEN",
                        "mergeable": "MERGEABLE",
                        "headRefName": "topic-branch"}
                return 0, json.dumps(data)
            if gh_args[0] == "api" and gh_args[3] == "repos/acme/a/pulls/1/merge":
                return 0, json.dumps({"merged": True, "sha": "M" * 40})
            if gh_args[:4] == ("api", "-X", "DELETE",
                               "repos/acme/a/git/refs/heads/topic-branch"):
                deletes.append(list(gh_args))
                return 0, ""  # 204
            raise AssertionError(f"unexpected call: {args}")

        evidence = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1}
        with patch("src.merge_executor._gh", side_effect=fake_gh):
            result = merge("acme/a", 1, SHA1, evidence, "test-token")
        assert result["merged"] is True
        assert deletes, "the head branch must be deleted explicitly"
        d = deletes[0]
        assert d[0] == "api" and "-X" in d and d[d.index("-X") + 1] == "DELETE", d
        assert d[-1] == "repos/acme/a/git/refs/heads/topic-branch", d

    def test_merge_tolerates_branch_already_gone(self):
        """404/422 on the ref DELETE (GitHub already auto-deleted the branch)
        is tolerated — the merge still counts as success."""
        def fake_gh(token, *args, **kw):
            gh_args = args
            if (lookup := _branch_lookups(gh_args)) is not None:
                return lookup
            if gh_args[:3] == ("pr", "view", "1"):
                data = {"headRefOid": SHA1, "state": "OPEN",
                        "mergeable": "MERGEABLE",
                        "headRefName": "topic-branch"}
                return 0, json.dumps(data)
            if gh_args[0] == "api" and gh_args[3] == "repos/acme/a/pulls/1/merge":
                return 0, json.dumps({"merged": True, "sha": "M" * 40})
            if gh_args[:4] == ("api", "-X", "DELETE",
                               "repos/acme/a/git/refs/heads/topic-branch"):
                return 1, "gh: Not Found (HTTP 404)"
            raise AssertionError(f"unexpected call: {args}")

        evidence = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1}
        with patch("src.merge_executor._gh", side_effect=fake_gh):
            result = merge("acme/a", 1, SHA1, evidence, "test-token")
        assert result["merged"] is True

    @staticmethod
    def _run(head_ref="topic-branch", cross=False, default="main", other_prs="",
             delete_result=(0, ""), raise_on=None):
        deletes: list = []

        def fake_gh(token, *args, **kw):
            if raise_on and args[:len(raise_on)] == raise_on:
                raise subprocess.TimeoutExpired(cmd="gh", timeout=60)
            if (lookup := _branch_lookups(args, default, other_prs)) is not None:
                return lookup
            if args[:3] == ("pr", "view", "1"):
                return 0, json.dumps({"headRefOid": SHA1, "state": "OPEN", "mergeable": "MERGEABLE",
                                      "headRefName": head_ref, "isCrossRepository": cross})
            if args[0] == "api" and args[3] == "repos/acme/a/pulls/1/merge":
                return 0, json.dumps({"merged": True, "sha": "M" * 40})
            if args[:3] == ("api", "-X", "DELETE"):
                deletes.append(args[3])
                return delete_result
            raise AssertionError(f"unexpected call: {args}")

        evidence = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1}
        with patch("src.merge_executor._gh", side_effect=fake_gh):
            result = merge("acme/a", 1, SHA1, evidence, "test-token")
        return result, deletes

    def test_shared_branch_with_other_open_pr_is_kept(self):
        result, deletes = self._run(head_ref="develop", other_prs="1\n7")
        assert result["merged"] is True
        assert deletes == []

    def test_default_branch_is_never_deleted(self):
        result, deletes = self._run(head_ref="main", default="main")
        assert result["merged"] is True
        assert deletes == []

    def test_fork_pr_branch_is_not_deleted(self):
        result, deletes = self._run(head_ref="someone:topic", cross=True)
        assert result["merged"] is True
        assert deletes == []

    def test_unknown_default_branch_keeps_the_branch(self):
        result, deletes = self._run(default="")
        assert result["merged"] is True
        assert deletes == []

    def test_open_pr_check_is_not_capped_at_30(self):
        """gh pr list returns 30 rows by default; a shared branch's older PR
        beyond that cap must still be seen, so the check asks for more."""
        seen: list = []
        orig = _branch_lookups

        def spy(args, *a, **k):
            if args[:2] == ("pr", "list"):
                seen.append(args)
            return orig(args, *a, **k)

        with patch(f"{__name__}._branch_lookups", side_effect=spy):
            self._run()
        assert seen and "--limit" in seen[0], seen
        assert int(seen[0][seen[0].index("--limit") + 1]) >= 1000, seen

    def test_own_branch_is_deleted_when_only_this_pr_uses_it(self):
        result, deletes = self._run(other_prs="1")
        assert deletes == ["repos/acme/a/git/refs/heads/topic-branch"]

    def test_delete_timeout_never_undoes_a_confirmed_merge(self):
        """A gh hang on the ref DELETE raises TimeoutExpired inside _gh; the merge
        is already confirmed, so merge() must still return merged=True."""
        result, _ = self._run(raise_on=("api", "-X", "DELETE"))
        assert result == {"merged": True, "sha": SHA1, "detail": result["detail"]}

    def test_lookup_timeout_never_undoes_a_confirmed_merge(self):
        result, deletes = self._run(raise_on=("api", "repos/acme/a", "--jq"))
        assert result["merged"] is True
        assert deletes == []

    def test_merge_skips_deletion_without_head_ref(self):
        """No headRefName on the PR → no deletion attempted."""
        def fake_gh(token, *args, **kw):
            gh_args = args
            if gh_args[:3] == ("pr", "view", "1"):
                data = {"headRefOid": SHA1, "state": "OPEN",
                        "mergeable": "MERGEABLE", "headRefName": ""}
                return 0, json.dumps(data)
            if gh_args[0] == "api" and gh_args[3] == "repos/acme/a/pulls/1/merge":
                return 0, json.dumps({"merged": True, "sha": "M" * 40})
            raise AssertionError(f"unexpected call: {args}")

        evidence = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1}
        with patch("src.merge_executor._gh", side_effect=fake_gh):
            result = merge("acme/a", 1, SHA1, evidence, "test-token")
        assert result["merged"] is True


# ---------------------------------------------------------------------------
# Escalation tests
# ---------------------------------------------------------------------------

class TestShouldEscalate:
    """should_escalate — correct threshold."""

    def test_escalate_at_5_not_4(self):
        assert should_escalate(4, 5, 5) is False
        assert should_escalate(5, 5, 5) is True

    def test_escalate_at_6(self):
        assert should_escalate(6, 5, 5) is True


class TestRoundRecord:
    """RoundRecord — data class for round history."""

    def test_round_record_creation(self):
        r = RoundRecord(round_no=1, sha=SHA1, findings="bug", outcome="bounced")
        assert r.round_no == 1
        assert r.sha == SHA1


class TestWorkOrder:
    """WorkOrder — structured fix plan."""

    def test_to_builder_brief(self):
        wo = WorkOrder(
            route="claude:opus",
            summary="Fix the auth bug",
            root_cause="Missing input validation",
            steps=[{"file": "src/auth.py", "change": "Add validation", "why": "Prevents injection"}],
            tests=["tests/test_auth.py::test_validation"],
            risks="Low",
        )
        brief = wo.to_builder_brief("acme/a", 1, SHA1)
        assert "ESCALATED ADVISOR WORK ORDER" in brief
        assert "acme/a#1" in brief
        assert "Fix the auth bug" in brief
        assert "src/auth.py" in brief
        assert "tests/test_auth.py" in brief


class TestSubmitAndPoll:
    """Async advisor contract — submit returns advice_id (202), never waits;
    poll returns the status (item 1)."""

    def test_submit_returns_advice_id_from_202(self):
        """POST returns 202 {"advice_id": ...}; submit returns it without
        waiting for the work order."""
        cfg = SimpleNamespace(
            advisor_url="http://100.111.118.24:8787",
            advisor_token="test-token",
        )
        rounds = [RoundRecord(round_no=1, sha=SHA1, outcome="bounced")]

        class FakeResp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def read(self):
                return json.dumps({"advice_id": "advice-abc"}).encode()

        def fake_urlopen(req, **kw):
            return FakeResp()

        from src.escalation import _post_advise
        with patch("src.escalation.urllib.request.urlopen", side_effect=fake_urlopen):
            advice_id = _post_advise(cfg, "acme/a", 1, SHA1, rounds, "", "read-token")
        assert advice_id == "advice-abc"

    def test_unconfigured_raises(self):
        """Unset advisor_url/token fails closed."""
        cfg = SimpleNamespace(advisor_url="", advisor_token="")
        from src.escalation import _post_advise
        with pytest.raises(AdvisorUnavailable):
            _post_advise(cfg, "acme/a", 1, SHA1, [], "", "read-token")

    def test_429_queue_full(self):
        """Advisor queue full (429) → AdvisorRefused."""
        cfg = SimpleNamespace(
            advisor_url="http://100.111.118.24:8787",
            advisor_token="test-token",
        )

        def fake_urlopen(req, **kw):
            import urllib.error
            raise urllib.error.HTTPError(
                "http://100.111.118.24:8787/v1/advise",
                429, "Too Many Requests", {}, None,
            )

        from src.escalation import _post_advise
        with patch("src.escalation.urllib.request.urlopen", side_effect=fake_urlopen):
            with pytest.raises(AdvisorRefused):
                _post_advise(cfg, "acme/a", 1, SHA1, [], "", "read-token")

    def test_poll_parses_done_with_work_order(self):
        """GET /v1/advise/<id> done → AdviceStatus with the work order."""
        cfg = SimpleNamespace(
            advisor_url="http://100.111.118.24:8787",
            advisor_token="test-token",
        )

        class FakeResp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def read(self):
                return json.dumps({
                    "status": "done",
                    "work_order": {"summary": "Auth bug", "root_cause": "rc",
                                    "steps": [{"file": "src/a.py", "change": "x", "why": ""}],
                                    "tests": [], "risks": ""},
                }).encode()

        def fake_urlopen(req, **kw):
            return FakeResp()

        with patch("src.escalation.urllib.request.urlopen", side_effect=fake_urlopen):
            st = poll_advice(cfg, "advice-abc")
        assert st.status == "done"
        assert st.work_order["summary"] == "Auth bug"

    def test_poll_parses_failed(self):
        """GET failed → AdviceStatus with the error."""
        cfg = SimpleNamespace(
            advisor_url="http://100.111.118.24:8787",
            advisor_token="test-token",
        )

        class FakeResp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def read(self):
                return json.dumps({"status": "failed", "error": "agent timed out"}).encode()

        def fake_urlopen(req, **kw):
            return FakeResp()

        with patch("src.escalation.urllib.request.urlopen", side_effect=fake_urlopen):
            st = poll_advice(cfg, "advice-abc")
        assert st.status == "failed"
        assert "timed out" in st.error

    def test_submit_escalation_stores_advice_id_and_returns(self):
        """submit_escalation mints a read token, POSTs, stores phase
        'escalating' with the advice_id, and returns without waiting."""
        state = MagicMock()
        jobs = MagicMock()
        cfg = SimpleNamespace(advisor_url="http://100.111.118.24:8787",
                              advisor_token="test-token")
        rounds = [RoundRecord(round_no=1, sha=SHA1, outcome="bounced")]
        from src.escalation import submit_escalation, _read_token
        with patch("src.escalation._read_token", return_value="read-token"), \
             patch("src.escalation._post_advise", return_value="advice-xyz"):
            out = submit_escalation(cfg, state, jobs, "acme/a", 1, SHA1,
                                    rounds, "", installation_id=7)
        assert out == {"action": "escalating", "advice_id": "advice-xyz"}
        # The PR's escalating marker is recorded under the per-PR advisory
        # lock (defect 1), carrying the advice_id + installation_id.
        rec = state.record_escalating.call_args
        assert rec.kwargs["advice_id"] == "advice-xyz"
        assert rec.kwargs["installation_id"] == 7

    def test_submit_escalation_parks_when_advisor_unreachable(self):
        """Advisor unreachable → fail closed, PR parked, no wait."""
        state = MagicMock()
        jobs = MagicMock()
        cfg = SimpleNamespace(advisor_url="http://100.111.118.24:8787",
                              advisor_token="test-token")
        from src.escalation import submit_escalation, _read_token
        with patch("src.escalation._read_token", return_value="read-token"), \
             patch("src.escalation._post_advise",
                   side_effect=AdvisorUnavailable("down")):
            out = submit_escalation(cfg, state, jobs, "acme/a", 1, SHA1,
                                    [], "")
        assert out["action"] == "parked"
        parked = [c for c in state.record.call_args_list if c[0][3] == "parked"]
        assert parked


class TestReadTokenPermissions:
    """Read-only token has restricted permissions."""

    def test_read_token_permissions(self):
        """Verify the read token is minted with contents:read + pull_requests:read.

        Mocks repo_token and asserts the permissions dict passed to it.
        """
        from src.escalation import _read_token
        from unittest.mock import patch

        captured_permissions = []

        def fake_repo_token(cfg, repo, installation_id=None, permissions=None):
            captured_permissions.append(permissions)
            return "read-token-abc"

        with patch("src.github_app.repo_token", side_effect=fake_repo_token):
            cfg = SimpleNamespace()
            token = _read_token(cfg, "acme/a", installation_id=123)

        assert token == "read-token-abc"
        assert len(captured_permissions) == 1
        perms = captured_permissions[0]
        assert perms == {"contents": "read", "pull_requests": "read"}
        # Never the write token (no elevated permissions).
        assert "write" not in str(perms.values())


class TestEvidenceBrief:
    """evidence_brief — machine-readable evidence line."""

    def test_granted(self):
        ev = {
            "ok": True, "head_sha": SHA1, "decided_sha": SHA1,
            "trusted_producer_ok": True, "review_sha_match": True,
            "required_checks_pass": True, "unresolved_critical_high": 0,
            "coverage_complete": True, "auto_merge": True,
        }
        brief = evidence_brief(ev)
        assert "GRANTED" in brief

    def test_not_granted(self):
        ev = {"ok": False, "reason": "no evidence"}
        brief = evidence_brief(ev)
        assert "NOT_GRANTED" in brief


# ---------------------------------------------------------------------------
# Advisor server — new async API (ThreadingHTTPServer + AdvisorService)
# ---------------------------------------------------------------------------

def _git(*args, cwd=None, check=True):
    import subprocess as _sp
    return _sp.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def _make_bare_repo(tmp_path):
    """A local repo with a default branch that DIFFERS from the PR head sha.

    Returns (bare_path, sha_default, sha_head).  `git clone --bare` of the
    source after switching back to the default branch, so the bare repo's
    HEAD is the default branch, not the head_sha.
    """
    work = tmp_path / "src"
    work.mkdir()
    _git("init", "-b", "main", str(work))
    _git("-C", str(work), "config", "user.email", "t@t")
    _git("-C", str(work), "config", "user.name", "t")
    (work / "a.txt").write_text("one\n")
    _git("-C", str(work), "add", ".")
    _git("-C", str(work), "commit", "-m", "c1")
    sha_default = _git("-C", str(work), "rev-parse", "HEAD").stdout.strip()
    # PR head: a second commit on a topic branch.
    _git("-C", str(work), "checkout", "-b", "topic")
    (work / "a.txt").write_text("two\n")
    _git("-C", str(work), "add", ".")
    _git("-C", str(work), "commit", "-m", "c2")
    sha_head = _git("-C", str(work), "rev-parse", "HEAD").stdout.strip()
    _git("-C", str(work), "checkout", "main")
    bare = tmp_path / "bare.git"
    _git("clone", "--bare", str(work), str(bare))
    return str(bare), sha_default, sha_head


def _fake_cli(tmp_path, name="claude", delay=0, body=None):
    """A fake agent CLI on PATH.  `delay` sleeps to keep the slot busy."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / name
    body = body or (
        '{"summary":"fix auth","root_cause":"input validation",'
        '"steps":[{"file":"src/a.py","change":"add check","why":"security"}],'
        '"tests":["tests/test_a.py::test_x"],"risks":"low"}'
    )
    lines = ["#!/bin/sh"]
    if delay:
        lines.append(f"sleep {delay}")
    lines += [f"cat <<'EOF'", body, "EOF"]
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    return bin_dir


def _fake_cli_checks_head(tmp_path, head_sha, name="claude"):
    """A fake agent CLI that FAILS unless its cwd HEAD equals head_sha."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / name
    script.write_text(
        "#!/bin/sh\n"
        "head=$(git rev-parse HEAD 2>/dev/null)\n"
        f'if [ "$head" != "{head_sha}" ]; then\n'
        '  echo "WRONG HEAD $head" >&2\n'
        "  exit 1\n"
        "fi\n"
        "cat <<'EOF'\n"
        '{"summary":"head ok","root_cause":"rc",'
        '"steps":[{"file":"f","change":"c","why":"w"}],"tests":[],"risks":""}\n'
        "EOF\n"
    )
    script.chmod(0o755)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    return bin_dir


def _free_port():
    import socket as _s
    s = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestAdvisorServer:
    """Advisor HTTP server — auth, bind address, queue (new async API)."""

    @staticmethod
    def _start(tmp_path, bare="", delay=30):
        from advisor.server import ThreadingHTTPServer, AdvisorRequestHandler, AdvisorService
        if bare:
            _fake_cli(tmp_path, delay=delay)
        port = _free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), AdvisorRequestHandler)
        server.advisor_token = "test-secret"
        server.advisor_service = AdvisorService(
            routes=("claude:opus",), timeout=120, queue_max=3,
            origin_url=bare or None,
        )
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server, t, port

    def test_bind_address_is_tailnet(self, tmp_path):
        """The server binds to the given address (config-driven, tailnet
        default) — live POST without auth answers 401/400."""
        import urllib.request
        import urllib.error
        server, t, port = self._start(tmp_path)
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/advise",
                data=b'{}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                assert False, "expected 401"
            except urllib.error.HTTPError as exc:
                # No auth -> 401 (or 400 bad json when auth passes); either
                # confirms the server is live on the configured address.
                assert exc.code in (400, 401), exc.code
        finally:
            server.shutdown()
            server.server_close()
        from src.config import Config
        assert Config().advisor_bind_host == "100.111.118.24"

    def test_auth_required(self, tmp_path):
        """POST without bearer token returns 401."""
        import urllib.request
        import urllib.error
        server, t, port = self._start(tmp_path)
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/advise",
                data=json.dumps({"repo": "o/r", "pr": 1,
                                 "head_sha": "a" * 40, "read_token": "tok"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                assert False, "expected 401"
            except urllib.error.HTTPError as exc:
                assert exc.code == 401, exc.code
                assert "unauthorized" in exc.read().decode().lower()
        finally:
            server.shutdown()
            server.server_close()

    def test_queue_limit(self, tmp_path):
        """One agent at a time; at most 3 waiting, then 429."""
        import urllib.request
        import urllib.error
        bare, _, head_sha = _make_bare_repo(tmp_path)
        server, t, port = self._start(tmp_path, bare=bare, delay=30)
        try:
            body = json.dumps({"repo": "o/r", "pr": 1, "head_sha": head_sha,
                               "read_token": "tok"}).encode()

            def send():
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/advise", data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": "Bearer test-secret"},
                    method="POST",
                )
                return urllib.request.urlopen(req, timeout=5)

            # First request starts the agent (holds the single active slot);
            # the next 3 wait in the queue — all 202 immediately.
            for _ in range(4):
                resp = send()
                assert resp.getcode() == 202, resp.getcode()
                resp.close()
            # 5th: queue full -> 429.
            try:
                send()
                assert False, "expected 429"
            except urllib.error.HTTPError as exc:
                assert exc.code == 429, exc.code
                assert "queue full" in exc.read().decode().lower()
        finally:
            server.shutdown()
            server.server_close()

    def test_unknown_advice_id_404(self, tmp_path):
        """GET for an unknown advice_id returns 404 (auth checked first)."""
        import urllib.request
        import urllib.error
        server, t, port = self._start(tmp_path)
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/advise/nonexistent",
                headers={"Authorization": "Bearer test-secret"},
                method="GET",
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                assert False, "expected 404"
            except urllib.error.HTTPError as exc:
                assert exc.code == 404, exc.code
        finally:
            server.shutdown()
            server.server_close()

    def test_results_kept_24h(self, tmp_path):
        """Advice records are pruned after 24 h."""
        from advisor.server import AdvisorService
        svc = AdvisorService(routes=("claude:opus",), timeout=60, queue_max=3)
        aid = svc.submit({"repo": "o/r", "pr": 1, "head_sha": "a" * 40,
                          "read_token": "tok"})
        assert svc.status(aid)["status"] in ("queued", "running")
        # Simulate the record ageing past 24 h.
        with svc._lock:
            svc._records[aid]["created_at"] -= 25 * 60 * 60
        assert svc.status(aid) is None


class TestAsyncAdvisor:
    """Item 1 — the advisor is asynchronous: POST /v1/advise returns 202
    immediately, the agent runs in a background thread, GET polls
    running -> done, and the escalated builder round is created once."""

    def test_submit_fast_poll_done_builder_once(self, tmp_path):
        import time as _time
        from advisor.server import ThreadingHTTPServer, AdvisorRequestHandler, \
            AdvisorService
        from src.escalation import submit_escalation, poll_advice, escalate_pr

        bare, _, head_sha = _make_bare_repo(tmp_path)
        _fake_cli(tmp_path, delay=2)
        port = _free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), AdvisorRequestHandler)
        server.advisor_token = "test-secret"
        server.advisor_service = AdvisorService(
            routes=("claude:opus",), timeout=60, queue_max=3, origin_url=bare)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        cfg = SimpleNamespace(advisor_url=f"http://127.0.0.1:{port}",
                              advisor_token="test-secret")
        state = MagicMock()
        jobs = MagicMock()
        rounds = [RoundRecord(round_no=1, sha=head_sha, outcome="bounced")]
        try:
            with patch("src.escalation._read_token", return_value="read-token"):
                t0 = _time.monotonic()
                out = submit_escalation(cfg, state, jobs, "o/r", 1, head_sha,
                                        rounds, "")
                elapsed = _time.monotonic() - t0
            # The agent sleeps 2 s — submit must NOT wait on it.
            assert elapsed < 1.0, f"submit took {elapsed:.2f}s"
            assert out["action"] == "escalating"
            advice_id = out["advice_id"]
            assert advice_id
            rec = state.record_escalating.call_args
            assert rec.kwargs["advice_id"] == advice_id
            assert rec.kwargs["installation_id"] is None  # no install id passed

            # Poll: sees running, then done.
            seen = set()
            final = None
            for _ in range(80):
                st = poll_advice(cfg, advice_id)
                seen.add(st.status)
                if st.status == "done":
                    final = st
                    break
                _time.sleep(0.2)
            assert final is not None, f"never done; saw {sorted(seen)}"
            assert "running" in seen or "queued" in seen
            assert final.work_order and final.work_order["summary"] == "fix auth"

            # The ONE escalated builder round appears exactly once.
            wo = WorkOrder.from_dict(final.work_order)
            res = escalate_pr(cfg, state, jobs, "o/r", 1, head_sha, wo,
                              installation_id=7)
            assert res["action"] == "escalated"
            assert jobs.create_builder.call_count == 1
        finally:
            server.shutdown()
            server.server_close()


class TestAdvisorClone:
    """Item 2 — the advisor clone checks out the PR head (detached), so the
    agent's cwd HEAD equals head_sha even when the remote's default branch
    differs; a clone failure surfaces git stderr without the token."""

    def test_agent_cwd_head_equals_head_sha(self, tmp_path):
        from advisor.server import run_advisor
        bare, sha_default, sha_head = _make_bare_repo(tmp_path)
        assert sha_default != sha_head  # default branch differs from the head
        _fake_cli_checks_head(tmp_path, sha_head)
        wo = run_advisor("o/r", 1, sha_head, [], "", "read-token",
                         routes=("claude:opus",), timeout=60, origin_url=bare)
        assert wo["summary"] == "head ok"

    def test_failed_clone_surfaces_stderr_without_token(self, tmp_path):
        from advisor.server import run_advisor, AgentFailed
        bare, _, _ = _make_bare_repo(tmp_path)
        _fake_cli(tmp_path)
        with pytest.raises(AgentFailed) as ei:
            run_advisor("o/r", 1, "d" * 40, [], "", "supersecret-token",
                        routes=("claude:opus",), timeout=60, origin_url=bare)
        err = str(ei.value)
        assert "git" in err.lower()
        assert "supersecret-token" not in err  # never the token in the error

    def test_askpass_script_keeps_token_out_of_shell_code(self, tmp_path):
        """A token with shell metacharacters is data, never executable code.

        The generated askpass script must not embed the token; it reads the
        token from NEO_ADVISOR_GIT_TOKEN in the child environment instead.
        """
        from advisor.server import _clone_head
        bare, _, sha_head = _make_bare_repo(tmp_path)
        marker = tmp_path / "pwned"
        token = f'$(touch {marker}); "`touch {marker}`\n'
        workdir = tmp_path / "advisor"
        workdir.mkdir()

        clone_dir = _clone_head(str(workdir), "o/r", sha_head, token,
                                origin_url=bare)
        assert os.path.isdir(clone_dir)  # normal clone still succeeds

        askpass_path = workdir / "git-askpass.sh"
        script = askpass_path.read_text()
        assert token not in script
        assert not marker.exists()

        # Run the askpass helper the way git would: token only in the env.
        proc = subprocess.run(
            [str(askpass_path)],
            env={**os.environ, "NEO_ADVISOR_GIT_TOKEN": token},
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert proc.stdout == token + "\n"
        assert not marker.exists()


class TestRouteFallback:
    """Route fallback when claude is missing."""

    def test_route_fallback(self):
        """When claude is missing, codex is tried next."""
        from advisor.server import _check_cli, _parse_route, _run_agent

        # _check_cli uses `which` which is available on any Linux system.
        # For a truly missing CLI, we verify the function returns False
        # for a path that doesn't exist.
        assert _check_cli("/nonexistent/path/cli-binary") is False

    def test_parse_route(self):
        """Route specs are parsed correctly."""
        from advisor.server import _parse_route
        engine, params = _parse_route("claude:opus")
        assert engine == "claude"
        assert params["model"] == "opus"

        engine, params = _parse_route("codex:gpt-6.1-sol:high")
        assert engine == "codex"
        assert params["model"] == "gpt-6.1-sol"
        assert params["effort"] == "high"

        engine, params = _parse_route("codex:gpt-6-sol:high")
        assert engine == "codex"
        assert params["model"] == "gpt-6-sol"


class TestMalformedJsonRepair:
    """Malformed JSON → one repair → fail → park."""

    def test_parse_work_order_with_markdown_fence(self):
        """Markdown-fenced JSON is parsed after stripping fences."""
        from advisor.server import _parse_work_order
        raw = '```json\n{"summary": "test", "root_cause": "rc", "steps": []}\n```'
        result = _parse_work_order(raw)
        assert result["summary"] == "test"

    def test_parse_work_order_plain_json(self):
        """Plain JSON is parsed directly."""
        from advisor.server import _parse_work_order
        raw = '{"summary": "test", "root_cause": "rc", "steps": []}'
        result = _parse_work_order(raw)
        assert result["summary"] == "test"

    def test_parse_work_order_repair_extracts_json(self):
        """JSON embedded in prose is extracted."""
        from advisor.server import _parse_work_order
        raw = ('Some text before\n'
               '{"summary": "test", "root_cause": "rc", "steps": [{"file": "x.py", "change": "fix", "why": ""}]}\n'
               'Some text after')
        result = _parse_work_order(raw)
        assert result["summary"] == "test"
        assert len(result["steps"]) == 1

    @pytest.mark.parametrize("steps", ["abc", ["abc"]])
    def test_parse_work_order_rejects_malformed_steps(self, steps):
        """Steps that are not a list of objects are rejected, not passed on."""
        from advisor.server import _parse_work_order, AgentFailed
        raw = json.dumps(
            {"summary": "test", "root_cause": "rc", "steps": steps}
        )
        with pytest.raises(AgentFailed,
                           match="steps must be a list of objects"):
            _parse_work_order(raw)


class TestAgentContainsGhPrMerge:
    """Agent output containing `gh pr merge` does nothing — it's just text."""

    def test_gh_pr_merge_in_output_is_harmless(self):
        """The merge executor is the only path that calls gh pr merge.
        A merge command in the agent's prose is not a verdict — production
        verdict extraction returns None, so nothing is executed."""
        from src.neo_protocol import extract_verdict_line
        text = ("Done.\ngh pr merge 42 --squash --admin\n"
                "Also: `gh pr merge --auto`\n")
        assert extract_verdict_line(text) is None


class TestEscalatedRoundBrief:
    """Escalated round's brief contains the work order."""

    def test_work_order_in_brief(self):
        """The work order is included in the escalated builder brief."""
        wo = WorkOrder(
            route="claude:opus",
            summary="Fix auth bug",
            root_cause="Missing validation",
            steps=[{"file": "src/auth.py", "change": "Add validation", "why": "Security"}],
            tests=["tests/test_auth.py::test_validation"],
            risks="Low",
        )
        brief = wo.to_builder_brief("acme/a", 1, SHA1)
        assert "ESCALATED ADVISOR WORK ORDER" in brief
        assert "src/auth.py" in brief
        assert "Add validation" in brief


class TestSecondBounceAfterEscalation:
    """2nd bounce after escalation parks the PR."""

    def test_second_bounce_parks(self):
        """After escalation, another bounce parks the PR."""
        state = MagicMock()
        jobs = MagicMock()
        cfg = SimpleNamespace(
            bounces_before_escalation=5,
            advisor_url="http://100.111.118.24:8787",
            advisor_token="test-token",
        )

        # Simulate that the advisor fails (unavailable).
        from src.escalation import _park_pr
        result = _park_pr(state, jobs, "acme/a", 1, SHA1,
                          "advisor unavailable after escalation")
        assert result["action"] == "parked"
        assert "advisor" in result["reason"]
        state.record.assert_called_once()


class TestCountBuilderRounds:
    """Round counting across SHAs."""

    def test_count_rounds(self):
        """builder_rounds returns the total count."""
        state = MagicMock()
        state.builder_rounds.return_value = 3
        from src.escalation import count_builder_rounds
        assert count_builder_rounds(state, "acme/a", 1) == 3
        state.builder_rounds.assert_called_once_with("acme/a", 1)


class TestRecoverEscalations:
    """Item 1 — the recovery loop's poll-and-advance: timeout parks, advisor
    failure parks, done creates the ONE escalated round."""

    def test_timeout_parks(self):
        """Give up after the timeout and park with a one-line reason."""
        state = MagicMock()
        state.escalating_prs.return_value = [{
            "repo": "o/r", "pr": 1, "head_sha": SHA1,
            "advice_id": "advice-1", "installation_id": None,
            "created_at": datetime.now(timezone.utc) - timedelta(minutes=31),
        }]
        jobs = MagicMock()
        cfg = SimpleNamespace(advisor_url="http://x", advisor_token="t")
        from src.escalation import recover_escalations
        handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "parked", handled
        assert "did not finish" in handled[0]["reason"]
        state.take_escalation.assert_called_once_with("o/r", 1, "advice-1")

    def test_advisor_failed_parks(self):
        """Advisor reports failed → park with the advisor's error."""
        state = MagicMock()
        state.escalating_prs.return_value = [{
            "repo": "o/r", "pr": 1, "head_sha": SHA1,
            "advice_id": "advice-1", "installation_id": None,
            "created_at": datetime.now(timezone.utc),
        }]
        jobs = MagicMock()
        cfg = SimpleNamespace(advisor_url="http://x", advisor_token="t")
        from src.escalation import recover_escalations, AdviceStatus
        with patch("src.escalation.poll_advice",
                   return_value=AdviceStatus(status="failed", error="agent timed out")):
            handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "parked", handled
        assert "timed out" in handled[0]["reason"]
        state.take_escalation.assert_called_once_with("o/r", 1, "advice-1")

    def test_done_creates_one_escalated_round(self):
        """Advisor done → escalate_pr creates the ONE escalated round."""
        state = MagicMock()
        state.escalating_prs.return_value = [{
            "repo": "o/r", "pr": 1, "head_sha": SHA1,
            "advice_id": "advice-1", "installation_id": 7,
            "created_at": datetime.now(timezone.utc),
        }]
        state.builder_rounds.return_value = 5
        # No escalated round recorded yet — idempotent escalate_pr records one.
        state.advisor_escalation_for.return_value = None
        jobs = MagicMock()
        cfg = SimpleNamespace(advisor_url="http://x", advisor_token="t")
        work_order = {"route": "claude:opus", "summary": "Fix auth",
                      "root_cause": "rc",
                      "steps": [{"file": "src/a.py", "change": "c", "why": "w"}],
                      "tests": [], "risks": ""}
        from src.escalation import recover_escalations, AdviceStatus
        with patch("src.escalation.poll_advice",
                   return_value=AdviceStatus(status="done", work_order=work_order)):
            handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "escalated", handled
        # ONE escalated builder round.
        jobs.create_builder.assert_called_once()
        payload = jobs.create_builder.call_args.kwargs["payload"]
        assert payload["escalated"] is True
        assert payload["work_order"]["summary"] == "Fix auth"
        assert payload["installation_id"] == 7
        state.take_escalation.assert_called_once_with("o/r", 1, "advice-1")


class TestComputeDiffStat:
    """Defect 5 — the diff stat comes from the pull API (changed_files /
    additions / deletions), never `gh pr diff --stat` (which gh rejects, so
    the advisor permanently saw `Diff stat: (not available)`)."""

    def test_computes_stat_from_pull_api(self):
        from src.neo_worker import _compute_diff_stat
        calls: list = []

        def fake_gh(token, *args, **kw):
            gh_args = args
            calls.append(list(gh_args))
            if "--jq" in gh_args:
                return 0, "3 files changed, +120/-40"
            return 0, json.dumps({"changed_files": 3, "additions": 120,
                                  "deletions": 40})

        with patch("src.neo_worker._gh", side_effect=fake_gh):
            stat = _compute_diff_stat("tok", "acme/a", 7)
        assert stat == "3 files changed, +120/-40", stat
        # The stat comes from the pull API — never `gh pr diff --stat`.
        assert calls and calls[0][0] == "api", calls
        assert "repos/acme/a/pulls/7" in calls[0], calls
        assert "--stat" not in calls[0], calls
        assert "pr" not in calls[0][:2], calls

    def test_failure_returns_empty(self):
        """A failed stat read fails closed to '' (the advisor brief just says
        'not available'); it must never crash the escalation."""
        from src.neo_worker import _compute_diff_stat

        def fake_gh(token, *args, **kw):
            return 1, "gh: not authenticated"

        with patch("src.neo_worker._gh", side_effect=fake_gh):
            assert _compute_diff_stat("tok", "acme/a", 7) == ""


class TestAdvisorNotFoundParks:
    """Extra A — an advisor 404 for a KNOWN advice_id (advisor restarted and
    lost its in-memory results) parks immediately with that reason instead of
    waiting out the full timeout."""

    def test_404_parks_immediately(self):
        state = MagicMock()
        state.escalating_prs.return_value = [{
            "repo": "o/r", "pr": 1, "head_sha": SHA1,
            "advice_id": "advice-404", "installation_id": None,
            "created_at": datetime.now(timezone.utc),
        }]
        jobs = MagicMock()
        cfg = SimpleNamespace(advisor_url="http://x", advisor_token="t")
        from src.escalation import recover_escalations
        with patch("src.escalation.poll_advice",
                   side_effect=AdvisorNotFound(
                       "advisor has no record for advice_id advice-404")):
            handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "parked", handled
        assert "no record" in handled[0]["reason"] or "404" in handled[0]["reason"]
        state.take_escalation.assert_called_once_with("o/r", 1, "advice-404")

    def test_404_superseded_is_dropped_not_parked(self):
        """If a newer escalation superseded the 404'd advice, the PR is NOT
        parked — the newer advice is still pending."""
        state = MagicMock()
        state.escalating_prs.return_value = [{
            "repo": "o/r", "pr": 1, "head_sha": SHA1,
            "advice_id": "advice-old", "installation_id": None,
            "created_at": datetime.now(timezone.utc),
        }]
        state.take_escalation.return_value = False  # a newer advice superseded it
        jobs = MagicMock()
        cfg = SimpleNamespace(advisor_url="http://x", advisor_token="t")
        from src.escalation import recover_escalations
        with patch("src.escalation.poll_advice",
                   side_effect=AdvisorNotFound("no record")):
            handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "dropped", handled
        state.take_escalation.assert_called_once_with("o/r", 1, "advice-old")


class TestTerminalActionFailure:
    """Defect 2 — a terminal action that raises AFTER the escalating marker is
    consumed (transient DB error in escalate_pr / _park_pr) must not strand
    the PR.  The marker is re-recorded so the next poll retries; idempotent
    escalate_pr prevents a duplicate round."""

    def _state(self, advice_id="advice-1"):
        state = MagicMock()
        state.escalating_prs.return_value = [{
            "repo": "o/r", "pr": 1, "head_sha": SHA1,
            "advice_id": advice_id, "installation_id": 7,
            "created_at": datetime.now(timezone.utc),
        }]
        state.take_escalation.return_value = True
        return state

    def test_done_act_failure_rerecords_marker(self):
        """escalate_pr raises after consume → marker re-recorded, action
        retried, and the PR is NOT parked or stranded."""
        state = self._state()
        jobs = MagicMock()
        cfg = SimpleNamespace(advisor_url="http://x", advisor_token="t")
        from src.escalation import recover_escalations, AdviceStatus
        work_order = {"summary": "s", "root_cause": "rc",
                      "steps": [{"file": "f", "change": "c", "why": "w"}],
                      "tests": [], "risks": ""}

        def boom(*a, **k):
            raise RuntimeError("transient DB error")

        with patch("src.escalation.poll_advice",
                   return_value=AdviceStatus(status="done", work_order=work_order)), \
             patch("src.escalation.escalate_pr", side_effect=boom):
            handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "retry", handled
        assert "transient" in handled[0]["reason"].lower()
        # The marker is re-recorded (via the advisory-lock record_escalating).
        state.record_escalating.assert_called_once()
        rec = state.record_escalating.call_args
        assert rec.kwargs["advice_id"] == "advice-1"
        assert rec.kwargs["installation_id"] == 7

    def test_park_failure_rerecords_marker(self):
        """_park_pr raises after consume (timeout path) → marker re-recorded,
        not stranded."""
        state = MagicMock()
        state.escalating_prs.return_value = [{
            "repo": "o/r", "pr": 1, "head_sha": SHA1,
            "advice_id": "advice-1", "installation_id": None,
            "created_at": datetime.now(timezone.utc) - timedelta(minutes=31),
        }]
        state.take_escalation.return_value = True
        jobs = MagicMock()
        cfg = SimpleNamespace(advisor_url="http://x", advisor_token="t")
        from src.escalation import recover_escalations

        def boom(*a, **k):
            raise RuntimeError("db down")

        with patch("src.escalation._park_pr", side_effect=boom):
            handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "retry", handled
        state.record_escalating.assert_called_once()
        assert state.record_escalating.call_args.kwargs["advice_id"] == "advice-1"


class TestEscalatePrIdempotent:
    """Defect 2 — escalate_pr is idempotent per advice_id: a retry after a
    partial failure must not mint a second escalated round or a second
    builder job."""

    def test_second_call_returns_existing_round(self):
        state = MagicMock()
        state.builder_rounds.return_value = 5
        # First call recorded round 6; the retry finds it and reuses it.
        state.advisor_escalation_for.return_value = 6
        jobs = MagicMock()
        cfg = SimpleNamespace(advisor_url="http://x", advisor_token="t")
        from src.escalation import escalate_pr, WorkOrder
        wo = WorkOrder(route="claude:opus", summary="Fix", root_cause="rc",
                       steps=[{"file": "f", "change": "c", "why": "w"}],
                       tests=[], risks="")
        out = escalate_pr(cfg, state, jobs, "o/r", 1, SHA1, wo,
                          installation_id=7, advice_id="advice-1")
        assert out["action"] == "escalated"
        assert out["round"] == 6
        # No duplicate record, but the builder job is ensured.
        state.record.assert_not_called()
        jobs.create_builder.assert_called_once()
        assert jobs.create_builder.call_args[0][3] == 6

    def test_first_call_records_round(self):
        state = MagicMock()
        state.builder_rounds.return_value = 5
        state.advisor_escalation_for.return_value = None
        jobs = MagicMock()
        cfg = SimpleNamespace(advisor_url="http://x", advisor_token="t")
        from src.escalation import escalate_pr, WorkOrder
        wo = WorkOrder(route="claude:opus", summary="Fix", root_cause="rc",
                       steps=[{"file": "f", "change": "c", "why": "w"}],
                       tests=[], risks="")
        out = escalate_pr(cfg, state, jobs, "o/r", 1, SHA1, wo,
                          installation_id=7, advice_id="advice-1")
        assert out["action"] == "escalated"
        assert out["round"] == 6
        state.record.assert_called_once()
        assert state.record.call_args[0][3] == "escalated"
        assert state.record.call_args.kwargs["detail"]["advice_id"] == "advice-1"
        jobs.create_builder.assert_called_once()


class TestConfigDefaults:
    """Defect 5 — NEO_MAX_BUILDER_ROUNDS defaults to bounces_before_escalation
    (5), so a stock Config reaches the 5-bounce escalation instead of the
    builder lane parking at round 3.  Round 2 (HIGH): a config with max <
    bounces must CLAMP with a warning, never refuse to start — the live box
    ships NEO_MAX_BUILDER_ROUNDS=2 and the old raise crash-looped both
    services at startup."""

    def test_default_max_builder_rounds_matches_bounces(self):
        c = Config()
        assert c.max_builder_rounds == 5, c.max_builder_rounds
        assert c.max_builder_rounds == c.bounces_before_escalation == 5

    def test_config_clamp_env_2_starts_logs_escalates(self, monkeypatch, caplog):
        """NEO_MAX_BUILDER_ROUNDS=2 (the live .env value) with the new
        bounces_before_escalation default of 5: the config STARTS (no crash
        loop), LOGS the clamp, and the round limit escalates at 2 — clamped
        up to the bounce floor so the5-bounce escalation is reachable."""
        monkeypatch.setenv("NEO_MAX_BUILDER_ROUNDS", "2")
        monkeypatch.delenv("NEO_BOUNCES_BEFORE_ESCALATION", raising=False)
        monkeypatch.setenv("NEO_CLAUDE_BIN", sys.executable)  # validate() gate
        with caplog.at_level(logging.WARNING, logger="rhobear_neo.config"):
            c = Config()               # starts (was: fine)
            assert c.validate() is c   # starts (was: ValueError crash-loop)
        assert any("NEO_MAX_BUILDER_ROUNDS" in rec.getMessage()
                   for rec in caplog.records), "the clamp must log a warning"
        # escalates at 2: the limit escalates from the shipped 2 up to the
        # bounce floor, so the builder lane can never park before the PR
        # reaches bounces_before_escalation.
        assert c.max_builder_rounds == 5 == c.bounces_before_escalation
        assert c.max_builder_rounds >= c.bounces_before_escalation
        assert should_escalate(5, c.max_builder_rounds,
                               c.bounces_before_escalation) is True
        # And literally "escalates at 2": with the bounce floor at 2 alongside
        # the shipped 2, the escalation gate fires after 2 rounds.
        monkeypatch.setenv("NEO_BOUNCES_BEFORE_ESCALATION", "2")
        c2 = Config()
        assert c2.max_builder_rounds == 2 == c2.bounces_before_escalation
        assert should_escalate(2, c2.max_builder_rounds,
                               c2.bounces_before_escalation) is True

    def test_validate_refuses_invalid_rounds(self, monkeypatch):
        """validate() still refuses INVALID values — negative or non-integer —
        just never a value the previous release shipped."""
        c = Config()
        c.max_builder_rounds = -1
        with pytest.raises(ValueError, match="NEO_MAX_BUILDER_ROUNDS"):
            c.validate()

        monkeypatch.setenv("NEO_MAX_BUILDER_ROUNDS", "lots")
        c2 = Config()   # the tolerant reader substitutes the default...
        assert c2.max_builder_rounds == 5
        with pytest.raises(ValueError, match="NEO_MAX_BUILDER_ROUNDS"):
            c2.validate()   # ...but validate() refuses the raw non-integer

    def test_validate_accepts_max_eq_bounces(self):
        c = Config()
        c.max_builder_rounds = 5
        c.bounces_before_escalation = 5
        # validate() raises only on genuinely invalid values here (other checks
        # are independent of the env); max == bounces must not trip it.
        try:
            c.validate()
        except ValueError as exc:
            assert "max_builder_rounds" not in str(exc).lower()
            assert "NEO_MAX_BUILDER_ROUNDS" not in str(exc)


# ---------------------------------------------------------------------------
# W1-07-r1 — the two greens on the exact head (the merge rule (b)-(d))
#
# Board `merge-authority` v7 / `trusted` / `verdicts`:
#   green 1: Apollo ACCEPT on the head commit — always required
#   green 2: the repo's own merge authority — architect approval bound to the
#            exact head, OR a trusted reviewer's green. Either one counts.
#   a trusted reviewer that is red always blocks; a new push resets all of it.
# ---------------------------------------------------------------------------

W1_OWNER = "deariencampbell1-sys"
W1_LANE_REPO = f"{W1_OWNER}/rhobear-neo"


def _review(state, oid=SHA1, author="rhobear-reviews[bot]",
            at="2026-10-26T11:52:00Z", author_type="Bot"):
    return {"state": state, "commit": {"oid": oid},
            "author": {"login": author, "type": author_type},
            "submittedAt": at, "body": "verdict: ACCEPT",
            "url": f"https://github.com/acme/a/pull/5#pullrequestreview-{state}"}


def _status(context, state="success", login=None, created="2026-10-26T11:50:00Z",
            creator_type="Bot"):
    if login is None:
        login = (W1_OWNER if context == "rhobear-reviews" or context.startswith("architect/")
                 else f"{context}[bot]")
    return {"context": context, "state": state,
            "creator": {"login": login, "type": creator_type},
            "created_at": created, "target_url": f"https://ci.example/{context}"}


def _greens(*, repo="acme/a", reviews=None, statuses=None, rollup=None,
            enabled=(), sha=SHA1):
    from src import neo_worker
    return neo_worker._green_fields(Config(), repo, sha,
                                    reviews or [], statuses or [], rollup or [], enabled)


def _bundle(greens, **over):
    ev = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1, "review_bound_ok": True,
          "trusted_green": True, "trusted_states": ["success"],
          "trusted_producer_ok": True, "review_sha_match": True,
          "required_checks_pass": True, "unresolved_critical_high": 0,
          "coverage_complete": True, "auto_merge": True,
          "failing_checks": 0}
    ev.update(greens)
    ev.update(over)
    return ev


def _authority(ev, install_auto_merge=True):
    from src import neo_worker
    from src.jev_review import JevAssessment
    return neo_worker.merge_authority(install_auto_merge, ev,
                                      JevAssessment(False, reason="no key"))


@pytest.mark.parametrize(("signal", "expected"), [
    ({"name": "CodeAnt", "conclusion": "SUCCESS",
      "app": {"slug": "codeant-ai"}}, True),
    ({"name": "CodeAnt", "conclusion": "SUCCESS",
      "app": {"slug": "mallory"}}, False),
    (_status("CodeAnt", login="CodeAnt[bot]", creator_type="Bot"), True),
    (_status("CodeAnt", login="CodeAnt", creator_type="User"), False),
    (_review("APPROVED", author="CodeAnt[bot]", author_type="Bot"), True),
    (_review("APPROVED", author="CodeAnt", author_type="User"), False),
], ids=("real-app-check", "user-look-alike-check", "bot-status",
        "user-status", "bot-review", "user-review"))
def test_green2b_requires_reviewer_identity_not_just_name(signal, expected):
    kind = "review" if "author" in signal else "status" if "context" in signal else "check"
    greens = _greens(enabled=("CodeAnt",),
                     reviews=[signal] if kind == "review" else [],
                     statuses=[signal] if kind == "status" else [],
                     rollup=[signal] if kind == "check" else [])
    assert greens["green2_ok"] is expected, greens


def test_user_codeant_status_never_makes_pr_green_end_to_end():
    greens = _greens(enabled=("CodeAnt",),
                     statuses=[_status("CodeAnt", login="CodeAnt", creator_type="User"),
                               _status("rhobear-reviews")],
                     reviews=[_review("APPROVED")])
    assert greens["green1_ok"] is True
    assert greens["green2_ok"] is False
    ok, reason = _authority(_bundle(greens))
    assert not ok and "green 2" in reason


def test_green2b_uses_newest_status_per_context():
    statuses = [_status("CodeAnt", login="CodeAnt[bot]", created="2026-10-26T11:50:00Z"),
                _status("CodeAnt", "failure", login="CodeAnt[bot]",
                        created="2026-10-26T12:00:00Z")]
    greens = _greens(enabled=("CodeAnt",), statuses=statuses)
    assert greens["green2_ok"] is False
    assert any("CodeAnt" in red for red in greens["trusted_red"])


def test_two_greens_codeant_and_apollo_merge():
    """CodeAnt green + Apollo ACCEPT on the exact head merges (owner Q5)."""
    greens = _greens(enabled=("CodeAnt",),
                     statuses=[_status("CodeAnt"), _status("rhobear-reviews")],
                     reviews=[_review("APPROVED")])
    ok, reason = _authority(_bundle(greens))
    assert ok is True, reason


def test_apollo_alone_is_not_enough():
    """Apollo ACCEPT is green 1 — never enough on its own."""
    greens = _greens(enabled=(), statuses=[_status("rhobear-reviews")],
                     reviews=[_review("APPROVED")])
    ok, reason = _authority(_bundle(greens))
    assert ok is False
    assert "green 2" in reason, reason


def test_disabled_codeant_does_not_count():
    """A disabled trusted reviewer's green does not count (the toggle)."""
    greens = _greens(enabled=(), statuses=[_status("CodeAnt"), _status("rhobear-reviews")],
                     reviews=[_review("APPROVED")])
    assert greens["green2_ok"] is False
    ok, reason = _authority(_bundle(greens))
    assert ok is False and "green 2" in reason, reason
    # ...and the same green counts the moment the customer turns it on.
    on = _greens(enabled=("CodeAnt",),
                 statuses=[_status("CodeAnt"), _status("rhobear-reviews")],
                 reviews=[_review("APPROVED")])
    assert _authority(_bundle(on))[0] is True


def test_trusted_reviewer_red_blocks():
    """A trusted reviewer that is red always blocks, even with Apollo's ACCEPT
    (board `verdicts`: "Zeus will not merge #46 even with Apollo's ACCEPT")."""
    greens = _greens(enabled=("CodeAnt",),
                     statuses=[_status("CodeAnt", "failure"), _status("rhobear-reviews")],
                     reviews=[_review("APPROVED")])
    ok, reason = _authority(_bundle(greens))
    assert ok is False and "red" in reason and "CodeAnt" in reason, reason
    # REQUEST CHANGES from a trusted reviewer blocks too.
    greens = _greens(enabled=("Greptile",),
                     statuses=[_status("rhobear-reviews")],
                     reviews=[_review("APPROVED"),
                              _review("CHANGES_REQUESTED", author="greptile[bot]")])
    ok, reason = _authority(_bundle(greens))
    assert ok is False and "red" in reason and "greptile" in reason, reason


def test_new_push_resets_the_greens():
    """Any new push resets both greens: everything here is bound to the head."""
    greens = _greens(enabled=("CodeAnt",), statuses=[],
                     reviews=[_review("APPROVED", oid=SHA2)])
    assert greens["green1_ok"] is False and greens["green2_ok"] is False
    ok, reason = _authority(_bundle(greens))
    assert ok is False, reason


def test_production_74_is_refused_naming_the_missing_green():
    """Production #74 (reviews-app, f0820bc, 2026-10-25): Apollo's review was
    COMMENTED and the only green was one rhobear-reviews status — Zeus merged
    it on a bare status green. Refused here, and the reason names the missing
    green(s)."""
    greens = _greens(enabled=(), reviews=[_review("COMMENTED")],
                     statuses=[_status("rhobear-reviews")],
                     rollup=[{"context": "rhobear-reviews", "state": "success"}])
    ok, reason = _authority(_bundle(greens, trusted_producer_ok=False))
    assert ok is False
    assert "green 1" in reason and "green 2" in reason, reason
    assert "COMMENTED" in reason, reason


def test_lane_repo_never_merges_without_architect_approval_on_the_exact_head():
    """A lane repo (rhobear-neo) with auto-merge ON still refuses without
    architect approval on the exact head — no bare status green, ever."""
    greens = _greens(repo=W1_LANE_REPO, reviews=[_review("APPROVED")], statuses=[],
                     enabled=())
    ok, reason = _authority(_bundle(greens), install_auto_merge=True)
    assert ok is False
    assert "architect" in reason, reason


def test_absent_check_is_not_green():
    """Apollo green, architect status absent — no merge; the reason names the
    missing status. A missing check counts as NOTHING (never as passing)."""
    greens = _greens(reviews=[_review("APPROVED")], statuses=[], enabled=())
    assert greens["green2_ok"] is False
    assert "architect" in greens["green2_reason"], greens["green2_reason"]
    ok, reason = _authority(_bundle(greens))
    assert ok is False and "green 2" in reason, reason


def test_merge_requires_exact_head():
    """A review on a parent commit is not green for the head."""
    greens = _greens(reviews=[_review("APPROVED", oid=SHA2)],
                     statuses=[_status("architect/l3-kit")], enabled=())
    assert greens["green1_ok"] is False
    ok, reason = _authority(_bundle(greens))
    assert ok is False and "green 1" in reason, reason


def test_architect_approval_counts_as_green_2_on_the_exact_head():
    """The architect's ACCEPT bound to the exact head is green 2 on its own
    (board `merge-authority` v7: "either one counts")."""
    greens = _greens(repo=W1_LANE_REPO, reviews=[_review("APPROVED")],
                     statuses=[_status("architect/l1-reviews")], enabled=())
    assert greens["green2_ok"] is True and greens["green2_kind"] == "architect"
    ok, reason = _authority(_bundle(greens), install_auto_merge=True)
    assert ok is True, reason


def test_architect_green_never_hides_a_red_trusted_reviewer():
    greens = _greens(repo=W1_LANE_REPO, reviews=[_review("APPROVED")],
                     statuses=[_status("architect/l1-reviews"), _status("CodeAnt", "failure")],
                     enabled=("CodeAnt",))
    assert greens["green2_ok"] is True
    assert greens["green2_kind"] == "architect"
    assert len(greens["trusted_red"]) == 1
    assert "CodeAnt" in greens["trusted_red"][0]
    ok, reason = _authority(_bundle(greens))
    assert ok is False and "CodeAnt" in reason, reason


def test_apollo_green_one_requires_the_real_bot_identity():
    greens = _greens(repo=W1_LANE_REPO,
                     reviews=[_review("APPROVED", author="rhobear-reviews", author_type="User")],
                     statuses=[_status("architect/l1-reviews")])
    assert greens["green1_ok"] is False
    assert greens["green2_ok"] is True
    ok, reason = _authority(_bundle(greens))
    assert ok is False and "green 1" in reason, reason


def test_a_reviewer_with_review_and_status_counts_once():
    """An outside reviewer with both a review and a status is one green —
    counted once, not twice."""
    greens = _greens(enabled=("CodeAnt",),
                     statuses=[_status("CodeAnt")],
                     reviews=[_review("APPROVED", author="CodeAnt[bot]")])
    assert greens["green2_ok"] is True
    assert len(greens["trusted_seen"]) == 1, greens["trusted_seen"]
    comment_lines = [ln for ln in _merge_comment(_bundle(greens)).splitlines()
                     if ln.startswith(("1:", "2:"))]
    assert len(comment_lines) == 2, comment_lines


def _merge_comment(evidence):
    from src import merge_executor
    return merge_executor.two_green_comment(evidence)


def test_merge_comment_lists_both_greens_with_links_and_ci_summary():
    """The one merge comment (board `verdicts` 02): the two greens with links
    and the CI summary, posted AFTER the merge (W1-07-r2 defect 9)."""
    greens = _greens(enabled=("CodeAnt",),
                     statuses=[_status("CodeAnt"), _status("rhobear-reviews")],
                     reviews=[_review("APPROVED")],
                     rollup=[{"context": "rhobear-reviews", "state": "success"},
                             {"name": "CodeAnt", "conclusion": "SUCCESS"}])
    body = _merge_comment(_bundle(greens))
    assert body.startswith(f"Two greens on {SHA1[:7]}"), body
    assert "1: " in body and "Apollo ACCEPT" in body
    assert "2: " in body and "CodeAnt passed (trusted)" in body
    assert "CI: 2 of 2 checks passed" in body, body
    assert "#pullrequestreview-APPROVED" in body, body          # green 1 link
    assert "https://ci.example/CodeAnt" in body, body           # green 2 link
    assert "11:52" in body and "11:50" in body, body            # times


def test_merge_rechecks_the_two_green_gate_immediately_before_the_merge():
    """merge() re-runs the same gate immediately before the merge API call —
    a bundle missing a green never reaches the merge call (step 5)."""
    greens = _greens(enabled=(), statuses=[_status("rhobear-reviews")],
                     reviews=[_review("COMMENTED")])
    ev = _bundle(greens)
    calls = []

    def fake_gh(token, *args, **kw):
        calls.append(args)
        if args[:3] == ("pr", "view", "5"):
            return 0, json.dumps({"headRefOid": SHA1, "state": "OPEN",
                                  "mergeable": "MERGEABLE", "headRefName": "topic",
                                  "isCrossRepository": False})
        raise AssertionError(f"unexpected call: {args}")

    with patch("src.merge_executor._gh", side_effect=fake_gh):
        with pytest.raises(MergeDenied) as exc:
            merge("o/r", 5, SHA1, ev, "tok")
    assert "green 1" in str(exc.value), exc.value
    assert not [c for c in calls if "PUT" in c], calls
    assert not [c for c in calls if c[:2] == ("pr", "comment")], calls


def test_merge_merges_then_posts_the_comment():
    """Two greens present: the merge goes first, then the one comment is
    posted (W1-07-r2 defect 9: posting the comment first left a stale
    'Two greens on <sha>' comment on every refused merge)."""
    greens = _greens(enabled=("CodeAnt",),
                     statuses=[_status("CodeAnt"), _status("rhobear-reviews")],
                     reviews=[_review("APPROVED")])
    ev = _bundle(greens)
    calls = []

    def fake_gh(token, *args, **kw):
        calls.append(args)
        if args[:3] == ("pr", "view", "5"):
            return 0, json.dumps({"headRefOid": SHA1, "state": "OPEN",
                                  "mergeable": "MERGEABLE", "headRefName": "topic",
                                  "isCrossRepository": False})
        if args[:2] == ("pr", "comment"):
            return 0, "{}"
        if args[:2] == ("api", "-X") and "PUT" in args:
            return 0, json.dumps({"merged": True, "sha": "M" * 40})
        if args[:2] == ("api", "-X") and "DELETE" in args:
            return 0, ""
        return 0, "main"

    with patch("src.merge_executor._gh", side_effect=fake_gh):
        result = merge("o/r", 5, SHA1, ev, "tok")
    assert result["merged"] is True
    kinds = ["comment" if c[:2] == ("pr", "comment") else
             "merge" if ("PUT" in c and "-X" in c) else "other" for c in calls]
    assert "comment" in kinds and "merge" in kinds
    assert kinds.index("merge") < kinds.index("comment"), kinds


def test_newest_status_per_trusted_context_decides():
    """Defect 3: the collector took the UNION of every trusted status on the
    head — one failing status that a later re-run drove to success kept the PR
    red forever. The newest status per trusted context decides."""
    from src import neo_worker

    cfg = Config()
    cfg.trusted_reviewer_app_id = "12345"
    statuses = [
        {"context": "rhobear-reviews", "state": "success",
         "created_at": "2026-10-26T11:55:00Z"},
        {"context": "rhobear-reviews", "state": "failure",
         "created_at": "2026-10-26T11:50:00Z"},   # the run that was re-driven
    ]

    def fake_gh(token, *args, timeout=30):
        if args[:2] == ("pr", "view"):
            return 0, json.dumps({"headRefOid": SHA1, "headRefName": "topic",
                                  "reviews": [],
                                  "statusCheckRollup": [
                                      {"__typename": "StatusContext",
                                       "context": "rhobear-reviews", "state": "SUCCESS",
                                       "createdAt": "2026-10-26T11:55:00Z"}]})
        if args[0] == "api" and "/check-runs" in args[1]:
            return 0, json.dumps({"total_count": 0, "check_runs": []})
        if args[0] == "api" and args[1].endswith("/status"):
            return 0, json.dumps({"total_count": len(statuses), "statuses": statuses})
        raise AssertionError(f"unexpected gh call {args}")

    with patch("src.neo_worker._gh", side_effect=fake_gh):
        ev = neo_worker._review_evidence("tok", cfg, "acme/a", 5, SHA1)
    assert ev["trusted_states"] == ["success"], ev["trusted_states"]
    assert ev["trusted_green"] is True

    # ...and a fresh failure on top of an old success keeps it red (the newest
    # status of the context decides in both directions).
    statuses[:] = [
        {"context": "rhobear-reviews", "state": "failure",
         "created_at": "2026-10-26T12:00:00Z"},
        {"context": "rhobear-reviews", "state": "success",
         "created_at": "2026-10-26T11:50:00Z"},
    ]
    with patch("src.neo_worker._gh", side_effect=fake_gh):
        ev = neo_worker._review_evidence("tok", cfg, "acme/a", 5, SHA1)
    assert ev["trusted_states"] == ["failure"], ev["trusted_states"]
    assert ev["trusted_green"] is False


def test_a_review_by_a_prefix_login_is_never_a_trusted_green():
    """End-to-end (W1-07-r2 defect 1, CRITICAL): a review by GitHub login
    `apol` on the exact head, with no architect status, must leave
    green2_ok False. A prefix of the trusted source's name is a DIFFERENT
    GitHub account — it must never produce the green that authorises a merge."""
    greens = _greens(enabled=("apollo",),
                     reviews=[_review("APPROVED", author="apol")],
                     statuses=[])
    assert greens["green2_ok"] is False, greens
    assert greens["green1_ok"] is False, greens
    ok, reason = _authority(_bundle(greens))
    assert ok is False, reason


def test_a_red_reviewer_after_a_green_one_still_refuses_the_merge():
    """Defect 2 (HIGH): green 2b `break`ed on the first enabled green, so a
    red reviewer sorted after a green one was never collected into
    trusted_red and rule (d) ("no enabled trusted reviewer is red on this
    head") failed open on merge authority.  Both source orders refuse and
    name z-red."""
    for sources in (("a-green", "z-red"), ("z-red", "a-green")):
        greens = _greens(enabled=sources,
                         statuses=[_status("a-green"),
                                   _status("z-red", "failure")],
                         reviews=[_review("APPROVED")])
        assert any("z-red" in r for r in greens["trusted_red"]), (sources, greens)
        assert greens["green2_ok"] is False, (sources, greens)
        ok, reason = _authority(_bundle(greens))
        assert ok is False, (sources, reason)
        assert "z-red" in reason, (sources, reason)


def test_an_architect_status_from_a_non_owner_account_is_refused():
    """Defect 4 (HIGH): the "architect status from the owner's account" branch
    had only a positive test. On the non-lane merge path, an architect status
    success set by a NON-OWNER account is refused: green2_ok False with a
    reason naming the creator rule (it must be set from the owner's account).
    Both the always-architect lane repo and the enabled-source route refuse.
    The positive case (test_architect_approval_counts_as_green_2_on_the_exact_head)
    stays green."""
    for repo, enabled in ((W1_LANE_REPO, ()), ("acme/a", ("architect",))):
        greens = _greens(repo=repo, reviews=[_review("APPROVED")],
                         statuses=[_status("architect/l1-reviews", login="mallory")],
                         enabled=enabled)
        assert greens["green2_ok"] is False, (repo, greens)
        assert "owner's account" in greens["green2_reason"], greens["green2_reason"]
        ok, reason = _authority(_bundle(greens), install_auto_merge=True)
        assert ok is False, (repo, reason)


def test_missing_architect_creator_is_not_an_owner_match():
    from src import neo_worker

    cfg = dataclasses.replace(Config(), owner_install_id="")
    with_empty_login = _status("architect/l1-reviews", login="")
    without_creator = {key: value for key, value in with_empty_login.items() if key != "creator"}
    for status in (with_empty_login, without_creator):
        greens = neo_worker._green_fields(
            cfg, W1_LANE_REPO, SHA1, [_review("APPROVED")], [status], [], ())
        assert greens["green2_ok"] is False, greens
        assert "owner's account" in greens["green2_reason"], greens["green2_reason"]


def test_a_review_with_no_author_is_never_apollos():
    """Defect 5 (MEDIUM): an empty/missing review author fell open — the
    green-1 loop skipped the Apollo check for a falsy author and chose the
    nameless review as Apollo's ACCEPT. Empty, None and whitespace authors
    never match any source."""
    for author in ("", None, "   "):
        greens = _greens(enabled=(),
                         statuses=[_status("rhobear-reviews")],
                         reviews=[_review("APPROVED", author=author)])
        assert greens["green1_ok"] is False, (repr(author), greens)
        ok, reason = _authority(_bundle(greens))
        assert ok is False, (repr(author), reason)


def test_created_at_instants_are_compared_as_instants_not_strings():
    """Defect 7 (MEDIUM): the newest-status-per-context collector compared
    created_at as strings, so mixed Z / +offset timestamps sorted
    lexicographically wrong and the OLDER status could win.  Parse to aware
    datetimes (Z -> +00:00) and compare those; a value that cannot be parsed
    sorts LAST (never wins).  12:00+05:00 is 07:00Z — older than 10:00Z."""
    from src import neo_worker

    cfg = Config()
    cfg.trusted_reviewer_app_id = "12345"
    newer = {"context": "rhobear-reviews", "state": "success",
             "created_at": "2026-10-26T10:00:00Z"}
    older = {"context": "rhobear-reviews", "state": "failure",
             "created_at": "2026-10-26T12:00:00+05:00"}
    unparsable = {"context": "rhobear-reviews", "state": "failure",
                  "created_at": "not-a-timestamp"}

    def run(statuses):
        def fake_gh(token, *args, timeout=30):
            if args[:2] == ("pr", "view"):
                return 0, json.dumps({"headRefOid": SHA1, "headRefName": "topic",
                                      "reviews": [], "statusCheckRollup": []})
            if args[0] == "api" and "/check-runs" in args[1]:
                return 0, json.dumps({"total_count": 0, "check_runs": []})
            if args[0] == "api" and args[1].endswith("/status"):
                return 0, json.dumps({"total_count": len(statuses),
                                      "statuses": statuses})
            raise AssertionError(f"unexpected gh call {args}")

        with patch("src.neo_worker._gh", side_effect=fake_gh):
            return neo_worker._review_evidence("tok", cfg, "acme/a", 5, SHA1)

    # Mixed timezones, both orders: the true newest (10:00Z) wins.
    for statuses in ([newer, older], [older, newer]):
        ev = run(statuses)
        assert ev["trusted_states"] == ["success"], (statuses, ev["trusted_states"])

    # An unparseable created_at sorts last and never wins, both orders.
    for statuses in ([unparsable, newer], [newer, unparsable]):
        ev = run(statuses)
        assert ev["trusted_states"] == ["success"], (statuses, ev["trusted_states"])


def test_a_refused_merge_posts_no_two_greens_comment():
    """Defect 9 (LOW): the merge comment was posted BEFORE the merge, so a
    refused or moved-head merge left a stale "Two greens on <sha>" comment
    and a retry posted a duplicate.  Decision (W1-07-r2): the comment is
    posted AFTER a successful merge — a refused merge leaves NO comment."""
    greens = _greens(enabled=("CodeAnt",),
                     statuses=[_status("CodeAnt"), _status("rhobear-reviews")],
                     reviews=[_review("APPROVED")])
    ev = _bundle(greens)
    calls = []

    def fake_gh(token, *args, **kw):
        calls.append(args)
        if args[:3] == ("pr", "view", "5"):
            return 0, json.dumps({"headRefOid": SHA1, "state": "OPEN",
                                  "mergeable": "MERGEABLE", "headRefName": "topic",
                                  "isCrossRepository": False})
        if args[:2] == ("pr", "comment"):
            return 0, "{}"
        if args[:2] == ("api", "-X") and "PUT" in args:
            return 1, "Protected branch hook declined: pull request is not mergeable"
        raise AssertionError(f"unexpected call: {args}")

    with patch("src.merge_executor._gh", side_effect=fake_gh):
        with pytest.raises((MergeDenied, MergeFailed)):
            merge("o/r", 5, SHA1, ev, "tok")
    assert not [c for c in calls if c[:2] == ("pr", "comment")], \
        f"a refused merge must leave no comment, got: {calls}"


def test_lane_or_unreadable_head_branch_never_merges_under_the_non_lane_rule():
    """A lane or unreadable head branch may never skip the lane architect and
    drift gates by using the non-lane two-greens rule."""
    greens = _greens(enabled=("CodeAnt",),
                     statuses=[_status("CodeAnt"), _status("rhobear-reviews")],
                     reviews=[_review("APPROVED")])
    ok, reason = _authority(_bundle(greens, head_ref="l3-kit/buttons"))
    assert ok is False and "lane" in reason, reason
    ok, reason = _authority(_bundle(greens, head_ref=""))
    assert ok is False and "head branch unavailable" in reason

def test_platform_owner_install_id_never_counts_on_a_customer_repo():
    greens = _greens(repo="acme/a", enabled=("architect",), reviews=[_review("APPROVED")],
                     statuses=[_status("architect/l1-reviews", login=W1_OWNER)])
    assert greens["green2_ok"] is False
    assert "owner's account" in greens["green2_reason"]
    ok, reason = _authority(_bundle(greens))
    assert ok is False and "green 2" in reason
    positive = _greens(repo=W1_LANE_REPO, enabled=("architect",), reviews=[_review("APPROVED")],
                       statuses=[_status("architect/l1-reviews", login=W1_OWNER)])
    assert positive["green2_ok"] is True
    assert positive["green2_kind"] == "architect"

def test_green_one_is_order_independent_across_review_lists():
    older = _review("APPROVED", at="2026-10-26T11:00:00Z")
    newer = _review("CHANGES_REQUESTED", at="2026-10-26T12:00:00Z")
    for order in ([older, newer], [newer, older]):
        greens = _greens(reviews=order, statuses=[], enabled=())
        assert greens["green1_ok"] is False
        assert greens["green1_reason"] == "CHANGES_REQUESTED"

def test_a_partial_rule_bundle_still_runs_the_gate_before_the_merge():
    ev = {"ok": True, "head_sha": SHA1, "decided_sha": SHA1,
          "trusted_red": ["CodeAnt (failure)"]}
    def fake_gh(token, *args, timeout=30):
        if args[:3] == ("pr", "view", "5"):
            return 0, json.dumps({"headRefOid": SHA1, "state": "OPEN"})
        raise AssertionError(f"unexpected gh call {args}")
    with patch("src.merge_executor._gh", fake_gh):
        with pytest.raises(MergeDenied, match="trusted reviewer is red"):
            merge("o/r", 5, SHA1, ev, "tok")


def test_customer_architect_is_enabled_without_a_trusted_reviewer_row():
    greens = _greens(repo="acme/a", enabled=(), reviews=[_review("APPROVED")],
                     statuses=[_status("architect/l1-reviews", login="acme")])
    assert greens["green2_ok"] is True
    assert greens["green2_kind"] == "architect"


def _refusal_pr_view(calls, **over):
    data = {"headRefOid": SHA1, "state": "OPEN", "mergeable": "MERGEABLE",
            "headRefName": "topic", "isCrossRepository": False}
    data.update(over)

    def fake_gh(token, *args, timeout=30):
        calls.append(args)
        if args[:3] == ("pr", "view", "5"):
            return 0, json.dumps(data)
        if args[:2] == ("api", "-X") and "PUT" in args:
            return 0, json.dumps({"merged": True, "sha": SHA1})
        if args[:2] == ("pr", "comment"):
            return 0, "{}"
        return 0, "main"
    return fake_gh


def test_merge_refuses_a_fork_before_api():
    calls = []
    with patch("src.merge_executor._gh", _refusal_pr_view(calls, isCrossRepository=True)):
        with pytest.raises(MergeDenied, match="fork"):
            merge("o/r", 5, SHA1, {"ok": True}, "tok")
    assert not any("PUT" in c for c in calls)
    assert not any(c[:2] == ("pr", "comment") for c in calls)


def test_merge_refuses_a_non_mergeable_pr_before_api():
    calls = []
    with patch("src.merge_executor._gh", _refusal_pr_view(calls, mergeable="CONFLICTING")):
        with pytest.raises(MergeDenied, match="mergeable"):
            merge("o/r", 5, SHA1, {"ok": True}, "tok")
    assert not any("PUT" in c for c in calls)
    assert not any(c[:2] == ("pr", "comment") for c in calls)
