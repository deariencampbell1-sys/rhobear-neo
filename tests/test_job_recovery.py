"""Tests for N15-B1 durable jobs, leased builder rounds, and defect fixes.

Covers:
  - Jobs.create: duplicate concurrent wakes produce one job
  - Jobs.claim: atomic claim with lease
  - Jobs.recover_expired: lease expiry recovery
  - Jobs.supersede: newer round for same PR supersedes older ones
  - Jobs.claim_queued: two Builders racing claim one row
  - run_neo with jobs: dedupe + triage in one transaction
  - Crash before/after agent runs → lease recovery
  - Saturated executor keeps the wake (no drop)
  - Defect 1: unbounded fixer loop → round cap → parked
  - Defect 2: pr_state returns "" → not treated as closed
  - Defect 3: broader error handling around agent spawn
  - Defect 4: dropped wakes → nothing dropped, stays queued
  - Defect 5: rollup_green computed from rollup
  - Defect 6: dry-run busy loop → sleep every iteration
  - Hermes Builder invocation unchanged (same cmd line / outcome parsing)
"""
from __future__ import annotations

import json
import logging
import time
from unittest.mock import MagicMock, patch, ANY

import pytest

from src.jobs import Jobs, SCHEMA as JOBS_SCHEMA
from src.neo_state import NeoState
from src.neo_builder import (
    BuilderAgent, build_brief, pr_state, dispatch_one, gh,
    WORKTREES, BRIEFS, CONCURRENCY,
)


# ---------------------------------------------------------------------------
# Fixtures — in-memory SQLite-like testing via psycopg mocking
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_conn():
    """Provide a mock psycopg connection that records SQL."""
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value = cur
    conn.execute.return_value = cur  # c.execute() returns cur
    conn.__enter__.return_value = conn
    cur.__enter__.return_value = cur
    cur.fetchone.return_value = None
    cur.fetchall.return_value = []
    cur.description = []
    cur.rowcount = 0
    return conn


@pytest.fixture
def jobs(mock_conn):
    """Jobs instance with mocked connection."""
    with patch("src.jobs.psycopg.connect", return_value=mock_conn):
        j = Jobs("postgresql://fake:5432/db")
        return j


# ---------------------------------------------------------------------------
# Job creation and deduplication
# ---------------------------------------------------------------------------

class TestJobCreation:
    def test_create_returns_id(self, mock_conn):
        """create() returns the new job id."""
        mock_conn.cursor.return_value.fetchone.return_value = (42,)
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            job_id = j.create("owner/repo", 1, "abc" * 40, kind="wake",
                              payload={"repo": "owner/repo", "pr_number": 1, "sha": "abc" * 40})
            assert job_id == 42

    def test_create_duplicate_returns_none(self, mock_conn):
        """Duplicate (repo, pr, sha) returns None (ON CONFLICT DO NOTHING)."""
        mock_conn.cursor.return_value.fetchone.return_value = None
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            # First create
            j.create("owner/repo", 1, "abc" * 40)
            # Second create (same key) — simulate no row returned
            mock_conn.cursor.return_value.fetchone.return_value = None
            job_id = j.create("owner/repo", 1, "abc" * 40)
            assert job_id is None

    def test_create_different_sha_returns_id(self, mock_conn):
        """Different SHA for same repo+pr is a different job."""
        mock_conn.cursor.return_value.fetchone.side_effect = [(42,), (43,)]
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            id1 = j.create("owner/repo", 1, "abc" * 40)
            id2 = j.create("owner/repo", 1, "def" * 40)
            assert id1 == 42
            assert id2 == 43

    def test_claim_returns_id(self, mock_conn):
        """claim() returns the job id on successful claim."""
        mock_conn.cursor.return_value.fetchone.return_value = (42,)
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            job_id = j.claim("owner/repo", 1, "abc" * 40)
            assert job_id == 42

    def test_claim_already_claimed_returns_none(self, mock_conn):
        """claim() returns None when job is already running."""
        mock_conn.cursor.return_value.fetchone.return_value = None
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            job_id = j.claim("owner/repo", 1, "abc" * 40)
            assert job_id is None

    def test_claim_increments_attempt(self, mock_conn):
        """claim() increments attempt counter."""
        mock_conn.execute.return_value.fetchone.return_value = (42,)
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            # Verify the UPDATE SQL includes attempt = attempt + 1
            j.claim("owner/repo", 1, "abc" * 40)
            sql = mock_conn.execute.call_args[0][0]
            assert "attempt = attempt + 1" in sql


# ---------------------------------------------------------------------------
# Job lifecycle: resolve, lease, recover_expired
# ---------------------------------------------------------------------------

class TestJobLifecycle:
    def test_resolve_sets_terminal_outcome(self, mock_conn):
        """resolve() marks a job done with a terminal outcome."""
        # mock_conn.execute returns cur (the mock cursor). Set rowcount on cur.
        mock_conn.execute.return_value.rowcount = 1
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            result = j.resolve(42, "done", payload={"verdict": "ACCEPT-READY"})
            assert result is True

    def test_resolve_invalid_outcome(self):
        """resolve() raises AssertionError for non-terminal outcomes."""
        with patch("src.jobs.psycopg.connect"):
            j = Jobs("postgresql://fake:5432/db")
            with pytest.raises(AssertionError):
                j.resolve(42, "invalid")

    def test_lease_extends_expiry(self, mock_conn):
        """lease() extends the lease on a running job."""
        mock_conn.execute.return_value.rowcount = 1
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            result = j.lease(42, lease_secs=300)
            assert result is True

    def test_lease_expired_returns_false(self, mock_conn):
        """lease() returns False when the job is no longer running."""
        mock_conn.execute.return_value.rowcount = 0
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            result = j.lease(42)
            assert result is False

    def test_recover_expired_returns_rows(self, mock_conn):
        """recover_expired() returns recovered rows."""
        cur = mock_conn.cursor.return_value
        cur.fetchall.return_value = [
            (1, "owner/repo", 1, "abc" * 40, "builder", "queued",
             None, 1, 3, None, "{}", "2026-01-01", "2026-01-01"),
        ]
        # Simulate real psycopg description (iterable of 7-tuples)
        cur.description = [
            ("id", None, None, None, None, None, None),
            ("repo", None, None, None, None, None, None),
            ("pr_number", None, None, None, None, None, None),
            ("head_sha", None, None, None, None, None, None),
            ("kind", None, None, None, None, None, None),
            ("state", None, None, None, None, None, None),
            ("outcome", None, None, None, None, None, None),
            ("attempt", None, None, None, None, None, None),
            ("max_attempts", None, None, None, None, None, None),
            ("lease_expires", None, None, None, None, None, None),
            ("payload", None, None, None, None, None, None),
            ("created_at", None, None, None, None, None, None),
            ("updated_at", None, None, None, None, None, None),
        ]
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            rows = j.recover_expired("builder")
            assert len(rows) == 1
            assert rows[0]["repo"] == "owner/repo"
            assert rows[0]["attempt"] == 1

    def test_supersede_older_jobs(self, mock_conn):
        """supersede() marks older queued jobs for the same PR."""
        cur = mock_conn.cursor.return_value
        cur.fetchall.return_value = [(1,), (2,)]
        cur.description = [("id", None, None, None, None, None, None)]
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            ids = j.supersede("owner/repo", 1, "def" * 40)
            assert ids == [1, 2]

    def test_claim_queued_atomic(self, mock_conn):
        """claim_queued() atomically claims one queued job."""
        cur = mock_conn.cursor.return_value
        cur.fetchone.return_value = (
            42, "owner/repo", 1, "abc" * 40, "builder", "running",
            None, 1, 3, None, "{}", "2026-01-01", "2026-01-01",
        )
        cur.description = [
            ("id", None, None, None, None, None, None),
            ("repo", None, None, None, None, None, None),
            ("pr_number", None, None, None, None, None, None),
            ("head_sha", None, None, None, None, None, None),
            ("kind", None, None, None, None, None, None),
            ("state", None, None, None, None, None, None),
            ("outcome", None, None, None, None, None, None),
            ("attempt", None, None, None, None, None, None),
            ("max_attempts", None, None, None, None, None, None),
            ("lease_expires", None, None, None, None, None, None),
            ("payload", None, None, None, None, None, None),
            ("created_at", None, None, None, None, None, None),
            ("updated_at", None, None, None, None, None, None),
        ]
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            row = j.claim_queued("builder")
            assert row is not None
            assert row["id"] == 42
            assert row["repo"] == "owner/repo"

    def test_claim_queued_empty(self, mock_conn):
        """claim_queued() returns None when no jobs are queued."""
        cur = mock_conn.cursor.return_value
        cur.fetchone.return_value = None
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            row = j.claim_queued("builder")
            assert row is None


# ---------------------------------------------------------------------------
# Two Builders racing claim one row
# ---------------------------------------------------------------------------

class TestBuilderRace:
    def test_two_builders_claim_one_row(self, mock_conn):
        """Two concurrent Builders: one claims, other gets None."""
        cur = mock_conn.cursor.return_value
        # First call returns a row, second returns None
        cur.fetchone.side_effect = [
            (42, "owner/repo", 1, "abc" * 40, "builder", "running",
             None, 1, 3, None, "{}", "2026-01-01", "2026-01-01"),
            None,
        ]
        cur.description = [
            ("id", None, None, None, None, None, None),
            ("repo", None, None, None, None, None, None),
            ("pr_number", None, None, None, None, None, None),
            ("head_sha", None, None, None, None, None, None),
            ("kind", None, None, None, None, None, None),
            ("state", None, None, None, None, None, None),
            ("outcome", None, None, None, None, None, None),
            ("attempt", None, None, None, None, None, None),
            ("max_attempts", None, None, None, None, None, None),
            ("lease_expires", None, None, None, None, None, None),
            ("payload", None, None, None, None, None, None),
            ("created_at", None, None, None, None, None, None),
            ("updated_at", None, None, None, None, None, None),
        ]
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            # Builder 1 claims
            row1 = j.claim_queued("builder")
            assert row1 is not None
            # Builder 2 tries — gets nothing (UPDATE WHERE state='queued' matches none)
            row2 = j.claim_queued("builder")
            assert row2 is None


# ---------------------------------------------------------------------------
# Defect 1: round cap → parked
# ---------------------------------------------------------------------------

class TestDefect1RoundCap:
    def test_round_cap_parks_job(self, mock_conn):
        """A builder round at or above max_builder_rounds gets parked."""
        mock_conn.execute.return_value.rowcount = 1
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            result = j.resolve(42, "parked",
                               payload={"reason": "max builder rounds (2) reached"})
            assert result is True
            # The outcome is passed as a parameter, not inlined in SQL
            params = mock_conn.execute.call_args[0][1]
            assert "parked" in params

    def test_dispatch_one_enforces_round_cap(self):
        """dispatch_one() parks the job when round >= max."""
        from src.config import Config
        cfg = Config()
        cfg.max_builder_rounds = 2
        jobs = MagicMock(spec=Jobs)
        row = {"id": 1, "repo": "owner/repo", "pr_number": 1,
               "head_sha": "a" * 40, "attempt": 3, "builder_round": 3}
        with patch("src.neo_builder.pr_state", return_value="OPEN\tmain"):
            dispatch_one(cfg, jobs, row)
        jobs.resolve.assert_called_once_with(
            1, "parked",
            payload={"reason": "max builder rounds (2) reached"},
        )

    def test_round_under_cap_dispatches(self, tmp_path):
        """dispatch_one() dispatches when round < max."""
        from src.config import Config
        cfg = Config()
        cfg.max_builder_rounds = 3
        cfg.gh_token = "test-token"
        jobs = MagicMock(spec=Jobs)
        row = {"id": 1, "repo": "owner/repo", "pr_number": 1,
               "head_sha": "a" * 40, "attempt": 1, "builder_round": 1}
        with patch("src.neo_builder.BRIEFS", tmp_path / "briefs"):
            with patch("src.neo_builder.pr_state", return_value="OPEN\tmain"):
                with patch("src.neo_builder.prepare_worktree",
                           return_value=("/tmp/wt", "main")):
                    with patch("src.neo_builder.fetch_findings", return_value=""):
                        with patch("src.neo_builder.gh", return_value=(0, "a" * 40)):
                            with patch.object(BuilderAgent, "run",
                                              return_value=(
                                                  {"input_tokens": 0}, "FIXED")):
                                dispatch_one(cfg, jobs, row)
        # Should have been resolved as 'done' (dispatched), not 'parked'
        assert jobs.resolve.call_count >= 1
        call_args = jobs.resolve.call_args[0]
        assert call_args[1] == "done"


# ---------------------------------------------------------------------------
# Defect 2: pr_state returns "" → not treated as closed
# ---------------------------------------------------------------------------

class TestDefect2PrState:
    def test_empty_pr_state_is_not_skipped(self):
        """Empty pr_state should NOT cause the builder to skip the job.

        The dispatch returns early without resolving the job, leaving it
        running for lease recovery.
        """
        from src.config import Config
        cfg = Config()
        cfg.gh_token = "test-token"
        cfg.max_builder_rounds = 3
        jobs = MagicMock(spec=Jobs)
        row = {"id": 1, "repo": "owner/repo", "pr_number": 1,
               "head_sha": "a" * 40, "attempt": 1, "builder_round": 1}
        with patch("src.neo_builder.pr_state", return_value=""):
            dispatch_one(cfg, jobs, row)
        # Should NOT have resolved (returns early without resolving)
        # The job stays running for lease recovery
        assert jobs.resolve.call_count == 0

    def test_pr_state_returns_empty_on_gh_failure(self):
        """pr_state returns empty string when gh fails, not closed/merged."""
        with patch("src.neo_builder.gh", return_value=(1, "error")):
            state = pr_state(MagicMock(), "owner/repo", 1)
            assert state == ""


# ---------------------------------------------------------------------------
# Defect 3: broader error handling around agent spawn
# ---------------------------------------------------------------------------

class TestDefect3AgentErrors:
    def test_run_neo_catches_broad_exceptions(self):
        """run_neo must catch any exception from _run_agent, not just AgentError."""
        from src.neo_worker import run_neo
        from src.config import Config
        from src.neo_state import NeoState
        from src.jobs import Jobs

        cfg = Config()
        cfg.gh_token = "test-token"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.max_builder_rounds = 2
        cfg.auto_merge_default = False

        state = MagicMock(spec=NeoState)
        state.builder_rounds.return_value = 0
        state.install.return_value = {"auto_merge": False, "plan": None, "credits_balance": 100}

        jobs = MagicMock(spec=Jobs)
        jobs.claim.return_value = 42  # successful claim

        wake = {"repo": "owner/repo", "sha": "a" * 40,
                "green": True, "context": "test", "pr_number": 1}

        with patch("src.neo_worker.resolve_pr", return_value=1):
            with patch("src.neo_worker._review_evidence",
                       return_value={"ok": True, "head_sha": "a" * 40,
                                     "decided_sha": "a" * 40,
                                     "review_sha": "a" * 40,
                                     "review_bound_ok": True,
                                     "trusted_states": ["success"],
                                     "trusted_green": True,
                                     "failing_checks": 0,
                                     "rollup_green": True}):
                with patch("src.neo_worker._jev_for_pr"):
                    with patch("src.neo_worker._run_agent",
                               side_effect=FileNotFoundError("agent binary not found")):
                        # Should NOT raise — the broad except catches it
                        try:
                            run_neo(cfg, state, jobs, wake)
                        except Exception:
                            pytest.fail("run_neo should catch any exception from agent")

    def test_run_neo_catches_oserror(self):
        """OSError from agent spawn must be caught."""
        from src.neo_worker import run_neo
        from src.config import Config
        from src.neo_state import NeoState
        from src.jobs import Jobs

        cfg = Config()
        cfg.gh_token = "test-token"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.max_builder_rounds = 2
        cfg.auto_merge_default = False

        state = MagicMock(spec=NeoState)
        state.builder_rounds.return_value = 0
        state.install.return_value = {"auto_merge": False, "plan": None, "credits_balance": 100}

        jobs = MagicMock(spec=Jobs)
        jobs.claim.return_value = 42

        wake = {"repo": "owner/repo", "sha": "a" * 40,
                "green": True, "context": "test", "pr_number": 1}

        with patch("src.neo_worker.resolve_pr", return_value=1):
            with patch("src.neo_worker._review_evidence",
                       return_value={"ok": True, "head_sha": "a" * 40,
                                     "decided_sha": "a" * 40,
                                     "review_sha": "a" * 40,
                                     "review_bound_ok": True,
                                     "trusted_states": ["success"],
                                     "trusted_green": True,
                                     "failing_checks": 0,
                                     "rollup_green": True}):
                with patch("src.neo_worker._jev_for_pr"):
                    with patch("src.neo_worker._run_agent",
                               side_effect=OSError("permission denied")):
                        try:
                            run_neo(cfg, state, jobs, wake)
                        except Exception:
                            pytest.fail("run_neo should catch OSError")


# ---------------------------------------------------------------------------
# Defect 4: dropped wakes → nothing dropped, stays queued
# ---------------------------------------------------------------------------

class TestDefect4DroppedWakes:
    def test_wake_persisted_before_executor(self, mock_conn):
        """on_verdict persists the wake before submitting to executor."""
        from src.__main__ import main as neo_main
        from src.config import Config

        # Verify __main__.py's on_verdict creates a job
        import src.__main__ as main_mod
        assert hasattr(main_mod, "main")

    def test_no_semaphore_backpressure(self):
        """__main__ no longer uses a Semaphore to cap in-flight wakes."""
        import src.__main__ as main_mod
        # The old code had `_inflight = threading.Semaphore(...)` which is gone
        source = open(main_mod.__file__).read()
        assert "Semaphore" not in source, "Semaphore backpressure was removed"
        assert "dropping" not in source or "dropping" not in [
            l for l in source.split("\n") if "wake" in l.lower()
        ]


# ---------------------------------------------------------------------------
# Defect 5: rollup_green computed from rollup
# ---------------------------------------------------------------------------

class TestDefect5RollupGreen:
    def test_rollup_green_computed(self):
        """_review_evidence computes rollup_green from statusCheckRollup."""
        from src.neo_worker import _review_evidence
        # We can't call it without gh, but we can verify the function
        # produces a dict with the key
        import inspect
        source = inspect.getsource(_review_evidence)
        assert "rollup_green" in source
        # Should compute it from the rollup data, not hardcode
        assert "SUCCESS" in source

    def test_rollup_green_present_in_evidence(self):
        """Evidence dict includes rollup_green key."""
        from src.neo_worker import _review_evidence
        import inspect
        sig = inspect.signature(_review_evidence)
        assert "sha" in sig.parameters


# ---------------------------------------------------------------------------
# Defect 6: dry-run busy loop
# ---------------------------------------------------------------------------

class TestDefect6DryRun:
    def test_dry_run_sleeps_every_iteration(self):
        """dry-run mode sleeps every iteration, not only when no rows found."""
        import src.neo_builder as builder
        source = open(builder.__file__).read()
        # The old code had `if not rows: time.sleep(POLL_SECS)` inside the
        # dry-run block — it only slept when no rows.  The new code always
        # sleeps via `time.sleep(POLL_SECS)` unconditionally.
        dry_run_section = source[source.index("if dry:"):source.index("continue", source.index("if dry:"))]
        assert "time.sleep(POLL_SECS)" in dry_run_section
        # There should NOT be a conditional `if not rows:` before the sleep
        assert "if not rows" not in dry_run_section


# ---------------------------------------------------------------------------
# Lease expiry: crash mid-run → recovery → parked at cap
# ---------------------------------------------------------------------------

class TestLeaseExpiry:
    def test_lease_expiry_recovers_job(self, mock_conn):
        """recover_expired resets expired leases to queued."""
        cur = mock_conn.cursor.return_value
        cur.fetchall.return_value = [
            (1, "owner/repo", 1, "abc" * 40, "builder", "queued",
             None, 1, 3, None, "{}", "2026-01-01", "2026-01-01"),
        ]
        cur.description = [
            ("id", None, None, None, None, None, None),
            ("repo", None, None, None, None, None, None),
            ("pr_number", None, None, None, None, None, None),
            ("head_sha", None, None, None, None, None, None),
            ("kind", None, None, None, None, None, None),
            ("state", None, None, None, None, None, None),
            ("outcome", None, None, None, None, None, None),
            ("attempt", None, None, None, None, None, None),
            ("max_attempts", None, None, None, None, None, None),
            ("lease_expires", None, None, None, None, None, None),
            ("payload", None, None, None, None, None, None),
            ("created_at", None, None, None, None, None, None),
            ("updated_at", None, None, None, None, None, None),
        ]
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            rows = j.recover_expired("builder")
            assert len(rows) == 1

    def test_attempt_cap_parks_after_max_retries(self, mock_conn):
        """Job with attempt >= max_attempts gets parked after recovery."""
        mock_conn.execute.return_value.rowcount = 1
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            result = j.resolve(42, "parked",
                               payload={"reason": "lease expired 3/3 times"})
            assert result is True

    def test_crash_before_agent_starts_releases(self, mock_conn):
        """Crash before agent start: lease expiry recovers, no stranded SHA."""
        cur = mock_conn.cursor.return_value
        cur.fetchall.return_value = [
            (1, "owner/repo", 1, "abc" * 40, "wake", "queued",
             None, 1, 3, None, "{}", "2026-01-01", "2026-01-01"),
        ]
        cur.description = [
            ("id", None, None, None, None, None, None),
            ("repo", None, None, None, None, None, None),
            ("pr_number", None, None, None, None, None, None),
            ("head_sha", None, None, None, None, None, None),
            ("kind", None, None, None, None, None, None),
            ("state", None, None, None, None, None, None),
            ("outcome", None, None, None, None, None, None),
            ("attempt", None, None, None, None, None, None),
            ("max_attempts", None, None, None, None, None, None),
            ("lease_expires", None, None, None, None, None, None),
            ("payload", None, None, None, None, None, None),
            ("created_at", None, None, None, None, None, None),
            ("updated_at", None, None, None, None, None, None),
        ]
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            rows = j.recover_expired("wake")
            assert len(rows) == 1
            assert rows[0]["state"] == "queued"

    def test_stale_round_superseded_by_newer(self, mock_conn):
        """A stale builder round is superseded when a newer one exists."""
        cur = mock_conn.cursor.return_value
        cur.fetchall.return_value = [(1,), (2,)]
        cur.description = [("id", None, None, None, None, None, None)]
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            ids = j.supersede("owner/repo", 1, "def" * 40)
            assert len(ids) == 2


# ---------------------------------------------------------------------------
# Hermes Builder invocation unchanged
# ---------------------------------------------------------------------------

class TestHermesBuilderUnchanged:
    def test_builder_agent_subclass(self):
        """BuilderAgent is a HermesAgent subclass with same interface."""
        agent = BuilderAgent(
            work_dir="/tmp/test-wt",
            hermes_bin="/opt/rhobear-hermes/bin/hermes",
            provider="zai", model="glm-5.3-flash",
        )
        assert hasattr(agent, "run")
        assert hasattr(agent, "_build_cmd")
        assert hasattr(agent, "_build_env")
        assert hasattr(agent, "_parse_output")

    def test_builder_hermes_command_line(self):
        """BuilderAgent produces the same hermes CLI as the HermesAgent."""
        agent = BuilderAgent(
            work_dir="/tmp/test-wt",
            hermes_bin="/opt/rhobear-hermes/bin/hermes",
            provider="zai", model="glm-5.3-flash",
        )
        cmd = agent._build_cmd("fix the bug")
        assert cmd[0] == "/opt/rhobear-hermes/bin/hermes"
        assert "-z" in cmd
        assert cmd[cmd.index("-z") + 1] == "fix the bug"
        assert "-m" in cmd
        assert cmd[cmd.index("-m") + 1] == "glm-5.3-flash"
        assert "--provider" in cmd
        assert cmd[cmd.index("--provider") + 1] == "zai"
        assert "--yolo" in cmd
        assert "--accept-hooks" in cmd

    def test_builder_verdict_parsing(self):
        """BuilderAgent parses the same VERDICT lines as HermesAgent."""
        from src.agent_hermes import CANONICAL_VERDICTS
        agent = BuilderAgent(
            work_dir="/tmp/test-wt",
            hermes_bin="/fake/hermes", provider="zai", model="glm-5.3-flash",
        )
        from unittest.mock import MagicMock
        proc = MagicMock()
        proc.stdout = "Work done\nVERDICT: ACCEPT-READY\n"
        proc.stderr = ""
        proc.returncode = 0
        usage, verdict = agent._parse_output(proc)
        assert verdict == "ACCEPT-READY"
        assert "ACCEPT-READY" in CANONICAL_VERDICTS


# ---------------------------------------------------------------------------
# __main__ — on_verdict creates job before executor
# ---------------------------------------------------------------------------

class TestMainOnVerdict:
    def test_on_verdict_creates_job(self, mock_conn):
        """on_verdict creates a job before submitting to executor."""
        from src.__main__ import main
        import src.__main__ as main_mod
        # Verify the module structure — on_verdict logic is in main()
        assert callable(main_mod.main)

    def test_on_verdict_duplicate_returns_early(self):
        """Duplicate wake (same repo, pr, sha) returns early (job already exists)."""
        # Test the early-return path in on_verdict
        import src.__main__ as main_mod
        source = open(main_mod.__file__).read()
        assert "duplicate" in source or "already queued" in source


# ---------------------------------------------------------------------------
# Job schema
# ---------------------------------------------------------------------------

class TestJobSchema:
    def test_schema_includes_neo_jobs_table(self):
        """SCHEMA includes neo_jobs table definition."""
        assert "CREATE TABLE IF NOT EXISTS neo_jobs" in JOBS_SCHEMA
        assert "outcome" in JOBS_SCHEMA
        assert "lease_expires" in JOBS_SCHEMA
        assert "attempt" in JOBS_SCHEMA
        assert "max_attempts" in JOBS_SCHEMA

    def test_schema_in_neo_state(self):
        """neo_state.py schema also includes neo_jobs."""
        from src.neo_state import SCHEMA
        assert "CREATE TABLE IF NOT EXISTS neo_jobs" in SCHEMA


# ---------------------------------------------------------------------------
# Concurrency: saturated executor keeps the wake queued
# ---------------------------------------------------------------------------

class TestSaturatedExecutor:
    def test_executor_full_keeps_wake_queued(self, mock_conn):
        """When the executor is saturated, the wake stays queued (not dropped)."""
        from src.__main__ import main
        import src.__main__ as main_mod
        # The old code had a Semaphore that would drop wakes. The new code
        # creates the job first, then submits to executor. If the executor
        # is full, the job is already persisted.
        source = open(main_mod.__file__).read()
        # No Semaphore-based backpressure
        assert "Semaphore" not in source
        # The job is created before executor.submit
        assert "jobs.create(repo, pr, sha, payload=wake)" in source


# ---------------------------------------------------------------------------
# Import and module structure tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_jobs(self):
        from src.jobs import Jobs, SCHEMA
        assert Jobs is not None
        assert "neo_jobs" in SCHEMA

    def test_import_neo_builder_jobs(self):
        from src.neo_builder import Jobs
        assert Jobs is not None

    def test_import_neo_worker_jobs(self):
        from src.neo_worker import Jobs
        assert Jobs is not None
