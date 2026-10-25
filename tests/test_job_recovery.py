"""N15-B1 — worker/builder wiring tests for durable jobs and the defect fixes.

The real Postgres semantics (ON CONFLICT, leases, SKIP LOCKED claims, the
migration) are covered by tests/test_jobs_pg.py against a real PG 16
(NEO_TEST_DSN).  This file covers the wiring that needs fakes — the module
contracts, SQL shape, and run_neo / dispatch_one / __main__ orchestration —
so the full suite runs without a database.

Covers:
  - Jobs.create_wake / create_builder: dedupe + outcome-based re-arm wiring
  - Jobs.claim / resolve / retry / park / lease / recover_expired /
    supersede_older / peek_queued contracts
  - Defect 1  round cap -> parked (never auto-dispatched)
  - Defect 2  transient pr_state is not treated as closed
  - Defect 3  any agent-spawn exception releases the job (retry)
  - Defect 4  wakes are persisted before the executor (nothing dropped)
  - Defect 5  rollup_green computed from the rollup
  - Defect 6  --dry-run sleeps every iteration and is read-only
  - GLM 1     status wakes (no PR yet) create a job (pr NULL)
  - GLM 2     a BOUNCE verdict creates the builder job
  - GLM 3     green after red on the same head is a new job (color in key)
  - GLM 4     failed wakes are retried, never stranded
  - GLM 5     unresolvable PR is released retryably, not poisoned
  - GLM 6     long agent runs renew the lease
  - GLM 7     newest-first claim + supersede OLDER rounds
  - GLM 8     transient pr_state leaves the job running (lease recovery)
  - GLM 9     --dry-run never mutates durable job state
  - One schema definition (neo_state delegates to jobs.ensure_schema)
  - Hermes Builder invocation unchanged (same cmd line / outcome parsing)
"""
from __future__ import annotations

import json
import inspect
from unittest.mock import MagicMock, patch, ANY

import pytest

from src.jobs import Jobs, SCHEMA as JOBS_SCHEMA, INDEXES, MIGRATION, TERMINAL_OUTCOMES
from src.github_app import repo_token as _real_repo_token


@pytest.fixture(autouse=True)
def _patch_repo_token():
    """All tests in this file use MagicMock(spec=Jobs) for jobs and don't
    need a real GitHubApp.  Patch repo_token to return a dummy token so the
    removal of the personal-token fallback doesn't break them."""
    def fake_repo_token(cfg, repo, installation_id=None):
        return "test-ghs-token"
    with patch("src.neo_worker.repo_token", side_effect=fake_repo_token), \
         patch("src.neo_builder.repo_token", side_effect=fake_repo_token):
        yield
from src.neo_builder import (
    BuilderAgent, build_brief, pr_state, dispatch_one, gh,
    WORKTREES, BRIEFS, CONCURRENCY, POLL_SECS,
)


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
    with patch("src.jobs.psycopg.connect", return_value=mock_conn):
        return Jobs("postgresql://fake:5432/db")


def _cols(*names):
    return [("id", None, None, None, None, None, None)]  # placeholder, len>0


# ---------------------------------------------------------------------------
# Job creation — dedupe on OUTCOME, color in the wake key
# ---------------------------------------------------------------------------

class TestJobCreation:
    def test_create_wake_returns_id(self, mock_conn):
        mock_conn.cursor.return_value.fetchone.return_value = (42,)
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            job_id = j.create_wake("o/r", "abc" * 40, True,
                                   pr_number=None, payload={"repo": "o/r"})
            assert job_id == 42

    def test_create_wake_conflict_target_has_color(self, mock_conn):
        """GLM 3: the wake key includes wake_color, so a green verdict after a
        red triage of the same head is a NEW job."""
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            j.create_wake("o/r", "abc" * 40, True)
            sql = mock_conn.execute.call_args[0][0]
            assert "ON CONFLICT (repo, head_sha, wake_color)" in sql
            assert "WHERE kind = 'wake'" in sql

    def test_create_wake_rearm_only_terminal(self, mock_conn):
        """dedupe on outcome: only done/parked/superseded (or expired-running)
        rows are re-armed; an active queued/running row suppresses."""
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            j.create_wake("o/r", "abc" * 40, False)
            sql = mock_conn.execute.call_args[0][0]
            assert "state IN ('done', 'parked', 'superseded')" in sql
            assert "lease_expires < now()" in sql

    def test_create_builder_key_includes_round(self, mock_conn):
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            j.create_builder("o/r", 3, "abc" * 40, 1)
            sql = mock_conn.execute.call_args[0][0]
            assert "ON CONFLICT (repo, pr_number, builder_round)" in sql
            assert "WHERE kind = 'builder'" in sql


class TestJobLifecycle:
    def test_claim_by_id(self, mock_conn):
        mock_conn.execute.return_value.fetchone.return_value = (42,)
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            assert j.claim(42, pr_number=7) == 42
            sql, params = mock_conn.execute.call_args[0]
            assert "attempt = attempt + 1" in sql
            assert "state = 'queued'" in sql
            assert params[0] == 7

    def test_resolve_terminal_only(self, mock_conn):
        mock_conn.execute.return_value.rowcount = 1
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            assert j.resolve(42, "done") is True
            # no_installation is the terminal refusal for a repo the GitHub App
            # is not installed on (never retried).
            assert j.resolve(42, "no_installation") is True
            for bad in ("failed", "retry", ""):
                # A real check, not an assert (asserts vanish under python -O).
                with pytest.raises(ValueError):
                    j.resolve(42, bad)
            assert TERMINAL_OUTCOMES == (
                "done", "no_change", "no_installation", "superseded", "parked")

    def test_retry_returns_queued_and_parks_at_cap(self, mock_conn):
        """failed(retryable) goes back to queued; at the cap it parks."""
        mock_conn.execute.return_value.fetchone.return_value = (2, 3)
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            assert j.retry(42, reason="boom") == "queued"
            sql, params = mock_conn.execute.call_args[0]
            assert "state = 'queued'" in sql and "outcome = 'failed'" in sql
            # at cap -> park
            mock_conn.execute.return_value.fetchone.return_value = (3, 3)
            assert j.retry(42, reason="boom") == "parked"

    def test_park_from_any_state(self, mock_conn):
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            j.park(42, reason="done")
            sql = mock_conn.execute.call_args[0][0]
            assert "outcome = 'parked'" in sql
            assert "state = 'running'" not in sql  # park is state-agnostic

    def test_claim_queued_newest_first(self, mock_conn):
        """GLM 7: the claim is ORDER BY created_at DESC — newest round wins."""
        mock_conn.cursor.return_value.fetchone.return_value = (1,) * 14
        mock_conn.cursor.return_value.description = [
            ("id", None, None, None, None, None, None)] * 14
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            j.claim_queued("builder")
            sql = mock_conn.execute.call_args[0][0]
            assert "ORDER BY created_at DESC, id DESC LIMIT 1" in sql
            assert "FOR UPDATE SKIP LOCKED" in sql

    def test_peek_queued_is_read_only(self, mock_conn):
        """GLM 9: dry-run peek must be a SELECT, never an UPDATE."""
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            j.peek_queued("builder")
            sql = mock_conn.execute.call_args[0][0]
            assert sql.lstrip().upper().startswith("SELECT")
            assert "UPDATE NEO_JOBS SET" not in sql.upper()  # no claim/lease mutation
            assert "DELETE" not in sql.upper()

    def test_supersede_older_rounds(self, mock_conn):
        """GLM 7: supersede marks OLDER rounds (builder_round < claimed)."""
        mock_conn.cursor.return_value.fetchall.return_value = [(1,), (2,)]
        mock_conn.cursor.return_value.description = [("id", None, None, None, None, None, None)]
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            ids = j.supersede_older("o/r", 3, 2)
            assert ids == [1, 2]
            sql, params = mock_conn.execute.call_args[0]
            assert "builder_round < %s" in sql
            assert "state = 'queued'" in sql


class TestSchemaSingleDefinition:
    def test_jobs_holds_one_schema(self):
        assert "CREATE TABLE IF NOT EXISTS neo_jobs" in JOBS_SCHEMA
        assert "wake_color" in JOBS_SCHEMA
        assert "builder_round" in JOBS_SCHEMA
        assert "UNIQUE (repo, pr_number, head_sha)" not in JOBS_SCHEMA  # legacy gone

    def test_neo_state_delegates_not_defines(self):
        """Only ONE schema definition for neo_jobs — neo_state no longer
        defines the table; its ensure_schema() calls jobs.ensure_schema()."""
        from src.neo_state import SCHEMA as STATE_SCHEMA, NeoState
        assert "CREATE TABLE IF NOT EXISTS neo_jobs" not in STATE_SCHEMA
        src = inspect.getsource(NeoState.ensure_schema)
        assert "Jobs(self.dsn).ensure_schema()" in src

    def test_ensure_schema_runs_migration(self, mock_conn):
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            Jobs("postgresql://fake:5432/db").ensure_schema()
            assert mock_conn.execute.call_count == 3  # SCHEMA, MIGRATION, INDEXES


# ---------------------------------------------------------------------------
# Defect 1 — unbounded fixer loop: round cap -> parked
# ---------------------------------------------------------------------------

class TestDefect1RoundCap:
    def _row(self, round_, id_=1):
        return {"id": id_, "repo": "o/r", "pr_number": 5, "head_sha": "a" * 40,
                "builder_round": round_, "attempt": round_}

    def test_dispatch_one_parks_when_over_cap(self):
        from src.config import Config
        cfg = Config()
        cfg.max_builder_rounds = 2
        jobs = MagicMock(spec=Jobs)
        with patch("src.neo_builder.pr_state", return_value="OPEN\tmain"):
            dispatch_one(cfg, jobs, self._row(round_=3))
        jobs.park.assert_called_once()
        jobs.resolve.assert_not_called()

    def test_dispatch_one_dispatches_at_cap(self, tmp_path):
        """round == max is allowed (rounds are 1-indexed: max=2 -> rounds 1,2)."""
        from src.config import Config
        cfg = Config()
        cfg.max_builder_rounds = 2
        cfg.gh_token = "test-token"
        jobs = MagicMock(spec=Jobs)
        with patch("src.neo_builder.BRIEFS", tmp_path / "briefs"), \
             patch("src.neo_builder.pr_state", return_value="OPEN\tmain"), \
             patch("src.neo_builder.prepare_worktree", return_value=("/tmp/wt", "main")), \
             patch("src.neo_builder.head_moved", return_value=None), \
             patch("src.neo_builder.fetch_findings", return_value=("{\"findings\": []}", None)), \
             patch("src.neo_builder.gh", return_value=(0, "a" * 40)), \
             patch.object(BuilderAgent, "run",
                          return_value=({"input_tokens": 0}, "FIXED")):
            dispatch_one(cfg, jobs, self._row(round_=2))
        assert jobs.resolve.call_count >= 1
        assert jobs.resolve.call_args[0][1] == "done"


# ---------------------------------------------------------------------------
# Defect 2 — transient gh failure must not consume a round
# ---------------------------------------------------------------------------

class TestDefect2PrState:
    def test_empty_pr_state_is_not_skipped(self):
        """Empty pr_state -> dispatch_one returns early WITHOUT resolving; the
        job stays running for lease recovery (GLM 8: never terminal 'failed')."""
        from src.config import Config
        cfg = Config()
        cfg.gh_token = "t"
        cfg.max_builder_rounds = 3
        jobs = MagicMock(spec=Jobs)
        row = {"id": 1, "repo": "o/r", "pr_number": 5, "head_sha": "a" * 40,
               "builder_round": 1, "attempt": 1}
        with patch("src.neo_builder.pr_state", return_value=""):
            dispatch_one(cfg, jobs, row)
        assert jobs.resolve.call_count == 0
        assert jobs.park.call_count == 0
        assert jobs.retry.call_count == 0

    def test_pr_state_returns_empty_on_gh_failure(self):
        with patch("src.neo_builder.gh", return_value=(1, "error")):
            assert pr_state(MagicMock(), "o/r", 1) == ""

    def test_closed_merged_resolves_no_change(self):
        from src.config import Config
        cfg = Config()
        cfg.gh_token = "t"
        cfg.max_builder_rounds = 3
        jobs = MagicMock(spec=Jobs)
        row = {"id": 1, "repo": "o/r", "pr_number": 5, "head_sha": "a" * 40,
               "builder_round": 1, "attempt": 1}
        with patch("src.neo_builder.pr_state", return_value="MERGED\tmain"):
            dispatch_one(cfg, jobs, row)
        jobs.resolve.assert_called_once_with(
            1, "no_change", worker_token=None, payload={"reason": "PR state: MERGED"})


# ---------------------------------------------------------------------------
# Defect 3 / GLM 4 — any agent-spawn exception releases the job
# ---------------------------------------------------------------------------

class TestDefect3AgentErrors:
    def _run(self, side_effect, jobs):
        from src.neo_worker import run_neo
        from src.config import Config
        cfg = Config()
        cfg.gh_token = "t"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.max_builder_rounds = 2
        cfg.auto_merge_default = False
        cfg.deepseek_timeout = 30
        state = MagicMock()
        state.builder_rounds.return_value = 0
        state.install.return_value = {"auto_merge": False, "plan": None, "credits_balance": 100}
        state.already_acted.return_value = False   # per-SHA idempotency gate must not short-circuit
        state.already_escalated.return_value = False  # PR has not escalated yet
        wake = {"repo": "o/r", "sha": "a" * 40, "green": True,
                "context": "test", "pr_number": 1, "job_id": 42}
        # Provide a realistic return for jobs.get (Defect 5: attempt_number).
        if hasattr(jobs, 'get') and not callable(getattr(jobs, 'get', None)):
            jobs.get = MagicMock(return_value={"attempt": 1})
        with patch("src.neo_worker.resolve_pr", return_value=1), \
             patch("src.neo_worker._review_evidence", return_value={
                 "ok": True, "head_sha": "a" * 40, "decided_sha": "a" * 40,
                 "review_sha": "a" * 40, "review_bound_ok": True,
                 "trusted_states": ["success"], "trusted_green": True,
                 "failing_checks": 0, "rollup_green": True}), \
             patch("src.neo_worker._jev_for_pr"), \
             patch("src.neo_worker._run_agent", side_effect=side_effect):
            run_neo(cfg, state, jobs, wake)
        return jobs

    def test_file_not_found_releases_job(self):
        jobs = MagicMock(spec=Jobs)
        jobs.create_wake.return_value = 42
        jobs.claim.return_value = 42
        self._run(FileNotFoundError("agent binary not found"), jobs)
        jobs.retry.assert_called_once()
        assert "agent crashed" in jobs.retry.call_args.kwargs["reason"]

    def test_oserror_releases_job(self):
        jobs = MagicMock(spec=Jobs)
        jobs.create_wake.return_value = 42
        jobs.claim.return_value = 42
        self._run(OSError("permission denied"), jobs)
        jobs.retry.assert_called_once()

    def test_no_verdict_releases_job(self):
        jobs = MagicMock(spec=Jobs)
        jobs.create_wake.return_value = 42
        jobs.claim.return_value = 42
        # _run_agent returns empty usage + empty verdict (AgentError path)
        self._run(lambda *a, **kw: ({}, "", None, None), jobs)
        jobs.retry.assert_called_once()


# ---------------------------------------------------------------------------
# Defect 4 — wakes persisted before the executor; nothing dropped
# ---------------------------------------------------------------------------

class TestDefect4DroppedWakes:
    def test_on_verdict_persists_before_submit(self):
        import src.__main__ as main_mod
        source = open(main_mod.__file__).read()
        assert "jobs.create_wake(repo, sha, green" in source
        assert "Semaphore" not in source  # backpressure cap removed

    def test_status_wake_without_pr_creates_job(self):
        """GLM 1: on_verdict passes pr_number (may be None) into create_wake —
        a `status` event with no PR number still gets a durable job."""
        import src.__main__ as main_mod
        source = open(main_mod.__file__).read()
        assert "pr_number=wake.get(\"pr_number\")" in source

    def test_recovery_loop_resubmits_expired_and_orphaned(self):
        """GLM 4: __main__ runs a lease-recovery loop that re-queues expired
        wake leases and re-submits orphaned queued wakes (no strand)."""
        import src.__main__ as main_mod
        source = open(main_mod.__file__).read()
        assert "jobs.recover_expired(\"wake\")" in source
        assert "jobs.stale_queued(\"wake\"" in source
        assert "_submit_wake(executor, jobs, wake, _run_wake)" in source

    def test_webhook_status_wake_has_no_pr(self):
        """The status webhook deliberately carries no pr_number; on_verdict
        creates the job anyway (pr resolved later via the head SHA)."""
        from src.config import Config
        from src.webhook_server import NeoDispatcher
        cfg = Config()
        cfg.orgs = ("o",)
        cfg.repo_allowlist = ()
        cfg.trusted_review_contexts = ("rhobear-reviews",)
        seen = []
        d = NeoDispatcher(cfg, seen.append)
        out = d.dispatch("status", {
            "repository": {"full_name": "o/r"},
            "context": "rhobear-reviews", "state": "success",
            "sha": "a" * 40,
        })
        assert out.code == 200
        assert len(seen) == 1
        assert "pr_number" not in seen[0] or seen[0].get("pr_number") is None
        assert seen[0]["repo"] == "o/r" and seen[0]["green"] is True


# ---------------------------------------------------------------------------
# Defect 5 — rollup_green computed from the rollup
# ---------------------------------------------------------------------------

class TestDefect5RollupGreen:
    def test_rollup_green_computed_from_rollup(self):
        from src.neo_worker import _review_evidence
        source = inspect.getsource(_review_evidence)
        assert "rollup_green" in source
        assert "SUCCESS" in source
        assert "statusCheckRollup" in source


# ---------------------------------------------------------------------------
# Defect 6 / GLM 9 — dry-run sleeps every iteration and is read-only
# ---------------------------------------------------------------------------

class TestDefect6DryRun:
    def test_dry_run_uses_read_only_peek(self):
        import src.neo_builder as builder
        source = open(builder.__file__).read()
        dry_section = source[source.index("if dry:"):source.index("continue", source.index("if dry:"))]
        assert "jobs.peek_queued" in dry_section
        assert "time.sleep(POLL_SECS)" in dry_section


# ---------------------------------------------------------------------------
# GLM 2 — a BOUNCE verdict creates the builder job
# ---------------------------------------------------------------------------

class TestGLM2BuilderCreated:
    def test_bounce_creates_builder_job(self):
        from src.neo_worker import run_neo
        from src.config import Config
        cfg = Config()
        cfg.gh_token = "t"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.max_builder_rounds = 2
        cfg.auto_merge_default = False
        cfg.deepseek_timeout = 30
        state = MagicMock()
        state.builder_rounds.return_value = 1   # one round already spent
        state.install.return_value = {"auto_merge": False, "plan": None, "credits_balance": 100}
        state.already_acted.return_value = False   # per-SHA idempotency gate must not short-circuit
        state.already_escalated.return_value = False  # PR has not escalated yet
        jobs = MagicMock(spec=Jobs)
        jobs.create_wake.return_value = 42
        jobs.claim.return_value = 42
        wake = {"repo": "o/r", "sha": "a" * 40, "green": False,
                "context": "test", "pr_number": 3, "job_id": 42}
        with patch("src.neo_worker.resolve_pr", return_value=3), \
             patch("src.neo_worker._review_evidence", return_value={
                 "ok": True, "head_sha": "a" * 40, "decided_sha": "a" * 40,
                 "review_sha": "a" * 40, "review_bound_ok": True,
                 "trusted_states": ["success"], "trusted_green": True,
                 "failing_checks": 0, "rollup_green": True}), \
             patch("src.neo_worker._jev_for_pr"), \
             patch("src.neo_worker._run_agent",
                   return_value=({"input_tokens": 1}, "BOUNCE-BUILDER fix it", 0, None)):
            run_neo(cfg, state, jobs, wake)
        jobs.create_builder.assert_called_once()
        call = jobs.create_builder.call_args
        assert call.args[0:4] == ("o/r", 3, "a" * 40, 2)   # rounds+1 = 2
        jobs.resolve.assert_called_once_with(42, "done", worker_token=ANY, payload=ANY)


# ---------------------------------------------------------------------------
# GLM 5 — unresolvable PR is released retryably
# ---------------------------------------------------------------------------

class TestGLM5UnresolvablePR:
    def test_no_pr_claims_then_retries(self):
        from src.neo_worker import run_neo
        from src.config import Config
        cfg = Config()
        cfg.gh_token = "t"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.max_builder_rounds = 2
        cfg.auto_merge_default = False
        cfg.deepseek_timeout = 30
        state = MagicMock()
        jobs = MagicMock(spec=Jobs)
        jobs.create_wake.return_value = 42
        jobs.claim.return_value = 42
        wake = {"repo": "o/r", "sha": "a" * 40, "green": True,
                "context": "test", "job_id": 42}
        with patch("src.neo_worker.resolve_pr", return_value=None):
            run_neo(cfg, state, jobs, wake)
        jobs.claim.assert_called_once()
        jobs.retry.assert_called_once_with(42, worker_token=ANY, reason="no open PR for head sha")
        state.record.assert_not_called()


# ---------------------------------------------------------------------------
# GLM 6 — lease renewal wraps the long agent runs
# ---------------------------------------------------------------------------

class TestGLM6LeaseRenewal:
    def test_run_neo_renews_lease_around_agent(self):
        from src.neo_worker import run_neo
        from src.config import Config
        cfg = Config()
        cfg.gh_token = "t"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.max_builder_rounds = 2
        cfg.auto_merge_default = False
        cfg.deepseek_timeout = 30
        state = MagicMock()
        state.builder_rounds.return_value = 0
        state.install.return_value = {"auto_merge": False, "plan": None, "credits_balance": 100}
        state.already_acted.return_value = False   # per-SHA idempotency gate must not short-circuit
        state.already_escalated.return_value = False  # PR has not escalated yet
        jobs = MagicMock(spec=Jobs)
        jobs.create_wake.return_value = 42
        jobs.claim.return_value = 42
        jobs.lease_renewal.return_value.__enter__ = MagicMock()
        jobs.lease_renewal.return_value.__exit__ = MagicMock(return_value=False)
        wake = {"repo": "o/r", "sha": "a" * 40, "green": True,
                "context": "test", "pr_number": 3, "job_id": 42}
        with patch("src.neo_worker.resolve_pr", return_value=3), \
             patch("src.neo_worker._review_evidence", return_value={
                 "ok": True, "head_sha": "a" * 40, "decided_sha": "a" * 40,
                 "review_sha": "a" * 40, "review_bound_ok": True,
                 "trusted_states": ["success"], "trusted_green": True,
                 "failing_checks": 0, "rollup_green": True}), \
             patch("src.neo_worker._jev_for_pr"), \
             patch("src.neo_worker._run_agent",
                   return_value=({"input_tokens": 1}, "ACCEPT-READY", 0, None)):
            run_neo(cfg, state, jobs, wake)
        assert jobs.lease_renewal.call_count == 1
        assert jobs.lease_renewal.call_args[0][0] == 42

    def test_dispatch_one_renews_lease_around_agent(self, tmp_path):
        from src.config import Config
        cfg = Config()
        cfg.max_builder_rounds = 3
        cfg.gh_token = "t"
        jobs = MagicMock(spec=Jobs)
        row = {"id": 1, "repo": "o/r", "pr_number": 5, "head_sha": "a" * 40,
               "builder_round": 1, "attempt": 1}
        with patch("src.neo_builder.pr_state", return_value="OPEN\tmain"), \
             patch("src.neo_builder.prepare_worktree", return_value=("/tmp/wt", "main")), \
             patch("src.neo_builder.head_moved", return_value=None), \
             patch("src.neo_builder.fetch_findings", return_value=("{\"findings\": []}", None)), \
             patch("src.neo_builder.gh", return_value=(0, "a" * 40)), \
             patch("src.neo_builder.BRIEFS", tmp_path / "briefs"), \
             patch.object(BuilderAgent, "run",
                          return_value=({"input_tokens": 0}, "FIXED")):
            dispatch_one(cfg, jobs, row)
        assert jobs.lease_renewal.call_count == 1


# ---------------------------------------------------------------------------
# GLM 8 — transient pr_state leaves the job running (never terminal 'failed')
# ---------------------------------------------------------------------------

class TestGLM8TransientPrState:
    def test_main_loop_does_not_resolve_failed(self):
        import src.neo_builder as builder
        source = open(builder.__file__).read()
        assert 'resolve(row["id"], "failed"' not in source
        assert "job stays running" in source


# ---------------------------------------------------------------------------
# Round cap enforced at the claim in main() (B1 brief)
# ---------------------------------------------------------------------------

class TestRoundCapAtClaim:
    def test_main_parks_over_cap(self, monkeypatch, caplog):
        """The real builder main() loop parks a claimed job whose round exceeds
        max_builder_rounds — it is never auto-dispatched."""
        import logging
        import sys
        import src.neo_builder as builder
        cfg = MagicMock()
        cfg.max_builder_rounds = 2
        cfg.repo_allowlist = ()
        cfg.gh_token = "t"
        cfg.hermes_bin = "/fake/hermes"
        cfg.hermes_provider = "zai"
        cfg.hermes_model = "glm-5.3-flash"
        cfg.deepseek_timeout = 30
        jobs = MagicMock(spec=Jobs)
        over_cap = {"id": 10, "repo": "o/r", "pr_number": 5,
                    "head_sha": "a" * 40, "builder_round": 3, "attempt": 3}
        jobs.claim_queued.side_effect = [over_cap, None]
        jobs.recover_expired.return_value = []
        monkeypatch.setattr(builder, "Config", lambda: cfg)
        monkeypatch.setattr(builder, "Jobs", lambda dsn: jobs)
        monkeypatch.setattr(builder, "pr_state", lambda *a: "OPEN\tmain")
        monkeypatch.setattr(sys, "argv", ["neo_builder", "--once"])
        with caplog.at_level(logging.INFO):
            builder.main()
        jobs.park.assert_any_call(
            10, reason="max builder rounds (2) reached (round 3)")
        assert jobs.claim_queued.call_count == 2  # over-cap row + empty
        assert not jobs.resolve.called

    def test_main_never_dispatches_second_round_of_inflight_pr(self, monkeypatch, caplog):
        """Integrator finding 1: two rounds of the SAME PR share one worktree
        (prepare_worktree keys its dest on repo+pr).  If a round is already
        running in another worker, a newly claimed round for the same PR must
        be RELEASED (not dispatched, not attempt-burning) so a second agent can
        never rm -rf the live worktree."""
        import logging
        import sys
        import src.neo_builder as builder
        cfg = MagicMock()
        cfg.max_builder_rounds = 3
        cfg.repo_allowlist = ()
        cfg.gh_token = "t"
        cfg.hermes_bin = "/fake/hermes"
        cfg.hermes_provider = "zai"
        cfg.hermes_model = "glm-5.3-flash"
        cfg.deepseek_timeout = 30
        jobs = MagicMock(spec=Jobs)
        row1 = {"id": 1, "repo": "o/r", "pr_number": 5, "head_sha": "a" * 40,
                "builder_round": 1, "attempt": 1}
        row2 = {"id": 2, "repo": "o/r", "pr_number": 5, "head_sha": "b" * 40,
                "builder_round": 2, "attempt": 1}
        jobs.claim_queued.side_effect = [row1, row2, None]
        jobs.recover_expired.return_value = []
        jobs.supersede_older.return_value = []
        dispatched: list[int] = []

        def fake_dispatch(cfg_, jobs_, row_):
            dispatched.append(row_["id"])

        monkeypatch.setattr(builder, "Config", lambda: cfg)
        monkeypatch.setattr(builder, "Jobs", lambda dsn: jobs)
        monkeypatch.setattr(builder, "pr_state", lambda *a: "OPEN\tmain")
        monkeypatch.setattr(builder, "dispatch_one", fake_dispatch)
        monkeypatch.setattr(sys, "argv", ["neo_builder", "--once"])
        with caplog.at_level(logging.INFO):
            builder.main()
        # round 2 of the same PR was claimed but released — never dispatched
        jobs.release.assert_called_once()
        assert jobs.release.call_args[0][0] == 2
        assert dispatched == [1]
        assert "already inflight" in caplog.text


# ---------------------------------------------------------------------------
# Hermes Builder invocation unchanged
# ---------------------------------------------------------------------------

class TestHermesBuilderUnchanged:
    def test_builder_agent_subclass(self):
        agent = BuilderAgent(work_dir="/tmp/wt", hermes_bin="/opt/rhobear-hermes/bin/hermes",
                             provider="zai", model="glm-5.3-flash")
        assert hasattr(agent, "run") and hasattr(agent, "_build_cmd")
        assert hasattr(agent, "_build_env") and hasattr(agent, "_parse_output")

    def test_builder_hermes_command_line(self):
        agent = BuilderAgent(work_dir="/tmp/wt", hermes_bin="/opt/rhobear-hermes/bin/hermes",
                             provider="zai", model="glm-5.3-flash")
        cmd = agent._build_cmd("fix the bug")
        assert cmd[0] == "/opt/rhobear-hermes/bin/hermes"
        assert "-z" in cmd and cmd[cmd.index("-z") + 1] == "fix the bug"
        assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "glm-5.3-flash"
        assert "--provider" in cmd and cmd[cmd.index("--provider") + 1] == "zai"
        assert "--yolo" in cmd and "--accept-hooks" in cmd

    def test_builder_verdict_parsing(self):
        """Block 3: the builder lane parses its OWN contract verdicts
        {FIXED, NO-CHANGE, BLOCKED} — the Neo protocol set (ACCEPT-READY etc.)
        is not the fixer's contract, and a completed run with an unparseable
        verdict returns "" (no_change, never a retry) instead of raising."""
        from src.agent_hermes import CANONICAL_VERDICTS
        from src.agent_hermes import NonZeroExit, MissingResult
        from src.neo_builder import BUILDER_CANONICAL_VERDICTS
        agent = BuilderAgent(work_dir="/tmp/wt", hermes_bin="/fake/hermes",
                             provider="zai", model="glm-5.3-flash")
        # the builder's canonical set is the fixer contract, not the triage one
        assert BUILDER_CANONICAL_VERDICTS == frozenset({"FIXED", "NO-CHANGE", "BLOCKED"})
        assert "ACCEPT-READY" not in BUILDER_CANONICAL_VERDICTS
        for verdict in ("FIXED", "NO-CHANGE", "BLOCKED"):
            proc = MagicMock()
            proc.stdout = f"Work done\nVERDICT: {verdict}\n"
            proc.stderr = ""
            proc.returncode = 0
            usage, got = agent._parse_output(proc)
            assert got == verdict, (verdict, got)
            assert usage == {"input_tokens": 0, "output_tokens": 0}
        # a clean exit with the triage-lane verdict (or garbage) is a COMPLETED
        # run: "" verdict + raw_tail recorded, never an exception / retry
        proc = MagicMock()
        proc.stdout = "Work done\nVERDICT: ACCEPT-READY\n"
        proc.stderr = ""
        proc.returncode = 0
        usage, got = agent._parse_output(proc)
        assert got == ""
        assert usage["raw_tail"].endswith("VERDICT: ACCEPT-READY")
        assert "ACCEPT-READY" in CANONICAL_VERDICTS  # triage set unchanged
        # a FAILED run still raises (agent failure -> retry)
        proc = MagicMock()
        proc.stdout = "VERDICT: FIXED\n"
        proc.stderr = "error"
        proc.returncode = 1
        import pytest as _pt
        with _pt.raises(NonZeroExit):
            agent._parse_output(proc)
        proc = MagicMock()
        proc.stdout = ""
        proc.stderr = ""
        proc.returncode = 0
        with _pt.raises(MissingResult):
            agent._parse_output(proc)


# ---------------------------------------------------------------------------
# Imports / module contracts
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_jobs(self):
        from src.jobs import Jobs
        assert Jobs is not None

    def test_import_neo_builder(self):
        from src.neo_builder import BuilderAgent, build_brief, fetch_findings, \
            prepare_worktree, dispatch_one, gh, pr_state, WORKTREES, BRIEFS, \
            POLL_SECS, CONCURRENCY
        assert callable(build_brief)
        assert POLL_SECS > 0 and CONCURRENCY > 0

    def test_import_neo_worker(self):
        from src.neo_worker import run_neo, _review_evidence, merge_authority, \
            evidence_brief, _jev_for_pr, credits_for, resolve_pr, _gh, _wake_lease
        assert callable(run_neo) and callable(merge_authority)


# ---------------------------------------------------------------------------
# MEDIUM-5 — required-checks gate fails closed on unknown rollup
# ---------------------------------------------------------------------------

class TestRequiredChecksFailClosed:
    """ACCEPT + rollup_green=None (no rollup data) must be denied by
    merge_authority and must never reach the merge executor."""

    def _run(self, rollup_green):
        from src.neo_worker import run_neo
        from src.config import Config
        from src.jev_review import JevAssessment

        cfg = Config()
        cfg.max_builder_rounds = 2
        cfg.auto_merge_default = False
        cfg.deepseek_timeout = 30
        state = MagicMock()
        state.builder_rounds.return_value = 0
        state.install.return_value = {"auto_merge": True, "plan": None,
                                      "credits_balance": 100}
        state.already_acted.return_value = False
        state.already_escalated.return_value = False
        jobs = MagicMock(spec=Jobs)
        jobs.claim.return_value = True
        jobs.get.return_value = {"attempt": 1}
        wake = {"repo": "o/r", "sha": "a" * 40, "green": True,
                "context": "rhobear-reviews", "pr_number": 5, "job_id": 42}
        evidence = {
            "ok": True, "reason": "", "head_sha": "a" * 40,
            "decided_sha": "a" * 40, "review_sha": "a" * 40,
            "review_bound_ok": True, "trusted_states": ["success"],
            "trusted_green": True, "trusted_producer_ok": True,
            "review_sha_match": True, "failing_checks": 0,
            "rollup_green": rollup_green,
            "unresolved_critical_high": 0, "coverage_complete": True,
        }
        with patch("src.neo_worker.resolve_pr", return_value=5), \
             patch("src.neo_worker._review_evidence", return_value=evidence), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent",
                   return_value=({"input_tokens": 1}, "ACCEPT", 0, None)), \
             patch("src.neo_worker.ClaudeAgent"), \
             patch("src.merge_executor.merge",
                   return_value={"merged": True, "sha": "a" * 40}) as merge:
            run_neo(cfg, state, jobs, wake)
        phases = [c.args[3] for c in state.record.call_args_list
                  if len(c.args) >= 4]
        return phases, merge

    def test_unknown_rollup_is_denied_and_never_merged(self):
        phases, merge = self._run(None)
        merge.assert_not_called()
        assert "ready" in phases
        assert "merged" not in phases

    def test_green_rollup_still_merges(self):
        phases, merge = self._run(True)
        merge.assert_called_once()
        assert "merged" in phases


# ---------------------------------------------------------------------------
# MEDIUM-9 — submit before stamping submitted_at
# ---------------------------------------------------------------------------

class TestSubmitWakeOrder:
    def test_submit_happens_before_stamp(self):
        from src.__main__ import _submit_wake
        events = []

        class Exec:
            def submit(self, fn, wake):
                events.append("submit")
                return MagicMock()

        class JobsFake:
            def mark_submitted(self, job_id):
                events.append(("stamp", job_id))

            def clear_submitted(self, job_id):
                events.append(("clear", job_id))

        ok = _submit_wake(Exec(), JobsFake(),
                          {"job_id": 7, "repo": "o/r"}, lambda w: None)
        assert ok is True
        assert events == ["submit", ("stamp", 7)]

    def test_shutdown_clears_stamp_instead_of_marking(self):
        from src.__main__ import _submit_wake
        events = []

        class ShutdownExecutor:
            def submit(self, fn, wake):
                events.append("submit")
                raise RuntimeError("cannot schedule new futures after shutdown")

        class JobsFake:
            def mark_submitted(self, job_id):
                events.append(("stamp", job_id))

            def clear_submitted(self, job_id):
                events.append(("clear", job_id))

        ok = _submit_wake(ShutdownExecutor(), JobsFake(),
                          {"job_id": 7, "repo": "o/r"}, lambda w: None)
        assert ok is False
        assert events == ["submit", ("clear", 7)]


# ---------------------------------------------------------------------------
# MEDIUM-6 — worktree locks live under WORKTREES, never /tmp
# ---------------------------------------------------------------------------

class TestWorktreeLock:
    def test_lock_under_worktrees_and_appended_not_truncated(self, tmp_path,
                                                            monkeypatch):
        from src import neo_builder as nb
        import subprocess as sp

        worktrees = tmp_path / "wts"
        monkeypatch.setattr(nb, "WORKTREES", worktrees)
        lock_dir = worktrees / ".locks"
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / "acme-app-7.lock"
        lock_path.write_text("held\n")

        def fake_run(cmd, *a, **kw):
            return sp.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(nb.subprocess, "run", fake_run)
        monkeypatch.setattr(nb, "pr_state", lambda *a, **kw: "OPEN\tmain")

        dest, branch = nb.prepare_worktree("tok", "acme/app", 7)

        assert dest == str(worktrees / "acme-app-7")
        assert branch == "main"
        assert lock_path.exists()
        assert lock_path.is_relative_to(worktrees)
        assert "/tmp/rhobear-wt-locks" not in str(lock_path)
        # "a" (not "w"): an existing lock file is never truncated.
        assert lock_path.read_text() == "held\n"
        assert lock_dir.stat().st_mode & 0o777 == 0o700


# ---------------------------------------------------------------------------
# MEDIUM-10 — a failed head read after the run is not a push
# ---------------------------------------------------------------------------

class TestHeadReadAfterRun:
    def _dispatch(self, after_response, tmp_path):
        from src.config import Config
        cfg = Config()
        cfg.max_builder_rounds = 3
        cfg.gh_token = "t"
        jobs = MagicMock(spec=Jobs)
        row = {"id": 1, "repo": "o/r", "pr_number": 5, "head_sha": "a" * 40,
               "builder_round": 1, "attempt": 1}
        responses = [(0, "a" * 40), after_response]

        def fake_gh(*a, **kw):
            return responses.pop(0)

        with patch("src.neo_builder.pr_state", return_value="OPEN\tmain"), \
             patch("src.neo_builder.prepare_worktree",
                   return_value=("/tmp/wt", "main")), \
             patch("src.neo_builder.head_moved", return_value=None), \
             patch("src.neo_builder.fetch_findings",
                   return_value=('{"findings": []}', None)), \
             patch("src.neo_builder.gh", side_effect=fake_gh), \
             patch("src.neo_builder.BRIEFS", tmp_path / "briefs"), \
             patch.object(BuilderAgent, "run",
                          return_value=({"input_tokens": 0}, "FIXED")):
            dispatch_one(cfg, jobs, row)
        return jobs

    def test_gh_failure_after_run_leaves_job_running(self, tmp_path):
        jobs = self._dispatch((1, "transient gh failure"), tmp_path)
        jobs.resolve.assert_not_called()
        jobs.retry.assert_not_called()

    def test_garbage_head_after_run_leaves_job_running(self, tmp_path):
        jobs = self._dispatch((0, "not-a-sha"), tmp_path)
        jobs.resolve.assert_not_called()
        jobs.retry.assert_not_called()

    def test_unchanged_head_still_resolves_fixed_no_push(self, tmp_path):
        jobs = self._dispatch((0, "a" * 40), tmp_path)
        jobs.resolve.assert_called_once()
        assert jobs.resolve.call_args[0][1] == "done"
        assert jobs.resolve.call_args[1]["payload"]["outcome"] == "fixed-no-push"

    def test_real_push_still_resolves_done_pushed(self, tmp_path):
        jobs = self._dispatch((0, "b" * 40), tmp_path)
        jobs.resolve.assert_called_once()
        assert jobs.resolve.call_args[0][1] == "done"
        assert jobs.resolve.call_args[1]["payload"]["outcome"] == "pushed"


# ---------------------------------------------------------------------------
# MEDIUM — a failed head read BEFORE the run is not a push
# ---------------------------------------------------------------------------

class TestHeadReadBeforeRun:
    """gh failure on the pre-run head read returns stderr, not a SHA.  The
    round must not run the agent or resolve pushed=True; the job stays running
    for lease recovery (mirrors the post-run read handling)."""

    def _dispatch(self, before_response, tmp_path):
        from src.config import Config
        cfg = Config()
        cfg.max_builder_rounds = 3
        cfg.gh_token = "t"
        jobs = MagicMock(spec=Jobs)
        row = {"id": 1, "repo": "o/r", "pr_number": 5, "head_sha": "a" * 40,
               "builder_round": 1, "attempt": 1}

        def fake_gh(*a, **kw):
            return before_response

        agent_run = MagicMock(return_value=({"input_tokens": 0}, "FIXED"))
        with patch("src.neo_builder.pr_state", return_value="OPEN\tmain"), \
             patch("src.neo_builder.prepare_worktree",
                   return_value=("/tmp/wt", "main")), \
             patch("src.neo_builder.head_moved", return_value=None), \
             patch("src.neo_builder.fetch_findings",
                   return_value=('{"findings": []}', None)), \
             patch("src.neo_builder.gh", side_effect=fake_gh), \
             patch("src.neo_builder.BRIEFS", tmp_path / "briefs"), \
             patch.object(BuilderAgent, "run", agent_run):
            dispatch_one(cfg, jobs, row)
        return jobs, agent_run

    def test_gh_failure_before_run_leaves_job_running(self, tmp_path):
        jobs, agent_run = self._dispatch((1, "transient gh failure"), tmp_path)
        jobs.resolve.assert_not_called()
        jobs.retry.assert_not_called()
        agent_run.assert_not_called()

    def test_garbage_head_before_run_leaves_job_running(self, tmp_path):
        jobs, agent_run = self._dispatch((0, "not-a-sha"), tmp_path)
        jobs.resolve.assert_not_called()
        jobs.retry.assert_not_called()
        agent_run.assert_not_called()
