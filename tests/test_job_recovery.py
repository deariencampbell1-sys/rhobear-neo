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
from unittest.mock import MagicMock, patch, ANY

import pytest

from src.jobs import (
    Jobs, SCHEMA as JOBS_SCHEMA, INDEXES, MIGRATION, SCHEMA_VERSION,
    TERMINAL_OUTCOMES,
)
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

    def test_retry_below_cap_requeues(self, mock_conn):
        """failed(retryable) goes back to queued — one statement."""
        mock_conn.execute.return_value.fetchone.return_value = ("queued",)
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            assert j.retry(42, reason="boom") == "queued"
            sql, params = mock_conn.execute.call_args[0]
            assert "state = 'running'" in sql and "outcome" in sql
            assert mock_conn.execute.call_count == 1

    def test_retry_at_cap_parks_in_the_same_statement(self, mock_conn):
        """W1-13-r1 Defect 1: the at-cap decision is made in the SAME statement
        that requeues (a CASE writing done/parked directly) — there is no
        second park() statement whose window a claiming worker could fall
        into.  Returns 'parked'."""
        mock_conn.execute.return_value.fetchone.return_value = ("done",)
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            assert j.retry(42, worker_token="w1", reason="boom") == "parked"
            # exactly ONE statement — never UPDATE-then-park
            assert mock_conn.execute.call_count == 1
            sql = mock_conn.execute.call_args[0][0]
            assert "CASE" in sql and "'parked'" in sql
            assert "worker_token = %s" in sql

    def test_retry_without_a_touchable_row_returns_question(self, mock_conn):
        mock_conn.execute.return_value.fetchone.return_value = None
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            assert j.retry(42, worker_token="w1") == "?"

    def test_park_guards_accept_token_and_expected_state(self, mock_conn):
        """W1-13-r1 Defect 1: park() accepts worker_token and/or
        expected_state.  With a token it only touches the claimant's
        queued/running row; with expected_state it only touches that state."""
        mock_conn.execute.return_value.rowcount = 1
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            assert j.park(42, worker_token="w1", reason="done") is True
            sql, params = mock_conn.execute.call_args[0]
            assert "worker_token = %s" in sql
            assert "state IN ('queued', 'running')" in sql
            assert "w1" in params
            assert j.park(42, expected_state="running", reason="done") is True
            sql, params = mock_conn.execute.call_args[0]
            assert "state = %s" in sql and "running" in params

    def test_park_bare_never_touches_a_held_row(self, mock_conn):
        """The bare form (the recovery loops' call shape) must never park a
        row another worker currently holds: a live lease means a claimant."""
        mock_conn.execute.return_value.rowcount = 1
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            j.park(42, reason="lease expired 2/2 times")
            sql = mock_conn.execute.call_args[0][0]
            assert "outcome = 'parked'" in sql
            assert "lease_expires < now()" in sql  # live-lease rows are refused
            assert "state IN ('queued', 'running')" in sql

    def test_park_returns_false_when_nothing_touched(self, mock_conn):
        mock_conn.execute.return_value.rowcount = 0
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            j = Jobs("postgresql://fake:5432/db")
            assert j.park(42, expected_state="running") is False

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

    def test_neo_state_delegates_not_defines(self, monkeypatch):
        """Only ONE schema definition for neo_jobs — neo_state no longer
        defines the table: running NeoState.ensure_schema() hands the table to
        jobs.ensure_schema() and never creates it itself."""
        from src.neo_state import SCHEMA as STATE_SCHEMA, NeoState
        assert "CREATE TABLE IF NOT EXISTS neo_jobs" not in STATE_SCHEMA
        executed, delegated = [], []

        class _Cursor:
            def execute(self, sql, *args):
                executed.append(sql)

        class _Ctx:
            def __enter__(self):
                return _Cursor()

            def __exit__(self, *exc):
                return False

        class _FakeJobs:
            def __init__(self, dsn):
                self.dsn = dsn

            def ensure_schema(self):
                delegated.append(self.dsn)

        monkeypatch.setattr(NeoState, "_conn", lambda self: _Ctx())
        monkeypatch.setattr("src.jobs.Jobs", _FakeJobs)
        NeoState("postgresql://unit").ensure_schema()
        assert delegated == ["postgresql://unit"], \
            "NeoState.ensure_schema() must delegate neo_jobs to jobs.ensure_schema()"
        assert all("neo_jobs" not in sql for sql in executed), \
            "NeoState must never define neo_jobs itself"

    def test_ensure_schema_runs_migration_once(self, mock_conn):
        """No neo_schema_version row (pre-B1 upgrade OR fresh install): the
        legacy migration runs once and stamps the version — under the
        advisory lock."""
        mock_conn.execute.return_value.fetchone.return_value = None
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            Jobs("postgresql://fake:5432/db").ensure_schema()
        executed = [c.args[0] for c in mock_conn.execute.call_args_list]
        assert any("pg_advisory_lock" in s for s in executed)
        assert JOBS_SCHEMA in executed            # create the table
        assert MIGRATION in executed              # destructive — once
        assert INDEXES in executed                # keys rebuilt after columns
        assert any("INSERT INTO neo_schema_version" in s for s in executed)

    def test_ensure_schema_second_call_is_noop(self, mock_conn):
        """Version row already at SCHEMA_VERSION: the migration is GATED OFF —
        a second ensure_schema() must never delete rows or rewrite constraints
        (the poison MIGRATION would raise if it ever ran)."""
        mock_conn.execute.return_value.fetchone.return_value = (SCHEMA_VERSION,)
        with patch("src.jobs.psycopg.connect", return_value=mock_conn):
            with patch("src.jobs.MIGRATION", "SELECT migration_must_not_run()"):
                Jobs("postgresql://fake:5432/db").ensure_schema()
        executed = [c.args[0] for c in mock_conn.execute.call_args_list]
        assert "SELECT migration_must_not_run()" not in executed
        assert MIGRATION not in executed
        assert INDEXES not in executed
        assert not any("INSERT INTO neo_schema_version" in s for s in executed)
        assert any("pg_advisory_unlock" in s for s in executed)


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

    def test_crash_retry_passes_the_claim_token(self):
        """The crash retry targets THIS claim: retry(42) presents the same
        worker_token that claimed the job."""
        jobs = MagicMock(spec=Jobs)
        jobs.claim.return_value = 42
        self._run(FileNotFoundError("agent binary not found"), jobs)
        jobs.retry.assert_called_once()
        assert jobs.retry.call_args.args == (42,)
        assert jobs.retry.call_args.kwargs["worker_token"] == \
            jobs.claim.call_args.kwargs["worker_token"]


# ---------------------------------------------------------------------------
# Defect 4 — wakes persisted before the executor; nothing dropped
# ---------------------------------------------------------------------------

class _ScriptedJobs:
    """Fake Jobs for driving __main__.main() and its _recovery_loop closure
    without a DB.  Script `expired` / `orphaned` / `dedup` per scenario;
    every call is appended to `events` in order."""
    events: list = []
    expired: list = []
    orphaned: list = []
    dedup: bool = False

    def __init__(self, dsn):
        pass

    def ensure_schema(self):
        pass

    def reset_queued_submissions(self, kind):
        self.events.append(("reset", kind))

    def create_wake(self, repo, sha, green, pr_number=None, payload=None):
        self.events.append(("create_wake", repo, sha, pr_number))
        if self.dedup:
            return None
        return len([e for e in self.events if e[0] == "create_wake"])

    def mark_submitted(self, job_id):
        self.events.append(("mark_submitted", job_id))

    def clear_submitted(self, job_id):
        self.events.append(("clear_submitted", job_id))

    def recover_expired(self, kind):
        self.events.append(("recover_expired", kind))
        return [dict(r) for r in self.expired]

    def stale_queued(self, kind, older_than_secs):
        self.events.append(("stale_queued", kind, older_than_secs))
        return [dict(r) for r in self.orphaned]

    def park(self, job_id, **kw):
        self.events.append(("park", job_id, kw.get("reason")))
        return True


class TestScriptedJobsIsolation:
    def test_two_scripted_jobs_instances_do_not_share_events(self):
        """W1-13-r2 defect 7: script state is per-INSTANCE (the script passed
        through the Jobs factory lambda), never on the class — two fakes never
        see each other's events, so one test can never leak rows into the
        next."""
        a = _ScriptedJobs("dsn")
        b = _ScriptedJobs("dsn")
        assert a.create_wake("o/r", "a" * 40, True) == 1
        assert [e[0] for e in a.events] == ["create_wake"]
        assert b.events == []


def _drive_wake_main(monkeypatch, events):
    """Run __main__.main() with fakes (no DB, no threads, no network) and
    return the REAL (on_verdict, _recovery_loop) closures it builds."""
    import concurrent.futures
    import src.__main__ as main_mod

    captured = {}
    cfg = MagicMock()
    cfg.orgs = ("o",)
    cfg.escalation_poll_secs = 3600.0
    cfg.escalation_timeout_secs = 60

    class _FakeState:
        def __init__(self, dsn):
            pass

        def ensure_schema(self):
            pass

    class _FakeUsage:
        def __init__(self, dsn):
            pass

        def ensure_schema(self):
            pass

    class _FakeExecutor:
        def __init__(self, **kw):
            pass

        def submit(self, fn, wake):
            events.append(("submit", wake.get("job_id")))

    class _FakeThread:
        def __init__(self, target=None, **kw):
            captured["recovery"] = target

        def start(self):
            pass

    _ScriptedJobs.events = events
    monkeypatch.setattr("src.__main__._config.load", lambda: cfg)
    monkeypatch.setattr("src.__main__.github_app_for", lambda c: {"id": 1})
    monkeypatch.setattr("src.__main__.NeoState", _FakeState)
    monkeypatch.setattr("src.__main__.Jobs", lambda dsn: _ScriptedJobs(dsn))
    monkeypatch.setattr("src.__main__.UsageStore", _FakeUsage)
    monkeypatch.setattr("src.__main__.run_neo", lambda *a, **kw: None)
    monkeypatch.setattr("src.__main__.serve", lambda *a, **kw: None)
    monkeypatch.setattr("src.__main__.NeoDispatcher",
                        lambda c, cb: captured.setdefault("on_verdict", cb))
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _FakeExecutor)
    monkeypatch.setattr("threading.Thread", _FakeThread)
    main_mod.main()
    return captured["on_verdict"], captured["recovery"]


class TestDefect4DroppedWakes:
    def test_on_verdict_persists_before_submit(self, monkeypatch):
        """The wake is persisted as a durable job BEFORE the executor ever
        sees it — and no backpressure cap drops any of them (32/32 handed
        off, each exactly once)."""
        events = []
        on_verdict, _ = _drive_wake_main(monkeypatch, events)
        for i in range(32):
            on_verdict({"repo": "o/r", "sha": f"{i:040x}", "green": True,
                        "context": "test", "pr_number": 1})
        assert [e[0] for e in events] == ["create_wake", "submit", "mark_submitted"] * 32

    def test_duplicate_wake_is_never_submitted(self, monkeypatch):
        """A deduped wake (another job owns the SHA) is never handed off."""
        events = []
        _ScriptedJobs.dedup = True
        try:
            on_verdict, _ = _drive_wake_main(monkeypatch, events)
            on_verdict({"repo": "o/r", "sha": "a" * 40, "green": True,
                        "context": "test", "pr_number": 1})
        finally:
            _ScriptedJobs.dedup = False
        assert [e[0] for e in events] == ["create_wake"]

    def test_status_wake_without_pr_creates_job(self, monkeypatch):
        """GLM 1: a `status` wake carries no pr_number — the durable job is
        created anyway (pr resolved later via the head SHA) and submitted."""
        events = []
        on_verdict, _ = _drive_wake_main(monkeypatch, events)
        on_verdict({"repo": "o/r", "sha": "a" * 40, "green": True,
                    "context": "test"})
        assert ("create_wake", "o/r", "a" * 40, None) in events
        assert ("submit", 1) in events and ("mark_submitted", 1) in events

    def test_recovery_loop_resubmits_expired_and_orphaned(self, monkeypatch):
        """GLM 4: one recovery pass parks ONLY the wake that is out of
        attempts and re-submits everything else exactly once (expired leases
        below the cap AND orphaned queued wakes)."""
        import src.__main__ as main_mod
        events = []
        _ScriptedJobs.events = events
        _ScriptedJobs.expired = [
            {"id": 1, "repo": "o/r", "head_sha": "a" * 40,
             "attempt": 2, "max_attempts": 2,
             "payload": {"repo": "o/r", "sha": "a" * 40}},
            {"id": 2, "repo": "o/r", "head_sha": "b" * 40,
             "attempt": 1, "max_attempts": 3,
             "payload": {"repo": "o/r", "sha": "b" * 40}},
        ]
        _ScriptedJobs.orphaned = [
            {"id": 3, "repo": "o/r", "head_sha": "c" * 40,
             "payload": {"repo": "o/r", "sha": "c" * 40}},
        ]
        try:
            _, recovery = _drive_wake_main(monkeypatch, events)
            sleeps = []

            def fake_sleep(secs):
                sleeps.append(secs)
                if len(sleeps) >= 2:
                    raise KeyboardInterrupt

            monkeypatch.setattr("time.sleep", fake_sleep)
            monkeypatch.setattr("src.escalation.recover_escalations",
                                lambda *a, **kw: None)
            with pytest.raises(KeyboardInterrupt):
                recovery()
        finally:
            _ScriptedJobs.expired = []
            _ScriptedJobs.orphaned = []
        # out-of-attempts wake: parked once (by name), never re-submitted
        assert [e for e in events if e[0] == "park"] == [
            ("park", 1, "lease expired 2/2 times")]
        # below-cap expired wake + orphan: each submitted exactly once
        assert [e for e in events if e[0] == "submit"] == [("submit", 2), ("submit", 3)]
        assert ("mark_submitted", 2) in events and ("mark_submitted", 3) in events
        assert ("reset", "wake") in events           # startup un-strand
        assert ("recover_expired", "wake") in events
        assert ("stale_queued", "wake", main_mod.RECOVERY_SECS) in events
        assert sleeps == [main_mod.RECOVERY_SECS, main_mod.RECOVERY_SECS]

    def test_recovery_loop_does_not_park_a_recoverable_wake(self, monkeypatch):
        """A wake with attempts left is never parked — it is re-submitted
        exactly once."""
        events = []
        _ScriptedJobs.events = events
        _ScriptedJobs.expired = [
            {"id": 7, "repo": "o/r", "head_sha": "d" * 40,
             "attempt": 1, "max_attempts": 3,
             "payload": {"repo": "o/r", "sha": "d" * 40}},
        ]
        try:
            _, recovery = _drive_wake_main(monkeypatch, events)
            sleeps = []

            def fake_sleep(secs):
                sleeps.append(secs)
                if len(sleeps) >= 2:
                    raise KeyboardInterrupt

            monkeypatch.setattr("time.sleep", fake_sleep)
            monkeypatch.setattr("src.escalation.recover_escalations",
                                lambda *a, **kw: None)
            with pytest.raises(KeyboardInterrupt):
                recovery()
        finally:
            _ScriptedJobs.expired = []
        assert not [e for e in events if e[0] == "park"]
        assert [e for e in events if e[0] == "submit"] == [("submit", 7)]

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
    def _evidence(self, rollup):
        """Run the REAL _review_evidence against a scripted gh CLI."""
        import json as _json
        from src.neo_worker import _review_evidence
        from src.config import Config
        cfg = Config()
        cfg.trusted_reviewer_app_id = "12345"
        cfg.trusted_review_contexts = ("rhobear-reviews",)
        sha = "a" * 40

        def fake_gh(token, *args, **kw):
            if args[0] == "pr":
                return 0, _json.dumps({"headRefOid": sha, "reviews": [],
                                       "statusCheckRollup": rollup})
            if "check-runs" in args[1]:
                run = {"name": "rhobear-reviews", "head_sha": sha,
                       "app": {"id": "12345"}, "status": "completed",
                       "conclusion": "success",
                       "output": {"summary": '{"unresolved_critical_high": 0, '
                                             '"coverage": "complete"}'}}
                return 0, _json.dumps({"check_runs": [run], "total_count": 1})
            return 0, _json.dumps({"statuses": [
                {"context": "rhobear-reviews", "state": "success"}]})

        with patch("src.neo_worker._gh", side_effect=fake_gh):
            return _review_evidence("tok", cfg, "o/r", 5, sha)

    def test_rollup_green_computed_from_the_rollup(self):
        """rollup_green is computed from the branch-protection check rollup:
        green only when every check is SUCCESS, red on any FAILURE, and
        UNKNOWN (None) when there is no rollup data at all."""
        ev = self._evidence([{"context": "ci/test", "conclusion": "SUCCESS"}])
        assert ev["rollup_green"] is True
        assert ev["required_checks_pass"] is True
        ev = self._evidence([{"context": "ci/test", "conclusion": "FAILURE"}])
        assert ev["rollup_green"] is False
        assert ev["required_checks_pass"] is False
        ev = self._evidence([])
        assert ev["rollup_green"] is None
        assert ev["required_checks_pass"] is None


# ---------------------------------------------------------------------------
# Defect 6 / GLM 9 — dry-run sleeps every iteration and is read-only
# ---------------------------------------------------------------------------

class TestDefect6DryRun:
    def _run(self, monkeypatch, argv, jobs, fake_sleep=None):
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
        monkeypatch.setattr(builder, "Config", lambda: cfg)
        monkeypatch.setattr(builder, "Jobs", lambda dsn: jobs)
        monkeypatch.setattr(sys, "argv", ["neo_builder"] + argv)
        if fake_sleep is not None:
            monkeypatch.setattr(builder.time, "sleep", fake_sleep)
        builder.main()

    def test_dry_run_is_read_only(self, monkeypatch):
        """A dry-run peeks at the queue (peek_queued) and mutates NOTHING —
        no claim, no resolve, no park, no retry, no release, no recovery."""
        jobs = MagicMock(spec=Jobs)
        jobs.peek_queued.return_value = [
            {"repo": "o/r", "pr_number": 5, "builder_round": 1, "attempt": 1}]
        self._run(monkeypatch, ["--once", "--dry-run"], jobs)
        jobs.peek_queued.assert_called_once()
        jobs.claim_queued.assert_not_called()
        jobs.resolve.assert_not_called()
        jobs.park.assert_not_called()
        jobs.retry.assert_not_called()
        jobs.release.assert_not_called()
        jobs.create_builder.assert_not_called()
        jobs.supersede_older.assert_not_called()
        jobs.recover_expired.assert_not_called()

    def test_dry_run_sleeps_every_iteration(self, monkeypatch):
        """The dry-run poll sleeps POLL_SECS on EVERY iteration (the bug was
        sleeping only on an empty queue)."""
        jobs = MagicMock(spec=Jobs)
        jobs.peek_queued.return_value = []
        sleeps = []

        def fake_sleep(secs):
            sleeps.append(secs)
            if len(sleeps) >= 3:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            self._run(monkeypatch, ["--dry-run"], jobs, fake_sleep)
        assert sleeps == [POLL_SECS] * 3
        assert jobs.peek_queued.call_count == 3


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
    def _cfg(self):
        cfg = MagicMock()
        cfg.max_builder_rounds = 3
        cfg.deepseek_timeout = 30
        cfg.hermes_bin = "/fake/hermes"
        return cfg

    def _row(self):
        return {"id": 1, "repo": "o/r", "pr_number": 5,
                "head_sha": "a" * 40, "builder_round": 1,
                "attempt": 1, "worker_token": "w1"}

    def test_transient_pr_state_never_resolves_the_job(self):
        """GLM 8: a transient empty pr_state (gh failure) leaves the job
        RUNNING for lease recovery — it is never resolved/failed terminally."""
        import src.neo_builder as builder
        jobs = MagicMock(spec=Jobs)
        with patch("src.neo_builder.repo_token", return_value="tok"), \
             patch("src.neo_builder.pr_state", return_value=""):
            builder.dispatch_one(self._cfg(), jobs, self._row())
        jobs.resolve.assert_not_called()
        jobs.retry.assert_not_called()
        jobs.park.assert_not_called()

    def test_transient_github_error_never_resolves_the_job(self):
        """A transient GitHub error (token mint / 5xx) also leaves the job
        running — never a terminal resolve."""
        from src.github_app import GitHubAppError
        import src.neo_builder as builder
        jobs = MagicMock(spec=Jobs)
        with patch("src.neo_builder.repo_token",
                   side_effect=GitHubAppError("502 bad gateway")), \
             patch("src.neo_builder.pr_state", return_value="OPEN\tmain"):
            builder.dispatch_one(self._cfg(), jobs, self._row())
        jobs.resolve.assert_not_called()
        jobs.retry.assert_not_called()
        jobs.park.assert_not_called()


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
                    "head_sha": "a" * 40, "builder_round": 3, "attempt": 3,
                    "worker_token": "wtok"}
        jobs.claim_queued.side_effect = [over_cap, None]
        jobs.recover_expired.return_value = []
        monkeypatch.setattr(builder, "Config", lambda: cfg)
        monkeypatch.setattr(builder, "Jobs", lambda dsn: jobs)
        monkeypatch.setattr(builder, "pr_state", lambda *a: "OPEN\tmain")
        monkeypatch.setattr(sys, "argv", ["neo_builder", "--once"])
        with caplog.at_level(logging.INFO):
            builder.main()
        jobs.park.assert_any_call(
            10, worker_token="wtok",
            reason="max builder rounds (2) reached (round 3)")
        assert jobs.claim_queued.call_count == 2  # over-cap row + empty
        assert not jobs.resolve.called

    def test_dispatch_one_round_cap_park_presents_the_claim_token(self):
        """dispatch_one's own round-cap gate parks the row it was claimed with:
        the park presents the claimant's worker_token (the same token
        claim_queued minted) and the "reached" reason.  A bare park() refuses
        the row's LIVE lease and the round sits running until lease expiry."""
        import src.neo_builder as builder
        cfg = MagicMock()
        cfg.max_builder_rounds = 2
        jobs = MagicMock(spec=Jobs)
        row = {"id": 10, "repo": "o/r", "pr_number": 5, "head_sha": "a" * 40,
               "builder_round": 3, "attempt": 3, "worker_token": "wtok",
               "payload": "{}"}
        builder.dispatch_one(cfg, jobs, row)
        jobs.park.assert_called_once()
        assert jobs.park.call_args.kwargs["worker_token"] == "wtok"
        assert "reached" in jobs.park.call_args.kwargs["reason"]
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


# ---------------------------------------------------------------------------
# W1-13-r1 Defect 5 — guard: none of the three test modules may assert on
# source text.  Substring assertions on module source pass on any refactor
# that keeps the substring and fail on any rename that keeps the behaviour —
# false confidence on exactly the recovery paths under test.
# ---------------------------------------------------------------------------

class TestNoSourceTextAssertions:
    FILES = ("test_job_recovery.py", "test_builder_evidence.py", "test_baseline.py")

    def test_no_test_module_asserts_on_source_text(self):
        import re
        from pathlib import Path
        here = Path(__file__).parent
        forbidden = (
            r"open\(.*" + "__file__" + r"\).*read\(\)",
            r"(?:src|source)\[" + r".*\.index\(",
            r"inspect\." + r"getsource",
            "__file__" + r"\)?.*read_text\(\)",
        )
        for name in self.FILES:
            path = str(here / name)
            text = open(path).read()
            for pat in forbidden:
                m = re.search(pat, text)
                assert m is None, (
                    f"{name} asserts on source text ({m.group(0)!r}) — "
                    "drive the real function and assert on calls made instead"
                )


# ---------------------------------------------------------------------------
# W1-13-r1 Defect 7 — the owner install id must come from Config
# ---------------------------------------------------------------------------

class TestOwnerInstallIdFromConfig:
    """A hard-coded install id in the worker is exactly what drifts per
    deployment: the owner workspace must be NEO_OWNER_INSTALL_ID on Config."""

    def test_owner_install_id_is_env_configured(self, monkeypatch):
        from src.config import Config
        monkeypatch.setenv("NEO_OWNER_INSTALL_ID", "deploy-specific-owner")
        assert Config().owner_install_id == "deploy-specific-owner"

    def test_owner_install_id_defaults_to_the_legacy_value(self):
        from src.config import Config
        from src.neo_worker import OWNER_INSTALL_ID
        assert Config().owner_install_id == OWNER_INSTALL_ID

    def test_metering_follows_the_config_value(self, monkeypatch):
        from src.config import Config
        from src.neo_worker import OWNER_INSTALL_ID, meters_credits
        monkeypatch.setenv("NEO_OWNER_INSTALL_ID", "deploy-specific-owner")
        cfg = Config()
        # the CONFIGURED owner never meters ...
        assert meters_credits(
            {"install_id": "deploy-specific-owner", "plan": "pro"}, cfg) is False
        # ... and the legacy hard-coded id decides nothing any more
        assert meters_credits(
            {"install_id": OWNER_INSTALL_ID, "plan": "pro"}, cfg) is True

    def test_credit_gate_uses_the_config_owner(self, monkeypatch):
        """run_neo's out-of-credits gate follows cfg.owner_install_id: the
        configured owner is exempt; the legacy hard-coded id is a customer."""
        from src.config import Config
        from src.jev_review import JevAssessment
        from src.neo_worker import OWNER_INSTALL_ID, run_neo
        monkeypatch.setenv("NEO_OWNER_INSTALL_ID", "custom-owner")
        cfg = Config()
        cfg.gh_token = "t"
        cfg.deepseek_model = "deepseek-v4-flash"
        cfg.max_builder_rounds = 3
        cfg.auto_merge_default = False
        cfg.deepseek_timeout = 30
        state = MagicMock()
        state.builder_rounds.return_value = 0
        state.already_acted.return_value = False
        state.already_escalated.return_value = False
        jobs = MagicMock(spec=Jobs)
        jobs.claim.return_value = 42
        wake = {"repo": "o/r", "sha": "a" * 40, "green": True,
                "context": "test", "pr_number": 5, "job_id": 42}
        evidence = {"ok": True, "head_sha": "a" * 40, "decided_sha": "a" * 40,
                    "review_sha": "a" * 40, "review_bound_ok": True,
                    "trusted_states": ["success"], "trusted_green": True,
                    "failing_checks": 0, "rollup_green": True,
                    "required_checks_pass": True, "coverage_complete": True,
                    "unresolved_critical_high": 0, "comments_addressed": True,
                    "review_state": "APPROVED"}

        def drive(inst):
            state.install.return_value = inst
            jobs.reset_mock()
            with patch("src.neo_worker.repo_token", return_value="tok"), \
                 patch("src.neo_worker.resolve_pr", return_value=5), \
                 patch("src.neo_worker._review_evidence", return_value=evidence), \
                 patch("src.neo_worker._jev_for_pr",
                       return_value=JevAssessment(False, reason="no key")), \
                 patch("src.neo_worker._run_agent",
                       return_value=({"input_tokens": 1}, "NO-CHANGE", 0, None)) as agent, \
                 patch("src.neo_worker._gh", return_value=(0, "")):
                run_neo(cfg, state, jobs, wake)
            return agent

        # the legacy hard-coded id with zero balance: a plain customer -> gated
        agent = drive({"install_id": OWNER_INSTALL_ID, "plan": "pro",
                       "credits_balance": 0})
        agent.assert_not_called()
        assert jobs.park.called, "customer install with no credits is parked"

        # the CONFIGURED owner with zero balance: never metered -> runs
        agent = drive({"install_id": "custom-owner", "plan": "pro",
                       "credits_balance": 0})
        agent.assert_called_once()
        assert not jobs.park.called
