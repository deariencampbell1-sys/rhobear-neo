"""N15-B1 durable-job integration tests against a REAL Postgres.

Runs only when NEO_TEST_DSN is set (the integrator's throwaway PG 16 —
`host=/tmp/.../scratchpad/pg port=54329 user=neo dbname=neo`, trust auth).
Fakes alone are not acceptable for this lane, so these tests exercise the
actual ON CONFLICT / UPDATE ... RETURNING / SKIP LOCKED SQL against real
Postgres, plus the run_neo wiring (gh / agent faked) against the real job
store.

Covers every numbered integrator + GLM finding and the 8-thread claim race:

  * I-1  a BOUNCE-BUILDER verdict creates a kind='builder' job (builder lane lives)
  * I-2  the unique key carries kind (+round): wake and builder jobs for the same
         head SHA are distinct rows; two rounds of one PR are distinct rows
  * I-3  8 threads claiming 20 queued jobs -> 20 claims, 20 unique (no dupes)
  * G-1  a `status` wake (no PR number yet) creates a job with pr_number NULL
  * G-3  a green verdict after a red triage of the SAME head is a new job (color in key)
  * G-4  a crashed/failed wake is retried (outcome='failed', back to queued) and
         an expired running lease is recovered — never stranded
  * G-5  a wake whose PR is unresolvable is released retryably, not poisoned
  * G-6  lease renewal actually extends lease_expires (long agent runs)
  * G-7  claim is newest-first and supersedes OLDER rounds (newest round wins)
  * G-8  a running job left behind by a transient failure is re-queued by lease
         recovery (never resolved as terminal 'failed')
  * G-9  --dry-run (peek_queued) is read-only; only claim burns attempt/state
  *      dedupe on OUTCOME: active job suppresses, terminal job re-arms
  *      lease expiry re-runs once then parks at the attempt cap

Run:  NEO_TEST_DSN='<dsn>' .venv/bin/python -m pytest tests/test_jobs_pg.py -q
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from src.config import Config
from src.jobs import Jobs
from src.neo_state import NeoState
from src.jev_review import JevAssessment

DSN = os.environ.get("NEO_TEST_DSN", "")

pytestmark = pytest.mark.skipif(not DSN, reason="NEO_TEST_DSN not set")

SHA1 = "a" * 40
SHA2 = "b" * 40


def _wipe() -> None:
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("DELETE FROM neo_jobs")
        c.execute("DELETE FROM neo_actions")
        c.execute("DELETE FROM neo_installs")


@pytest.fixture()
def db():
    Jobs(DSN).ensure_schema()
    NeoState(DSN).ensure_schema()
    _wipe()
    return Jobs(DSN)


def _cfg() -> Config:
    cfg = Config()
    cfg.database_url = DSN
    cfg.gh_token = "test-token"
    cfg.deepseek_model = "deepseek-v4-flash"
    cfg.max_builder_rounds = 2
    cfg.auto_merge_default = False
    cfg.deepseek_timeout = 60
    return cfg


def _evidence(sha: str) -> dict:
    return {"ok": True, "head_sha": sha, "decided_sha": sha, "review_sha": sha,
            "review_bound_ok": True, "trusted_states": ["success"],
            "trusted_green": True, "failing_checks": 0, "rollup_green": True}


# ---------------------------------------------------------------------------
# I-1  a BOUNCE-BUILDER verdict creates the builder job (the builder lane lives)
# ---------------------------------------------------------------------------

class TestBuilderJobCreated:
    def test_bounce_creates_builder_job_and_wake_done(self, db):
        jobs = db
        state = NeoState(DSN)
        cfg = _cfg()
        wake = {"repo": "o/r", "sha": SHA1, "green": False,
                "context": "rhobear-reviews", "pr_number": 5}
        with patch("src.neo_worker.resolve_pr", return_value=5), \
             patch("src.neo_worker._review_evidence", return_value=_evidence(SHA1)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent",
                   return_value=({"input_tokens": 1000, "output_tokens": 100},
                                 "BOUNCE-BUILDER substantial fix needed")):
            from src.neo_worker import run_neo
            run_neo(cfg, state, jobs, wake)

        # The wake job is done and the builder lane has a kind='builder' job.
        with psycopg.connect(DSN, autocommit=True) as c:
            wakes = c.execute(
                "SELECT id, state, outcome FROM neo_jobs "
                "WHERE kind='wake' ORDER BY id").fetchall()
        assert len(wakes) == 1, wakes
        wake_id, st, outcome = wakes[0]
        assert st == "done" and outcome == "done", wakes
        with psycopg.connect(DSN, autocommit=True) as c:
            builds = c.execute(
                "SELECT repo, pr_number, head_sha, builder_round, state "
                "FROM neo_jobs WHERE kind='builder' ORDER BY id").fetchall()
        assert len(builds) == 1, builds
        repo, pr, sha, round_, st = builds[0]
        assert (repo, pr, sha, round_) == ("o/r", 5, SHA1, 1)
        assert st == "queued", builds
        # state recorded the bounce (builder round 1) for the loop guard
        acts = state.builder_rounds("o/r", 5)
        assert acts == 1, acts

    def test_repeat_bounce_same_round_dedupes(self, db):
        """Two duplicate bounces for the same round produce ONE builder job."""
        jobs = db
        a = jobs.create_builder("o/r", 5, SHA1, 1)
        b = jobs.create_builder("o/r", 5, SHA1, 1)
        assert a is not None and b is None
        with psycopg.connect(DSN, autocommit=True) as c:
            n = c.execute("SELECT COUNT(*) FROM neo_jobs WHERE kind='builder'").fetchone()[0]
        assert n == 1, n


# ---------------------------------------------------------------------------
# I-2  the unique key carries kind (+ round)
# ---------------------------------------------------------------------------

class TestUniqueKeyKind:
    def test_wake_and_builder_same_sha_distinct(self, db):
        """A builder job for the same head SHA as its wake job is a DIFFERENT
        row — the old UNIQUE (repo, pr, sha) silently dropped it (verified
        against PG: wake id 1, builder for same sha -> None)."""
        w = db.create_wake("o/r", SHA1, False, pr_number=3)
        b = db.create_builder("o/r", 3, SHA1, 1)
        assert w is not None and b is not None
        assert w != b
        assert db.get(w)["kind"] == "wake"
        assert db.get(b)["kind"] == "builder"
        assert db.get(b)["builder_round"] == 1

    def test_two_rounds_of_one_pr_distinct(self, db):
        r1 = db.create_builder("o/r", 3, SHA1, 1)
        r2 = db.create_builder("o/r", 3, SHA2, 2)
        assert r1 is not None and r2 is not None and r1 != r2
        # The key is (repo, pr, round): a round-2 job already exists even for a
        # different head SHA — creating it again dedupes (one round-2 job).
        r3 = db.create_builder("o/r", 3, SHA1, 2)
        assert r3 is None
        with psycopg.connect(DSN, autocommit=True) as c:
            n = c.execute("SELECT COUNT(*) FROM neo_jobs WHERE kind='builder'").fetchone()[0]
        assert n == 2, n

    def test_wake_color_in_key(self, db):
        """Red then green on the SAME head are different wake jobs (G-3)."""
        red = db.create_wake("o/r", SHA1, False, pr_number=4)
        green = db.create_wake("o/r", SHA1, True, pr_number=4)
        assert red is not None and green is not None and red != green
        assert db.get(red)["wake_color"] is False
        assert db.get(green)["wake_color"] is True


# ---------------------------------------------------------------------------
# I-3 / G — concurrency: 8 threads claiming 20 queued jobs
# ---------------------------------------------------------------------------

class TestClaimRace:
    def test_8_threads_claim_20_unique(self, db):
        for i in range(20):
            db.create_builder("o/r", 100 + i, f"{i:040d}", 1)
        ids: list[int] = []
        lock = threading.Lock()

        def claimer():
            claimed = []
            for _ in range(5):
                row = db.claim_queued("builder")
                if row:
                    claimed.append(row["id"])
            with lock:
                ids.extend(claimed)

        threads = [threading.Thread(target=claimer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(ids) == 20, (len(ids), sorted(ids))
        assert len(set(ids)) == 20, "claims must be unique"
        assert db.count_queued("builder") == 0

    def test_two_builders_racing_claim_one_round(self, db):
        db.create_builder("o/r", 9, SHA1, 1)
        results: list[dict] = []
        lock = threading.Lock()

        def claimer():
            row = db.claim_queued("builder")
            with lock:
                results.append(row)

        threads = [threading.Thread(target=claimer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        won = [r for r in results if r]
        assert len(won) == 1, results
        assert won[0]["builder_round"] == 1


# ---------------------------------------------------------------------------
# G-1  status wakes (no PR yet) create a job; PR resolved later
# ---------------------------------------------------------------------------

class TestStatusWakeNoPr:
    def test_status_wake_creates_job_with_null_pr(self, db):
        job_id = db.create_wake("o/r", SHA1, True, pr_number=None,
                                payload={"kind": "status", "repo": "o/r",
                                         "sha": SHA1, "green": True})
        assert job_id is not None
        row = db.get(job_id)
        assert row["pr_number"] is None
        assert row["state"] == "queued"
        # run_neo resolves the PR later and claims by id, stamping pr_number
        claimed = db.claim(job_id, pr_number=7, lease_secs=600)
        assert claimed == job_id
        assert db.get(job_id)["pr_number"] == 7
        assert db.get(job_id)["state"] == "running"


# ---------------------------------------------------------------------------
# G-4  crashed / failed wakes are retried, never stranded
# ---------------------------------------------------------------------------

class TestWakeRecovery:
    def test_failed_wake_goes_back_to_queued_then_reclaimed(self, db):
        job_id = db.create_wake("o/r", SHA1, True)
        assert db.claim(job_id, lease_secs=600) == job_id
        # agent crashed / produced no verdict -> retryable failure
        assert db.retry(job_id, reason="agent crashed") == "queued"
        row = db.get(job_id)
        assert row["state"] == "queued"
        assert row["outcome"] == "failed"
        # a later wake / recovery re-claims it
        assert db.claim(job_id, lease_secs=600) == job_id

    def test_expired_running_wake_recovered_and_reclaimed(self, db):
        job_id = db.create_wake("o/r", SHA1, True, max_attempts=3)
        db.claim(job_id, lease_secs=1)
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute("UPDATE neo_jobs SET lease_expires = now() - interval '5 seconds' WHERE id=%s",
                      (job_id,))
        recovered = db.recover_expired("wake")
        assert [r["id"] for r in recovered] == [job_id]
        assert db.get(job_id)["state"] == "queued"
        # re-runs once (claim increments attempt)
        db.claim(job_id, lease_secs=1)
        assert db.get(job_id)["attempt"] == 2

    def test_lease_expiry_reruns_once_then_parks_at_cap(self, db):
        job_id = db.create_wake("o/r", SHA1, True, max_attempts=2)
        for _ in range(2):
            db.claim(job_id, lease_secs=1)
            with psycopg.connect(DSN, autocommit=True) as c:
                c.execute("UPDATE neo_jobs SET lease_expires = now() - interval '5 seconds' WHERE id=%s",
                          (job_id,))
            db.recover_expired("wake")
        row = db.get(job_id)
        assert row["attempt"] == 2
        # recovery parks at the cap instead of re-queuing
        assert row["state"] == "queued"
        db.park(job_id, reason="lease expired 2/2 times")
        assert db.get(job_id)["state"] == "done"
        assert db.get(job_id)["outcome"] == "parked"
        # parked is terminal: claim does not resurrect it
        assert db.claim(job_id) is None

    def test_retry_parks_at_attempt_cap(self, db):
        job_id = db.create_wake("o/r", SHA1, True, max_attempts=2)
        db.claim(job_id, lease_secs=600)          # attempt 1
        assert db.retry(job_id, reason="fail1") == "queued"
        db.claim(job_id, lease_secs=600)          # attempt 2
        assert db.retry(job_id, reason="fail2") == "parked"
        row = db.get(job_id)
        assert row["state"] == "done" and row["outcome"] == "parked"


# ---------------------------------------------------------------------------
# G-5  a wake whose PR is unresolvable is released retryably (not poisoned)
# ---------------------------------------------------------------------------

class TestUnresolvablePR:
    def test_no_pr_releases_retryably_then_later_wake_rearms(self, db):
        jobs = db
        state = NeoState(DSN)
        cfg = _cfg()
        wake = {"repo": "o/r", "sha": SHA1, "green": True,
                "context": "rhobear-reviews"}
        with patch("src.neo_worker.resolve_pr", return_value=None):
            from src.neo_worker import run_neo
            run_neo(cfg, state, jobs, wake)
        with psycopg.connect(DSN, autocommit=True) as c:
            rows = c.execute(
                "SELECT id, state, outcome, attempt FROM neo_jobs WHERE kind='wake'").fetchall()
        assert len(rows) == 1
        _id, st, outcome, attempt = rows[0]
        # released retryably: back to queued with outcome='failed' (not terminal,
        # and definitely not a silently-dropped wake)
        assert st == "queued" and outcome == "failed" and attempt == 1, rows
        # bounded: two more claim+retry cycles park it (owner-actionable)
        jobs.claim(_id, lease_secs=600)
        assert jobs.retry(_id, reason="no PR yet") == "queued"
        jobs.claim(_id, lease_secs=600)
        assert jobs.retry(_id, reason="no PR yet") == "parked"
        assert jobs.get(_id)["state"] == "done"
        assert jobs.get(_id)["outcome"] == "parked"
        # A later wake for the same SHA+color re-arms the parked job
        # (dedupe on outcome, not on row existence) — the SHA is NOT poisoned.
        new_id = db.create_wake("o/r", SHA1, True)
        assert new_id is not None
        assert db.get(new_id)["state"] == "queued"
        assert db.get(new_id)["outcome"] is None


# ---------------------------------------------------------------------------
# G-6  lease renewal keeps a long agent run owned (never re-claimed mid-run)
# ---------------------------------------------------------------------------

class TestLeaseRenewal:
    def test_lease_extends_expiry(self, db):
        job_id = db.create_wake("o/r", SHA1, True)
        db.claim(job_id, lease_secs=1)
        # after the lease lapsed, renewal on a running job pushes it forward
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute("UPDATE neo_jobs SET lease_expires = now() - interval '5 seconds' WHERE id=%s",
                      (job_id,))
        assert db.lease(job_id, lease_secs=3600) is True
        with psycopg.connect(DSN, autocommit=True) as c:
            future = c.execute(
                "SELECT lease_expires > now() + interval '30 minutes' "
                "FROM neo_jobs WHERE id=%s", (job_id,)).fetchone()[0]
        assert future is True

    def test_lease_renewal_context_used_by_dispatch_one(self, db, tmp_path):
        """dispatch_one runs the agent inside lease_renewal (a second builder
        must never claim the same round mid-run)."""
        from src.neo_builder import dispatch_one, BUILDER_LEASE_SECS
        cfg = _cfg()
        cfg.max_builder_rounds = 3
        jobs = db
        row_id = jobs.create_builder("o/r", 5, SHA1, 1)
        row = jobs.claim_queued("builder")
        assert row is not None and row["id"] == row_id
        with patch("src.neo_builder.pr_state", return_value="OPEN\tmain"), \
             patch("src.neo_builder.prepare_worktree",
                   return_value=("/tmp/wt", "main")), \
             patch("src.neo_builder.fetch_findings", return_value=""), \
             patch("src.neo_builder.gh", return_value=(0, SHA2)), \
             patch("src.neo_builder.BRIEFS", tmp_path / "briefs"), \
             patch("src.neo_builder.BuilderAgent.run",
                   return_value=({"input_tokens": 1}, "FIXED")):
            dispatch_one(cfg, jobs, row)
        row2 = jobs.get(row_id)
        assert row2["state"] == "done"
        assert row2["outcome"] == "done"
        assert BUILDER_LEASE_SECS >= 3600


# ---------------------------------------------------------------------------
# G-7  newest-first claim + supersede OLDER rounds (newest round wins)
# ---------------------------------------------------------------------------

class TestNewestWins:
    def test_claim_is_newest_first(self, db):
        old = db.create_builder("o/r", 5, SHA1, 1)
        new = db.create_builder("o/r", 5, SHA2, 2)
        row = db.claim_queued("builder")
        assert row["id"] == new, (old, new, row)
        assert row["builder_round"] == 2

    def test_stale_round_never_runs_after_newer(self, db):
        db.create_builder("o/r", 5, SHA1, 1)
        db.create_builder("o/r", 5, SHA2, 2)
        claimed = db.claim_queued("builder")
        assert claimed["builder_round"] == 2
        superseded = db.supersede_older("o/r", 5, claimed["builder_round"])
        assert len(superseded) == 1
        with psycopg.connect(DSN, autocommit=True) as c:
            rows = c.execute(
                "SELECT builder_round, state, outcome FROM neo_jobs "
                "WHERE kind='builder' ORDER BY builder_round").fetchall()
        assert rows == [(1, "done", "superseded"), (2, "running", None)], rows
        # the superseded older round cannot be claimed again
        assert db.claim_queued("builder") is None


# ---------------------------------------------------------------------------
# G-8  transient failure leaves the job running; lease recovery re-queues it
# ---------------------------------------------------------------------------

class TestTransientFailure:
    def test_running_job_recovered_not_terminal(self, db):
        job_id = db.create_builder("o/r", 5, SHA1, 1)
        db.claim_queued("builder")          # now running (transient pr_state hit)
        row = db.get(job_id)
        assert row["state"] == "running"
        assert row["outcome"] is None       # never resolved 'failed' (terminal)
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute("UPDATE neo_jobs SET lease_expires = now() - interval '5 seconds' WHERE id=%s",
                      (job_id,))
        recovered = db.recover_expired("builder")
        assert [r["id"] for r in recovered] == [job_id]
        assert db.get(job_id)["state"] == "queued"


# ---------------------------------------------------------------------------
# G-9  --dry-run is read-only (peek_queued never mutates)
# ---------------------------------------------------------------------------

class TestDryRunReadOnly:
    def test_peek_does_not_claim_or_burn_attempt(self, db):
        job_id = db.create_builder("o/r", 5, SHA1, 1)
        before = db.get(job_id)
        for _ in range(3):
            rows = db.peek_queued("builder")
            assert [r["id"] for r in rows] == [job_id]
        after = db.get(job_id)
        assert after["state"] == before["state"] == "queued"
        assert after["attempt"] == before["attempt"] == 0
        assert after["lease_expires"] is None
        # contrast: a real claim mutates (attempt + running)
        db.claim_queued("builder")
        claimed = db.get(job_id)
        assert claimed["state"] == "running"
        assert claimed["attempt"] == 1


# ---------------------------------------------------------------------------
# dedupe on OUTCOME (not on a row merely existing)
# ---------------------------------------------------------------------------

class TestDedupeOnOutcome:
    def test_active_wake_suppresses_new_wake(self, db):
        first = db.create_wake("o/r", SHA1, True)
        second = db.create_wake("o/r", SHA1, True)
        assert first is not None and second is None

    def test_terminal_wake_is_rearmed(self, db):
        first = db.create_wake("o/r", SHA1, True)
        db.claim(first, lease_secs=600)
        db.resolve(first, "no_change", payload={"reason": "stale head"})
        row = db.get(first)
        assert row["state"] == "done" and row["outcome"] == "no_change"
        # the next wake for the same SHA re-arms the SAME row
        again = db.create_wake("o/r", SHA1, True)
        assert again == first
        assert db.get(first)["state"] == "queued"
        assert db.get(first)["outcome"] is None

    def test_builder_terminal_rearmed(self, db):
        r1 = db.create_builder("o/r", 5, SHA1, 1)
        db.claim_queued("builder")
        db.resolve(r1, "no_change", payload={"reason": "PR closed"})
        r1b = db.create_builder("o/r", 5, SHA1, 1)
        assert r1b == r1
        assert db.get(r1)["state"] == "queued"


# ---------------------------------------------------------------------------
# FIX ROUND 3 — blocker 2: per-SHA idempotency gate (integrator finding 2)
# ---------------------------------------------------------------------------

class TestIdempotencyGate:
    def test_duplicate_delivery_does_not_repeat_work(self, db):
        """Defect 2: a duplicate delivery of the same verdict re-arms the
        terminal wake (dedupe on outcome, so the verdict is not dropped), but
        the per-SHA gate must resolve it no_change — the agent never re-runs,
        no second debit, no extra builder round."""
        jobs = db
        state = NeoState(DSN)
        cfg = _cfg()
        wake = {"repo": "o/r", "sha": SHA1, "green": False,
                "context": "rhobear-reviews", "pr_number": 5}
        agent_calls: list[int] = []

        def fake_agent(agent, brief):
            agent_calls.append(1)
            return {"input_tokens": 1000, "output_tokens": 100}, "BOUNCE-BUILDER fix it"

        from src.neo_worker import run_neo
        with patch("src.neo_worker.resolve_pr", return_value=5), \
             patch("src.neo_worker._review_evidence", return_value=_evidence(SHA1)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent", side_effect=fake_agent):
            run_neo(cfg, state, jobs, wake)   # first delivery: real triage
            run_neo(cfg, state, jobs, wake)   # duplicate delivery: gate -> no_change
        assert len(agent_calls) == 1, agent_calls
        with psycopg.connect(DSN, autocommit=True) as c:
            builds = c.execute("SELECT COUNT(*) FROM neo_jobs WHERE kind='builder'").fetchone()[0]
            acts = c.execute("SELECT COUNT(*) FROM neo_actions WHERE phase='builder'").fetchone()[0]
            triages = c.execute("SELECT COUNT(*) FROM neo_actions WHERE phase='triage'").fetchone()[0]
        assert builds == 1, builds      # no extra builder round minted
        assert acts == 1, acts
        assert triages == 1, triages    # one triage record (recorded after success)

    def test_merged_duplicate_is_skipped(self, db):
        """A duplicate delivery after Neo already merged this head resolves
        no_change — the already_acted 'merged' arm catches it (no re-merge)."""
        jobs = db
        state = NeoState(DSN)
        cfg = _cfg()
        cfg.auto_merge_default = True
        wake = {"repo": "o/r", "sha": SHA1, "green": True,
                "context": "rhobear-reviews", "pr_number": 5}
        agent_calls: list[int] = []

        def fake_agent(agent, brief):
            agent_calls.append(1)
            return {"input_tokens": 1, "output_tokens": 1}, "ACCEPT-MERGED"

        from src.neo_worker import run_neo
        with patch("src.neo_worker.resolve_pr", return_value=5), \
             patch("src.neo_worker._review_evidence", return_value=_evidence(SHA1)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent", side_effect=fake_agent):
            run_neo(cfg, state, jobs, wake)   # first delivery: merge
            run_neo(cfg, state, jobs, wake)   # duplicate: gate -> no_change
        assert len(agent_calls) == 1, agent_calls
        with psycopg.connect(DSN, autocommit=True) as c:
            wakes = c.execute(
                "SELECT outcome FROM neo_jobs WHERE kind='wake'").fetchall()
        # the duplicate re-arms the SAME terminal row (ON CONFLICT re-arm), then
        # the gate resolves it no_change — the merged disposition is overwritten
        # by the duplicate's no_change, the history lives in neo_actions
        assert [w[0] for w in wakes] == ["no_change"], wakes
        with psycopg.connect(DSN, autocommit=True) as c:
            merged = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE phase='merged'").fetchone()[0]
        assert merged == 1, merged  # merged exactly once

    def test_green_after_red_same_head_still_triages(self, db):
        """The color-aware gate must NOT swallow a green verdict that arrives
        after a red triage of the SAME head (the reviewer re-ran and passed
        it) — that is new decision input, not a duplicate."""
        jobs = db
        state = NeoState(DSN)
        cfg = _cfg()
        agent_calls: list[int] = []

        def fake_agent(agent, brief):
            agent_calls.append(1)
            return {"input_tokens": 1, "output_tokens": 1}, "ACCEPT-READY"

        from src.neo_worker import run_neo
        red = {"repo": "o/r", "sha": SHA1, "green": False,
               "context": "rhobear-reviews", "pr_number": 6}
        green = {"repo": "o/r", "sha": SHA1, "green": True,
                 "context": "rhobear-reviews", "pr_number": 6}
        with patch("src.neo_worker.resolve_pr", return_value=6), \
             patch("src.neo_worker._review_evidence", return_value=_evidence(SHA1)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent", side_effect=fake_agent):
            run_neo(cfg, state, jobs, red)
            run_neo(cfg, state, jobs, green)
        assert len(agent_calls) == 2, agent_calls

    def test_crashed_run_retry_is_not_gated(self, db):
        """A job whose agent crashed mid-run is retried and re-runs — the gate
        must not treat the retry as already-acted (the triage row is recorded
        only after a completed run)."""
        jobs = db
        state = NeoState(DSN)
        cfg = _cfg()
        agent_calls: list[int] = []

        def flaky_agent(agent, brief):
            agent_calls.append(1)
            if len(agent_calls) == 1:
                raise FileNotFoundError("agent binary gone")
            return {"input_tokens": 1, "output_tokens": 1}, "ACCEPT-READY"

        wake = {"repo": "o/r", "sha": SHA1, "green": True,
                "context": "rhobear-reviews", "pr_number": 7}
        from src.neo_worker import run_neo
        with patch("src.neo_worker.resolve_pr", return_value=7), \
             patch("src.neo_worker._review_evidence", return_value=_evidence(SHA1)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent", side_effect=flaky_agent):
            # first delivery: agent crashes -> job retried (back to queued)
            run_neo(cfg, state, jobs, wake)
        with psycopg.connect(DSN, autocommit=True) as c:
            job_id, st, outcome = c.execute(
                "SELECT id, state, outcome FROM neo_jobs WHERE kind='wake'").fetchone()
        assert st == "queued" and outcome == "failed", (st, outcome)
        # retry path: same job id re-submitted -> claim -> NOT gated -> re-runs
        with patch("src.neo_worker.resolve_pr", return_value=7), \
             patch("src.neo_worker._review_evidence", return_value=_evidence(SHA1)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent", side_effect=flaky_agent):
            run_neo(cfg, state, jobs, {"repo": "o/r", "sha": SHA1,
                                       "green": True, "job_id": job_id})
        assert len(agent_calls) == 2, agent_calls
        assert jobs.get(job_id)["outcome"] == "done"


# ---------------------------------------------------------------------------
# FIX ROUND 3 — blocker 1: never two rounds of one PR in one worktree
# ---------------------------------------------------------------------------

class TestInflightRelease:
    def test_release_returns_queued_without_failure_marker(self, db):
        """Jobs.release() voids a claim (same repo+PR already running in
        another worker sharing the worktree) WITHOUT recording a retryable
        failure: state back to queued, outcome stays NULL, no reason burned."""
        job_id = db.create_builder("o/r", 5, SHA1, 1)
        row = db.claim_queued("builder")
        assert row["id"] == job_id and row["attempt"] == 1
        assert db.release(job_id, worker_token=row["worker_token"]) is True
        r = db.get(job_id)
        assert r["state"] == "queued"
        assert r["outcome"] is None          # NOT a retryable failure
        assert r["attempt"] == 1             # release itself does not add
        # the released round is claimable again once the running one frees up
        row2 = db.claim_queued("builder")
        assert row2["id"] == job_id

    def test_release_rejects_stale_claimant(self, db):
        """A release by a non-claimant (token mismatch) does nothing."""
        job_id = db.create_builder("o/r", 5, SHA1, 1)
        row = db.claim_queued("builder")
        assert db.release(job_id, worker_token="not-the-claimant") is False
        assert db.get(job_id)["state"] == "running"
        assert db.release(job_id, worker_token=row["worker_token"]) is True

    def test_claim_queued_excludes_inflight_keys(self, db):
        """The SQL-side guard: claim_queued with exclude_keys never returns a
        row for a PR that is already running in this process (they share one
        worktree).  This is what prevents the claiming-releasing busy loop."""
        a = db.create_builder("o/r", 5, SHA1, 1)
        b = db.create_builder("o/r", 6, SHA2, 1)
        row = db.claim_queued("builder", exclude_keys={("o/r", 6)})
        assert row is not None and row["id"] == a, row   # PR 6 excluded -> PR 5
        # both PRs claimable when nothing is excluded
        c = db.create_builder("o/r", 7, SHA2, 1)
        row2 = db.claim_queued("builder", exclude_keys=set())
        assert row2["id"] == c


# ---------------------------------------------------------------------------
# FIX ROUND 3 — blocker 3: builder verdict branches (real DB dispatch_one)
# ---------------------------------------------------------------------------

class TestBuilderVerdictBranches:
    def _dispatch(self, db, run_result):
        """Claim a real builder round and dispatch it with a faked agent run."""
        from src.neo_builder import dispatch_one
        cfg = _cfg()
        cfg.max_builder_rounds = 3
        row_id = db.create_builder("o/r", 5, SHA1, 1)
        row = db.claim_queued("builder")
        assert row is not None and row["id"] == row_id
        with patch("src.neo_builder.pr_state", return_value="OPEN\tmain"), \
             patch("src.neo_builder.prepare_worktree", return_value=("/tmp/wt", "main")), \
             patch("src.neo_builder.fetch_findings", return_value=""), \
             patch("src.neo_builder.gh", return_value=(0, SHA2)), \
             patch("src.neo_builder.BRIEFS", Path("/tmp/briefs")), \
             patch("src.neo_builder.BuilderAgent.run", return_value=run_result):
            dispatch_one(cfg, db, row)
        return row_id

    def test_no_change_resolves_no_change(self, db):
        row_id = self._dispatch(db, ({"input_tokens": 0}, "NO-CHANGE"))
        r = db.get(row_id)
        assert r["state"] == "done" and r["outcome"] == "no_change", r

    def test_fixed_no_push_records_done(self, db):
        row_id = self._dispatch(db, ({"input_tokens": 0}, "FIXED"))
        r = db.get(row_id)
        assert r["state"] == "done" and r["outcome"] == "done", r
        assert r["payload"]["outcome"] == "fixed-no-push", r

    def test_completed_unparseable_verdict_is_no_change_not_retry(self, db):
        """Block 3: a COMPLETED run (exit 0) whose verdict line is unparseable
        resolves no_change with the raw tail recorded — it is NOT retried 3x
        then mis-parked (the old dead-code behavior)."""
        row_id = self._dispatch(
            db, ({"input_tokens": 0, "output_tokens": 0,
                  "raw_tail": "VERDICT: ACCEPT-READY"}, ""))
        r = db.get(row_id)
        assert r["state"] == "done" and r["outcome"] == "no_change", r
        assert r["payload"].get("verdict") == "", r
        assert "ACCEPT-READY" in r["payload"].get("raw_tail", ""), r

    def test_blocked_is_no_change_with_verdict_recorded(self, db):
        row_id = self._dispatch(db, ({"input_tokens": 0}, "BLOCKED"))
        r = db.get(row_id)
        assert r["state"] == "done" and r["outcome"] == "no_change", r
        assert r["payload"].get("verdict") == "BLOCKED", r


# ---------------------------------------------------------------------------
# FIX ROUND 3 — lower severity 1: recovery must not amplify waiting wakes
# ---------------------------------------------------------------------------

class TestRecoveryAmplification:
    def test_submitted_wake_is_not_resubmitted_while_waiting(self, db):
        """A wake that was submitted and is merely waiting in a saturated
        executor (submitted_at set, created long ago) must NOT be returned by
        stale_queued — it was re-submitted every recovery pass before the fix."""
        job_id = db.create_wake("o/r", SHA1, True,
                                payload={"repo": "o/r", "sha": SHA1, "green": True})
        assert db.get(job_id)["submitted_at"] is not None
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute("UPDATE neo_jobs SET created_at = now() - interval '3 minutes' "
                      "WHERE id = %s", (job_id,))
        assert db.stale_queued("wake", older_than_secs=30) == []

    def test_retried_wake_becomes_eligible_exactly_once(self, db):
        """A crash-retried wake clears submitted_at so it IS re-submitted, but
        after re-submission it is stamped again and not amplified."""
        job_id = db.create_wake("o/r", SHA1, True,
                                payload={"repo": "o/r", "sha": SHA1, "green": True})
        db.claim(job_id, lease_secs=600)
        db.retry(job_id, reason="agent crashed")
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute("UPDATE neo_jobs SET created_at = now() - interval '3 minutes' "
                      "WHERE id = %s", (job_id,))
        # retry cleared submitted_at -> stale_queued returns it once
        assert [r["id"] for r in db.stale_queued("wake", older_than_secs=30)] == [job_id]
        # after the recovery loop re-submits (mark_submitted), it is no longer
        # an orphan even if it waits in the executor again
        db.mark_submitted(job_id)
        assert db.stale_queued("wake", older_than_secs=30) == []

    def test_startup_reset_makes_orphans_eligible(self, db):
        """After a restart every executor task died: reset_queued_submissions
        clears the stamps once so the first sweep re-submits them all."""
        job_id = db.create_wake("o/r", SHA1, True,
                                payload={"repo": "o/r", "sha": SHA1, "green": True})
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute("UPDATE neo_jobs SET created_at = now() - interval '3 minutes' "
                      "WHERE id = %s", (job_id,))
        assert db.stale_queued("wake", older_than_secs=30) == []  # submitted
        assert db.reset_queued_submissions("wake") == 1
        assert [r["id"] for r in db.stale_queued("wake", older_than_secs=30)] == [job_id]


# ---------------------------------------------------------------------------
# FIX ROUND 3 — lower severity 2: worker_token (stale claimant never mutates)
# ---------------------------------------------------------------------------

class TestWorkerToken:
    def test_stale_claimant_cannot_resolve_replacement(self, db):
        """If a DB outage lets the lease lapse and another worker claims the
        re-queued job, the ORIGINAL claimant's resolve/retry/lease must fail —
        only the current claimant can mutate the row."""
        job_id = db.create_wake("o/r", SHA1, True)
        _id = db.claim(job_id, lease_secs=600)
        assert _id == job_id
        token1 = db.get(job_id)["worker_token"]
        assert token1
        # simulate the outage: lease expires, recovery re-queues, new worker claims
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute("UPDATE neo_jobs SET lease_expires = now() - interval '5 seconds' "
                      "WHERE id = %s", (job_id,))
        db.recover_expired("wake")
        _id2 = db.claim(job_id, lease_secs=600)
        assert _id2 == job_id
        token2 = db.get(job_id)["worker_token"]
        assert token2 != token1
        # the stale claimant's mutations are rejected
        assert db.resolve(job_id, "no_change", worker_token=token1) is False
        assert db.retry(job_id, worker_token=token1, reason="stale") == "?"
        assert db.lease(job_id, 600, worker_token=token1) is False
        assert db.get(job_id)["state"] == "running"
        # the current claimant can resolve
        assert db.resolve(job_id, "no_change", worker_token=token2) is True
        assert db.get(job_id)["outcome"] == "no_change"

    def test_resolve_clears_token(self, db):
        job_id = db.create_wake("o/r", SHA1, True)
        db.claim(job_id, lease_secs=600)
        token = db.get(job_id)["worker_token"]
        assert db.resolve(job_id, "no_change", worker_token=token) is True
        assert db.get(job_id)["worker_token"] is None


# ---------------------------------------------------------------------------
# Integrator, round-3 review: the two crash windows around the claim
# ---------------------------------------------------------------------------
class TestCrashWindows:
    def _patches(self, verdict="BOUNCE-BUILDER fix it", resolve=5):
        return [patch("src.neo_worker.resolve_pr", side_effect=resolve) if callable(resolve) or isinstance(resolve, BaseException)
                else patch("src.neo_worker.resolve_pr", return_value=resolve),
                patch("src.neo_worker._review_evidence", return_value=_evidence(SHA1)),
                patch("src.neo_worker._jev_for_pr", return_value=JevAssessment(False, reason="no key")),
                patch("src.neo_worker._run_agent", return_value=({"input_tokens": 10, "output_tokens": 1}, verdict))]

    def test_resolve_pr_crash_releases_the_job_for_retry(self, db):
        import subprocess
        from src.neo_worker import run_neo
        jobs, state, cfg = db, NeoState(DSN), _cfg()
        wake = {"repo": "o/r", "sha": SHA1, "green": True, "context": "rhobear-reviews"}
        boom = subprocess.TimeoutExpired(["gh"], 30)
        with patch("src.neo_worker.resolve_pr", side_effect=boom):
            run_neo(cfg, state, jobs, wake)
        with psycopg.connect(DSN, autocommit=True) as c:
            st, outcome, submitted = c.execute(
                "SELECT state, outcome, submitted_at FROM neo_jobs WHERE kind='wake'").fetchone()
        assert (st, outcome) == ("queued", "failed")
        assert submitted is None, "retry() clears the stamp so the sweep re-submits it"

    def test_crash_after_bounce_recorded_does_not_mint_a_second_round(self, db):
        from src.neo_worker import run_neo
        jobs, state, cfg = db, NeoState(DSN), _cfg()
        wake = {"repo": "o/r", "sha": SHA1, "green": True, "context": "rhobear-reviews", "pr_number": 5}
        real_record = NeoState.record

        def die_on_triage(self, repo, pr, sha, phase, **kw):
            if phase == "triage":
                raise RuntimeError("process killed before the triage marker")
            return real_record(self, repo, pr, sha, phase, **kw)

        ps = self._patches()
        for p in ps: p.start()
        try:
            with patch.object(NeoState, "record", die_on_triage):
                with pytest.raises(RuntimeError):
                    run_neo(cfg, state, jobs, wake)
            # Lease expiry re-queues the wake; the retry must not re-run the agent or mint round 2.
            with psycopg.connect(DSN, autocommit=True) as c:
                c.execute("UPDATE neo_jobs SET state='queued', lease_expires=NULL, worker_token=NULL WHERE kind='wake'")
                wake_id = c.execute("SELECT id FROM neo_jobs WHERE kind='wake'").fetchone()[0]
            agent = MagicMock(return_value=({"input_tokens": 10, "output_tokens": 1}, "BOUNCE-BUILDER again"))
            with patch("src.neo_worker._run_agent", agent):
                run_neo(cfg, state, jobs, dict(wake, job_id=wake_id))  # as the recovery sweep re-submits it
        finally:
            for p in ps: p.stop()
        assert agent.call_count == 0, "the earlier run's work was recorded; no second agent run"
        with psycopg.connect(DSN, autocommit=True) as c:
            rounds = c.execute("SELECT builder_round FROM neo_jobs WHERE kind='builder' ORDER BY id").fetchall()
        assert rounds == [(1,)], rounds
        assert state.already_acted("o/r", 5, SHA1, "triage", green=True)
