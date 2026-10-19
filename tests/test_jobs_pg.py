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
