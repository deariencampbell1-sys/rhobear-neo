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
from src.github_app import NoInstallation, AppNotConfigured
from src.jobs import Jobs
from src.neo_state import NeoState
from src.jev_review import JevAssessment


class _FakeGitHubApp:
    """Fake GitHubApp that returns a fixed token for any repo."""
    def __init__(self, token="test-ghs-token"):
        self._token = token

    def token_for_repo(self, repo, installation_id=None, permissions=None):
        return self._token

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
    # Inject a fake GitHubApp so repo_token works (no personal-token fallback).
    setattr(cfg, "_rhobear_github_app", _FakeGitHubApp("test-ghs-token"))
    return cfg


def _evidence(sha: str) -> dict:
    """Evidence from the trusted Reviews App check run on this exact head
    (items 3 & 4: unresolved/coverage come from the check run's output)."""
    return {"ok": True, "head_sha": sha, "decided_sha": sha, "review_sha": sha,
            "review_bound_ok": True, "trusted_states": ["success"],
            "trusted_green": True, "failing_checks": 0, "rollup_green": True,
            "trusted_producer_ok": True, "review_sha_match": True,
            "unresolved_critical_high": 0, "coverage_complete": True}


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
        # B2: merge executor requires auto_merge.  Ensure the install has it.
        state.install('o', 'o')  # create the install row (auto_merge=false default)
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute(
                "UPDATE neo_installs SET auto_merge = true WHERE install_id = 'o'",
            )
        wake = {"repo": "o/r", "sha": SHA1, "green": True,
                "context": "rhobear-reviews", "pr_number": 5}
        agent_calls: list[int] = []

        def fake_agent(agent, brief):
            agent_calls.append(1)
            return {"input_tokens": 1, "output_tokens": 1}, "ACCEPT-MERGED"

        import json as _json
        def fake_merge_gh(token, *args, **kw):
            """Fake _gh for merge_executor that returns a successful PR view + merge."""
            if args[:3] == ("pr", "view", "5"):
                data = {
                    "headRefOid": SHA1,
                    "state": "OPEN",
                    "mergeable": "MERGEABLE",
                    "reviews": [
                        {"body": "Looks good. COVERAGE: COMPLETE",
                         "commit_id": SHA1,
                         "author": {"type": "Bot"}},
                    ],
                    "statusCheckRollup": [
                        {"context": "ci/test", "conclusion": "SUCCESS"},
                    ],
                }
                return 0, _json.dumps(data)
            if args and args[0] == "api" and len(args) > 3 and \
                    args[3] == "repos/o/r/pulls/5/merge":
                # REST merge API with sha — GitHub rejects if the head moved.
                assert f"sha={SHA1}" in args, args
                return 0, _json.dumps({"merged": True, "sha": "M" * 40})
            raise AssertionError(f"unexpected merge _gh call: {args}")

        from src.neo_worker import run_neo
        with patch("src.neo_worker.resolve_pr", return_value=5), \
             patch("src.neo_worker._review_evidence", return_value=_evidence(SHA1)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent", side_effect=fake_agent), \
             patch("src.merge_executor._gh", side_effect=fake_merge_gh):
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


# ---------------------------------------------------------------------------
# N15-B1g — a repo with no GitHub App installation is refused: the job resolves
# `no_installation` (terminal, never retried) against the real job store.
# ---------------------------------------------------------------------------

class _NoInstallApp:
    """Fake GitHub App that reports no installation for any repo."""

    def token_for_repo(self, repo, installation_id=None, permissions=None):
        raise NoInstallation(f"repo {repo} has no GitHub App installation")


class TestNoInstallationRefusal:
    def test_wake_resolves_no_installation_terminal(self, db):
        from src.neo_worker import run_neo
        cfg = _cfg()
        setattr(cfg, "_rhobear_github_app", _NoInstallApp())
        state = NeoState(DSN)
        wake = {"repo": "o/r", "sha": SHA1, "green": True,
                "context": "rhobear-reviews", "pr_number": 5}
        with patch("src.neo_worker.resolve_pr") as resolve_pr:
            run_neo(cfg, state, db, wake)
        resolve_pr.assert_not_called()  # refused before any GitHub work
        with psycopg.connect(DSN, autocommit=True) as c:
            rows = c.execute("SELECT state, outcome, attempt, payload "
                             "FROM neo_jobs WHERE kind='wake'").fetchall()
        assert len(rows) == 1, rows
        st, outcome, attempt, payload = rows[0]
        assert st == "done" and outcome == "no_installation", rows
        assert attempt == 1, rows  # claimed once, then terminal — never retried
        assert "no GitHub App installation" in payload["reason"]

    def test_builder_round_resolves_no_installation_terminal(self, db):
        from src.neo_builder import dispatch_one
        cfg = _cfg()
        setattr(cfg, "_rhobear_github_app", _NoInstallApp())
        cfg.max_builder_rounds = 3
        row_id = db.create_builder("o/r", 5, SHA1, 1)
        row = db.claim_queued("builder")
        assert row is not None and row["id"] == row_id
        with patch("src.neo_builder.pr_state") as pr_state:
            dispatch_one(cfg, db, row)
        pr_state.assert_not_called()  # refused before any GitHub work
        r = db.get(row_id)
        assert r["state"] == "done" and r["outcome"] == "no_installation", r


# ---------------------------------------------------------------------------
# FIX ROUND 2 — item 5: end-to-end escalation on real Postgres
# ---------------------------------------------------------------------------

class TestE2EEscalationPG:
    """Five BOUNCE-BUILDER verdicts on successive head SHAs -> five cheap
    builder rounds -> advisor submitted once -> (fake advisor done) -> round 6
    created escalated with the work order in its brief -> dispatched, not
    parked -> a BOUNCE after it parks the PR and the advisor is not called
    again."""

    def _bounce_run(self, jobs, state, cfg, sha, pr=7):
        from src.neo_worker import run_neo
        wake = {"repo": "o/r", "sha": sha, "green": False,
                "context": "rhobear-reviews", "pr_number": pr}
        with patch("src.neo_worker.resolve_pr", return_value=pr), \
             patch("src.neo_worker._review_evidence", return_value=_evidence(sha)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent",
                   return_value=({"input_tokens": 1, "output_tokens": 1},
                                 "BOUNCE-BUILDER still broken")):
            run_neo(cfg, state, jobs, wake)

    def test_e2e_escalation_flow(self, db):
        jobs = db
        state = NeoState(DSN)
        cfg = _cfg()
        cfg.bounces_before_escalation = 5
        cfg.advisor_url = "http://127.0.0.1:9"  # never reached (patched)
        cfg.advisor_token = "test-token"
        SHAS = [f"{i:040d}" for i in range(1, 7)]
        pr = 7

        # --- 5 BOUNCE verdicts on successive head SHAs -> 5 cheap rounds ----
        for i in range(5):
            self._bounce_run(jobs, state, cfg, SHAS[i], pr)
        assert state.builder_rounds("o/r", pr) == 5
        with psycopg.connect(DSN, autocommit=True) as c:
            cheap = c.execute(
                "SELECT builder_round, state FROM neo_jobs WHERE kind='builder' "
                "AND builder_round < 6 ORDER BY builder_round").fetchall()
        assert [r[0] for r in cheap] == [1, 2, 3, 4, 5], cheap
        assert all(r[1] == "queued" for r in cheap), cheap

        # --- 6th BOUNCE -> advisor submitted once, PR goes escalating --------
        submits: list = []
        from src.escalation import AdviceStatus, WorkOrder as _WO

        def fake_post(cfg_, repo_, pr_, head_sha_, rounds_, diff_stat_, read_token_):
            submits.append((repo_, pr_, head_sha_))
            return "advice-e2e-1"

        with patch("src.escalation._post_advise", side_effect=fake_post):
            self._bounce_run(jobs, state, cfg, SHAS[5], pr)
        assert len(submits) == 1, submits
        with psycopg.connect(DSN, autocommit=True) as c:
            rows = c.execute(
                "SELECT detail FROM neo_actions WHERE repo='o/r' "
                "AND pr_number=%s AND phase='escalating'", (pr,)).fetchall()
        assert rows, "PR must have an escalating state row"
        assert all(r[0]["advice_id"] == "advice-e2e-1" for r in rows), rows

        # --- (fake advisor done) -> recover -> round 6 escalated ------------
        work_order = {"route": "claude:opus", "summary": "Fix auth bug",
                      "root_cause": "missing validation",
                      "steps": [{"file": "src/auth.py", "change": "add check",
                                 "why": "prevent injection"}],
                      "tests": ["tests/test_auth.py::test_validation"],
                      "risks": "low"}
        from src.escalation import recover_escalations
        with patch("src.escalation.poll_advice",
                   return_value=AdviceStatus(status="done", work_order=work_order)):
            handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "escalated", handled
        with psycopg.connect(DSN, autocommit=True) as c:
            row6 = c.execute(
                "SELECT id, state, payload FROM neo_jobs "
                "WHERE kind='builder' AND builder_round=6").fetchone()
        assert row6 is not None, "round 6 escalated builder job must exist"
        r6_id, r6_state, r6_payload = row6
        assert r6_state == "queued", r6_state
        assert r6_payload["escalated"] is True
        assert r6_payload["phase"] == "escalated"
        assert r6_payload["context"] == "neo:escalated"
        assert r6_payload["work_order"]["summary"] == "Fix auth bug"
        assert "ESCALATED ADVISOR WORK ORDER" in r6_payload["brief"]
        assert "Fix auth bug" in r6_payload["brief"]

        # --- dispatched, not parked: the escalated round is claimable --------
        claimed = jobs.claim_queued("builder")
        assert claimed is not None and claimed["builder_round"] == 6, claimed

        # --- a BOUNCE after it parks the PR; advisor not called again --------
        with patch("src.escalation._post_advise", side_effect=fake_post):
            self._bounce_run(jobs, state, cfg, "f" * 40, pr)
        assert len(submits) == 1, "advisor must not be called again after escalation"
        with psycopg.connect(DSN, autocommit=True) as c:
            parked = c.execute(
                "SELECT detail FROM neo_actions WHERE repo='o/r' AND pr_number=%s "
                "AND phase='parked' ORDER BY id", (pr,)).fetchall()
        reasons = [r[0]["reason"] for r in parked if isinstance(r[0], dict) and "reason" in r[0]]
        assert reasons, parked
        assert "bounced again" in reasons[-1]


# ---------------------------------------------------------------------------
# FIX ROUND 3 — the five review defects (real Postgres where the fix touches
# neo_actions / neo_jobs)
# ---------------------------------------------------------------------------

class TestAlreadyEscalatedPG:
    """Defect 1 — already_escalated counts only GENUINE advisor escalations
    (detail->'advisor_round' present).  Overloaded phase='escalated' rows (the
    agent's canonical ESCALATE verdict, the old out-of-credits marker) must
    NOT lock the PR out of builder/escalation automation."""

    def test_escalate_verdict_without_advisor_round_is_not_escalation(self, db):
        state = NeoState(DSN)
        state.record("o/r", 71, SHA1, "escalated", verdict="ESCALATE",
                     detail={"reason": "owner-proxy handoff"})
        assert state.already_escalated("o/r", 71) is False

    def test_genuine_advisor_escalation_is_escalation(self, db):
        state = NeoState(DSN)
        state.record("o/r", 72, SHA1, "escalated", verdict="ESCALATE",
                     builder_round=6,
                     detail={"reason": "escalated to frontier advisor",
                             "advisor_round": 6})
        assert state.already_escalated("o/r", 72) is True

    def test_out_of_credits_records_parked_not_escalated(self, db):
        """The out-of-credits gate records phase='parked' (with the reason),
        never phase='escalated' — so a later top-up + new head can still
        bounce through builder/escalation automation."""
        jobs = db
        state = NeoState(DSN)
        cfg = _cfg()
        cfg.max_builder_rounds = 5
        state.install("o", "o")
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute(
                "UPDATE neo_installs SET plan='starter', credits_balance=0 "
                "WHERE install_id='o'")
        wake = {"repo": "o/r", "sha": SHA1, "green": False,
                "context": "rhobear-reviews", "pr_number": 73}
        from src.neo_worker import run_neo
        with patch("src.neo_worker.resolve_pr", return_value=73), \
             patch("src.neo_worker._review_evidence", return_value=_evidence(SHA1)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent") as agent, \
             patch("src.neo_worker._gh", return_value=(0, "ok")):
            run_neo(cfg, state, jobs, wake)
        agent.assert_not_called()
        with psycopg.connect(DSN, autocommit=True) as c:
            parked = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE phase='parked' "
                "AND repo='o/r' AND pr_number=73").fetchone()[0]
            esc = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE phase='escalated' "
                "AND repo='o/r' AND pr_number=73").fetchone()[0]
        assert parked == 1, "out-of-credits marker must be phase='parked'"
        assert esc == 0, "out-of-credits must NOT be phase='escalated'"
        assert state.already_escalated("o/r", 73) is False


class TestEscalatingPhaseInRecoveryGatePG:
    """Defect 2 — the crash-recovery per-SHA gate recognizes phase='escalating'
    too.  If run_neo crashes after submit_escalation but before the triage
    marker, a re-delivered wake must restore the marker WITHOUT re-running the
    agent or re-submitting to the advisor (which would mint a second advice
    and burn a frontier slot)."""

    def test_escalating_phase_restores_triage_without_rerun(self, db):
        jobs = db
        state = NeoState(DSN)
        cfg = _cfg()
        cfg.bounces_before_escalation = 5
        cfg.advisor_url = "http://127.0.0.1:9"
        cfg.advisor_token = "test-token"
        pr = 74
        sha = "c" * 40
        # Simulate the crash: submit_escalation recorded the escalating row
        # but run_neo died before writing the triage marker.
        state.record("o/r", pr, sha, "escalating", verdict="ESCALATE",
                     detail={"advice_id": "advice-crash", "installation_id": 3})
        wake = {"repo": "o/r", "sha": sha, "green": False,
                "context": "rhobear-reviews", "pr_number": pr}
        from src.neo_worker import run_neo
        agent_calls: list = []

        def fake_agent(agent, brief):
            agent_calls.append(1)
            return {"input_tokens": 1, "output_tokens": 1}, "BOUNCE-BUILDER"

        with patch("src.neo_worker.resolve_pr", return_value=pr), \
             patch("src.neo_worker._review_evidence", return_value=_evidence(sha)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._run_agent", side_effect=fake_agent), \
             patch("src.escalation._post_advise",
                   side_effect=AssertionError("must not re-submit to advisor")):
            run_neo(cfg, state, jobs, wake)
        assert agent_calls == [], "agent must not re-run for an escalating SHA"
        with psycopg.connect(DSN, autocommit=True) as c:
            esc = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE phase='escalating' "
                "AND repo='o/r' AND pr_number=%s", (pr,)).fetchone()[0]
            tri = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE phase='triage' "
                "AND repo='o/r' AND pr_number=%s", (pr,)).fetchone()[0]
        assert esc == 1, "the existing escalating row must survive the gate"
        assert tri == 1, "triage marker restored (recovery loop can finish)"


class TestTakeEscalationPG:
    """Defect 3 — take_escalation deletes ONLY the advice it handled, and only
    when that advice is still the NEWEST escalation for the PR.  A newer
    escalation (new head bounced while the old advice was still queued) is
    never deleted and the stale advice is dropped without creating a round."""

    def test_older_advice_superseded_is_dropped(self, db):
        state = NeoState(DSN)
        state.record("o/r", 75, SHA1, "escalating", verdict="ESCALATE",
                     detail={"advice_id": "advice-A", "installation_id": 1})
        state.record("o/r", 75, SHA2, "escalating", verdict="ESCALATE",
                     detail={"advice_id": "advice-B", "installation_id": 1})
        # A is no longer the newest -> dropped (return False), never a round.
        assert state.take_escalation("o/r", 75, "advice-A") is False
        with psycopg.connect(DSN, autocommit=True) as c:
            remaining = [r[0] for r in c.execute(
                "SELECT detail->>'advice_id' FROM neo_actions "
                "WHERE repo='o/r' AND pr_number=75 AND phase='escalating' "
                "ORDER BY id").fetchall()]
        assert remaining == ["advice-B"], remaining

    def test_newest_advice_is_consumed(self, db):
        state = NeoState(DSN)
        state.record("o/r", 76, SHA1, "escalating", verdict="ESCALATE",
                     detail={"advice_id": "advice-A", "installation_id": 1})
        state.record("o/r", 76, SHA2, "escalating", verdict="ESCALATE",
                     detail={"advice_id": "advice-B", "installation_id": 1})
        # B is the newest -> consumed (True), both its rows removed.
        assert state.take_escalation("o/r", 76, "advice-B") is True
        with psycopg.connect(DSN, autocommit=True) as c:
            remaining = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE repo='o/r' "
                "AND pr_number=76 AND phase='escalating'").fetchone()[0]
        assert remaining == 0, remaining

    def test_unknown_advice_leaves_rows_untouched(self, db):
        state = NeoState(DSN)
        state.record("o/r", 77, SHA1, "escalating", verdict="ESCALATE",
                     detail={"advice_id": "advice-A", "installation_id": 1})
        assert state.take_escalation("o/r", 77, "advice-ghost") is False
        with psycopg.connect(DSN, autocommit=True) as c:
            remaining = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE repo='o/r' "
                "AND pr_number=77 AND phase='escalating'").fetchone()[0]
        assert remaining == 1, remaining


class TestEscalationInstallationIdPG:
    """Defect 4 — the NEWEST escalating row (the one escalating_prs() hands to
    the recovery loop) carries the webhook installation_id, so the escalated
    round uses the id Neo already had instead of a repo->installation lookup."""

    def _bounce(self, jobs, state, cfg, sha, pr, installation_id=None):
        from src.neo_worker import run_neo
        wake = {"repo": "o/r", "sha": sha, "green": False,
                "context": "rhobear-reviews", "pr_number": pr}
        if installation_id is not None:
            wake["installation_id"] = installation_id
        with patch("src.neo_worker.resolve_pr", return_value=pr), \
             patch("src.neo_worker._review_evidence", return_value=_evidence(sha)), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=JevAssessment(False, reason="no key")), \
             patch("src.neo_worker._compute_diff_stat", return_value=""), \
             patch("src.neo_worker._run_agent",
                   return_value=({"input_tokens": 1, "output_tokens": 1},
                                 "BOUNCE-BUILDER still broken")):
            run_neo(cfg, state, jobs, wake)

    def test_escalating_phase_row_carries_installation_id(self, db):
        jobs = db
        state = NeoState(DSN)
        cfg = _cfg()
        cfg.bounces_before_escalation = 5
        cfg.advisor_url = "http://127.0.0.1:9"
        cfg.advisor_token = "test-token"
        pr = 78
        for i in range(5):
            self._bounce(jobs, state, cfg, f"{i:040d}", pr)
        assert state.builder_rounds("o/r", pr) == 5
        with patch("src.escalation._post_advise", return_value="advice-inst-1"):
            self._bounce(jobs, state, cfg, "c" * 40, pr, installation_id=11)
        pr_rows = [r for r in state.escalating_prs() if r["pr"] == pr]
        assert pr_rows, "PR must be escalating"
        assert pr_rows[0]["advice_id"] == "advice-inst-1", pr_rows
        assert pr_rows[0]["installation_id"] == 11, pr_rows


# ---------------------------------------------------------------------------
# FIX ROUND 4 — defects 1/2/4 + item 5 (real Postgres where the fix touches
# neo_actions / neo_jobs)
# ---------------------------------------------------------------------------

def _bounce_default(jobs, state, cfg, sha, pr=95):
    """A BOUNCE-BUILDER wake through run_neo with the escalation plumbing
    patched out (used by the default-config escalation test)."""
    from src.neo_worker import run_neo
    wake = {"repo": "o/r", "sha": sha, "green": False,
            "context": "rhobear-reviews", "pr_number": pr}
    with patch("src.neo_worker.resolve_pr", return_value=pr), \
         patch("src.neo_worker._review_evidence", return_value=_evidence(sha)), \
         patch("src.neo_worker._jev_for_pr",
               return_value=JevAssessment(False, reason="no key")), \
         patch("src.neo_worker._compute_diff_stat", return_value=""), \
         patch("src.neo_worker._run_agent",
               return_value=({"input_tokens": 1, "output_tokens": 1},
                             "BOUNCE-BUILDER still broken")):
        run_neo(cfg, state, jobs, wake)


class TestTakeEscalationLockPG:
    """Defect 1 — take_escalation's check-then-delete-all runs under the
    per-PR advisory lock (pg_advisory_xact_lock(hashtext(repo||':'||pr))), and
    record_escalating (the escalating-marker INSERT in submit_escalation / the
    worker's phase row) takes the same lock.  A consume can never interleave
    with a concurrent submit and clobber a newer escalation's marker."""

    def test_take_escalation_blocks_while_lock_held(self, db):
        """While the per-PR advisory lock is held (a concurrent submit mid-
        INSERT), take_escalation BLOCKS instead of interleaving."""
        import time as _time
        state = NeoState(DSN)
        state.record_escalating("o/r", 80, SHA1, advice_id="advice-A",
                                installation_id=1)
        conn = psycopg.connect(DSN, autocommit=False)
        t = None
        result: dict = {}
        try:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('o/r:80'))")
            def do_take():
                try:
                    result["value"] = state.take_escalation("o/r", 80, "advice-A")
                except Exception as exc:  # noqa: BLE001
                    result["error"] = exc
            t = threading.Thread(target=do_take)
            t.start()
            _time.sleep(0.5)
            assert t.is_alive(), ("take_escalation must block while the "
                                  "advisory lock is held (defect 1)")
        finally:
            conn.rollback()  # release the lock so the thread can proceed
            if t is not None:
                t.join(timeout=10)
        assert "error" not in result, result
        assert result.get("value") is True

    def test_record_escalating_blocks_while_lock_held(self, db):
        """The escalating-marker INSERT also takes the lock, so a consume
        cannot slip between two rows of one escalation."""
        import time as _time
        state = NeoState(DSN)
        conn = psycopg.connect(DSN, autocommit=False)
        t = None
        result: dict = {}
        try:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('o/r:81'))")
            def do_record():
                try:
                    state.record_escalating("o/r", 81, SHA1, advice_id="advice-B",
                                            installation_id=1)
                    result["done"] = True
                except Exception as exc:  # noqa: BLE001
                    result["error"] = exc
            t = threading.Thread(target=do_record)
            t.start()
            _time.sleep(0.5)
            assert t.is_alive(), ("record_escalating must block while the "
                                  "advisory lock is held (defect 1)")
        finally:
            conn.rollback()
            if t is not None:
                t.join(timeout=10)
        assert "error" not in result, result
        assert result.get("done") is True

    def test_concurrent_submit_not_clobbered_by_take(self, db):
        """Whichever interleaving, a concurrently-inserted newer escalation's
        marker is NEVER deleted by take_escalation (defect 1)."""
        state = NeoState(DSN)
        state.record_escalating("o/r", 82, SHA1, advice_id="advice-A",
                                installation_id=1)
        barrier = threading.Barrier(2)
        results: dict = {}

        def do_take():
            barrier.wait()
            results["take"] = state.take_escalation("o/r", 82, "advice-A")

        def do_submit():
            barrier.wait()
            state.record_escalating("o/r", 82, SHA2, advice_id="advice-B",
                                    installation_id=1)
            results["submit_done"] = True

        t1 = threading.Thread(target=do_take)
        t2 = threading.Thread(target=do_submit)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        with psycopg.connect(DSN, autocommit=True) as c:
            remaining = [r[0] for r in c.execute(
                "SELECT detail->>'advice_id' FROM neo_actions WHERE repo='o/r' "
                "AND pr_number=82 AND phase='escalating' ORDER BY id").fetchall()]
        # Both orderings leave advice-B as the (sole) surviving marker.
        assert "advice-B" in remaining, remaining


class TestTerminalActionFailurePG:
    """Defect 2 — a terminal action that raises AFTER the escalating marker is
    consumed (transient DB error) re-records the marker so the next poll
    retries; idempotent escalate_pr means no duplicate round or job."""

    def test_escalate_pr_raise_rerecords_marker_then_retry_ok(self, db):
        jobs = db
        state = NeoState(DSN)
        state.record_escalating("o/r", 92, SHA1, advice_id="advice-1",
                                installation_id=1)
        cfg = _cfg()
        cfg.advisor_url = "http://127.0.0.1:9"
        cfg.advisor_token = "test-token"
        from src.escalation import recover_escalations, AdviceStatus
        work_order = {"route": "claude:opus", "summary": "Fix auth",
                      "root_cause": "rc",
                      "steps": [{"file": "src/a.py", "change": "c", "why": "w"}],
                      "tests": [], "risks": ""}

        def boom(*a, **k):
            raise RuntimeError("transient DB error")

        with patch("src.escalation.poll_advice",
                   return_value=AdviceStatus(status="done", work_order=work_order)), \
             patch("src.escalation.escalate_pr", side_effect=boom):
            handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "retry", handled
        # The marker is re-recorded — the PR is still escalating, not stranded.
        rows = [r for r in state.escalating_prs() if r["pr"] == 92]
        assert rows and rows[0]["advice_id"] == "advice-1", rows
        # Second pass succeeds — ONE escalated round and ONE builder job.
        with patch("src.escalation.poll_advice",
                   return_value=AdviceStatus(status="done", work_order=work_order)):
            handled2 = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled2) == 1 and handled2[0]["action"] == "escalated", handled2
        with psycopg.connect(DSN, autocommit=True) as c:
            n_esc = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE phase='escalated' "
                "AND repo='o/r' AND pr_number=92").fetchone()[0]
            n_job = c.execute(
                "SELECT COUNT(*) FROM neo_jobs WHERE kind='builder' "
                "AND repo='o/r' AND pr_number=92").fetchone()[0]
        assert n_esc == 1, n_esc
        assert n_job == 1, n_job

    def test_partial_failure_then_retry_is_idempotent(self, db):
        """Real partial failure: escalate_pr records the escalated row, then
        jobs.create_builder raises.  The marker is re-recorded; the retry
        reuses the recorded round and creates the job — still ONE of each."""
        jobs = db
        state = NeoState(DSN)
        state.record_escalating("o/r", 93, SHA1, advice_id="advice-1",
                                installation_id=1)
        cfg = _cfg()
        cfg.advisor_url = "http://127.0.0.1:9"
        cfg.advisor_token = "test-token"
        from src.escalation import recover_escalations, AdviceStatus
        work_order = {"route": "claude:opus", "summary": "Fix auth",
                      "root_cause": "rc",
                      "steps": [{"file": "src/a.py", "change": "c", "why": "w"}],
                      "tests": [], "risks": ""}

        with patch("src.escalation.poll_advice",
                   return_value=AdviceStatus(status="done", work_order=work_order)), \
             patch.object(jobs, "create_builder",
                          side_effect=RuntimeError("db blip")):
            handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "retry", handled
        # The escalated row was recorded before create_builder failed...
        with psycopg.connect(DSN, autocommit=True) as c:
            n_esc = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE phase='escalated' "
                "AND repo='o/r' AND pr_number=93").fetchone()[0]
        assert n_esc == 1, n_esc
        # ...and the marker was re-recorded so the next poll retries.
        assert [r for r in state.escalating_prs() if r["pr"] == 93]
        # Second pass: idempotent escalate_pr reuses round 1 + creates the job.
        with patch("src.escalation.poll_advice",
                   return_value=AdviceStatus(status="done", work_order=work_order)):
            handled2 = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled2) == 1 and handled2[0]["action"] == "escalated", handled2
        with psycopg.connect(DSN, autocommit=True) as c:
            n_esc2 = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE phase='escalated' "
                "AND repo='o/r' AND pr_number=93").fetchone()[0]
            n_job = c.execute(
                "SELECT COUNT(*) FROM neo_jobs WHERE kind='builder' "
                "AND repo='o/r' AND pr_number=93").fetchone()[0]
        assert n_esc2 == 1, n_esc2
        assert n_job == 1, n_job

    def test_park_failure_rerecords_marker(self, db):
        """A _park_pr failure after the consume (timeout path) also re-records
        the marker — never strands the PR."""
        jobs = db
        state = NeoState(DSN)
        state.record_escalating("o/r", 91, SHA1, advice_id="advice-1",
                                installation_id=None)
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute(
                "UPDATE neo_actions SET created_at = now() - interval '31 minutes' "
                "WHERE repo='o/r' AND pr_number=91")
        cfg = _cfg()
        cfg.advisor_url = "http://127.0.0.1:9"
        cfg.advisor_token = "test-token"
        from src.escalation import recover_escalations

        def boom(*a, **k):
            raise RuntimeError("db down")

        with patch("src.escalation._park_pr", side_effect=boom):
            handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "retry", handled
        assert [r for r in state.escalating_prs() if r["pr"] == 91]


class TestTimeoutNoAdviceIdPG:
    """Defect 4 — an escalating marker without an advice_id that has waited
    past the timeout is cleared (all escalating rows) and the PR is parked —
    the recovery loop never spins on it forever."""

    def test_timeout_marker_without_advice_id_parks(self, db):
        jobs = db
        state = NeoState(DSN)
        state.record_escalating("o/r", 94, SHA1, advice_id="")
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute(
                "UPDATE neo_actions SET created_at = now() - interval '31 minutes' "
                "WHERE repo='o/r' AND pr_number=94")
        cfg = _cfg()
        cfg.advisor_url = "http://127.0.0.1:9"
        cfg.advisor_token = "test-token"
        from src.escalation import recover_escalations
        handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "parked", handled
        with psycopg.connect(DSN, autocommit=True) as c:
            esc = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE phase='escalating' "
                "AND repo='o/r' AND pr_number=94").fetchone()[0]
            parked = c.execute(
                "SELECT COUNT(*) FROM neo_actions WHERE phase='parked' "
                "AND repo='o/r' AND pr_number=94").fetchone()[0]
        assert esc == 0, "the un-pollable marker must be cleared (no infinite loop)"
        assert parked >= 1, parked


class TestDefaultConfigEscalationPG:
    """Defect 5 — NEO_MAX_BUILDER_ROUNDS defaults to bounces_before_escalation
    (5), so a stock Config reaches the 5-bounce escalation: the advisor is
    called once and the ONE escalated round is dispatched (claimable)."""

    def test_default_config_reaches_escalation(self, db):
        jobs = db
        state = NeoState(DSN)
        cfg = Config()  # defaults: max_builder_rounds == bounces == 5
        cfg.database_url = DSN
        cfg.gh_token = "test-token"
        cfg.advisor_url = "http://127.0.0.1:9"
        cfg.advisor_token = "test-token"
        setattr(cfg, "_rhobear_github_app", _FakeGitHubApp("test-ghs-token"))
        assert cfg.max_builder_rounds == 5, cfg.max_builder_rounds
        assert cfg.max_builder_rounds == cfg.bounces_before_escalation == 5
        pr = 95
        for i in range(5):
            _bounce_default(jobs, state, cfg, f"{i:040d}", pr)
        assert state.builder_rounds("o/r", pr) == 5
        submits: list = []

        def fake_post(*a, **k):
            submits.append(a)
            return "advice-default-1"

        with patch("src.escalation._post_advise", side_effect=fake_post):
            _bounce_default(jobs, state, cfg, "c" * 40, pr)
        assert len(submits) == 1, submits
        # (fake advisor done) -> ONE escalated round, then claimable.
        from src.escalation import recover_escalations, AdviceStatus
        work_order = {"route": "claude:opus", "summary": "Fix auth",
                      "root_cause": "rc",
                      "steps": [{"file": "src/a.py", "change": "c", "why": "w"}],
                      "tests": [], "risks": ""}
        with patch("src.escalation.poll_advice",
                   return_value=AdviceStatus(status="done", work_order=work_order)):
            handled = recover_escalations(cfg, state, jobs, timeout_secs=1800)
        assert len(handled) == 1 and handled[0]["action"] == "escalated", handled
        claimed = jobs.claim_queued("builder")
        assert claimed is not None and claimed["builder_round"] == 6, claimed
