"""r4 MEDIUM 2 — a COMMITTED escalation decision must never lose its builder
job, and must never be parked.

`escalate_pr` writes the escalation DECISION (the 'escalated' row) BEFORE
the builder JOB is created.  When the job insert fails, the decision exists
without its effect: the retry marker is re-recorded, and the recovery loop
can then take the 'the advisor never finished' path and PARK the PR — a
successful advice parked and its round lost (the review's exact wording:
"the escalated outcome still loses its builder job / may append PARKED").

The rule these tests pin:

  * the recovery REPAIRS a committed decision whose builder job is missing
    (idempotently — the job of the recorded round is created, never a second
    round, never a second decision);
  * a committed decision is NEVER parked;
  * the stale reclaim treats an escalated row as a SETTLED outcome only when
    its builder job exists — a partial outcome is re-driven, not cleared.

Real Postgres: the decision row, the retry marker, the builder job and the
reclaim are all real rows.

Run: NEO_TEST_DSN='<dsn>' .venv/bin/python -m pytest \
     tests/test_escalation_partial_outcome_pg.py -q
"""
from __future__ import annotations

import os
import time

import psycopg
import pytest
from psycopg.rows import dict_row

from src import escalation as esc
from src.escalation import WorkOrder, escalate_pr
from src.jobs import Jobs
from src.neo_state import ESCALATION_INFLIGHT_PHASE, NeoState

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO_TEST_DSN"),
    reason="NEO_TEST_DSN not set (no throwaway Postgres for this run)",
)

HEAD = "a" * 40


@pytest.fixture()
def dsn():
    return os.environ["NEO_TEST_DSN"]


@pytest.fixture()
def fresh(dsn):
    NeoState(dsn).ensure_schema()
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("TRUNCATE neo_actions")
        c.execute("TRUNCATE neo_jobs")
    yield dsn
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("TRUNCATE neo_actions")
        c.execute("TRUNCATE neo_jobs")


class _Cfg:
    advisor_url = "http://127.0.0.1:9"
    advisor_token = "t"


def _phases(dsn, repo, pr):
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        return [r["phase"] for r in c.execute(
            "SELECT phase FROM neo_actions WHERE repo=%s AND pr_number=%s "
            "ORDER BY id", (repo, pr)).fetchall()]


def _decision_round(dsn, repo, pr):
    """The builder round of the committed decision (or None)."""
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        rows = c.execute(
            "SELECT builder_round FROM neo_actions WHERE repo=%s AND "
            "pr_number=%s AND phase='escalated' ORDER BY id",
            (repo, pr)).fetchall()
    return int(rows[0]["builder_round"]) if rows else None


def _builder_jobs(dsn, repo, pr):
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        return c.execute(
            "SELECT id, builder_round, state FROM neo_jobs WHERE "
            "kind='builder' AND repo=%s AND pr_number=%s ORDER BY id",
            (repo, pr)).fetchall()


def _age_escalating(dsn, repo, pr, *, seconds):
    """The retry marker keeps the ORIGINAL start — age it so the recovery
    loop takes the 'the advisor never finished' path."""
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "UPDATE neo_actions SET detail = jsonb_set(detail, "
            "'{escalation_started}', to_jsonb(%s::double precision)) "
            "WHERE repo=%s AND pr_number=%s AND phase='escalating'",
            (time.time() - seconds, repo, pr))


def _age_inflight(dsn, repo, pr, *, seconds):
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "UPDATE neo_actions SET detail = jsonb_set(detail, "
            "'{consumed_at}', to_jsonb(%s::double precision)) "
            "WHERE repo=%s AND pr_number=%s AND phase=%s",
            (time.time() - seconds, repo, pr, ESCALATION_INFLIGHT_PHASE))


def _work_order():
    return WorkOrder(
        route="claude:opus",
        summary="keep the field the handler drops",
        root_cause="the handler drops the field",
        steps=[{"file": "src/app.py", "change": "keep the field",
                "why": "the test asserts it"}],
        tests=["tests/test_app.py::test_field"],
        risks="none")


class _FailingJobs:
    """The builder job insert fails AFTER the decision committed."""
    def create_builder(self, *a, **k):
        raise RuntimeError("the builder job insert failed")


def _commit_decision_without_its_job(state, repo, pr, advice_id):
    """The real flow: the decision commits, the builder job insert fails,
    the marker is re-recorded for the next poll."""
    out = esc._consume_and_act(
        state, repo, pr, HEAD, advice_id, 5,
        lambda: escalate_pr(_Cfg(), state, _FailingJobs(), repo, pr, HEAD,
                            _work_order(), installation_id=5,
                            advice_id=advice_id),
        detail={"advice_id": advice_id})
    assert out["action"] == "retry", out
    assert _phases(state.dsn, repo, pr).count("escalated") == 1, \
        "the decision was not committed"
    assert _builder_jobs(state.dsn, repo, pr) == [], \
        "the builder job must still be missing"


def test_a_committed_decision_without_its_job_is_repaired_never_parked(
        fresh, monkeypatch):
    """The recovery REPAIRS the partial outcome: the builder job of the
    committed round is created (so the round runs), the decision is never
    duplicated and the successful advice is NEVER parked."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 340, HEAD, advice_id="adv-repair",
                            installation_id=5)
    _commit_decision_without_its_job(state, "o/r", 340, "adv-repair")

    # The retry marker kept the ORIGINAL start: the recovery loop now takes
    # the 'the advisor never finished' path.
    _age_escalating(fresh, "o/r", 340, seconds=7200)

    def _no_poll(cfg, advice_id):
        raise AssertionError("the repair must not need the advisor")

    monkeypatch.setattr(esc, "poll_advice", _no_poll)

    handled = esc.recover_escalations(_Cfg(), state, Jobs(fresh),
                                      timeout_secs=1800)
    assert handled, "the stranded advice was never handled"

    jobs = _builder_jobs(fresh, "o/r", 340)
    assert len(jobs) == 1, f"the builder job was not repaired: {jobs}"
    assert jobs[0]["builder_round"] == _decision_round(fresh, "o/r", 340), jobs

    phases = _phases(fresh, "o/r", 340)
    assert phases.count("escalated") == 1, \
        f"the repair wrote a second decision: {phases}"
    assert "parked" not in phases, \
        f"a successful advice was parked: {phases}"


def test_the_repair_is_idempotent(fresh, monkeypatch):
    """Recovery runs again and again: the repair NEVER creates a second
    builder job, never a second decision and never a second round."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 342, HEAD, advice_id="adv-idem",
                            installation_id=5)
    _commit_decision_without_its_job(state, "o/r", 342, "adv-idem")
    _age_escalating(fresh, "o/r", 342, seconds=7200)

    def _no_poll(cfg, advice_id):
        raise AssertionError("the repair must not need the advisor")

    monkeypatch.setattr(esc, "poll_advice", _no_poll)

    for _ in range(3):
        esc.recover_escalations(_Cfg(), state, Jobs(fresh), timeout_secs=1800)

    assert len(_builder_jobs(fresh, "o/r", 342)) == 1, \
        _builder_jobs(fresh, "o/r", 342)
    assert _phases(fresh, "o/r", 342).count("escalated") == 1, \
        _phases(fresh, "o/r", 342)


def test_a_decision_without_its_builder_job_is_not_settled(fresh):
    """The stale reclaim's settled check: an escalated row is a SETTLED
    outcome only when its builder job exists.  A decision WITHOUT its job
    is a partial outcome — it is re-driven (so the repair can create the
    job), never cleared as if the round were done."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 341, HEAD, advice_id="adv-partial")
    assert state.take_escalation("o/r", 341, "adv-partial") is True
    state.record("o/r", 341, HEAD, "escalated", verdict="ESCALATE",
                 builder_round=1,
                 detail={"advice_id": "adv-partial", "advisor_round": 1})
    assert _builder_jobs(fresh, "o/r", 341) == []
    _age_inflight(fresh, "o/r", 341, seconds=7200)

    reclaimed = state.reclaim_stale_inflight(stale_secs=3600)
    assert reclaimed == [{"repo": "o/r", "pr": 341,
                          "action": "re-recorded"}], reclaimed


def test_a_settled_decision_with_its_job_is_cleared(fresh):
    """The other half of the rule: a decision whose builder job EXISTS is
    settled — the stale marker is cleared and nothing is re-driven."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 343, HEAD, advice_id="adv-settled")
    assert state.take_escalation("o/r", 343, "adv-settled") is True
    state.record("o/r", 343, HEAD, "escalated", verdict="ESCALATE",
                 builder_round=2,
                 detail={"advice_id": "adv-settled", "advisor_round": 2})
    Jobs(fresh).create_builder("o/r", 343, HEAD, 2, payload={"round": 2})
    _age_inflight(fresh, "o/r", 343, seconds=7200)

    reclaimed = state.reclaim_stale_inflight(stale_secs=3600)
    assert reclaimed == [{"repo": "o/r", "pr": 343,
                          "action": "cleared (outcome already recorded)"}], \
        reclaimed
    with psycopg.connect(fresh, row_factory=dict_row) as c:
        rows = c.execute(
            "SELECT phase FROM neo_actions WHERE repo='o/r' AND "
            "pr_number=343 AND phase=%s",
            (ESCALATION_INFLIGHT_PHASE,)).fetchall()
    assert rows == [], rows
