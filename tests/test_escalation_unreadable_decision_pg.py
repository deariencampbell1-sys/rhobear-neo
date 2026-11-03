"""r5 HIGH 2 — an UNREADABLE committed decision is deferred, never parked.

The ASTRA review of head 3ff1018 (comment 6609225440) found, in
`src/escalation.py`:

    _committed_decision() swallows *every* exception from
    advisor_escalation_detail() and returns None.  If escalating_prs()
    already returned an aged retry marker after the escalated decision
    committed but its builder job failed, a one-call lookup failure sets
    decided=False and the >1800s path writes a PARKED row.

So a TRANSIENT read error is read as "there is no decision" and a
successful escalation is parked — its builder round lost, and the
decision/job mismatch never repaired.

The rule this test pins:

  * a committed decision that cannot be READ is UNKNOWN, not ABSENT;
  * an unknown committed state DEFERS: no PARKED row, no terminal
    outcome, and the marker survives for the next poll;
  * once the read succeeds, the missing builder job is repaired exactly
    once (one job, one decision).

Real Postgres: the decision, the marker, the job and the recovery are
all real rows; only the decision READ is faulted (a transient failure).

Run: NEO_TEST_DSN='<dsn>' .venv/bin/python -m pytest \
     tests/test_escalation_unreadable_decision_pg.py -q
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
from src.neo_state import NeoState

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO_TEST_DSN"),
    reason="NEO_TEST_DSN not set (no throwaway Postgres for this run)",
)

HEAD = "d" * 40


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


class _FailingJobs:
    """The builder job insert fails AFTER the decision committed."""
    def create_builder(self, *a, **k):
        raise RuntimeError("the builder job insert failed")


def _work_order():
    return WorkOrder(
        route="claude:opus",
        summary="keep the field the handler drops",
        root_cause="the handler drops the field",
        steps=[{"file": "src/app.py", "change": "keep the field",
                "why": "the test asserts it"}],
        tests=["tests/test_app.py::test_field"],
        risks="none")


def _phases(dsn, repo, pr):
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        return [r["phase"] for r in c.execute(
            "SELECT phase FROM neo_actions WHERE repo=%s AND pr_number=%s "
            "ORDER BY id", (repo, pr)).fetchall()]


def _builder_jobs(dsn, repo, pr):
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        return c.execute(
            "SELECT id, builder_round, state FROM neo_jobs WHERE "
            "kind='builder' AND repo=%s AND pr_number=%s ORDER BY id",
            (repo, pr)).fetchall()


def _age_escalating(dsn, repo, pr, *, seconds):
    """The retry marker keeps the ORIGINAL start — age it so the recovery
    loop would take the 'the advisor never finished' path."""
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "UPDATE neo_actions SET detail = jsonb_set(detail, "
            "'{escalation_started}', to_jsonb(%s::double precision)) "
            "WHERE repo=%s AND pr_number=%s AND phase='escalating'",
            (time.time() - seconds, repo, pr))


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


def _fail_the_next_decision_read(state, monkeypatch):
    """The committed-decision lookup fails ONCE (a transient read error)."""
    real = state.advisor_escalation_detail
    calls = {"n": 0}

    def _flaky(repo, pr, advice_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient read failure")
        return real(repo, pr, advice_id)

    monkeypatch.setattr(state, "advisor_escalation_detail", _flaky)
    return calls


def test_an_unreadable_committed_decision_is_deferred_never_parked(
        fresh, monkeypatch):
    """HIGH 2 — the decision read fails once on an AGED marker: the
    recovery must NOT write a PARKED row (the escalation was successful),
    must not consume the marker, and must leave the round repairable."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 960, HEAD, advice_id="adv-unreadable",
                            installation_id=5)
    _commit_decision_without_its_job(state, "o/r", 960, "adv-unreadable")
    _age_escalating(fresh, "o/r", 960, seconds=7200)
    _fail_the_next_decision_read(state, monkeypatch)

    def _no_poll(cfg, advice_id):
        raise AssertionError("an unreadable decision must not be polled")

    monkeypatch.setattr(esc, "poll_advice", _no_poll)

    handled = esc.recover_escalations(_Cfg(), state, Jobs(fresh),
                                      timeout_secs=1800)

    phases = _phases(fresh, "o/r", 960)
    assert "parked" not in phases, \
        f"a successful escalation was parked on a read failure: {phases}"
    assert "escalating" in phases, \
        f"the retry marker was consumed on a read failure: {phases}"
    assert [h["action"] for h in handled] == ["deferred"], handled


def test_the_decision_is_repaired_once_after_the_read_recovers(
        fresh, monkeypatch):
    """The next poll reads the decision: the missing builder job is
    repaired EXACTLY once — one job, one decision, never parked."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 961, HEAD, advice_id="adv-recover",
                            installation_id=5)
    _commit_decision_without_its_job(state, "o/r", 961, "adv-recover")
    _age_escalating(fresh, "o/r", 961, seconds=7200)
    _fail_the_next_decision_read(state, monkeypatch)

    def _no_poll(cfg, advice_id):
        raise AssertionError("the repair must not need the advisor")

    monkeypatch.setattr(esc, "poll_advice", _no_poll)

    esc.recover_escalations(_Cfg(), state, Jobs(fresh), timeout_secs=1800)
    assert "parked" not in _phases(fresh, "o/r", 961), \
        _phases(fresh, "o/r", 961)

    # the read recovers -> the committed decision repairs its builder job
    esc.recover_escalations(_Cfg(), state, Jobs(fresh), timeout_secs=1800)

    jobs = _builder_jobs(fresh, "o/r", 961)
    assert len(jobs) == 1, f"the builder job was not repaired: {jobs}"
    phases = _phases(fresh, "o/r", 961)
    assert phases.count("escalated") == 1, \
        f"the repair wrote a second decision: {phases}"
    assert "parked" not in phases, \
        f"a successful advice was parked: {phases}"
