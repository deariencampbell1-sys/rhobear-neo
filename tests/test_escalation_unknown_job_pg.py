"""r5 HIGH 1 — an UNKNOWN builder-job state is NEVER a settled escalation.

The ASTRA review of head 3ff1018 (comment 6609225440) found, in
`src/neo_state.py`:

    _builder_job_exists() explicitly returns None when neo_jobs is
    unavailable or a round cannot be parsed; the caller only continues
    when the result `is False`.  None therefore falls through to
    `settled=True` and DELETEs the stale in-flight row.

So when the builder job's existence cannot be TOLD (the supported
`to_regclass('neo_jobs')` probe cannot see the table, or the recorded
round cannot be parsed), the stale reclaim treats the escalated
DECISION as complete and deletes the only recovery marker — the escalated
builder round is lost without its job ever being created.

The rule this test pins:

  * only an explicit TRUE may settle an escalation;
  * a FALSE (the job is missing) and a NONE (the job state is unknown)
    both keep the marker: it is re-driven, never cleared as if the round
    were done.

Real Postgres: the escalated decision, the in-flight marker and the
reclaim are all real rows.  The "unknown" job state is produced by a
schema whose search_path has `neo_actions` but NOT `neo_jobs` — exactly
the supported to_regclass path the review names.

Run: NEO_TEST_DSN='<dsn>' .venv/bin/python -m pytest \
     tests/test_escalation_unknown_job_pg.py -q
"""
from __future__ import annotations

import os
import time

import psycopg
import pytest
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

from src.neo_state import (ESCALATION_INFLIGHT_PHASE, SCHEMA, NeoState,
                           _builder_job_exists)

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO_TEST_DSN"),
    reason="NEO_TEST_DSN not set (no throwaway Postgres for this run)",
)

HEAD = "c" * 40

#: A schema with neo_actions but WITHOUT neo_jobs: the state cannot tell
#: whether the builder job exists.
NOJOBS_SCHEMA = "neo_nojobs_r5"


@pytest.fixture()
def dsn():
    return os.environ["NEO_TEST_DSN"]


@pytest.fixture()
def nojobs_dsn(dsn):
    """A DSN whose search_path holds neo_actions but NOT neo_jobs."""
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(f"DROP SCHEMA IF EXISTS {NOJOBS_SCHEMA} CASCADE")
        c.execute(f"CREATE SCHEMA {NOJOBS_SCHEMA}")
    conn = make_conninfo(dsn, options=f"-c search_path={NOJOBS_SCHEMA}")
    # NeoState.SCHEMA only (NOT ensure_schema: that would create neo_jobs).
    with psycopg.connect(conn, autocommit=True) as c:
        c.execute(SCHEMA)
    yield conn
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(f"DROP SCHEMA IF EXISTS {NOJOBS_SCHEMA} CASCADE")


def _age_inflight(dsn, repo, pr, *, seconds):
    """Age the in-flight marker so the stale reclaim considers it.

    The consumed_at is a real PAST timestamp (not 0): the reclaim's ABA
    guard refuses a marker whose live consumed_at is <= 0, so aging it to
    0 would skip the row instead of exercising the settle path."""
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "UPDATE neo_actions SET detail = jsonb_set(detail, "
            "'{consumed_at}', to_jsonb(%s::double precision)) "
            "WHERE repo=%s AND pr_number=%s AND phase=%s",
            (time.time() - seconds, repo, pr, ESCALATION_INFLIGHT_PHASE))


def _rows(dsn, repo, pr):
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        return c.execute(
            "SELECT id, phase, detail FROM neo_actions WHERE repo=%s AND "
            "pr_number=%s ORDER BY id", (repo, pr)).fetchall()


def test_the_probe_reports_unknown_when_the_table_is_not_visible(nojobs_dsn):
    """The supported to_regclass probe reports the job state as UNKNOWN
    (None) when neo_jobs is not reachable — never as "the job is there"."""
    state = NeoState(nojobs_dsn)
    with state._conn() as c:
        assert _builder_job_exists(c, "o/r", 1, 1) is None, \
            "the probe must not answer 'the job exists' when it cannot tell"


def test_the_probe_reports_unknown_for_a_malformed_round(dsn):
    """A round that cannot be parsed is UNKNOWN, not 'the job exists'."""
    NeoState(dsn).ensure_schema()
    with psycopg.connect(dsn) as c:
        assert _builder_job_exists(c, "o/r", 1, "not-a-round") is None


def test_an_unknown_builder_job_state_never_clears_the_marker(nojobs_dsn):
    """HIGH 1 — the stale reclaim must NOT call the decision complete when
    the builder job's existence cannot be told: the marker survives (the
    round is re-driven), and the escalated decision is not lost."""
    state = NeoState(nojobs_dsn)
    state.record_escalating("o/r", 900, HEAD, advice_id="adv-unknown")
    assert state.take_escalation("o/r", 900, "adv-unknown") is True
    # the decision committed; its builder job was never created
    state.record("o/r", 900, HEAD, "escalated", verdict="ESCALATE",
                 builder_round=1,
                 detail={"advice_id": "adv-unknown", "advisor_round": 1})
    _age_inflight(nojobs_dsn, "o/r", 900, seconds=7200)

    reclaimed = state.reclaim_stale_inflight(stale_secs=3600)

    assert [r["action"] for r in reclaimed] != [
        "cleared (outcome already recorded)"], \
        f"an UNKNOWN job state settled the escalation: {reclaimed}"
    rows = _rows(nojobs_dsn, "o/r", 900)
    assert len(rows) == 2, f"the recovery marker was deleted: {rows}"
    assert rows[0]["id"] == 1, f"the recovery marker was deleted: {rows}"
    assert rows[1]["phase"] == "escalated", rows
