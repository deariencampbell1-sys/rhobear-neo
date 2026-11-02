"""W1-08-r2 — the judge's ONLY writer against REAL Postgres (HIGH 2 + 6).

Runs only when NEO_TEST_DSN is set (the same throwaway PG the jobs tests
use — see tests/test_jobs_pg.py). Fakes alone are not acceptable for the
atomicity claims, so these exercise the actual advisory-lock + CAS SQL of
src/kronos/store.py against the real schema:

- HIGH 2: a `yes` really records the escalated neo_actions row AND the
  round-6 builder job in ONE transaction (the r1 code called
  db.record()/db.create_builder() on a read-only rows adapter that has
  neither);
- HIGH 6: the round-six ceiling, the escalating marker and the escalated
  marker are re-checked INSIDE the same locked transaction (compare-and-set)
  — a lost race defers, never a 7th round, never a second escalated row;
- the stop marker is the same shape Zeus's already_escalated() consumes.

Run:  NEO_TEST_DSN='<dsn>' .venv/bin/python -m pytest tests/test_kronos_store_pg.py -q
"""
from __future__ import annotations

import json
import os
import threading

import psycopg
import pytest
from psycopg.rows import dict_row

from src.jobs import Jobs
from src.neo_state import NeoState

# The store lands with the r2 fix commit; on the red commit this module is
# skipped as a whole instead of erroring at collection.
_store = pytest.importorskip(
    "src.kronos.store",
    reason="src/kronos/store.py not present yet (red-first commit)")
SqlWrites = _store.SqlWrites

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO_TEST_DSN"),
    reason="NEO_TEST_DSN not set (no throwaway Postgres for this run)",
)


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


def _builder_rounds(dsn, repo: str, pr: int, n: int, *,
                    start: int = 1) -> None:
    state = NeoState(dsn)
    for i in range(start, start + n):
        state.record(repo, pr, f"{'a' * 39}{chr(97 + i % 26)}", "builder",
                     verdict="BOUNCE", builder_round=i,
                     detail={"tree_sha": f"tree{i}"})


def _escalated_rows(dsn, repo: str, pr: int) -> list[dict]:
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        return c.execute(
            "SELECT * FROM neo_actions WHERE repo=%s AND pr_number=%s "
            "AND phase='escalated'", (repo, pr)).fetchall()


def test_yes_records_escalated_row_and_builder_job_in_one_write(fresh):
    repo, pr = "o/r", 52
    _builder_rounds(fresh, repo, pr, 5)
    writes = SqlWrites(fresh)
    ok, note = writes.escalate(
        repo, pr, "h" * 40, advisor_round=6,
        detail={"reason": "kronos judge: one escalated round",
                "advisor_round": 6, "advice_id": f"kronos:{repo}#{pr}",
                "kronos": True},
        builder_payload={"verdict": "BOUNCE-BUILDER", "phase": "escalated",
                         "round": 6, "escalated": True,
                         "work_order": {"summary": "the plan"}})
    assert ok, note
    rows = _escalated_rows(fresh, repo, pr)
    assert len(rows) == 1 and rows[0]["builder_round"] == 6
    detail = rows[0]["detail"]
    if isinstance(detail, str):
        detail = json.loads(detail)
    assert detail["advisor_round"] == 6
    job = Jobs(fresh)
    with psycopg.connect(fresh, row_factory=psycopg.rows.dict_row) as c:
        jrow = c.execute(
            "SELECT * FROM neo_jobs WHERE kind='builder' AND repo=%s "
            "AND pr_number=%s AND builder_round=6", (repo, pr)).fetchone()
    assert jrow is not None, "the round-6 builder job was not created"
    assert jrow["state"] == "queued"
    payload = jrow["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    assert payload["escalated"] is True
    # the worker can actually claim it (real claim path)
    claimed = job.claim_queued("builder")
    assert claimed is not None and claimed["builder_round"] == 6


def test_second_escalate_defers_never_a_second_row(fresh):
    repo, pr = "o/r", 7
    _builder_rounds(fresh, repo, pr, 5)
    writes = SqlWrites(fresh)
    ok1, _ = writes.escalate(repo, pr, "h" * 40, advisor_round=6,
                             detail={"advisor_round": 6},
                             builder_payload={"escalated": True})
    assert ok1
    ok2, note2 = writes.escalate(repo, pr, "h" * 40, advisor_round=6,
                                 detail={"advisor_round": 6},
                                 builder_payload={"escalated": True})
    assert not ok2, "a second `yes` wrote a second escalated row"
    assert "deferred" in note2.lower() or "already" in note2.lower(), note2
    assert len(_escalated_rows(fresh, repo, pr)) == 1


def test_round_six_ceiling_is_enforced_inside_the_lock(fresh):
    repo, pr = "o/r", 8
    _builder_rounds(fresh, repo, pr, 6)
    writes = SqlWrites(fresh)
    ok, note = writes.escalate(repo, pr, "h" * 40, advisor_round=7,
                               detail={}, builder_payload={})
    assert not ok, "a 7th round was written"
    assert "6" in note or "ceiling" in note.lower(), note
    assert _escalated_rows(fresh, repo, pr) == []


def test_escalating_marker_defers_inside_the_lock(fresh):
    repo, pr = "o/r", 9
    _builder_rounds(fresh, repo, pr, 5)
    NeoState(fresh).record_escalating(repo, pr, "h" * 40, advice_id="zeus-1")
    writes = SqlWrites(fresh)
    ok, note = writes.escalate(repo, pr, "h" * 40, advisor_round=6,
                               detail={}, builder_payload={})
    assert not ok, "Kronos escalated over Zeus's in-flight advisor path"
    assert "escalating" in note.lower() or "zeus" in note.lower(), note
    assert _escalated_rows(fresh, repo, pr) == []


def test_stop_records_the_marker_zeus_consumes(fresh):
    repo, pr = "o/r", 10
    _builder_rounds(fresh, repo, pr, 5)
    writes = SqlWrites(fresh)
    ok, _note = writes.stop(repo, pr, "h" * 40, advisor_round=6,
                            detail={"reason": "kronos judge stopped the loop",
                                    "advisor_round": 6, "kronos_stop": True})
    assert ok
    rows = _escalated_rows(fresh, repo, pr)
    assert len(rows) == 1
    detail = rows[0]["detail"]
    if isinstance(detail, str):
        detail = json.loads(detail)
    assert detail["kronos_stop"] is True
    assert detail["advisor_round"] == 6, \
        "already_escalated() keys on detail->advisor_round"
    with psycopg.connect(fresh) as c:
        n = c.execute(
            "SELECT COUNT(*) FROM neo_jobs WHERE kind='builder' AND repo=%s",
            (repo,)).fetchone()[0]
    assert n == 0, "a stop never creates a builder round"


def test_second_stop_defers(fresh):
    repo, pr = "o/r", 11
    _builder_rounds(fresh, repo, pr, 5)
    writes = SqlWrites(fresh)
    ok1, _ = writes.stop(repo, pr, "h" * 40, advisor_round=6, detail={})
    ok2, note2 = writes.stop(repo, pr, "h" * 40, advisor_round=6, detail={})
    assert ok1 and not ok2, "two stop markers were written"
    assert len(_escalated_rows(fresh, repo, pr)) == 1


def test_concurrent_escalations_produce_exactly_one_row(fresh):
    """The atomicity claim, raced for real: two Kronos passes (threads)
    escalate the same PR at the same time — exactly ONE escalated row and
    ONE builder job may survive."""
    repo, pr = "o/r", 12
    _builder_rounds(fresh, repo, pr, 5)
    writes_a = SqlWrites(fresh)
    writes_b = SqlWrites(fresh)
    results: list[tuple[bool, str]] = []

    def _go(writes):
        results.append(writes.escalate(
            repo, pr, "h" * 40, advisor_round=6,
            detail={"advisor_round": 6}, builder_payload={"escalated": True}))

    threads = [threading.Thread(target=_go, args=(writes_a,)),
               threading.Thread(target=_go, args=(writes_b,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wins = [ok for ok, _note in results]
    assert sorted(wins, reverse=True) == [True, False], results
    assert len(_escalated_rows(fresh, repo, pr)) == 1
    with psycopg.connect(fresh) as c:
        n = c.execute(
            "SELECT COUNT(*) FROM neo_jobs WHERE kind='builder' "
            "AND repo=%s AND pr_number=%s AND builder_round=6",
            (repo, pr)).fetchone()[0]
    assert n == 1, "two builder jobs for the same round"
