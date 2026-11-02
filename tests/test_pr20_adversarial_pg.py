"""Red-first review cases for rhobear-neo PR #20 @ 814a3c92.

Run against a throwaway real PostgreSQL: NEO_TEST_DSN=... python -m pytest -q
    tests/test_pr20_adversarial_pg.py
This file must remain red until the per-advice lifecycle is corrected.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from psycopg.rows import dict_row

from src.escalation import _consume_and_act, recover_escalations
from src.neo_state import ESCALATION_INFLIGHT_PHASE, NeoState

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO_TEST_DSN"), reason="requires throwaway real PostgreSQL"
)
HEAD_A, HEAD_B = "a" * 40, "b" * 40


@pytest.fixture()
def fresh():
    dsn = os.environ["NEO_TEST_DSN"]
    NeoState(dsn).ensure_schema()
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE neo_actions, neo_jobs")
    yield dsn
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE neo_actions, neo_jobs")


def _owners(dsn, pr):
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT phase, detail->>'advice_id' AS advice_id FROM neo_actions "
            "WHERE repo='o/r' AND pr_number=%s AND phase IN "
            "('escalating', %s) ORDER BY id", (pr, ESCALATION_INFLIGHT_PHASE)
        ).fetchall()
    return [(r["advice_id"], r["phase"]) for r in rows]


def _age(dsn, pr, advice_id):
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "UPDATE neo_actions SET detail=jsonb_set(detail, '{consumed_at}', "
            "to_jsonb(%s::double precision)) WHERE repo='o/r' "
            "AND pr_number=%s AND phase=%s AND detail->>'advice_id'=%s",
            (time.time() - 7200, pr, ESCALATION_INFLIGHT_PHASE, advice_id)
        )


def test_completing_advice_a_preserves_concurrent_advice_b(fresh):
    """Two independently connected workers: B is held in flight as A finishes."""
    a, b = NeoState(fresh), NeoState(fresh)
    a.record_escalating("o/r", 901, HEAD_A, advice_id="A")
    entered, release = threading.Event(), threading.Event()

    def act_a():
        entered.set()
        assert release.wait(timeout=10)
        a.record("o/r", 901, HEAD_A, "parked", detail={"advice_id": "A"})
        return {"action": "parked"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(_consume_and_act, a, "o/r", 901, HEAD_A,
                             "A", 1, act_a)
        try:
            assert entered.wait(timeout=10), "A did not reach the action barrier"
            b.record_escalating("o/r", 901, HEAD_B, advice_id="B")
            assert b.take_escalation("o/r", 901, "B") is True
        finally:
            release.set()
        assert future.result(timeout=10)["action"] == "parked"

    assert _owners(fresh, 901) == [("B", ESCALATION_INFLIGHT_PHASE)], \
        "A's completion erased B's durable ownership"
    _age(fresh, 901, "B")
    b.reclaim_stale_inflight()
    assert _owners(fresh, 901) == [("B", "escalating")], \
        "B must remain recoverable after A completes"


def test_two_stale_advices_reclaim_only_newest(fresh):
    state = NeoState(fresh)
    state.record_escalating("o/r", 902, HEAD_A, advice_id="A")
    assert state.take_escalation("o/r", 902, "A")
    state.record_escalating("o/r", 902, HEAD_B, advice_id="B")
    assert state.take_escalation("o/r", 902, "B")
    _age(fresh, 902, "A")
    _age(fresh, 902, "B")
    state.reclaim_stale_inflight()
    assert _owners(fresh, 902) == [("B", "escalating")], \
        "old A must not be revived while newer B is discarded"


class _Cfg:
    advisor_url = "http://127.0.0.1:9"
    advisor_token = "unused"


def _terminal_outcome(state, pr):
    state.record("o/r", pr, HEAD_A, "escalated", verdict="ESCALATE",
                 builder_round=6,
                 detail={"advice_id": "A", "advisor_round": 6})
    return {"action": "escalated"}


def _assert_recovery_does_not_park(fresh, pr):
    _age(fresh, pr, "A")
    recover_escalations(_Cfg(), NeoState(fresh), None, timeout_secs=1800)
    assert NeoState(fresh).advisor_escalation_for("o/r", pr, "A") == 6
    assert _owners(fresh, pr) == [], "settled marker must be cleared"
    with psycopg.connect(fresh) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM neo_actions WHERE repo='o/r' "
            "AND pr_number=%s AND phase='parked'", (pr,)
        ).fetchone()[0]
    assert count == 0, "successful escalation was followed by contradictory PARKED"


def test_crash_after_terminal_commit_before_cleanup(fresh):
    state = NeoState(fresh)
    state.record_escalating("o/r", 903, HEAD_A, advice_id="A",
                            extra={"escalation_started": time.time() - 7200})
    assert state.take_escalation("o/r", 903, "A")
    _terminal_outcome(state, 903)  # process dies here, before cleanup
    _assert_recovery_does_not_park(fresh, 903)


def test_cleanup_exception_after_terminal_commit(fresh, monkeypatch):
    state = NeoState(fresh)
    state.record_escalating("o/r", 904, HEAD_A, advice_id="A",
                            extra={"escalation_started": time.time() - 7200})
    with monkeypatch.context() as patch:
        patch.setattr(state, "clear_inflight_markers",
                      lambda *args: (_ for _ in ()).throw(OSError("disk fault")))
        result = _consume_and_act(
            state, "o/r", 904, HEAD_A, "A", 1,
            lambda: _terminal_outcome(state, 904))
    assert result["action"] == "escalated"
    _assert_recovery_does_not_park(fresh, 904)
