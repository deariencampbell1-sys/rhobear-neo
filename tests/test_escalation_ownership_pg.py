"""Adversarial real-PostgreSQL ownership regressions for W1-08-r4.

These tests intentionally use independent psycopg sessions. They are gated by
NEO_TEST_DSN and must not be represented as run when that DSN is unavailable.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest

from src.escalation import _consume_and_act
from src.kronos.store import SqlWrites
from src.neo_state import ESCALATION_INFLIGHT_PHASE, NeoState

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO_TEST_DSN"),
    reason="NEO_TEST_DSN not set (real PostgreSQL required)",
)
REPO = "test/ownership"
HEAD_A = "a" * 40
HEAD_B = "b" * 40


@pytest.fixture
def dsn():
    dsn = os.environ["NEO_TEST_DSN"]
    NeoState(dsn).ensure_schema()
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE neo_actions")
        conn.execute("TRUNCATE neo_jobs")
    yield dsn
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE neo_actions")
        conn.execute("TRUNCATE neo_jobs")


def _owners(dsn, pr):
    with psycopg.connect(dsn) as conn:
        return conn.execute(
            "SELECT phase, detail->>'advice_id' FROM neo_actions "
            "WHERE repo=%s AND pr_number=%s "
            "AND phase IN ('escalating', %s) ORDER BY id",
            (REPO, pr, ESCALATION_INFLIGHT_PHASE),
        ).fetchall()


def _age_inflight(dsn, pr):
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "UPDATE neo_actions SET detail=jsonb_set(detail, "
            "'{consumed_at}', to_jsonb(%s::double precision)) "
            "WHERE repo=%s AND pr_number=%s AND phase=%s",
            (time.time() - 7200, REPO, pr, ESCALATION_INFLIGHT_PHASE),
        )


def test_two_sessions_a_completion_cannot_erase_b_ownership(dsn):
    """A completes while B is paused between take and outcome. B must still
    block Kronos on a quiet PR, then survive simulated process death and be
    eligible for stale reclaim. Every actor opens its own DB connection."""
    state = NeoState(dsn)
    pr = 9101
    a_inside = threading.Event()
    a_finish = threading.Event()
    b_inside = threading.Event()
    b_crash = threading.Event()

    def a_act():
        a_inside.set()
        assert a_finish.wait(15), "A barrier timed out"
        state.record(REPO, pr, HEAD_A, "parked", detail={"reason": "A ended"})
        return {"action": "parked"}

    def b_act():
        b_inside.set()
        assert b_crash.wait(15), "B barrier timed out"
        raise SystemExit("simulate worker B crash before outcome")

    def b_worker():
        try:
            return _consume_and_act(state, REPO, pr, HEAD_B, "advice-B",
                                    2, b_act)
        except SystemExit:
            return "crashed"

    state.record_escalating(REPO, pr, HEAD_A, advice_id="advice-A")
    with ThreadPoolExecutor(max_workers=2) as pool:
        a_future = pool.submit(_consume_and_act, state, REPO, pr,
                               HEAD_A, "advice-A", 1, a_act)
        try:
            assert a_inside.wait(15), "A did not consume its advice"
            state.record_escalating(REPO, pr, HEAD_B, advice_id="advice-B")
            b_future = pool.submit(b_worker)
            assert b_inside.wait(15), "B did not consume its advice"
            assert _owners(dsn, pr) == [
                (ESCALATION_INFLIGHT_PHASE, "advice-A"),
                (ESCALATION_INFLIGHT_PHASE, "advice-B"),
            ]
            a_finish.set()
            assert a_future.result(timeout=15) == {"action": "parked"}
            assert _owners(dsn, pr) == [
                (ESCALATION_INFLIGHT_PHASE, "advice-B")
            ], "A cleanup erased B's still-running marker"

            # Eliminate the quiet-window fallback: only B's marker may block.
            with psycopg.connect(dsn, autocommit=True) as conn:
                conn.execute(
                    "UPDATE neo_actions SET created_at=now()-interval '2 hours' "
                    "WHERE repo=%s AND pr_number=%s", (REPO, pr))
            ok, reason = SqlWrites(dsn).stop(
                REPO, pr, HEAD_B, advisor_round=6, detail={})
            assert not ok and "mid-flight" in reason, (ok, reason)
        finally:
            a_finish.set()
            b_crash.set()
        assert b_future.result(timeout=15) == "crashed"

    _age_inflight(dsn, pr)
    reclaimed = state.reclaim_stale_inflight()
    assert reclaimed == [{"repo": REPO, "pr": pr, "action": "re-recorded"}]
    assert _owners(dsn, pr) == [("escalating", "advice-B")]


def test_two_stale_inflight_reclaims_newest_not_oldest(dsn):
    """Scanning stale A before stale B must not resurrect A and delete B."""
    state = NeoState(dsn)
    pr = 9102
    state.record_escalating(REPO, pr, HEAD_A, advice_id="advice-A")
    assert state.take_escalation(REPO, pr, "advice-A")
    state.record_escalating(REPO, pr, HEAD_B, advice_id="advice-B")
    assert state.take_escalation(REPO, pr, "advice-B")
    _age_inflight(dsn, pr)

    state.reclaim_stale_inflight()
    assert _owners(dsn, pr) == [("escalating", "advice-B")], (
        "the obsolete A must not supersede newer B")
    assert state.take_escalation(REPO, pr, "advice-A") is False
    assert state.take_escalation(REPO, pr, "advice-B") is True
    assert _owners(dsn, pr) == [
        (ESCALATION_INFLIGHT_PHASE, "advice-B")
    ]


def test_take_newest_does_not_convert_older_queued_advice_to_inflight(dsn):
    """Only the advice actually consumed may become an in-flight owner."""
    state = NeoState(dsn)
    pr = 9103
    state.record_escalating(REPO, pr, HEAD_A, advice_id="advice-A")
    state.record_escalating(REPO, pr, HEAD_B, advice_id="advice-B")
    assert state.take_escalation(REPO, pr, "advice-B")
    assert _owners(dsn, pr) == [
        (ESCALATION_INFLIGHT_PHASE, "advice-B")
    ]
    state.clear_inflight_markers(REPO, pr, "advice-B")
    assert _owners(dsn, pr) == []
