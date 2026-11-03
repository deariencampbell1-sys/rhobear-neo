"""W1-08-r4 HIGH C — Zeus's consume-then-act gap is sealed by a durable
in-flight marker (AUTHORIZED Zeus interop change, Morpheus ruling 2026-11-02).

The ruling: take_escalation() must not leave a gap with no marker. The
consumed advice's rows are re-phased to 'escalation-inflight' (carrying a
consumed_at timestamp) INSTEAD of being deleted; the marker is removed only
when escalate_pr() records its outcome, and it is reclaimable after a stated
stale timeout (ESCALATION_INFLIGHT_STALE_SECS = 3600s) so a crashed Zeus
cannot block a PR forever. Kronos defers whenever the marker exists.

Zeus's behaviour is otherwise identical: the superseded branch still deletes
only that advice's rows (a newer escalation's marker exists — no gap), the
return values are unchanged, and no existing test was weakened.

Run:  NEO_TEST_DSN='<dsn>' .venv/bin/python -m pytest tests/test_escalation_inflight_pg.py -q
"""
from __future__ import annotations

import json
import os
import time

import psycopg
import pytest
from psycopg.rows import dict_row

from src.neo_state import (
    ESCALATION_INFLIGHT_PHASE,
    ESCALATION_INFLIGHT_STALE_SECS,
    NeoState,
)

_escalation = pytest.importorskip(
    "src.escalation",
    reason="src/escalation.py not importable (red-first commit)")
_consume_and_act = _escalation._consume_and_act

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


def _age_inflight(dsn, repo: str, pr: int, *, seconds: float) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "UPDATE neo_actions SET detail = jsonb_set(detail, "
            "'{consumed_at}', to_jsonb(%s::double precision)) "
            "WHERE repo=%s AND pr_number=%s AND phase=%s",
            (time.time() - seconds, repo, pr, ESCALATION_INFLIGHT_PHASE))


def _marker_phases(dsn, repo: str, pr: int) -> list[tuple[str, dict]]:
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        rows = c.execute(
            "SELECT phase, detail FROM neo_actions WHERE repo=%s "
            "AND pr_number=%s AND phase IN ('escalating', %s) ORDER BY id",
            (repo, pr, ESCALATION_INFLIGHT_PHASE)).fetchall()
    out = []
    for r in rows:
        d = r["detail"]
        if isinstance(d, str):
            d = json.loads(d)
        out.append((r["phase"], d))
    return out


def test_stale_secs_is_stated():
    """The ruling: reclaimable after a STATED stale timeout — it is a named
    constant, 3600 s, exported for the recovery loop and the docs."""
    assert ESCALATION_INFLIGHT_STALE_SECS == 3600


def test_take_escalation_leaves_a_durable_in_flight_marker(fresh):
    state = NeoState(fresh)
    state.record_escalating("o/r", 300, HEAD, advice_id="adv-1",
                            installation_id=7)
    assert state.take_escalation("o/r", 300, "adv-1") is True
    rows = _marker_phases(fresh, "o/r", 300)
    inflight = [r for r in rows if r[0] == ESCALATION_INFLIGHT_PHASE]
    assert len(inflight) == 1, f"no durable in-flight marker: {rows}"
    detail = inflight[0][1]
    assert detail.get("advice_id") == "adv-1", detail
    assert detail.get("installation_id") == 7, detail
    assert float(detail.get("consumed_at") or 0) > 0, \
        "the marker carries no consumed_at timestamp"
    assert not [r for r in rows if r[0] == "escalating"], \
        "the escalating marker survived the take"


def test_consume_and_act_clears_the_marker_when_the_outcome_records(fresh):
    """The real flow: the recovery loop's _consume_and_act takes the marker
    (which re-phases it to in-flight), act() records the outcome, and the
    marker is cleared — it never outlives the outcome."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 301, HEAD, advice_id="adv-2",
                            installation_id=1)
    out = _consume_and_act(
        state, "o/r", 301, HEAD, "adv-2", 1,
        lambda: {"action": "escalated"}, detail={"advice_id": "adv-2"})
    assert out == {"action": "escalated"}
    # the marker existed while act() ran — prove the take re-phases it:
    # take a second advice and check the phases directly
    state.record_escalating("o/r", 301, "b" * 40, advice_id="adv-2b")
    assert state.take_escalation("o/r", 301, "adv-2b") is True
    rows = _marker_phases(fresh, "o/r", 301)
    assert [r[0] for r in rows] == [ESCALATION_INFLIGHT_PHASE], rows
    state.clear_inflight_markers("o/r", 301)
    assert _marker_phases(fresh, "o/r", 301) == [], \
        "the in-flight marker outlived the recorded outcome"


def test_park_outcome_clears_the_marker_too(fresh):
    """The give-up / failed-advice paths record a parked row — that outcome
    clears the marker the same way."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 306, HEAD, advice_id="adv-9")
    out = _consume_and_act(
        state, "o/r", 306, HEAD, "adv-9", 1,
        lambda: _escalation._park_pr(state, None, "o/r", 306, HEAD, "gave up"),
        detail={"advice_id": "adv-9"})
    assert out["action"] == "parked"
    assert _marker_phases(fresh, "o/r", 306) == []
    with psycopg.connect(fresh, row_factory=dict_row) as c:
        rows = c.execute(
            "SELECT detail FROM neo_actions WHERE repo='o/r' AND "
            "pr_number=306 AND phase='parked'").fetchall()
    assert len(rows) == 1


def test_consume_and_act_failure_keeps_the_marker_and_re_records(fresh):
    """A raised act() re-records the escalating marker (the existing retry
    contract); the in-flight marker stays until an outcome records — Kronos
    defers while Zeus retries."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 302, HEAD, advice_id="adv-3",
                            installation_id=1)

    def _boom():
        raise RuntimeError("transient DB error")

    out = _consume_and_act(state, "o/r", 302, HEAD, "adv-3", 1, _boom,
                           detail={"advice_id": "adv-3"})
    assert out == {"action": "retry",
                   "reason": "transient failure: transient DB error"}, out
    rows = _marker_phases(fresh, "o/r", 302)
    assert [r[0] for r in rows] == [ESCALATION_INFLIGHT_PHASE, "escalating"], \
        rows


def test_superseded_take_still_drops_only_the_old_advice(fresh):
    """Unchanged semantics: a SUPERSEDED advice is dropped (return False) —
    a newer escalating marker exists, so there is no gap to seal."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 303, HEAD, advice_id="advice-A")
    state.record_escalating("o/r", 303, "b" * 40, advice_id="advice-B")
    assert state.take_escalation("o/r", 303, "advice-A") is False
    rows = _marker_phases(fresh, "o/r", 303)
    assert [r[0] for r in rows] == ["escalating"], rows
    assert rows[0][1].get("advice_id") == "advice-B", rows


def test_fresh_in_flight_marker_is_not_reclaimed(fresh):
    state = NeoState(fresh)
    state.record_escalating("o/r", 304, HEAD, advice_id="adv-4")
    assert state.take_escalation("o/r", 304, "adv-4") is True
    reclaimed = state.reclaim_stale_inflight(stale_secs=3600)
    assert reclaimed == []
    rows = _marker_phases(fresh, "o/r", 304)
    assert [r[0] for r in rows] == [ESCALATION_INFLIGHT_PHASE], rows


def test_stale_in_flight_marker_is_reclaimed_to_escalating(fresh):
    """Past the stated stale timeout the marker is re-phased to 'escalating'
    with its original detail (advice_id preserved, consumed_at stripped) —
    the advice re-enters Zeus's normal recovery loop, and escalate_pr's
    idempotency (defect 2) prevents a duplicate round."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 305, HEAD, advice_id="adv-5",
                            installation_id=3)
    assert state.take_escalation("o/r", 305, "adv-5") is True
    _age_inflight(fresh, "o/r", 305, seconds=7200)
    reclaimed = state.reclaim_stale_inflight(stale_secs=3600)
    assert reclaimed == [{"repo": "o/r", "pr": 305,
                          "action": "re-recorded"}], reclaimed
    rows = _marker_phases(fresh, "o/r", 305)
    assert [r[0] for r in rows] == ["escalating"], rows
    detail = rows[0][1]
    assert "consumed_at" not in detail, detail
    assert detail.get("advice_id") == "adv-5", detail
    assert detail.get("installation_id") == 3, detail


def test_reclaim_never_touches_a_newer_escalations_marker(fresh):
    """If a newer escalating marker took over the PR while the old in-flight
    marker aged out, reclaiming DROPS the stale marker instead of
    resurrecting the old advice over it."""
    state = NeoState(fresh)
    state.record_escalating("o/r", 307, HEAD, advice_id="adv-old")
    assert state.take_escalation("o/r", 307, "adv-old") is True
    _age_inflight(fresh, "o/r", 307, seconds=7200)
    # meanwhile a newer escalation is submitted (a new head bounced)
    state.record_escalating("o/r", 307, "c" * 40, advice_id="adv-new")
    reclaimed = state.reclaim_stale_inflight(stale_secs=3600)
    assert reclaimed == [{"repo": "o/r", "pr": 307,
                          "action": "dropped (a newer escalation owns the PR)"}]
    rows = _marker_phases(fresh, "o/r", 307)
    assert [r[0] for r in rows] == ["escalating"], rows
    assert rows[0][1].get("advice_id") == "adv-new", rows


def test_recover_escalations_reclaims_a_stale_marker_end_to_end(
        fresh, monkeypatch):
    """The full crash-recovery loop, through Zeus's own recovery entrypoint:
    stale marker reclaimed -> the recovery loop polls the advice -> the
    advisor lost it (404) -> park -> outcome recorded, NO marker left."""
    from src.escalation import AdvisorNotFound

    state = NeoState(fresh)
    state.record_escalating("o/r", 308, HEAD, advice_id="adv-6")
    assert state.take_escalation("o/r", 308, "adv-6") is True
    _age_inflight(fresh, "o/r", 308, seconds=7200)

    def _lost(cfg, advice_id):
        raise AdvisorNotFound("advisor has no record")

    monkeypatch.setattr(_escalation, "poll_advice", _lost)

    class _Cfg:
        advisor_url = "http://127.0.0.1:9"
        advisor_token = "t"

    out = _escalation.recover_escalations(_Cfg(), state, None,
                                          timeout_secs=1800)
    assert out, "the reclaimed advice was never handled"
    assert out[0]["action"] == "parked", out
    assert _marker_phases(fresh, "o/r", 308) == [], \
        "markers survived the recorded outcome"
    with psycopg.connect(fresh, row_factory=dict_row) as c:
        rows = c.execute(
            "SELECT phase FROM neo_actions WHERE repo='o/r' AND "
            "pr_number=308").fetchall()
    assert [r["phase"] for r in rows] == ["parked"], rows


# --- MEDIUM 2: outcome committed, cleanup failed/crashed -> stale reclaim ---

class _CleanupFails(NeoState):
    def clear_inflight_markers(self, *a, **k):
        raise RuntimeError("cleanup crashed after the outcome committed")


def _phases(dsn, repo, pr):
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        return [r["phase"] for r in c.execute(
            "SELECT phase FROM neo_actions WHERE repo=%s AND pr_number=%s "
            "ORDER BY id", (repo, pr)).fetchall()]


class _Cfg:
    advisor_url = "http://127.0.0.1:9"
    advisor_token = "t"


def _no_poll(monkeypatch):
    def _boom(cfg, advice_id):
        raise AssertionError("a settled advice must not be re-polled")
    monkeypatch.setattr(_escalation, "poll_advice", _boom)


def test_cleanup_failure_after_escalated_outcome_never_parks(
        fresh, monkeypatch):
    state = _CleanupFails(fresh)
    state.record_escalating("o/r", 320, HEAD, advice_id="adv-m2")

    def _act():
        state.record("o/r", 320, HEAD, "escalated", verdict="ESCALATE",
                     builder_round=6,
                     detail={"advice_id": "adv-m2", "reason": "escalated"})
        return {"action": "escalated"}

    out = _consume_and_act(state, "o/r", 320, HEAD, "adv-m2", 1, _act,
                           detail={"advice_id": "adv-m2"})
    assert out == {"action": "escalated"}
    assert _marker_phases(fresh, "o/r", 320), "marker should linger"
    _age_inflight(fresh, "o/r", 320, seconds=7200)
    _no_poll(monkeypatch)

    _escalation.recover_escalations(_Cfg(), NeoState(fresh), None,
                                    timeout_secs=1800)
    assert _phases(fresh, "o/r", 320) == ["escalated"], \
        "stale recovery appended a decision after a successful escalation"


def test_crash_after_parked_outcome_clears_marker_without_acting(
        fresh, monkeypatch):
    state = NeoState(fresh)
    state.record_escalating("o/r", 321, HEAD, advice_id="adv-m2b")
    assert state.take_escalation("o/r", 321, "adv-m2b") is True
    _escalation._park_pr(state, None, "o/r", 321, HEAD, "gave up")
    _age_inflight(fresh, "o/r", 321, seconds=7200)
    _no_poll(monkeypatch)

    _escalation.recover_escalations(_Cfg(), state, None, timeout_secs=1800)
    assert _phases(fresh, "o/r", 321) == ["parked"], \
        _phases(fresh, "o/r", 321)
