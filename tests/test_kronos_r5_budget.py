"""r5 MEDIUM 1 — the pass deadline holds in the LAST FRACTIONAL SECOND.

The ASTRA review of head 3ff1018 (comment 6609225440) left MEDIUM 1 open:

    _mint_timeout and _http_timeout turn e.g. 0.2 seconds left into a
    1-second network timeout.  SqlWrites._timeouts likewise rounds its
    connect timeout to at least 1 second, then allows connection + SET
    LOCAL + multiple statements under independently bounded timeouts.

The rule these tests pin:

  * an HTTP call whose timeout is whole seconds is never given MORE than
    the remaining budget: a sub-second remainder is honoured EXACTLY
    (urllib's socket timeout takes a fraction);
  * a DB write whose remaining budget cannot pay for a whole-second
    connect does not START at all (libpq truncates a fractional
    connect_timeout to 0, which means "wait forever"), and its
    statement timeouts are re-read from the budget at every stage
    boundary, divided by the write's statement count so their SUM fits
    in what is left;
  * a refused write leaves NOTHING behind.

Real Postgres for the write paths; the budget is a scripted clock.

Run: NEO_TEST_DSN='<dsn>' .venv/bin/python -m pytest \
     tests/test_kronos_r5_budget.py -q
"""
from __future__ import annotations

import os

import psycopg
import pytest

from src.kronos import BudgetSpent
from src.kronos import __main__ as km
from src.kronos.store import SqlWrites

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO_TEST_DSN"),
    reason="NEO_TEST_DSN not set (no throwaway Postgres for this run)",
)

HEAD = "e" * 40


@pytest.fixture()
def dsn():
    return os.environ["NEO_TEST_DSN"]


@pytest.fixture()
def fresh(dsn):
    # The SAME shape as tests/test_kronos_store_pg.py: ensure_schema (the
    # unique indexes the writer's ON CONFLICT needs are version-gated and
    # are NOT recreated after a DROP) and a TRUNCATE.
    from src.jobs import Jobs
    from src.neo_state import NeoState
    NeoState(dsn).ensure_schema()
    Jobs(dsn).ensure_schema()
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("TRUNCATE neo_actions")
        c.execute("TRUNCATE neo_jobs")
    yield dsn
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("TRUNCATE neo_actions")
        c.execute("TRUNCATE neo_jobs")


def _scripted(values):
    """A budget that answers with `values` in order (the last value
    repeats).  It stands in for the pass clock."""
    seq = list(values)

    def _left():
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return _left


# --------------------------------------------------------------------------
# the whole-second helpers (HTTP: urllib's socket timeout takes a fraction)
# --------------------------------------------------------------------------

def test_a_sub_second_http_budget_is_honoured_exactly():
    """r5 MEDIUM 1 — 0.2 s left used to become a ONE-SECOND network
    timeout: the call outlived the deadline it protects.  The exact
    remainder is used instead."""
    assert km._http_timeout(lambda: 0.2, 30) == pytest.approx(0.2)
    assert km._mint_timeout(lambda: 0.2) == pytest.approx(0.2)
    assert km._http_timeout(lambda: 0.75, 30) == pytest.approx(0.75)


def test_a_whole_second_timeout_never_exceeds_the_remaining_budget():
    """While the budget lasts, the timeout is what is LEFT of it — never
    more (a one-second timeout with 0.9 s left would overrun)."""
    assert km._http_timeout(lambda: 5.0, 30) == 5
    assert km._http_timeout(lambda: 2.4, 30) == 2
    assert km._mint_timeout(lambda: 5.0) == 5
    assert km._mint_timeout(lambda: 2.4) == 2


def test_an_exhausted_budget_starts_no_call_at_all():
    """r4 MEDIUM 1 — nothing left means the call does not start."""
    for value in (0.0, -1.0):
        with pytest.raises(BudgetSpent):
            km._mint_timeout(lambda v=value: v)
        with pytest.raises(BudgetSpent):
            km._http_timeout(lambda v=value: v, cap=30)


class _Cfg:
    alert_repo = "o/alert"
    alert_label = "kronos"
    alert_window_secs = 3600
    zeus_app_id = "1"
    zeus_app_key = "key"


class _RecordingApp:
    instances: list = []

    def __init__(self, app_id, key_pem, *a, http_timeout=30, **k):
        self.http_timeout = http_timeout
        _RecordingApp.instances.append(self)

    def token_for_repo(self, repo, installation_id=None, permissions=None):
        return "tok"

    def jwt(self):
        return "jwt"


def test_the_alert_http_call_never_gets_more_than_the_budget(monkeypatch):
    """The owner's alert is the last side effect of a pass: its HTTP
    timeout is the budget that is LEFT — never a free-standing second
    that outlives the deadline."""
    import src.github_app as github_app
    monkeypatch.setattr(github_app, "GitHubApp", _RecordingApp)
    _RecordingApp.instances.clear()
    seen: list[float] = []

    class _Svc:
        def http(self, method, url, *, headers=None, body=None, timeout=30):
            seen.append(timeout)
            return 200, "[]"

    state = km.KronosState(":memory:")
    km._alerter(_Cfg(), state, lambda: 1000.0, services=_Svc(),
                left_fn=lambda: 0.25).alert("fp-r5", "subject", "body")
    assert seen, "the alerter never made an HTTP call"
    assert max(seen) <= 0.25, (
        f"the alert HTTP timeout outlived the deadline: {seen}")


# --------------------------------------------------------------------------
# the DB write
# --------------------------------------------------------------------------

def test_a_write_refuses_to_start_below_the_floor(dsn):
    """The CONNECT is whole-second work: a write whose remaining budget
    cannot pay for it does not start at all (libpq truncates a fractional
    connect_timeout to 0, which means "wait forever")."""
    writes = SqlWrites(dsn, budget_left=lambda: 0.2)
    with pytest.raises(BudgetSpent):
        writes.escalate("o/r", 969, HEAD, advisor_round=1,
                        detail={"reason": "x"},
                        builder_payload={"round": 1})
    writes = SqlWrites(dsn, budget_left=lambda: 2.5)
    with pytest.raises(BudgetSpent):
        writes.escalate("o/r", 969, HEAD, advisor_round=1,
                        detail={"reason": "x"},
                        builder_payload={"round": 1})


def test_the_connect_and_statements_fit_in_the_remaining_budget(dsn):
    """While the budget lasts, the connect timeout is what is LEFT of it
    and the statements SHARE what the connect leaves — their sum fits in
    the remaining budget."""
    writes = SqlWrites(dsn, budget_left=lambda: 5.0)
    connect, statement_ms = writes._timeouts()
    assert connect <= 5, f"the connect can overrun the deadline: {connect} s"
    assert statement_ms <= 5000, \
        f"the statement can overrun the deadline: {statement_ms} ms"
    # the statements' SUM fits in what the connect leaves
    assert statement_ms * 6 <= 5000, \
        f"the statements' sum overruns the deadline: {statement_ms} ms x6"


def test_a_slow_connect_crossing_the_deadline_starts_no_write(fresh):
    """The connect itself consumes budget: when it leaves too little, the
    write does not start — and NOTHING is written."""
    writes = SqlWrites(fresh, budget_left=_scripted([5.0, 0.2]))
    with pytest.raises(BudgetSpent):
        writes.escalate("o/r", 970, HEAD, advisor_round=1,
                        detail={"reason": "x"},
                        builder_payload={"round": 1})
    with psycopg.connect(fresh, autocommit=True) as c:
        assert c.execute("SELECT COUNT(*) FROM neo_actions").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM neo_jobs").fetchone()[0] == 0


def test_the_budget_is_rechecked_before_the_write(fresh):
    """The budget is re-read at every stage boundary: a budget spent
    between the connect and the INSERT refuses the write — and NOTHING is
    written."""
    writes = SqlWrites(fresh, budget_left=_scripted([5.0, 5.0, 0.2]))
    with pytest.raises(BudgetSpent):
        writes.escalate("o/r", 971, HEAD, advisor_round=1,
                        detail={"reason": "x"},
                        builder_payload={"round": 1})
    with psycopg.connect(fresh, autocommit=True) as c:
        assert c.execute("SELECT COUNT(*) FROM neo_actions").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM neo_jobs").fetchone()[0] == 0


def test_a_write_that_fits_the_budget_still_works(fresh):
    """The refusal is about the deadline, not about the write: a write
    that fits is written exactly as before."""
    writes = SqlWrites(fresh, budget_left=lambda: 30.0)
    ok, note = writes.escalate("o/r", 972, HEAD, advisor_round=1,
                               detail={"reason": "x"},
                               builder_payload={"round": 1})
    assert ok, note
    with psycopg.connect(fresh, autocommit=True) as c:
        assert c.execute(
            "SELECT COUNT(*) FROM neo_actions WHERE phase='escalated'"
        ).fetchone()[0] == 1
        assert c.execute(
            "SELECT COUNT(*) FROM neo_jobs WHERE kind='builder'"
        ).fetchone()[0] == 1
