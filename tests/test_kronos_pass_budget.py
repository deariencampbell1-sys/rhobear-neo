"""r4 MEDIUM 1 — the pass deadline is enforced on EVERY side effect.

The r4 review (ASTRA, 2026-11-03, written for head 95da67e1) left MEDIUM 1
OPEN at the previous head: `src/kronos/__main__.py` floors an EXHAUSTED pass
budget to a ONE-SECOND timeout, so a token mint (and the owner alert's HTTP
call) can still START after the deadline; and the durable delivered-flag
write is bounded by a fixed statement timeout instead of by what is LEFT of
the pass budget, so it can overrun the deadline.

The rule these tests pin:

  * an EXHAUSTED budget is never floored to one second — the call does not
    start at all (BudgetSpent), exactly like the judge's read-only mint;
  * the DB work of a pass is bounded by what is LEFT of the budget.

Run: .venv/bin/python -m pytest tests/test_kronos_pass_budget.py -q
"""
from __future__ import annotations

import pytest

from src.kronos import KronosState
from src.kronos import __main__ as km
from src.kronos.store import SqlWrites


class _Cfg:
    """The KronosConfig fields the alerter reads."""
    alert_repo = "o/reviews-app"
    alert_label = "kronos"
    zeus_app_id = 1
    zeus_app_key = "key"
    alert_window_secs = 6 * 3600


class _RecordingApp:
    """Stands in for GitHubApp: records every mint (and its timeout)."""
    instances: list = []

    def __init__(self, app_id, key_pem, *a, http_timeout=30, **k):
        self.http_timeout = http_timeout
        _RecordingApp.instances.append(self)

    def token_for_repo(self, repo, installation_id=None, permissions=None):
        return "tok"

    def jwt(self):
        return "jwt"


class _Services:
    """Records every HTTP call the alerter would have made."""
    def __init__(self):
        self.calls: list = []

    def http(self, method, url, *, headers=None, body=None, timeout=30):
        self.calls.append((method, url, timeout))
        return 200, "[]"


def test_an_exhausted_budget_never_becomes_a_one_second_mint():
    """The mint's HTTP timeout is what is LEFT of the budget.  An exhausted
    budget is NOT floored to one second: the mint does not start."""
    with pytest.raises(km.BudgetSpent):
        km._mint_timeout(lambda: 0.0)
    with pytest.raises(km.BudgetSpent):
        km._mint_timeout(lambda: -1.0)


def test_the_alerters_mint_and_http_never_start_past_the_deadline(
        monkeypatch):
    """The owner alert is the last side effect of a pass: with the budget
    gone NEITHER the token mint NOR the alert's HTTP call may start."""
    import src.github_app as github_app
    monkeypatch.setattr(github_app, "GitHubApp", _RecordingApp)
    _RecordingApp.instances.clear()
    services = _Services()
    alerter = km._alerter(_Cfg(), KronosState(":memory:"), lambda: 1000.0,
                          services=services, left_fn=lambda: 0.0)
    with pytest.raises(km.BudgetSpent):
        alerter.alert("fp-1", "subject", "body")
    assert _RecordingApp.instances == [], "a mint was STARTED after the deadline"
    assert services.calls == [], "an HTTP call STARTED after the deadline"


def test_the_alerters_http_timeout_is_what_is_left_of_the_budget(
        monkeypatch):
    """While the budget lasts, the alert's HTTP call is bounded by what is
    LEFT of it (never a free-standing 30 s)."""
    import src.github_app as github_app
    monkeypatch.setattr(github_app, "GitHubApp", _RecordingApp)
    _RecordingApp.instances.clear()
    services = _Services()
    alerter = km._alerter(_Cfg(), KronosState(":memory:"), lambda: 1000.0,
                          services=services, left_fn=lambda: 2.0)
    alerter.alert("fp-2", "subject", "body")
    assert services.calls, "no HTTP call was made"
    assert services.calls[0][2] == 2, \
        f"the alert's HTTP timeout ignored the budget: {services.calls}"


def test_the_delivered_flag_write_is_bounded_by_the_remaining_budget():
    """The durable delivered-flag write is DB work of the pass: its timeouts
    are what is LEFT of the budget, never a free-standing 15 s that can
    overrun the deadline."""
    writes = SqlWrites("postgresql://localhost/none", budget_left=lambda: 0.5)
    connect, statement_ms = writes._timeouts()
    assert statement_ms <= 500, \
        f"the write can overrun the deadline: {statement_ms} ms"
    assert connect <= 1, f"the connect can overrun the deadline: {connect} s"


def test_a_write_without_a_budget_keeps_its_own_timeouts():
    """No pass budget (the store used outside a pass) keeps the plain
    timeouts — the bound only ever TIGHTENS them."""
    writes = SqlWrites("postgresql://localhost/none")
    connect, statement_ms = writes._timeouts()
    assert (connect, statement_ms) == (15, 15000), (connect, statement_ms)
