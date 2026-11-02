"""Red-first budget checks for PR #20 @ 814a3c92 (no database required)."""
import pytest

from src.kronos.__main__ import BudgetSpent, _mint_timeout
from src.kronos.judge import _outbox_entry, drain_notify_outbox


def test_deadline_expires_between_comment_and_owner_alert():
    remaining = [1.0]
    posted, alerts, delivered = [], [], []
    entry = _outbox_entry("o/r", 905, comment="stop notice",
                          alert_fp="loop:o/r#905", alert_subject="stop",
                          alert_body="owner must be told")

    class State:
        slots = {"notify_outbox": {"o/r#905": entry}, "open_alerts": {}}

        def slot(self, name):
            return self.slots[name]

    class Alerter:
        def alert(self, *args):
            alerts.append(args)
            return "created"

    def comment(*args):
        posted.append(args)
        remaining[0] = 0.0  # synchronous comment spent the last budget

    state = State()
    drain_notify_outbox(state, comment, Alerter(),
                        budget_left=lambda: remaining[0],
                        on_delivered=delivered.append)
    assert len(posted) == 1
    assert alerts == [], "alert transport started after the deadline"
    assert entry["needs_comment"] is False
    assert entry["needs_alert"] is True
    assert "o/r#905" in state.slot("notify_outbox")
    assert not delivered, "notification cannot be marked delivered"


def test_empty_budget_does_not_mint_with_one_second_grace():
    with pytest.raises(BudgetSpent):
        _mint_timeout(lambda: 0.0)
