"""W1-08-r1 red-first — nothing watches the units today.

Edge case 1: a unit that is ``failed`` is reported with the exit status and
the last 20 journal lines.

Edge case 2: a unit that restarts five times in ten minutes is reported ONCE,
with the count — not five times.

Every external touch (systemctl, journalctl, HTTP) is a fake built inside the
test: no real unit is restarted, no real journal is read, no network.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.kronos.checks import (
    BROKEN,
    OK,
    flap_status,
    journal_tail,
    unit_active,
)


@dataclass
class FakeResult:
    """What a subprocess runner returns (rc/stdout shape of subprocess.run)."""

    rc: int
    out: str = ""
    err: str = ""


class FakeRunner:
    """argv -> FakeResult, keyed on the full argv tuple."""

    def __init__(self, table: dict[tuple[str, ...], FakeResult]):
        self.table = table
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: list[str]) -> FakeResult:
        self.calls.append(tuple(argv))
        return self.table[tuple(argv)]


# ---------------------------------------------------------------------------
# Edge 1 — a dead unit is reported
# ---------------------------------------------------------------------------

def test_dead_unit_is_reported():
    """`systemctl is-active` says failed -> BROKEN, and the report carries the
    exit status plus the last 20 journal lines."""
    runner = FakeRunner({
        ("systemctl", "is-active", "rhobear-neo"): FakeResult(3, "failed\n"),
        ("systemctl", "show", "rhobear-neo", "-p", "ExecMainStatus",
         "-p", "ActiveState", "-p", "SubState", "--value"): FakeResult(
             0, "1\nfailed\nfailed\n"),
    })
    res = unit_active("zeus", "rhobear-neo", runner=runner)
    assert res.status == BROKEN, res
    assert "exit status 1" in res.reason, res.reason
    assert "failed" in res.reason, res.reason


def test_dead_unit_report_carries_last_20_journal_lines():
    """The one-line report the alerter posts includes the journal tail."""
    runner = FakeRunner({
        ("journalctl", "-u", "rhobear-neo", "-n", "20", "--no-pager",
         "-o", "short-iso"): FakeResult(0, "2026-11-01T10:00:00+00:00 boom 1\n"
                                            "2026-11-01T10:00:01+00:00 boom 2\n"),
    })
    tail = journal_tail("rhobear-neo", runner=runner, lines=20)
    assert "boom 1" in tail and "boom 2" in tail
    # exactly two lines were asked for and returned
    assert len(tail.strip().splitlines()) == 2


def test_journal_permission_falls_back_to_status():
    """Edge 5 companion: without journal read rights the tail still arrives —
    via `systemctl status` (systemd prints the recent lines itself)."""
    runner = FakeRunner({
        ("journalctl", "-u", "rhobear-neo", "-n", "20", "--no-pager",
         "-o", "short-iso"): FakeResult(1, "", "Hint: You are currently not "
                                              "seeing messages from other users and the system\n"),
        ("systemctl", "status", "rhobear-neo", "-n", "20", "--no-pager", "-l"):
            FakeResult(0, "● rhobear-neo.service - ...\nNov 01 10:00:00 box boom[1]: died\n"),
    })
    tail = journal_tail("rhobear-neo", runner=runner, lines=20)
    assert "died" in tail


def test_healthy_unit_is_ok():
    runner = FakeRunner({
        ("systemctl", "is-active", "rhobear-reviews"): FakeResult(0, "active\n"),
        ("systemctl", "show", "rhobear-reviews", "-p", "ExecMainStatus",
         "-p", "ActiveState", "-p", "SubState", "--value"): FakeResult(
             0, "0\nactive\nrunning\n"),
    })
    res = unit_active("reviews", "rhobear-reviews", runner=runner)
    assert res.status == OK, res
    assert res.reason


# ---------------------------------------------------------------------------
# Edge 2 — a flapping unit is reported once
# ---------------------------------------------------------------------------

def test_flapping_unit_is_reported_once():
    """NRestarts climbs 5+ inside the 10-minute window: exactly one report,
    carrying the count — a second call in the same window must not re-report."""
    state: dict = {}
    now = 1000.0
    alerts = []
    for i in range(5):
        should, state = flap_status(
            state, "rhobear-neo", nrestarts=i, now=now + i * 60)
        if should:
            alerts.append(i)
    assert alerts == [], "no flap before the threshold"

    should, state = flap_status(state, "rhobear-neo", nrestarts=5,
                                now=now + 5 * 60)
    assert should is True
    should2, state2 = flap_status(state, "rhobear-neo", nrestarts=6,
                                  now=now + 6 * 60)
    assert should2 is False, "a flap already reported in this window re-alerted"

    # outside the window the detector re-arms from the current count
    should3, state3 = flap_status(state2, "rhobear-neo", nrestarts=11,
                                  now=now + 6 * 60 + 601)
    assert should3 is False, "11 restarts with base 6 is 5, but the window restarted"
    should4, _ = flap_status(state3, "rhobear-neo", nrestarts=12,
                             now=now + 6 * 60 + 660)
    assert should4 is True


def test_flap_report_carries_the_count():
    """The flap report text says how many restarts happened."""
    state: dict = {}
    for i in range(6):
        should, state = flap_status(state, "rhobear-neo", nrestarts=i,
                                    now=1000.0 + i * 60)
    assert should is True
    # the report body is built by the alerter from the flap facts
    from src.kronos.checks import flap_report_body
    body = flap_report_body("rhobear-neo", count=5, window_secs=600)
    assert "5" in body and "rhobear-neo" in body


def test_flap_is_per_unit():
    state: dict = {}
    for i in range(6):
        should, state = flap_status(state, "rhobear-neo", nrestarts=i,
                                    now=1000.0 + i * 60)
    assert should is True
    should_other, state2 = flap_status(state, "rhobear-reviews",
                                       nrestarts=1, now=1060.0)
    assert should_other is False, "one unit's flap leaked into another unit"


@pytest.mark.parametrize("argv_unit", ["rhobear-neo"])
def test_unit_names_come_from_the_allow_list(argv_unit):
    """The four watched units are exactly the allow-list from the brief."""
    from src.kronos.remedies import ALLOWED_UNITS
    assert set(ALLOWED_UNITS) == {
        "rhobear-reviews", "rhobear-reviews-dash",
        "rhobear-neo", "rhobear-neo-builder",
    }
