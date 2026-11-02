"""W1-08-r2 — unit health checks with real subprocess shapes.

Edge case 1: a unit that is ``failed`` is reported with the exit status and
the last 20 journal lines.

Edge case 2: a unit that restarts five times in ten minutes is reported ONCE,
with the count — not five times.

r2 (MEDIUM 2): runner fakes are subprocess.CompletedProcess (the REAL result
shape — r1's fakes carried an ``rc`` field the real object does not have);
HTTP health makes a bounded real HTTP request (TCP-connect-only marked a
service that binds but never answers as healthy); the watched-unit list is
the config's, and the remedies module is gone (r2 ruling: read-only).

Every external touch (systemctl, journalctl, HTTP) is a fake built inside the
test: no real unit is touched, no real journal is read, no network.
"""
from __future__ import annotations

import subprocess
from urllib.error import URLError

from src.kronos.checks import (
    BROKEN,
    OK,
    flap_status,
    http_health,
    journal_tail,
    unit_active,
)


def proc(argv, returncode: int, stdout: str = "", stderr: str = ""):
    """A REAL subprocess.run result shape (r1 fakes lied with ``rc``)."""
    return subprocess.CompletedProcess(argv, returncode, stdout=stdout,
                                       stderr=stderr)


class FakeRunner:
    """argv -> CompletedProcess, keyed on the full argv tuple."""

    def __init__(self, table: dict[tuple[str, ...],
                                   subprocess.CompletedProcess]):
        self.table = table
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(tuple(argv))
        return self.table[tuple(argv)]


# ---------------------------------------------------------------------------
# Edge 1 — a dead unit is reported
# ---------------------------------------------------------------------------

def test_dead_unit_is_reported():
    """`systemctl is-active` says failed -> BROKEN, and the report carries the
    exit status plus the last 20 journal lines."""
    runner = FakeRunner({
        ("systemctl", "is-active", "rhobear-neo"): proc(
            ["systemctl"], 3, "failed\n"),
        ("systemctl", "show", "rhobear-neo", "-p", "ExecMainStatus",
         "-p", "ActiveState", "-p", "SubState", "--value"): proc(
             ["systemctl"], 0, "1\nfailed\nfailed\n"),
    })
    res = unit_active("zeus", "rhobear-neo", runner=runner)
    assert res.status == BROKEN, res
    assert "exit status 1" in res.reason, res.reason
    assert "failed" in res.reason, res.reason


def test_dead_unit_report_carries_last_20_journal_lines():
    """The one-line report the alerter posts includes the journal tail."""
    runner = FakeRunner({
        ("journalctl", "-u", "rhobear-neo", "-n", "20", "--no-pager",
         "-o", "short-iso"): proc(["journalctl"], 0,
                                  "2026-11-01T10:00:00+00:00 boom 1\n"
                                  "2026-11-01T10:00:01+00:00 boom 2\n"),
    })
    tail = journal_tail("rhobear-neo", runner=runner, lines=20)
    assert "boom 1" in tail and "boom 2" in tail
    # exactly two lines were asked for and returned
    assert len(tail.strip().splitlines()) == 2


def test_journal_tail_reads_the_real_returncode_field():
    """MEDIUM 2 — the tail decision reads subprocess' REAL field
    (returncode). A failed journalctl (returncode=1) falls back to
    `systemctl status` even when stdout is empty."""
    runner = FakeRunner({
        ("journalctl", "-u", "rhobear-neo", "-n", "20", "--no-pager",
         "-o", "short-iso"): proc(["journalctl"], 1, stdout="",
                                  stderr="Hint: You are currently not "
                                         "seeing messages from other users"),
        ("systemctl", "status", "rhobear-neo", "-n", "20", "--no-pager",
         "-l"): proc(["systemctl"], 0,
                     "● rhobear-neo.service - ...\n"
                     "Nov 01 10:00:00 box boom[1]: died\n"),
    })
    tail = journal_tail("rhobear-neo", runner=runner, lines=20)
    assert "died" in tail


def test_journal_permission_falls_back_to_status():
    """Edge 5 companion: without journal read rights the tail still arrives —
    via `systemctl status` (systemd prints the recent lines itself)."""
    runner = FakeRunner({
        ("journalctl", "-u", "rhobear-neo", "-n", "20", "--no-pager",
         "-o", "short-iso"): proc(["journalctl"], 1, stderr="Hint: You are "
                                  "currently not seeing messages"),
        ("systemctl", "status", "rhobear-neo", "-n", "20", "--no-pager",
         "-l"): proc(["systemctl"], 0,
                     "● rhobear-neo.service - ...\n"
                     "Nov 01 10:00:00 box boom[1]: died\n"),
    })
    tail = journal_tail("rhobear-neo", runner=runner, lines=20)
    assert "died" in tail


def test_healthy_unit_is_ok():
    runner = FakeRunner({
        ("systemctl", "is-active", "rhobear-reviews"): proc(
            ["systemctl"], 0, "active\n"),
        ("systemctl", "show", "rhobear-reviews", "-p", "ExecMainStatus",
         "-p", "ActiveState", "-p", "SubState", "--value"): proc(
             ["systemctl"], 0, "0\nactive\nrunning\n"),
    })
    res = unit_active("reviews", "rhobear-reviews", runner=runner)
    assert res.status == OK, res
    assert res.reason


# ---------------------------------------------------------------------------
# MEDIUM 2 — HTTP health is a real bounded HTTP request
# ---------------------------------------------------------------------------

def test_http_health_ok_when_the_service_answers():
    """Any HTTP response (even 404 on a POST-only webhook) proves the process
    answers — that is the health signal."""
    calls = []

    def request(url, timeout):
        calls.append((url, timeout))
        return 404, "not found"

    res = http_health("webhook", "http://127.0.0.1:8766", timeout=5,
                      request=request)
    assert res.status == OK, res
    assert calls and calls[0][0].startswith("http://127.0.0.1:8766")
    assert "404" in res.reason, res.reason


def test_http_health_detects_a_hung_responder():
    """MEDIUM 2 — a service that BINDS the port but never answers is BROKEN.
    TCP-connect-only (the r1 check) called it healthy."""
    def request(url, timeout):
        raise TimeoutError("no response inside the budget")

    res = http_health("zeus", "http://127.0.0.1:8767", timeout=5,
                      request=request)
    assert res.status == BROKEN, res


def test_http_health_broken_on_connection_refused():
    def request(url, timeout):
        raise URLError("connection refused")

    res = http_health("routegate", "http://127.0.0.1:8645", timeout=5,
                      request=request)
    assert res.status == BROKEN, res
    assert "URLError" in res.reason or "refused" in res.reason, res.reason


def test_http_health_broken_on_timeout_exception():
    import socket

    def request(url, timeout):
        raise socket.timeout("timed out")

    res = http_health("webhook", "http://127.0.0.1:8766", timeout=5,
                      request=request)
    assert res.status == BROKEN, res


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
    assert should3 is False, "the window restarted from the current count"
    should4, _ = flap_status(state3, "rhobear-neo", nrestarts=16,
                             now=now + 6 * 60 + 660)
    assert should4 is True, "5 more restarts inside the new window flap again"


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


def test_watched_units_come_from_the_config():
    """The four watched units are the brief's — and there is no restart
    allow-list anymore (r2 ruling: Kronos never restarts anything)."""
    import importlib.util

    from src.kronos.config import KronosConfig
    cfg = KronosConfig()
    assert cfg.systemd_units == (
        "rhobear-reviews", "rhobear-reviews-dash",
        "rhobear-neo", "rhobear-neo-builder")
    assert importlib.util.find_spec("src.kronos.remedies") is None, \
        "the remedies module must be gone in r2 (read-only ruling)"
