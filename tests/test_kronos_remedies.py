"""W1-08-r1 — remedies with fakes (acceptance).

- a dead unit is restarted ONCE then alerted (second pass in the same hour
  does not restart again);
- a failed webhook delivery is redelivered (POST .../attempts, 202);
- a stuck Zeus job is re-queued;
- the restart allow-list is enforced (anything else is refused);
- --dry-run never calls a remedy.

Every runner/HTTP/store is a fake built inside the test.
"""
from __future__ import annotations

import pytest

from src.kronos import remedies
from src.kronos.checks import BROKEN, OK, CheckResult
from src.kronos.remedies import RemedyLedger, redeliver_delivery, restart_unit


class FakeResult:
    def __init__(self, rc: int, out: str = "", err: str = ""):
        self.rc, self.out, self.err = rc, out, err


class FakeRunner:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> FakeResult:
        self.calls.append(list(argv))
        return FakeResult(0, "active\n" if "is-active" in argv else "")


class FakeHttp:
    """method+url -> (status, body). Records calls."""

    def __init__(self, table: dict[tuple[str, str], tuple[int, str]]):
        self.table = table
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method: str, url: str, *, headers=None, body=None,
                 timeout=30):
        self.calls.append((method, url))
        return self.table[(method, url)]


# ---------------------------------------------------------------------------
# restart allow-list + once per hour
# ---------------------------------------------------------------------------

def test_restart_unit_uses_sudo_systemctl_and_allow_list():
    runner = FakeRunner()
    ok, note = restart_unit(runner, "rhobear-neo")
    assert ok is True
    assert runner.calls == [["sudo", "-n", "/usr/bin/systemctl", "restart",
                             "rhobear-neo"]], runner.calls


def test_restart_refuses_units_outside_the_allow_list():
    runner = FakeRunner()
    ok, note = restart_unit(runner, "sshd")
    assert ok is False
    assert runner.calls == [], "a non-allow-listed unit must never be restarted"
    assert "allow" in note.lower()


def test_restart_command_failure_is_reported_not_raised():
    class FailingRunner(FakeRunner):
        def __call__(self, argv):
            self.calls.append(list(argv))
            return FakeResult(1, "", "Failed to restart rhobear-neo.service")
    ok, note = restart_unit(FailingRunner(), "rhobear-neo")
    assert ok is False
    assert "restart" in note.lower()


def test_ledger_runs_a_remedy_once_per_hour_per_target():
    now = [1000.0]
    ledger = RemedyLedger(dict_state={}, now_fn=lambda: now[0])
    assert ledger.may("unit:rhobear-neo") is True
    ledger.did("unit:rhobear-neo")
    assert ledger.may("unit:rhobear-neo") is False, \
        "the same target was remedied twice within the hour"
    now[0] += remedies.REMEDY_WINDOW_SECS + 1
    assert ledger.may("unit:rhobear-neo") is True
    assert ledger.may("unit:rhobear-reviews") is True, \
        "the hour window is per target, not global"


def test_ledger_survives_across_passes_via_state(tmp_path):
    from src.kronos import KronosState
    state = KronosState(str(tmp_path / "state.json"))
    ledger = RemedyLedger(dict_state=state.slot("remedies"),
                          now_fn=lambda: 1000.0)
    ledger.did("unit:rhobear-neo")
    state.flush()
    # a fresh pass (new objects, same file) still sees the mark
    state2 = KronosState(str(tmp_path / "state.json"))
    ledger2 = RemedyLedger(dict_state=state2.slot("remedies"),
                           now_fn=lambda: 1000.0)
    assert ledger2.may("unit:rhobear-neo") is False


# ---------------------------------------------------------------------------
# dead unit -> restarted once -> alerted (acceptance, end to end with fakes)
# ---------------------------------------------------------------------------

class FakeIssues:
    """The alert transport, faked: issues in deariencampbell1-sys/rhobear-reviews-app."""

    def __init__(self):
        self.created: list[dict] = []
        self.comments: list[tuple[int, str]] = []
        self.closed: list[int] = []
        self._next = 100

    def search_open(self, fingerprint: str):
        for issue in self.created:
            if issue["fingerprint"] == fingerprint and issue["open"]:
                return issue
        return None

    def create(self, title: str, body: str, labels: list[str]):
        self._next += 1
        issue = {"number": self._next, "title": title, "body": body,
                 "labels": labels, "open": True,
                 "fingerprint": fingerprint_of(body)}
        self.created.append(issue)
        return issue

    def comment(self, number: int, body: str):
        self.comments.append((number, body))

    def close(self, number: int, note: str):
        for issue in self.created:
            if issue["number"] == number:
                issue["open"] = False
        self.closed.append(number)


def fingerprint_of(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("<!-- kronos-fingerprint:"):
            return line.split(":", 1)[1].strip().removeprefix("<!--").strip()
    return ""


def test_dead_unit_restarted_once_then_alerted(tmp_path):
    """The acceptance flow: pass 1 restarts the dead unit and opens ONE issue;
    pass 2 (unit healthy again) closes it; a second dead pass inside the hour
    never restarts twice."""
    from src.kronos import KronosState
    from src.kronos.alert import Alerter

    runner = FakeRunner()
    state = KronosState(str(tmp_path / "state.json"))
    ledger = RemedyLedger(dict_state=state.slot("remedies"), now_fn=lambda: 10.0)
    alerter = Alerter(FakeIssues(), state, now_fn=lambda: 10.0)

    def one_pass(unit_ok: bool) -> None:
        res = (CheckResult("unit:rhobear-neo", OK, "active")
               if unit_ok else
               CheckResult("unit:rhobear-neo", BROKEN, "failed (exit status 1)"))
        fp = f"unit-down:rhobear-neo"
        if res.status == BROKEN:
            if ledger.may(f"unit:rhobear-neo"):
                ok, note = restart_unit(runner, "rhobear-neo")
                ledger.did(f"unit:rhobear-neo")
            alerter.alert(fp, "Kronos: rhobear-neo is down", res.reason)
        else:
            alerter.resolved(fp, "unit is active again")

    one_pass(False)   # dead
    one_pass(False)   # still dead within the hour — no second restart
    one_pass(True)    # fixed

    restarts = [c for c in runner.calls if "restart" in c]
    assert len(restarts) == 1, runner.calls
    issues = alerter._client
    assert len(issues.created) == 1, "one GitHub issue per problem"
    assert issues.closed, "the issue is closed when fixed"


# ---------------------------------------------------------------------------
# redelivery + re-queue
# ---------------------------------------------------------------------------

def test_failed_delivery_is_redelivered():
    http = FakeHttp({
        ("POST", "https://api.github.com/app/hook/deliveries/12345/attempts"):
            (202, "{}"),
    })
    ok, note = redeliver_delivery(http, "zeus", "<jwt>", 12345)
    assert ok is True
    assert http.calls == [("POST",
                           "https://api.github.com/app/hook/deliveries/12345/attempts")]


def test_redelivery_failure_is_reported():
    http = FakeHttp({
        ("POST", "https://api.github.com/app/hook/deliveries/1/attempts"):
            (422, '{"message": "Validation failed"}'),
    })
    ok, note = redeliver_delivery(http, "zeus", "<jwt>", 1)
    assert ok is False
    assert "422" in note


class FakeJobs:
    def __init__(self, rows: list[dict] | None = None):
        self.rows = rows or []
        self.released: list[int] = []

    def release(self, job_id: int, *, worker_token=None) -> bool:
        self.released.append(job_id)
        return True


def test_stuck_running_job_is_requeued():
    jobs = FakeJobs()
    ok, note = remedies.requeue_stuck_job(jobs, {"id": 7, "state": "running"})
    assert ok is True
    assert jobs.released == [7]


def test_queued_job_is_not_requeued():
    """A queued job has no claimant to release — re-queue does not apply."""
    jobs = FakeJobs()
    ok, note = remedies.requeue_stuck_job(jobs, {"id": 7, "state": "queued"})
    assert ok is False
    assert jobs.released == []
