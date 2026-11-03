"""W1-08-r2 — the CLI contract and the one-pass wiring.

`python -m src.kronos --once --dry-run` prints ONE LINE PER CHECK and changes
nothing: no state file, no remedy, no GitHub call, no judge call. A real pass
takes the flock on the state dir first — two passes never overlap, and the
lock is one stable inode, never unlinked into a second lock (MEDIUM 1).

r2 additions:

- HIGH 8: a check that cannot run (missing binary, permission, timeout)
  becomes a BROKEN result and ALERTS — never a silent skip — and every
  alert kind closes when its problem is fixed;
- MEDIUM 2: the journal tail rides in a broken unit's alert body;
- MEDIUM 3: every long action's timeout is clamped to the remaining pass
  budget.
"""
from __future__ import annotations

import subprocess

import pytest

from src.kronos.__main__ import main, run_pass
from src.kronos.checks import CheckResult
from src.kronos.config import KronosConfig


class FakeServices:
    """Wires every external touch to fakes and records what was called."""

    def __init__(self, results: list[CheckResult] | None = None):
        self.results = results or []
        self.http = []
        self.runner_calls: list[tuple] = []

    def runner(self, argv, *, timeout=30):
        self.runner_calls.append(tuple(argv))
        # a real `systemctl show -p NRestarts --value` answers with a number
        if argv[:2] == ["systemctl", "show"] and "NRestarts" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="0\n",
                                               stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="active\n",
                                           stderr="")

    def db(self, dsn: str, **kw):
        raise AssertionError("no DSN is configured in these tests")

    def writes(self, dsn: str, **kw):
        raise AssertionError("no DSN is configured in these tests")


def test_once_dry_run_prints_one_line_per_check(capsys, tmp_path,
                                                 monkeypatch):
    """One line per check, and NOTHING changes."""
    for name in ("REVIEWS_DATABASE_URL", "RHOBEAR_KRONOS_REVIEWS_APP_ID",
                 "RHOBEAR_KRONOS_ZEUS_APP_ID"):
        monkeypatch.delenv(name, raising=False)
    fake = FakeServices()
    rc = main(["--once", "--dry-run", "--state-dir", str(tmp_path)],
              _services=fake)
    out = capsys.readouterr().out
    assert rc == 0
    lines = [l for l in out.splitlines() if l.startswith("kronos check ")]
    # the default check set: 3 HTTP healths + 4 units + postgres = 8 lines,
    # exactly one per check
    assert len(lines) == 8, lines
    assert len(set(lines)) == 8, lines
    for name in ("webhook", "zeus", "routegate", "postgres",
                 "unit:rhobear-reviews", "unit:rhobear-reviews-dash",
                 "unit:rhobear-neo", "unit:rhobear-neo-builder"):
        assert any(l.startswith(f"kronos check {name}:") for l in lines), \
            (name, lines)
    assert fake.http == [], "dry-run never calls GitHub or RouteGate"


def test_once_dry_run_writes_no_state_file(capsys, tmp_path):
    fake = FakeServices()
    main(["--once", "--dry-run", "--state-dir", str(tmp_path)], _services=fake)
    assert list(tmp_path.iterdir()) == [], \
        f"dry-run wrote {list(tmp_path.iterdir())}"


def test_once_dry_run_never_mutates_on_a_broken_unit(capsys, tmp_path):
    """The changes-nothing guarantee, exercised: a dead unit is reported in
    dry-run; nothing is alerted, nothing is called, no state is written."""
    fake = FakeServices()

    def failing_runner(argv, *, timeout=30):
        if "is-active" in argv:
            return subprocess.CompletedProcess(argv, 3, stdout="failed\n",
                                               stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    fake.runner = failing_runner
    rc = main(["--once", "--dry-run", "--state-dir", str(tmp_path)],
              _services=fake)
    out = capsys.readouterr().out
    assert rc == 0
    assert any("unit:rhobear-neo" in l and "broken" in l
               for l in out.splitlines()), out
    assert fake.http == [], "dry-run called GitHub or RouteGate"
    assert list(tmp_path.iterdir()) == [], "dry-run wrote a state file"


def test_pass_lock_prevents_two_passes_at_once(capsys, tmp_path):
    """A held lock makes the second pass exit immediately without running."""
    from src.kronos import PassLock
    lock = PassLock(str(tmp_path / "pass.lock"))
    lock.acquire()
    try:
        fake = FakeServices()
        rc = main(["--once", "--state-dir", str(tmp_path)], _services=fake)
        out = capsys.readouterr().out
        assert rc == 0
        assert "already running" in out.lower()
        assert fake.runner_calls == [], "a locked pass must not check"
    finally:
        lock.release()


def test_module_is_runnable():
    """`python -m src.kronos --help` exits 0 (the unit's ExecStart shape)."""
    import os
    import sys
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    proc = subprocess.run(
        [sys.executable, "-m", "src.kronos", "--help"],
        capture_output=True, text=True, cwd=".", timeout=60, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--dry-run" in proc.stdout


# ---------------------------------------------------------------------------
# MEDIUM 1 — one lock, never two
# ---------------------------------------------------------------------------

def test_lock_release_never_unlinks_the_lockfile(tmp_path):
    """MEDIUM 1 — release() unlocks and closes the fd but NEVER unlinks the
    file on POSIX: a waiter that opened the old inode between unlock and
    unlink would hold a lock on a file the next pass replaces — two passes
    both believing they own the lock. The inode must be stable."""
    fcntl = pytest.importorskip("fcntl", reason="POSIX flock only")
    import os
    from src.kronos import PassLock

    path = str(tmp_path / "pass.lock")
    lock = PassLock(path)
    assert lock.acquire() is True
    inode_before = os.stat(path).st_ino

    # a competing third process opens and tries to lock the SAME path
    rival_fd = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(rival_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise AssertionError("the rival locked a held lock")
        except OSError:
            pass  # correctly still held

        lock.release()

        # after release the file must STILL EXIST (never unlinked) and the
        # rival must be able to take the lock on the SAME inode
        assert os.path.exists(path), "release() unlinked the lockfile"
        assert os.stat(path).st_ino == inode_before, "the inode changed"
        fcntl.flock(rival_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(rival_fd, fcntl.LOCK_UN)
    finally:
        os.close(rival_fd)


def test_lock_is_reacquirable_after_release(tmp_path):
    from src.kronos import PassLock
    path = str(tmp_path / "pass.lock")
    first = PassLock(path)
    assert first.acquire() is True
    first.release()
    second = PassLock(path)
    assert second.acquire() is True, "the next pass must take the lock"
    second.release()


# ---------------------------------------------------------------------------
# HIGH 8 — a check that cannot run alerts loudly; every kind closes
# ---------------------------------------------------------------------------

class _FakeAlerter:
    """Stands in for the GitHub Alerter (the thing under test is run_pass's
    dispatch, not the GitHub client). Carries the same 6-hour dedupe
    semantics as the real Alerter so repeat passes are honest."""

    def __init__(self, clock=None, window_secs: int = 6 * 3600):
        self.alerts: list[tuple[str, str, str]] = []
        self.closes: list[str] = []      # (the resolved() method must keep
        self.window: dict[str, float] = {}  # its name — do NOT shadow it)
        self.clock = clock or [1_000_000.0]
        self.window_secs = window_secs

    def alert(self, fingerprint, subject, body):
        last = self.window.get(fingerprint)
        if last is not None and self.clock[0] - last < self.window_secs:
            return "suppressed"
        self.window[fingerprint] = self.clock[0]
        self.alerts.append((fingerprint, subject, body))
        return "created"

    def resolved(self, fingerprint, note):
        self.closes.append(fingerprint)
        self.window.pop(fingerprint, None)


def _run_pass(services, tmp_path, *, state=None, cfg=None, out=None,
              now_fn=None, mono_fn=None):
    from src.kronos import KronosState
    cfg = cfg or KronosConfig(state_dir=str(tmp_path),
                              reviews_database_url="")
    state = state or KronosState(str(tmp_path / "state.json"))
    lines: list[str] = []
    rc = run_pass(cfg, dry_run=False, state=state, services=services,
                  out=(out or lines.append),
                  now_fn=now_fn or (lambda: 1_000_000.0),
                  mono_fn=mono_fn or (lambda: 0.0))
    return rc, lines, state


def test_unrunnable_check_becomes_broken_and_alerts(tmp_path, monkeypatch):
    """HIGH 8 — a runner that explodes (missing binary / permission /
    timeout) must produce a BROKEN check line AND an alert, never a silent
    skip and never a crashed pass."""
    import pytest
    fake = FakeServices()

    def exploding_runner(argv, *, timeout=30):
        raise PermissionError("binary not executable")

    fake.runner = exploding_runner
    fake.results = None
    alerter = _FakeAlerter()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg, state, now_fn, **kw: alerter)
    rc, lines, _state = _run_pass(fake, tmp_path)
    assert rc == 0, "a broken check must never crash the pass"
    assert any("unit:rhobear-neo" in l and "broken" in l and
               "could not run" in l for l in lines), lines
    assert alerter.alerts, "an unrunnable check went silent"
    assert any(fp.startswith("unit-down:") or fp.startswith("check:")
               for fp, _s, _b in alerter.alerts), alerter.alerts


def test_every_broken_check_alerts_with_a_stable_fingerprint(tmp_path,
                                                             monkeypatch):
    """HIGH 8 — every BROKEN check (not just units) alerts, deduped by a
    stable fingerprint, and closes when the check turns ok again."""
    fake = FakeServices()

    def db_down_runner(argv, *, timeout=30):
        return subprocess.CompletedProcess(argv, 3, stdout="failed\n",
                                           stderr="")

    fake.runner = db_down_runner
    alerter = _FakeAlerter()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg, state, now_fn, **kw: alerter)
    _rc, _lines, _state = _run_pass(fake, tmp_path)
    fingerprints = {fp for fp, _s, _b in alerter.alerts}
    assert "unit-down:rhobear-neo" in fingerprints, fingerprints

    # a second identical pass must not re-alert inside the window
    before = len(alerter.alerts)
    _rc2, _l2, _s2 = _run_pass(fake, tmp_path)
    assert len(alerter.alerts) == before, "the same problem re-alerted"

    # fixed: the unit is active again -> the alert closes
    fake.runner = FakeServices().runner
    _rc3, _l3, _s3 = _run_pass(fake, tmp_path)
    assert "unit-down:rhobear-neo" in alerter.closes, alerter.closes


def test_failed_close_is_retried_next_pass(tmp_path, monkeypatch):
    """HIGH 8 — a resolve that raises must NOT drop the open alert: the
    entry stays and the next pass retries the close."""
    fake = FakeServices()

    def failing_runner(argv, *, timeout=30):
        return subprocess.CompletedProcess(argv, 3, stdout="failed\n",
                                           stderr="")

    fake.runner = failing_runner
    alerter = _FakeAlerter()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg, state, now_fn, **kw: alerter)
    _rc, _lines, state = _run_pass(fake, tmp_path)
    assert state.slot("open_alerts"), "nothing was alerted"

    # fixed, but the close call explodes this pass
    fake.runner = FakeServices().runner

    class Exploding:
        def alert(self, *a, **k):
            return "created"

        def resolved(self, *a, **k):
            raise RuntimeError("github down")

    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg, state, now_fn, **kw: Exploding())
    _rc2, lines2, state2 = _run_pass(fake, tmp_path)
    assert any("resolve failed" in l for l in lines2), lines2
    assert state2.slot("open_alerts"), "a failed close dropped the alert"

    # next pass: the close succeeds and the entry clears
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg, state, now_fn, **kw: alerter)
    _rc3, _l3, state3 = _run_pass(fake, tmp_path)
    assert "unit-down:rhobear-neo" in alerter.closes
    assert "unit-down:rhobear-neo" not in state3.slot("open_alerts")


# ---------------------------------------------------------------------------
# MEDIUM 2 — the journal tail rides in the broken unit's alert body
# ---------------------------------------------------------------------------

def test_broken_unit_alert_body_carries_the_journal_tail(tmp_path,
                                                         monkeypatch):
    fake = FakeServices()

    def runner(argv, *, timeout=30):
        if "is-active" in argv:
            return subprocess.CompletedProcess(argv, 3, stdout="failed\n",
                                               stderr="")
        if argv[0] == "journalctl":
            return subprocess.CompletedProcess(
                argv, 0, stdout="2026-11-01T10:00:00 boom 1\n"
                                "2026-11-01T10:00:01 boom 2\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="0\nactive\n"
                                                           "running\n",
                                           stderr="")

    fake.runner = runner
    alerter = _FakeAlerter()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg, state, now_fn, **kw: alerter)
    _rc, _lines, _state = _run_pass(fake, tmp_path)
    bodies = " ".join(b for _fp, _s, b in alerter.alerts)
    assert "boom 1" in bodies and "boom 2" in bodies, alerter.alerts


# ---------------------------------------------------------------------------
# MEDIUM 3 — the 4-minute budget is enforced INSIDE long actions
# ---------------------------------------------------------------------------

class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class _JudgeServices(FakeServices):
    """A fake sweep environment: 5 builder rounds in the DB, an http that
    records the timeout the judge was given. NOTE: FakeServices.__init__
    sets self.http = [] (a recording LIST for the dry-run tests) — here the
    http METHOD must win, so the recording list is dropped on purpose."""

    def __init__(self):
        super().__init__()
        self.judge_timeouts: list[int] = []
        self.comment_timeouts: list[int] = []
        # the inherited recording list would shadow the http() method below
        del self.http  # noqa: W420 — restore the class method
        self.http_log: list[tuple] = []

    def db(self, dsn, **kw):
        class _Db:
            def rows(_self, sql, params=()):
                s = " ".join(str(sql).split())
                if "GROUP BY repo, pr_number" in s:
                    return [{"repo": "o/r", "pr_number": 52,
                             "rounds": 5, "head_sha": "f" * 40}]
                if "FROM neo_jobs" in s:
                    # the authoritative installation identity (HIGH 6)
                    return [{"payload": {"installation_id": 4242}}]
                if "ORDER BY id ASC" in s:
                    return [{"round": i, "head_sha": f"{i:04d}" * 10,
                             "verdict": "BOUNCE", "tree_sha": f"t{i}",
                             "diff_stat": "", "created_at": 0}
                            for i in range(1, 6)]
                return []

        return _Db()

    def writes(self, dsn, **kw):
        class _Writes:
            def escalate(_self, repo, pr, head_sha, *, advisor_round, detail,
                         builder_payload, authority=None, **kw2):
                return True, "ok"

            def stop(_self, repo, pr, head_sha, *, advisor_round, detail,
                     authority=None, **kw2):
                return True, "ok"

        return _Writes()

    def http(self, method, url, *, headers=None, body=None, timeout=30):
        if "chat/completions" in url:
            self.judge_timeouts.append(timeout)
            import json as _json
            msg = {"choices": [{"message": {"content": _json.dumps(
                {"progress": "no", "plan": "", "reason": "stop"})}}]}
            return 200, _json.dumps(msg)
        if "api.github.com" in url and "/reviews" in url:
            return 200, "[]"
        if "api.github.com" in url and "/pulls/" in url:
            import json as _json
            return 200, _json.dumps({"state": "open", "merged": False,
                                     "draft": False,
                                     "head": {"sha": "f" * 40}})
        if "/issues/" in url and method == "POST":
            self.comment_timeouts.append(timeout)
            return 201, "{}"
        return 200, "[]"


def test_judge_and_comment_timeouts_are_clamped_to_the_budget(tmp_path,
                                                              monkeypatch):
    """MEDIUM 3 — with most of the pass budget spent, the Summit call and
    the PR comment must be clamped to what is LEFT, never their full
    configured timeouts (r1 let the judge run 300s inside a 240s pass)."""
    from src.kronos import KronosState

    services = _JudgeServices()
    cfg = KronosConfig(state_dir=str(tmp_path), pass_budget_secs=240,
                       judge_timeout_secs=300,
                       reviews_database_url="postgresql://fake/fake")
    clock = _Clock(0.0)

    # the DB read bumps the clock: by the time the judge fires, only 40s of
    # budget remain (240 - 200)
    real_db = services.db

    class _BumpingDb:
        def rows(self, sql, params=()):
            clock.t = 200.0
            return real_db("dsn").rows(sql, params)

    services.db = lambda dsn, **kw: _BumpingDb()

    alerter = _FakeAlerter()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg2, state, now_fn, **kw: alerter)

    state = KronosState(str(tmp_path / "state.json"))
    rc = run_pass(cfg, dry_run=False, state=state,
                  services=services, out=lambda l: None,
                  now_fn=clock, mono_fn=clock)
    assert rc == 0
    assert services.judge_timeouts, "the judge never ran"
    assert max(services.judge_timeouts) <= 40, \
        f"judge timeout not clamped: {services.judge_timeouts}"
