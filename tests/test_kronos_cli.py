"""W1-08-r1 — the CLI contract (acceptance).

`python -m src.kronos --once --dry-run` prints ONE LINE PER CHECK and changes
nothing: no state file, no remedy call, no GitHub call, no judge call. The
pass lock (flock on the state dir) makes two passes never overlap.
"""
from __future__ import annotations

import subprocess
import sys

from src.kronos.__main__ import main
from src.kronos.checks import CheckResult


class FakeServices:
    """Wires every external touch to fakes and records what was called."""

    def __init__(self, results: list[CheckResult] | None = None):
        self.results = results or [
            CheckResult("webhook", "ok", "listening"),
            CheckResult("unit:rhobear-neo", "ok", "active"),
        ]
        self.restart_calls: list[list[str]] = []
        http_calls: list = []
        self.http = http_calls
        self.check_names_run: list[str] = []

    def runner(self, argv):
        if "restart" in argv:
            self.restart_calls.append(list(argv))
        return type("R", (), {"rc": 0, "stdout": "active\n", "stderr": ""})()


def test_once_dry_run_prints_one_line_per_check(capsys, tmp_path):
    """One line per check, and NOTHING changes."""
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
    assert fake.restart_calls == [], "dry-run never restarts anything"
    assert fake.http == [], "dry-run never calls GitHub or RouteGate"


def test_once_dry_run_writes_no_state_file(capsys, tmp_path):
    fake = FakeServices()
    main(["--once", "--dry-run", "--state-dir", str(tmp_path)], _services=fake)
    assert list(tmp_path.iterdir()) == [], \
        f"dry-run wrote {list(tmp_path.iterdir())}"


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
        assert not fake.check_names_run, "a locked pass must not check"
    finally:
        lock.release()


def test_module_is_runnable():
    """`python -m src.kronos --help` exits 0 (the unit's ExecStart shape)."""
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    proc = subprocess.run(
        [sys.executable, "-m", "src.kronos", "--help"],
        capture_output=True, text=True, cwd=".", timeout=60, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--dry-run" in proc.stdout
