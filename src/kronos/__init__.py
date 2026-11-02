"""Kronos — watches every service, and judges bad fix loops.

W1-08-r1. One daemon, two jobs (owner, 2026-10-25: "Kronos and the escalator
are supposed to be one and the same"):

  1. Watches every review service on a 5-minute timer (one pass per firing,
     4-minute budget, never two passes at once — flock on the state dir):
     unit health, HTTP health of the three loopback services, Postgres,
     failed webhook deliveries for BOTH Apps, PRs with a delivery but no
     Apollo review after 30 minutes, Zeus jobs stuck in a non-final phase,
     and installs gated for credits (alerted, never silent). Remedies knock
     heads loose: restart the dead unit (allow-list only, once per hour per
     target), redeliver failed hook deliveries, re-queue a stuck job — then
     alert on GitHub (one issue per problem in the reviews-app repo, label
     `kronos`, opened as the Zeus App; commented and closed when fixed;
     never the same fingerprint twice inside 6 hours).

  2. Judges bad fix loops: when a PR reaches 5 Zeus fix rounds without two
     greens, Kronos sends the round history to Summit through RouteGate
     (consumer `reviews:kronos/<install_id>`, read-only installation token).
     `yes` -> one escalated Zeus round whose brief is the plan; `no`, a bad
     answer or a timeout -> Zeus stops on that PR, one comment with the
     reason, the owner alerted. Never a 7th round.

Read-only on other services' CODE: this package edits only its own files and
shares state with Zeus only through the database rows Zeus already consumes
(`already_escalated`, neo_jobs) — src/escalation.py and the reviews-app are
imported read-only, never modified.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class KronosState:
    """Small JSON-file state store (alert dedupe windows, remedy ledger,
    flap windows, judged markers).

    `:memory:` keeps everything in a dict (tests). Otherwise the file is
    loaded lazily, written atomically on flush(), and every slot is an
    independent dict so callers can hand a slot to a ledger without sharing
    the whole store.
    """

    def __init__(self, path: str):
        self._memory = path == ":memory:"
        self._path = None if self._memory else Path(path)
        self._data: dict = {}
        if not self._memory and self._path is not None and self._path.exists():
            try:
                self._data = json.loads(
                    self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # A corrupt state file must never stop the watchdog: start
                # empty (worst case, one repeat alert after a corruption).
                self._data = {}
        if not isinstance(self._data, dict):
            self._data = {}

    def slot(self, name: str) -> dict:
        return self._data.setdefault(name, {})

    def flush(self) -> None:
        if self._memory or self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self._path)


class PassLock:
    """The one-pass-at-a-time lock: flock (POSIX) with an O_EXCL fallback for
    platforms without fcntl. A pass that cannot take the lock exits
    immediately — the timer fires again in 5 minutes."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._fd = None

    def acquire(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl  # noqa: F401 — proven available above
        except ImportError:
            fcntl = None  # type: ignore[assignment]
        else:
            self._fd = os.open(str(self._path),
                               os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError:
                os.close(self._fd)
                self._fd = None
                return False
        # Non-POSIX fallback: exclusive create. A stale lock from a killed
        # pass is older than any pass budget, so it is broken after 10 min.
        try:
            self._fd = os.open(str(self._path),
                               os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.write(self._fd, str(os.getpid()).encode())
            return True
        except FileExistsError:
            try:
                age = time.time() - self._path.stat().st_mtime
            except OSError:
                return False
            if age > 600:
                try:
                    self._path.unlink()
                except OSError:
                    return False
                return self.acquire()
            return False

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            import fcntl
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None
        try:
            self._path.unlink()
        except OSError:
            pass
