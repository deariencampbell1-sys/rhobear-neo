"""W1-08-r1 — the Kronos systemd unit files and timer: spec, not deployment.

Nothing here starts, enables or installs a unit (the lane manager owns the
deploy steps). These tests pin the SPEC the manager installs:

- the service runs as its own user `rhobear-kronos`, keys delivered by
  /opt/rhobear/bin/vault-env (ExecStartPre, root), missing helper -> loud;
- the restart remedy needs `sudo`, so NoNewPrivileges=true is deliberately
  ABSENT from this unit (kernel: setuid exec is impossible under it — sudo
  can never elevate, the brief's restart remedy would be dead code). The
  sudoers line the manager installs (M-11) is the privilege boundary instead,
  and it is quoted in the unit so the manager can copy it;
- the timer wakes every 5 minutes and the unit runs exactly one pass.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
SERVICE = DEPLOY / "rhobear-kronos.service"
TIMER = DEPLOY / "rhobear-kronos.timer"

# The hardening set that CAN hold alongside sudo-based restarts
# (NoNewPrivileges is the one directive that cannot — see module docstring).
HARDENING = {
    "ProtectSystem": "full",
    "ProtectHome": "true",
    "PrivateTmp": "true",
    "RestrictAddressFamilies": "AF_UNIX AF_INET AF_INET6",
}


def _directives(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "[")):
            continue
        key, _, value = line.partition("=")
        out.setdefault(key.strip(), []).append(value.strip())
    return out


def test_service_runs_as_its_own_user():
    d = _directives(SERVICE.read_text(encoding="utf-8"))
    assert d.get("User") == ["rhobear-kronos"], d.get("User")
    assert d.get("Type") == ["oneshot"], "one timer firing = one pass"
    assert d.get("RuntimeDirectory") == ["rhobear-kronos"]


def test_service_takes_keys_from_vault_env_not_its_own_loader():
    d = _directives(SERVICE.read_text(encoding="utf-8"))
    pre = " ".join(d.get("ExecStartPre", []))
    assert "/opt/rhobear/bin/vault-env rhobear-kronos" in pre, pre
    assert pre.startswith("+"), "vault-env needs root, like the sibling units"
    envfile = " ".join(d.get("EnvironmentFile", []))
    assert "/run/rhobear-kronos/env" in envfile


def test_service_hardening_without_nonewprivileges():
    d = _directives(SERVICE.read_text(encoding="utf-8"))
    for key, want in HARDENING.items():
        assert d.get(key) == [want], f"{key}={want} required, got {d.get(key)}"
    assert d.get("NoNewPrivileges") is None, (
        "NoNewPrivileges=true would kill the brief's sudo restart remedy "
        "(setuid exec is impossible under it) — the sudoers line is the "
        "boundary instead; see the unit's comment for M-11"
    )
    rw = " ".join(d.get("ReadWritePaths", []))
    assert "/run/rhobear-kronos" in rw, "the state dir must be writable"


def test_unit_quotes_the_sudoers_line_for_the_manager():
    text = SERVICE.read_text(encoding="utf-8")
    assert "systemctl restart rhobear-neo.service" in text, (
        "the exact sudoers line for M-11 must be in the unit comment"
    )


def test_timer_wakes_every_five_minutes():
    d = _directives(TIMER.read_text(encoding="utf-8"))
    every5 = any(
        v.replace(" ", "").startswith("5min")
        for v in d.get("OnUnitActiveSec", []))
    calendar5 = any("0/5" in v or "*/5" in v
                    for v in d.get("OnCalendar", []))
    assert every5 or calendar5, "the timer must wake every 5 minutes"
    assert d.get("Unit") == ["rhobear-kronos.service"]


def test_timer_is_persistent():
    """A sleeping box must not skip the wake (the watchdog was dead for a
    month and nobody noticed — Persistent=true is the point of Kronos)."""
    d = _directives(TIMER.read_text(encoding="utf-8"))
    assert d.get("Persistent") == ["true"]


def test_units_verify_clean():
    """`systemd-analyze verify` exits 0. ExecStart binaries live under the
    deployed /opt tree, which CI boxes do not have — systemd-analyze then
    only notes the missing interpreter (exit 0), which is what the sibling
    units' test accepts too."""
    if shutil.which("systemd-analyze") is None:
        pytest.skip("systemd-analyze not available")
    proc = subprocess.run(
        ["systemd-analyze", "verify", str(SERVICE), str(TIMER)],
        capture_output=True, text=True,
    )
    out = (proc.stdout + proc.stderr).strip()
    assert proc.returncode == 0, f"systemd-analyze verify failed:\n{out}"
    real_warnings = [
        line for line in out.splitlines()
        if "warning" in line.lower()
        and "not executable" not in line
    ]
    assert not real_warnings, out
