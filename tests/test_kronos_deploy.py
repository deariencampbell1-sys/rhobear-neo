"""W1-08-r2 — the Kronos systemd unit files and timer: spec, not deployment.

Nothing here starts, enables or installs a unit (the lane manager owns the
deploy steps). These tests pin the SPEC the manager installs:

- r2 RULING (Morpheus, 2026-11-02): Kronos is READ-ONLY on other services —
  it observes; it never restarts, never kills, never edits a unit. The r1
  sudoers line and restart privilege are GONE, so NoNewPrivileges=true holds
  again (the r1 deviation existed only for the sudo remedy);
- HIGH 1: the alert dedupe, flap windows and judged markers must survive the
  oneshot's exit and a reboot — they live in a durable StateDirectory, not
  in the RuntimeDirectory systemd wipes when the service stops;
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

# The full hardening set — NoNewPrivileges is BACK: with no restart remedy
# there is nothing that needs setuid elevation.
HARDENING = {
    "NoNewPrivileges": "true",
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


def test_service_takes_keys_from_vault_env_not_its_own_loader():
    d = _directives(SERVICE.read_text(encoding="utf-8"))
    pre = " ".join(d.get("ExecStartPre", []))
    assert "/opt/rhobear/bin/vault-env rhobear-kronos" in pre, pre
    assert pre.startswith("+"), "vault-env needs root, like the sibling units"
    envfile = " ".join(d.get("EnvironmentFile", []))
    assert "/run/rhobear-kronos/env" in envfile


def test_service_is_read_only_no_restart_privilege_anywhere():
    """RULING — Kronos never restarts, kills or edits a unit. The unit file
    must carry NO sudoers line and NO systemctl restart privilege."""
    text = SERVICE.read_text(encoding="utf-8")
    assert "sudo" not in text.lower(), "a sudoers line survived the r2 ruling"
    assert "systemctl restart" not in text.lower(), (
        "a restart privilege or remedy note survived the r2 ruling"
    )
    d = _directives(text)
    assert d.get("NoNewPrivileges") == ["true"], (
        "with no privilege to elevate, NoNewPrivileges=true must hold"
    )


def test_service_hardening_set():
    d = _directives(SERVICE.read_text(encoding="utf-8"))
    for key, want in HARDENING.items():
        assert d.get(key) == [want], f"{key}={want} required, got {d.get(key)}"


def test_state_is_durable_across_oneshot_restarts():
    """HIGH 1 — the ledger (alert dedupe, flaps, judged markers) lives in a
    StateDirectory that systemd keeps across oneshot exits and reboots; the
    state dir env points there; the unit may write it."""
    d = _directives(SERVICE.read_text(encoding="utf-8"))
    assert d.get("StateDirectory") == ["rhobear-kronos"], d.get(
        "StateDirectory")
    assert d.get("StateDirectoryMode") == ["0700"], d.get(
        "StateDirectoryMode")
    env = " ".join(d.get("Environment", []))
    assert "RHOBEAR_KRONOS_STATE_DIR=/var/lib/rhobear-kronos" in env, env
    rw = " ".join(d.get("ReadWritePaths", []))
    assert "/var/lib/rhobear-kronos" in rw, \
        "the durable state dir must be writable"


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


def test_units_verify_clean(tmp_path):
    """`systemd-analyze verify` clean.

    The ExecStart binaries live under the deployed /opt tree, which CI boxes
    do not have — verify there only notes the missing interpreter (and exits
    1, like it does for the sibling rhobear-neo units). So when the deployed
    layout is absent, the verify runs inside a SYNTHETIC root
    (--root=): the units plus the exact ExecStart paths as stub
    executables. Exit 0 with zero output is the clean verdict; no root and
    no systemd unit is started.
    """
    if shutil.which("systemd-analyze") is None:
        pytest.skip("systemd-analyze not available")
    proc = subprocess.run(
        ["systemd-analyze", "verify", str(SERVICE), str(TIMER)],
        capture_output=True, text=True,
    )
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode == 0:
        assert not [l for l in out.splitlines()
                    if "warning" in l.lower() and "not executable" not in l], out
        return
    # Deployed layout absent -> synthetic root (the not-executable note is
    # the trigger; anything else failing means a real unit problem).
    if not all("not executable" in l for l in out.splitlines()):
        pytest.fail(f"systemd-analyze verify failed:\n{out}")
    root = tmp_path / "fakeroot"
    (root / "etc/systemd/system").mkdir(parents=True)
    (root / "usr/lib/systemd/system").mkdir(parents=True)
    (root / "opt/rhobear/rhobear-kronos/.venv/bin").mkdir(parents=True)
    (root / "opt/rhobear/bin").mkdir(parents=True)
    (root / "usr/bin").mkdir(parents=True)
    shutil.copy(SERVICE, root / "etc/systemd/system/rhobear-kronos.service")
    shutil.copy(TIMER, root / "etc/systemd/system/rhobear-kronos.timer")
    for target, source in (
        ("opt/rhobear/rhobear-kronos/.venv/bin/python", "/usr/bin/python3"),
        ("usr/bin/env", "/usr/bin/env"),
    ):
        if os.path.exists(source):
            shutil.copy(source, root / target)
            os.chmod(root / target, 0o755)
    stub = root / "opt/rhobear/bin/vault-env"
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(stub, 0o755)
    for unit in ("sysinit.target", "basic.target"):
        src = f"/usr/lib/systemd/system/{unit}"
        if os.path.exists(src):
            shutil.copy(src, root / "usr/lib/systemd/system" / unit)
    proc2 = subprocess.run(
        ["systemd-analyze", "verify",
         f"--root={root}", "rhobear-kronos.service", "rhobear-kronos.timer"],
        capture_output=True, text=True,
    )
    out2 = (proc2.stdout + proc2.stderr).strip()
    assert proc2.returncode == 0, f"systemd-analyze verify failed:\n{out2}"
    assert out2 == "", f"verify must be clean, got: {out2}"
