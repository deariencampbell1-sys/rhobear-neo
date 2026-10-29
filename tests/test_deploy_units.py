"""MEDIUM 1 / HIGH 2.2 (W1-10-r1) — the systemd units must be hardened.

`deploy/rhobear-neo-builder.service` runs the `--yolo --accept-hooks` fixer
with a push token in a worktree of third-party PR content.  The advisor unit
must run the frontier advisor as a dedicated unprivileged user
(`rhobear-advisor`) confined to its own clone dir — never as the service user
that can read `/opt/rhobear/rhobear-neo/.env` and `app-key.pem`.

A unit with a missing `User=` or a missing hardening directive fails HERE,
not at review time and not on the box.

Edge case covered: `systemd-analyze verify` on both units — zero warnings.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
BUILDER_UNIT = DEPLOY / "rhobear-neo-builder.service"
ADVISOR_UNIT = DEPLOY / "rhobear-neo-advisor.service"

# The hardening set every unit in this lane must carry.
HARDENING = {
    "NoNewPrivileges": "true",
    "ProtectSystem": "strict",
    "ProtectHome": "true",
    "PrivateTmp": "true",
    "RestrictAddressFamilies": "AF_UNIX AF_INET AF_INET6",
}

# Paths that must never be writable from either unit.
FORBIDDEN_WRITABLE = (".env", "app-key.pem")


def _directives(text: str) -> dict[str, list[str]]:
    """Parse a unit file into {key: [values]} (comments/sections skipped)."""
    out: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "[")):
            continue
        key, _, value = line.partition("=")
        out.setdefault(key.strip(), []).append(value.strip())
    return out


def _writable_paths(directives: dict[str, list[str]]) -> set[str]:
    paths: set[str] = set()
    for value in directives.get("ReadWritePaths", []):
        paths.update(value.split())
    return paths


def _assert_hardened(name: str, text: str) -> dict[str, list[str]]:
    d = _directives(text)
    assert d.get("User"), f"{name}: a unit without User= fails here, not on the box"
    for key, want in HARDENING.items():
        assert d.get(key) == [want], (
            f"{name}: {key}={want} required, got {d.get(key)}"
        )
    return d


def test_builder_unit_is_hardened():
    """MEDIUM 1 — the --yolo fixer unit carries the full hardening set and
    writable paths derived from the code, not guessed."""
    text = BUILDER_UNIT.read_text(encoding="utf-8")
    d = _assert_hardened("builder", text)
    assert d.get("User") == ["slang"], d.get("User")

    # ReadWritePaths must be EXACTLY the paths src/neo_builder.py writes:
    # WORKTREES (the per-repo-PR git clones + .locks) and BRIEFS (brief.md).
    from src.neo_builder import BRIEFS, WORKTREES
    rw = _writable_paths(d)
    assert rw == {str(WORKTREES), str(BRIEFS)}, (
        f"ReadWritePaths must match the code's write targets exactly: {rw}"
    )
    for path in rw:
        for forbidden in FORBIDDEN_WRITABLE:
            assert not path.endswith(forbidden), path


def test_advisor_unit_is_hardened():
    """HIGH 2.2 — the advisor runs as its own user, confined to its clone
    dir; the main deploy dir, .env and the App key paths are not writable
    (and W1-03's user separation keeps them unreadable)."""
    text = ADVISOR_UNIT.read_text(encoding="utf-8")
    d = _assert_hardened("advisor", text)
    assert d.get("User") == ["rhobear-advisor"], (
        "the advisor must NOT run as the service user"
    )

    rw = _writable_paths(d)
    assert rw, "the advisor needs exactly its own clone dir writable"
    for path in rw:
        for forbidden in FORBIDDEN_WRITABLE:
            assert not path.endswith(forbidden), path
        assert path != "/opt/rhobear/rhobear-neo", (
            "the main deploy dir (with .env and app-key.pem) must not be writable"
        )
        assert not path.startswith("/opt/rhobear/rhobear-neo/"), (
            "nothing inside the main deploy dir may be writable: " + path
        )


def test_units_verify_clean():
    """Edge 4 — `systemd-analyze verify` on both units reports zero
    warnings/errors.  Skipped where systemd-analyze or the deployed /opt
    layout is absent (verify checks ExecStart executability against the
    box layout)."""
    if shutil.which("systemd-analyze") is None:
        pytest.skip("systemd-analyze not available")
    for unit in (BUILDER_UNIT, ADVISOR_UNIT):
        directives = _directives(unit.read_text(encoding="utf-8"))
        for key in ("ExecStart", "ExecStartPre"):
            for cmd in directives.get(key, []):
                binary = cmd.split()[0].lstrip("-@:+!")
                if not os.path.exists(binary):
                    pytest.skip(f"deployed layout not present ({binary})")
    proc = subprocess.run(
        ["systemd-analyze", "verify", str(BUILDER_UNIT), str(ADVISOR_UNIT)],
        capture_output=True, text=True,
    )
    out = (proc.stdout + proc.stderr).strip()
    assert proc.returncode == 0, f"systemd-analyze verify failed:\n{out}"
    assert "warning" not in out.lower(), out
