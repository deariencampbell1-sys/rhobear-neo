"""W1-03-r1 — vault-backed service environments without unversioned drop-ins.

Fourteen tests over the installer-generated ``vault-env`` helper, the two owned
units and the secret ignore rules.  Every fixture is fabricated inside the
test: no real PEM, no real DSN, no real vault file, no network, no write to a
production path, and no systemd is started — the helper is exercised as a
plain POSIX ``sh`` script in pytest's tmp_path.  No product code is imported.
"""

from __future__ import annotations

import base64
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
INSTALLER = DEPLOY / "install.sh"
GITIGNORE = ROOT / ".gitignore"

HELPER_MARKER = "VAULT_ENV_HELPER"
NAMES_MARKER = "VAULT_ENV_NAMES"

# The versioned allowlists, exactly (aliases preserve the live code's names).
SERVICES = ("rhobear-neo", "rhobear-neo-builder")
NAMES_FILE_LINES = [
    "NEO_GITHUB_APP_KEY_PATH=RHOBEAR_ZEUS_GH_APP_KEY_B64",
    "RHOBEAR_NEO_WEBHOOK_SECRET=RHOBEAR_ZEUS_WEBHOOK_SECRET",
    "DATABASE_URL=REVIEWS_DATABASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPINFRA_API_KEY",
]
VAULT_NAMES = [
    "RHOBEAR_ZEUS_GH_APP_KEY_B64",
    "RHOBEAR_ZEUS_WEBHOOK_SECRET",
    "REVIEWS_DATABASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPINFRA_API_KEY",
]
APP_ID_LINE = "NEO_GITHUB_APP_ID=4347986"

UNITS = {
    "rhobear-neo.service": "rhobear-neo",
    "rhobear-neo-builder.service": "rhobear-neo-builder",
}

# Restart behavior is kept exactly as merged (the units' own settings).
RESTART_SEC = {"rhobear-neo.service": "3", "rhobear-neo-builder.service": "15"}

IGNORE_PATTERNS = [".env", ".dburl", ".dskey", ".ghtoken", ".webhooksecret", "*.pem", "app-key.pem"]

# The merged W1-10-r1 hardening that must remain intact (checked statically).
BUILDER_HARDENING = {
    "User": ["slang"],
    "NoNewPrivileges": ["true"],
    "ProtectSystem": ["strict"],
    "ProtectHome": ["true"],
    "PrivateTmp": ["true"],
    "RestrictAddressFamilies": ["AF_UNIX AF_INET AF_INET6"],
}
BUILDER_WRITABLE = {"/opt/rhobear/neo-builder-worktrees", "/opt/rhobear/neo-builder-briefs"}


# --- installer helpers -------------------------------------------------------

def _installer_text() -> str:
    return INSTALLER.read_text(encoding="utf-8")


def _heredoc(text: str, marker: str) -> str:
    """Extract the body of a quoted heredoc `<<'MARKER'` ... `MARKER`."""
    match = re.search(
        rf"<<'{re.escape(marker)}'\n(.*?)^{re.escape(marker)}[ \t]*$",
        text,
        re.S | re.M,
    )
    assert match, (
        f"installer heredoc {marker} absent — no installer helper markers (red)"
    )
    return match.group(1)


def _helper_body() -> str:
    return _heredoc(_installer_text(), HELPER_MARKER)


def _install_helper(tmp_path: Path) -> Path:
    """Extract the quoted helper heredoc from the installer to a temp dir."""
    helper = tmp_path / "vault-env"
    helper.write_text(_helper_body(), encoding="utf-8")
    helper.chmod(0o755)
    return helper


def _run_helper(
    helper: Path, service: str, *, vault: Path, run_dir: Path, names_dir: Path,
    extra_env=None, path_prefix=None,
):
    """Run the helper as plain POSIX sh with only the documented overrides."""
    env = {
        "VAULT_FILE": str(vault),
        "RUN_DIR": str(run_dir),
        "NAMES_DIR": str(names_dir),
        "PATH": (f"{path_prefix}:" if path_prefix else "") + "/usr/bin:/bin",
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["/bin/sh", str(helper), service],
        env=env,
        capture_output=True,
        text=True,
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _directives(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "[")):
            continue
        key, _, value = line.partition("=")
        out.setdefault(key.strip(), []).append(value.strip())
    return out


# --- 1. missing vault name fails closed --------------------------------------

def test_missing_vault_name_fails_closed_without_value(tmp_path):
    """A name absent from the vault aborts non-zero naming only the name, and
    the fabricated sentinel sitting in the vault never reaches any output."""
    sentinel = "SECRET-SENTINEL-9f3a7c-NOT-A-REAL-VALUE"
    vault = _write(tmp_path / "vault", f"OTHER_SECRET={sentinel}\n")
    names_dir = tmp_path / "names"
    names_dir.mkdir()
    _write(names_dir / "rhobear-neo.names", "RHOBEAR_ZEUS_GH_APP_KEY_B64\n")
    run_dir = tmp_path / "run"
    helper = _install_helper(tmp_path)

    proc = _run_helper(
        helper, "rhobear-neo", vault=vault, run_dir=run_dir, names_dir=names_dir
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, "missing vault name must fail closed"
    assert "missing vault name RHOBEAR_ZEUS_GH_APP_KEY_B64" in proc.stderr
    assert sentinel not in combined, "a vault value leaked into the output"
    assert not (run_dir / "rhobear-neo" / "env").exists(), "no usable env file may appear"
    assert list(run_dir.glob("**/*.pem")) == [], "no key file may appear"


# --- 2. aliases, PEM decode, atomic env, modes --------------------------------

def test_aliases_pem_decode_atomic_env_and_modes(tmp_path):
    """The service-specific aliases decode the App key to a pem and export its
    absolute path; values are data; modes are 0700/0600; no stdout on success."""
    key_bytes = (
        b"-----BEGIN FAKE PRIVATE KEY-----\n"
        b"0123456789abcdefB64MATERIAL0000\n"
        b"-----END FAKE PRIVATE KEY-----\n"
    )
    marker = tmp_path / "executed-marker"
    values = {
        "RHOBEAR_ZEUS_GH_APP_KEY_B64": base64.b64encode(key_bytes).decode(),
        "RHOBEAR_ZEUS_WEBHOOK_SECRET": f"wh=sec#ret@$(touch {marker}) `id` ;x",
        "REVIEWS_DATABASE_URL": "postgres://u:p=ss#w@rd@db.local/db?ssl=x",
        "DEEPSEEK_API_KEY": "sk-fake-zeus-key-not-real",
        "DEEPINFRA_API_KEY": "di-fake-key-not-real",
    }
    vault = _write(tmp_path / "vault", "".join(f"{k}={v}\n" for k, v in values.items()))
    run_dir = tmp_path / "run"

    for service in SERVICES:
        names_dir = tmp_path / f"names-{service}"
        names_dir.mkdir()
        _write(names_dir / f"{service}.names", "".join(l + "\n" for l in NAMES_FILE_LINES))
        bin_dir = tmp_path / f"bin-{service}"
        bin_dir.mkdir()
        helper = _install_helper(bin_dir)

        proc = _run_helper(
            helper, service, vault=vault, run_dir=run_dir, names_dir=names_dir
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == "", "no success stdout is allowed"
        assert proc.stderr == "", proc.stderr
        combined = proc.stdout + proc.stderr
        for value in values.values():
            assert value not in combined, "a secret value leaked into the output"

        svc_dir = run_dir / service
        env_file = svc_dir / "env"
        assert env_file.exists()
        assert stat.S_IMODE(svc_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600

        expected_pem = svc_dir / "neo_github_app_key_path.pem"
        assert expected_pem.is_file()
        assert expected_pem.read_bytes() == key_bytes, "decoded PEM bytes must match"
        assert expected_pem.is_absolute()
        assert stat.S_IMODE(expected_pem.stat().st_mode) == 0o600

        lines = env_file.read_text(encoding="utf-8").splitlines()
        assert lines == [
            f'NEO_GITHUB_APP_KEY_PATH="{expected_pem}"',
            f'RHOBEAR_NEO_WEBHOOK_SECRET="{_esc(values["RHOBEAR_ZEUS_WEBHOOK_SECRET"])}"',
            f'DATABASE_URL="{_esc(values["REVIEWS_DATABASE_URL"])}"',
            f'DEEPSEEK_API_KEY="{_esc(values["DEEPSEEK_API_KEY"])}"',
            f'DEEPINFRA_API_KEY="{_esc(values["DEEPINFRA_API_KEY"])}"',
        ], 'the five service-specific aliases must be emitted exactly as NAME="<escaped>"'
        assert not any(re.match(r".*_B64=", ln) for ln in lines), "no _B64 name may be exported"
        for ln in lines:
            key, _, value = ln.partition("=")
            assert key in {
                "NEO_GITHUB_APP_KEY_PATH",
                "RHOBEAR_NEO_WEBHOOK_SECRET",
                "DATABASE_URL",
                "DEEPSEEK_API_KEY",
                "DEEPINFRA_API_KEY",
            }
            assert value, f"empty value for {key}"

        # atomic delivery: output is staged in a sibling temp dir and renamed
        body = _helper_body()
        assert "mktemp -d" in body and ".vault-env." in body, (
            "helper must stage output in a sibling temp dir"
        )
        assert "mv -T -f" in body, "helper must rename temp output into place"
        assert "env.tmp." not in body and ".tmp.$$" not in body, (
            "no temp file may be created inside the service dir"
        )

    assert not marker.exists(), (
        "a metacharacter value was executed instead of treated as data"
    )


# --- 3. invalid base64 / duplicate outputs leave nothing behind ---------------

def test_invalid_base64_and_duplicate_output_leave_no_partial_files(tmp_path):
    """Both failure modes exit non-zero, write no final env/PEM, and never
    print the fabricated values."""
    bad_b64 = "!!not-base64-at-all!!"
    dup_secret = "dup-sentinel-value-4411"
    run_dir = tmp_path / "run"
    helper = _install_helper(tmp_path)

    # a) invalid base64 in the _B64 record
    names_a = tmp_path / "names-a"
    names_a.mkdir()
    _write(names_a / "svc.names", "APP_KEY_PATH=FAKE_KEY_B64\n")
    vault_a = _write(tmp_path / "vault-a", f"FAKE_KEY_B64={bad_b64}\n")
    proc = _run_helper(helper, "svc", vault=vault_a, run_dir=run_dir, names_dir=names_a)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, "invalid base64 must fail non-zero"
    assert bad_b64 not in combined, "the invalid value leaked into the output"
    assert not (run_dir / "svc" / "env").exists(), "no final env after invalid base64"
    assert list(run_dir.glob("**/*.pem")) == [], "no pem after invalid base64"

    # b) duplicate output names in the allowlist
    names_b = tmp_path / "names-b"
    names_b.mkdir()
    _write(names_b / "svc.names", "OUT_NAME=FAKE_A\nOUT_NAME=FAKE_B\n")
    vault_b = _write(tmp_path / "vault-b", f"FAKE_A={dup_secret}\nFAKE_B={dup_secret}\n")
    proc = _run_helper(helper, "svc", vault=vault_b, run_dir=run_dir, names_dir=names_b)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, "a duplicate output name must fail non-zero"
    assert "duplicate output name OUT_NAME" in proc.stderr
    assert dup_secret not in combined, "a vault value leaked into the output"
    assert not (run_dir / "svc" / "env").exists(), "no final env after duplicate output"
    assert list(run_dir.glob("**/*.pem")) == [], "no pem after duplicate output"


# --- 4. units use the versioned runtime environment --------------------------

def test_units_use_versioned_runtime_environment():
    """Every owned unit carries the versioned vault wiring and no /opt env or
    repository-env claim; the neo units keep their own stop behavior."""
    for unit_name, service in UNITS.items():
        text = (DEPLOY / unit_name).read_text(encoding="utf-8")
        lines = [ln.strip() for ln in text.splitlines()]

        assert lines.count(f"EnvironmentFile=-/run/{service}/env") == 1, (
            f"{unit_name}: EnvironmentFile=-/run wiring absent (red)"
        )
        assert lines.count(f"EnvironmentFile=/run/{service}/env") == 0, (
            f"{unit_name}: EnvironmentFile without the - fails on a fresh install"
        )
        expected_pre = (
            f"ExecStartPre=+/usr/bin/env PATH=/usr/bin:/bin "
            f"/opt/rhobear/bin/vault-env {service}"
        )
        assert lines.count(expected_pre) == 1, f"{unit_name}: exact {expected_pre} once"
        pre_lines = [ln for ln in lines if ln.startswith("ExecStartPre=")]
        assert len(pre_lines) == 1 and pre_lines[0].startswith("ExecStartPre=+/"), (
            f"{unit_name}: ExecStartPre must start with +/ and never a - (red)"
        )
        assert not any(ln.startswith("EnvironmentFile=/opt") for ln in lines), (
            f"{unit_name}: unversioned /opt EnvironmentFile must be gone"
        )
        assert ".env" not in text, f"{unit_name}: a repository .env claim remains"

        runtime_dirs = [ln for ln in lines if ln.startswith("RuntimeDirectory=")]
        assert runtime_dirs == [f"RuntimeDirectory={service} {service}-gh"], (
            f"{unit_name}: runtime directory wiring"
        )
        assert lines.count("RuntimeDirectoryMode=0700") == 1

        assert any(APP_ID_LINE in ln for ln in lines), f"{unit_name}: versioned App ID"
        assert "Environment=RHOBEAR_GH_APP_OWNER=deariencampbell1-sys" in lines
        assert f"Environment=RHOBEAR_GH_APP_CACHE=/run/{service}-gh" in lines
        path_lines = [ln for ln in lines if ln.startswith("Environment=PATH=")]
        assert len(path_lines) == 1, f"{unit_name}: exactly one PATH"
        path_value = path_lines[0].split("=", 2)[2]
        first = path_value.split(":")[0]
        assert first == "/opt/rhobear/gh-app", f"{unit_name}: gh-app wrapper must be first on PATH"

        env_lines = [ln for ln in lines if ln.startswith("Environment=")]
        assert set(env_lines) == {
            f"Environment={APP_ID_LINE}",
            "Environment=RHOBEAR_GH_APP_OWNER=deariencampbell1-sys",
            f"Environment=RHOBEAR_GH_APP_CACHE=/run/{service}-gh",
            f"Environment=PATH={path_value}",
        }, f"{unit_name}: exactly the drop-in settings, no provider/model drift"
        assert lines.count("Restart=always") == 1, unit_name
        assert lines.count(f"RestartSec={RESTART_SEC[unit_name]}") == 1, (
            f"{unit_name}: restart behavior must stay as merged"
        )

        assert "TimeoutStopSec" not in text, f"{unit_name}: neo keeps its own stop behavior"
        exec_starts = [ln for ln in lines if ln.startswith("ExecStart=")]
        assert exec_starts, f"{unit_name}: ExecStart preserved"

    assert "ExecStart=/opt/rhobear/rhobear-neo/.venv/bin/python -m src" in [
        ln.strip() for ln in (DEPLOY / "rhobear-neo.service").read_text(encoding="utf-8").splitlines()
    ]
    assert "ExecStart=/opt/rhobear/rhobear-neo/.venv/bin/python -m src.neo_builder" in [
        ln.strip()
        for ln in (DEPLOY / "rhobear-neo-builder.service").read_text(encoding="utf-8").splitlines()
    ]


# --- 5. installer has no repository-secret bootstrap -------------------------

def test_installer_has_no_repository_secret_bootstrap():
    """The installer installs the quoted helper and the versioned name files,
    and never creates, copies, edits or instructs editing a repository .env or
    app-key.pem."""
    text = _installer_text()

    helper = _helper_body()
    assert helper.startswith("#!/bin/sh"), "the helper must be a POSIX sh script"
    assert "chmod 0755 /opt/rhobear/bin/vault-env" in text, "helper must be installed 0755"

    names_body = _heredoc(text, NAMES_MARKER)
    assert names_body.splitlines() == NAMES_FILE_LINES, (
        "the versioned name file content must match the mandate exactly"
    )
    for service in SERVICES:
        assert f"{service}" in text, f"name file for {service} must be installed"
    assert "vault-env.d" in text, "name files install under /etc/rhobear/vault-env.d"
    assert "chmod 0644" in text, "name files must be installed 0644"

    # no repository-secret bootstrap of any kind
    assert "app-key.pem" not in text, "installer must not touch app-key.pem"
    assert ".env.example" not in text, "installer must not copy the example env"
    assert not re.search(r"(?<![A-Za-z0-9_.\-])\.env(?![A-Za-z0-9_])", text), (
        "installer must not create, copy, edit or instruct editing a repository .env"
    )


# --- 6. ignore patterns and the existing hardening ---------------------------

def test_secret_ignore_patterns_and_existing_hardening():
    """All seven secret ignore patterns are present, and the merged builder /
    advisor hardening remains intact (statically, without importing code)."""
    lines = {ln.strip() for ln in GITIGNORE.read_text(encoding="utf-8").splitlines()}
    for pattern in IGNORE_PATTERNS:
        assert pattern in lines, f".gitignore is missing the exact pattern {pattern}"

    builder = _directives((DEPLOY / "rhobear-neo-builder.service").read_text(encoding="utf-8"))
    for key, want in BUILDER_HARDENING.items():
        assert builder.get(key) == want, f"builder hardening changed: {key}={builder.get(key)}"
    rw = set()
    for value in builder.get("ReadWritePaths", []):
        rw.update(value.split())
    assert rw == BUILDER_WRITABLE, f"exact writable paths changed: {rw}"

    advisor = _directives((DEPLOY / "rhobear-neo-advisor.service").read_text(encoding="utf-8"))
    assert advisor.get("User") == ["rhobear-advisor"]
    for key, want in {
        "NoNewPrivileges": ["true"],
        "ProtectSystem": ["strict"],
        "ProtectHome": ["true"],
        "PrivateTmp": ["true"],
        "RestrictAddressFamilies": ["AF_UNIX AF_INET AF_INET6"],
    }.items():
        assert advisor.get(key) == want, f"advisor hardening changed: {key}"


# --- r3: outputs take the runtime directory owner ----------------------------

def _esc(value: str) -> str:
    """The r3 env-file escaping: every backslash doubled, then every quote
    backslashed."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _systemd_unquote(quoted: str) -> str:
    r"""Systemd's env-file double-quote rules: strip the outer quotes, then
    \X yields X."""
    assert quoted[:1] == quoted[-1:] == '"', repr(quoted)
    return re.sub(r"\\(.)", r"\1", quoted[1:-1], flags=re.S)


KEY_BYTES = (b"-----BEGIN FAKE PRIVATE KEY-----\nB64MATERIAL0000\n"
             b"-----END FAKE PRIVATE KEY-----\n")


def _r3_fixture(tmp_path: Path):
    """Shared r3 fixture: a names file and vault with one ordinary name and one
    _B64 name, plus the extracted helper and its run dir."""
    ord_value = "ord-sentinel-7c1e"
    vault = _write(tmp_path / "vault",
                   f"ORD_VAULT={ord_value}\nFAKE_KEY_B64={base64.b64encode(KEY_BYTES).decode()}\n")
    names_dir = tmp_path / "names"
    names_dir.mkdir()
    _write(names_dir / "svc.names", "ORD_OUT=ORD_VAULT\nKEY_PATH=FAKE_KEY_B64\n")
    helper_dir = tmp_path / "bin"
    helper_dir.mkdir()
    return _install_helper(helper_dir), tmp_path / "run", vault, names_dir, ord_value


def test_outputs_take_runtime_dir_owner(tmp_path):
    """r3 — the helper derives ownership from the service dir
    (``chown --reference=<svc_dir>``) before the rename: env and pem end up
    with the service directory's uid/gid, the right contents and modes, and
    the final paths are regular files. No fake `chown` is needed — ownership
    is compared directly against the service directory."""
    helper, run_dir, vault, names_dir, ord_value = _r3_fixture(tmp_path)
    proc = _run_helper(helper, "svc", vault=vault, run_dir=run_dir, names_dir=names_dir)
    assert proc.returncode == 0, proc.stderr
    svc_dir = run_dir / "svc"
    env_file = svc_dir / "env"
    pem = svc_dir / "key_path.pem"
    assert env_file.is_file() and pem.is_file(), "both outputs must be installed"
    ref = (svc_dir.stat().st_uid, svc_dir.stat().st_gid)
    assert (env_file.stat().st_uid, env_file.stat().st_gid) == ref, (
        "env must take the service directory's owner"
    )
    assert (pem.stat().st_uid, pem.stat().st_gid) == ref, (
        "pem must take the service directory's owner"
    )
    assert "chown --reference=" in _helper_body(), (
        "ownership must still be derived from the service directory"
    )
    assert stat.S_IMODE(svc_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(pem.stat().st_mode) == 0o600
    assert env_file.read_text(encoding="utf-8").splitlines() == [
        f'ORD_OUT="{ord_value}"',
        f'KEY_PATH="{pem}"',
    ]
    assert pem.read_bytes() == KEY_BYTES


def test_chown_failure_fails_closed(tmp_path):
    """r3 — if the owner cannot be set the helper fails closed: non-zero exit,
    `cannot set owner`, no final env or pem, no temp files left, and the fake
    sentinel value never printed. Failure is injected by replacing the
    helper's absolute ``/usr/bin/chown`` in the extracted test copy with a
    nonexistent temporary path — never a PATH-planted fake."""
    helper, run_dir, vault, names_dir, sentinel = _r3_fixture(tmp_path)
    body = helper.read_text(encoding="utf-8")
    assert "/usr/bin/chown" in body, (
        "helper must call chown at its absolute path for fail-closed injection (red)"
    )
    missing_chown = tmp_path / "no-such-chown"
    helper.write_text(body.replace("/usr/bin/chown", str(missing_chown)),
                      encoding="utf-8")
    proc = _run_helper(helper, "svc", vault=vault, run_dir=run_dir, names_dir=names_dir)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, "a failed chown must fail closed"
    assert "cannot set owner" in proc.stderr
    assert sentinel not in combined, "a vault value leaked into the output"
    assert not (run_dir / "svc" / "env").exists(), "no final env may appear"
    assert list(run_dir.glob("**/*.pem")) == [], "no pem may appear"
    assert list(run_dir.glob("**/.vault-env.*")) == [], "no temp file may survive"
    assert [p.name for p in (run_dir / "svc").iterdir()] == [], "no partial output"


def test_env_values_survive_systemd_quoting(tmp_path):
    """r3 — every env line is exactly NAME="<escaped>", and a parser applying
    systemd's double-quote rules (strip outer quotes; a backslash escapes
    the next character) returns
    every original value byte for byte: a leading quote, an embedded
    backslash, a trailing space, a mid-line # and an =."""
    values = {
        "V_LEAD": '"leading-quote',
        "V_BACK": "back\\slash",
        "V_TRAIL": "trailing-space ",
        "V_HASH": "mid#dle",
        "V_EQ": "eq=sign",
    }
    vault = _write(tmp_path / "vault", "".join(f"Q_{k[2:]}={v}\n" for k, v in values.items()))
    names_dir = tmp_path / "names"
    names_dir.mkdir()
    _write(names_dir / "svc.names", "".join(f"{k}=Q_{k[2:]}\n" for k in values))
    helper_dir = tmp_path / "bin"
    helper_dir.mkdir()
    helper = _install_helper(helper_dir)
    proc = _run_helper(helper, "svc", vault=vault, run_dir=tmp_path / "run",
                       names_dir=names_dir)
    assert proc.returncode == 0, proc.stderr
    parsed = {}
    for ln in (tmp_path / "run" / "svc" / "env").read_text(encoding="utf-8").splitlines():
        name, _, quoted = ln.partition("=")
        assert name in values, f"unexpected env name {name}"
        assert quoted == f'"{_esc(values[name])}"', (
            f'{name}: env line must be exactly NAME="<escaped>", got {quoted!r}'
        )
        parsed[name] = _systemd_unquote(quoted)
    assert parsed == values, "systemd's parser must return every value byte for byte"


def test_service_name_dot_rejected(tmp_path):
    """r3 — `vault-env .` and `vault-env ..` exit non-zero with the
    `invalid service name` diagnostic and create no files."""
    vault = _write(tmp_path / "vault", "ORD_VAULT=not-for-dot-names\n")
    names_dir = tmp_path / "names"
    names_dir.mkdir()
    run_dir = tmp_path / "run"
    helper_dir = tmp_path / "bin"
    helper_dir.mkdir()
    helper = _install_helper(helper_dir)
    for service in (".", ".."):
        proc = _run_helper(helper, service, vault=vault, run_dir=run_dir,
                           names_dir=names_dir)
        combined = proc.stdout + proc.stderr
        assert proc.returncode != 0, f"vault-env {service} must be rejected"
        assert "invalid service name" in proc.stderr
        assert "not-for-dot-names" not in combined, "a vault value leaked into the output"
    assert not run_dir.exists() or list(run_dir.iterdir()) == [], "no files may appear"


def test_preplaced_symlinks_are_not_followed(tmp_path):
    """r3 — a service pre-placing symlinks at every temp and final name in its
    own runtime dir gains nothing: every sentinel is byte-identical after the
    run, and env and the pem are regular 0600 files with the right contents."""
    helper, run_dir, vault, names_dir, ord_value = _r3_fixture(tmp_path)
    svc_dir = run_dir / "svc"
    svc_dir.mkdir(parents=True)
    sentinels = {}
    for name, tag in (("env", "env-final"), ("key_path.pem", "pem-final")):
        s = _write(tmp_path / f"sentinel-{tag}", f"SENTINEL-{tag}")
        (svc_dir / name).symlink_to(s)
        sentinels[tag] = s
    tmp_sentinel = _write(tmp_path / "sentinel-tmp", "SENTINEL-tmp")
    pids = set(range(1, 2049))
    pids.update(range(os.getpid() - 32, os.getpid() + 256))
    for n in sorted(pids):
        (svc_dir / f"env.tmp.{n}").symlink_to(tmp_sentinel)
    proc = _run_helper(helper, "svc", vault=vault, run_dir=run_dir, names_dir=names_dir)
    assert proc.returncode == 0, proc.stderr
    assert tmp_sentinel.read_text() == "SENTINEL-tmp", "a temp-name sentinel was written through"
    for tag, s in sentinels.items():
        assert s.read_text() == f"SENTINEL-{tag}", f"the {tag} sentinel was written through"
    env_file = svc_dir / "env"
    assert env_file.is_file() and not env_file.is_symlink(), "env must replace the link"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert env_file.read_text(encoding="utf-8").splitlines() == [
        f'ORD_OUT="{ord_value}"',
        f'KEY_PATH="{svc_dir / "key_path.pem"}"',
    ]
    pem = svc_dir / "key_path.pem"
    assert pem.is_file() and not pem.is_symlink(), "the pem must replace the link"
    assert pem.read_bytes() == KEY_BYTES
    assert stat.S_IMODE(pem.stat().st_mode) == 0o600
    assert list(run_dir.glob("**/.vault-env.*")) == [], "no temp dir may survive the run"


# --- F2-r1: pinned helper PATH + installer freeze guard ---------------------

PATH_PIN = "PATH=/usr/bin:/bin; export PATH"
RENDER_FN = "install_vault_env_helper"
INSTALL_FN = "install_vault_env"


def _render_function_text() -> str:
    """Extract only the installer's helper-rendering function, verbatim."""
    text = _installer_text()
    start = text.find(f"{RENDER_FN}() {{")
    assert start != -1, f"{RENDER_FN}() absent from the installer (red)"
    end = text.find(f"\n{INSTALL_FN}()", start)
    assert end != -1, f"{INSTALL_FN}() must follow {RENDER_FN}() (red)"
    return text[start:end]


def _run_render(func_text: str, dest: Path, tag: str):
    """Invoke the extracted render function with /bin/bash, destination $1."""
    script = dest.parent / f"render-{tag}.sh"
    script.write_text(f"{func_text}\n{RENDER_FN} \"$1\"\n", encoding="utf-8")
    return subprocess.run(["/bin/bash", str(script), str(dest)],
                          capture_output=True, text=True)


def test_installer_refuses_helper_without_path_pin(tmp_path):
    """F3 — the render function refuses to create its destination when the
    captured helper body does not open with the shebang plus the exact PATH
    pin and a newline; the unmodified function still writes the pinned
    helper whose first two lines are the shebang and that pin."""
    func_text = _render_function_text()

    # positive control: the unmodified render function writes a pinned helper
    dest_ok = tmp_path / "vault-env-pinned"
    proc = _run_render(func_text, dest_ok, "pinned")
    assert proc.returncode == 0, proc.stderr
    assert dest_ok.is_file(), "the valid helper must be written"
    assert dest_ok.read_text(encoding="utf-8").splitlines()[:2] == [
        "#!/bin/sh", PATH_PIN
    ], "the installed helper must open with the shebang and the exact PATH pin"

    # negative control: remove only the helper's PATH line from the function
    lines = func_text.splitlines(keepends=True)
    hit = next(i for i, ln in enumerate(lines) if ln.rstrip("\n") == PATH_PIN)
    reduced = "".join(lines[:hit] + lines[hit + 1:])
    dest_bad = tmp_path / "vault-env-unpinned"
    proc = _run_render(reduced, dest_bad, "unpinned")
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, "a helper body without the PATH pin must be refused (red)"
    assert "refusing" in combined and "PATH" in combined, combined
    assert not dest_bad.exists(), "the refused destination must not be created"


@pytest.mark.parametrize("util", ["chown", "mv"])
def test_path_pin_ignores_planted_utilities(tmp_path, util):
    """The helper's own PATH pin keeps a hostile prefix powerless: an
    executable fake <util> planted first on PATH (unique marker, exit 0)
    never runs while both the ordinary and the _B64 output paths complete —
    rc 0, no marker, no fake output, exact env/PEM contents and quoting,
    modes 0700/0600."""
    helper, run_dir, vault, names_dir, ord_value = _r3_fixture(tmp_path)
    bin_dir = tmp_path / "planted-bin"
    bin_dir.mkdir()
    marker = tmp_path / f"planted-{util}.marker"
    fake = _write(
        bin_dir / util,
        '#!/bin/sh\n: > "$PLANTED_MARKER"\n'
        f'printf "planted {util} ran\\n"\nexit 0\n',
    )
    fake.chmod(0o755)

    proc = _run_helper(
        helper, "svc", vault=vault, run_dir=run_dir, names_dir=names_dir,
        extra_env={"PLANTED_MARKER": str(marker)}, path_prefix=str(bin_dir),
    )
    assert proc.returncode == 0, proc.stderr
    assert not marker.exists(), (
        f"planted {util} executed — the helper PATH is not pinned (red)"
    )
    assert proc.stdout == "", f"planted {util} wrote to stdout: {proc.stdout!r}"
    assert proc.stderr == "", f"planted {util} wrote to stderr: {proc.stderr!r}"
    combined = proc.stdout + proc.stderr
    assert ord_value not in combined, "a vault value leaked into the output"

    svc_dir = run_dir / "svc"
    env_file = svc_dir / "env"
    pem = svc_dir / "key_path.pem"
    assert env_file.is_file(), f"{util} rename path must still install env"
    assert pem.is_file(), f"{util} rename path must still install the pem"
    assert stat.S_IMODE(svc_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(pem.stat().st_mode) == 0o600
    assert env_file.read_text(encoding="utf-8").splitlines() == [
        f'ORD_OUT="{ord_value}"',
        f'KEY_PATH="{pem}"',
    ], 'both outputs must keep exact NAME="<escaped>" quoting'
    assert pem.read_bytes() == KEY_BYTES
    assert list(run_dir.glob("**/.vault-env.*")) == [], "no temp dir may survive"
