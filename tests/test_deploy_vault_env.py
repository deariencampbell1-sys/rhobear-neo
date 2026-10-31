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
    assert 'atomic_write "$dest" 0755 1' in text, (
        "the helper must be installed 0755 through the atomic writer"
    )

    names_body = _heredoc(text, NAMES_MARKER)
    assert names_body.splitlines() == NAMES_FILE_LINES, (
        "the versioned name file content must match the mandate exactly"
    )
    for service in SERVICES:
        assert f"{service}" in text, f"name file for {service} must be installed"
    assert "vault-env.d" in text, "name files install under /etc/rhobear/vault-env.d"
    assert 'atomic_write "/etc/rhobear/vault-env.d/${svc}.names" 0644 0' in text, (
        "name files must be installed 0644 through the atomic writer"
    )

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
    """r2 — restoration of the base chown-target proof, without PATH
    interception: the extracted helper text has only the token
    ``/usr/bin/chown`` substituted with a logging fake (which appends its
    arguments to a log, then execs the real chown), so the log shows exactly
    what the helper chowns and in what order. The base properties must hold:
    the helper exits 0, the log holds exactly 2 calls, every call's first
    argument is ``--reference=<svc_dir>``, and every target is a temp path
    under ``RUN_DIR/.vault-env.`` — never the final ``$svc_dir/env`` or
    ``$svc_dir/key_path.pem``. The substitution is on this fixture only; the
    shipped helper keeps ``/usr/bin/chown``."""
    helper, run_dir, vault, names_dir, ord_value = _r3_fixture(tmp_path)
    svc_dir = run_dir / "svc"
    body = helper.read_text(encoding="utf-8")
    assert "/usr/bin/chown" in body, (
        "the helper must chown at its absolute path for token substitution"
    )
    log = tmp_path / "chown-calls.log"
    fake_chown = tmp_path / "logging-chown"
    fake_chown.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\t\' "$@" >> "$CHOWN_LOG"\n'
        'printf \'\\n\' >> "$CHOWN_LOG"\n'
        'exec /usr/bin/chown "$@"\n',
        encoding="utf-8",
    )
    fake_chown.chmod(0o755)
    helper.write_text(body.replace("/usr/bin/chown", str(fake_chown)), encoding="utf-8")

    proc = _run_helper(helper, "svc", vault=vault, run_dir=run_dir,
                       names_dir=names_dir, extra_env={"CHOWN_LOG": str(log)})
    assert proc.returncode == 0, proc.stderr

    calls = [ln.rstrip("\t").split("\t")
             for ln in log.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 2, f"expected exactly two chown calls, got {calls}"
    for args in calls:
        assert len(args) == 2, f"unexpected chown invocation: {args}"
        assert args[0] == f"--reference={svc_dir}", (
            f"every chown must reference the service directory: {args}"
        )
        target = args[1]
        assert target.startswith(f"{run_dir}/.vault-env."), (
            f"chown target outside the helper's staging dir: {target}"
        )
        assert target not in (str(svc_dir / "env"), str(svc_dir / "key_path.pem")), (
            f"FINAL PATH used as a chown target: {target}"
        )

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
WRITER_FN = "atomic_write"
CLEANUP_FN = "cleanup_tmp"


def _function_text(name: str) -> str:
    """One top-level ``name() { ... }`` definition from the installer,
    verbatim.  Quoted heredoc bodies are skipped: the helper body contains
    column-0 ``}`` lines that must not end the enclosing function early."""
    lines = _installer_text().splitlines(keepends=True)
    start = next(
        (i for i, ln in enumerate(lines) if ln.startswith(f"{name}() {{")), None
    )
    assert start is not None, f"{name}() absent from the installer (red)"
    depth, out, heredoc = 0, [], None
    for ln in lines[start:]:
        out.append(ln)
        if heredoc is not None:
            if ln.rstrip("\n") == heredoc:
                heredoc = None
            continue
        match = re.search(r"<<'([^']+)'", ln)
        if match:
            heredoc = match.group(1)
            continue
        depth += ln.count("{") - ln.count("}")
        if depth == 0:
            return "".join(out)
    raise AssertionError(f"unterminated {name}() definition (red)")


def _render_function_text() -> str:
    """Extract only the installer's atomic writer and helper-rendering
    function definitions, verbatim (no top-level installer code)."""
    return _function_text(WRITER_FN) + "\n" + _function_text(RENDER_FN)


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

    # r2: a second PATH= assignment anywhere in the body is refused, and no
    # destination is written — the guard may not stop at the first two lines
    doubled = func_text.replace(
        "set -eu; set +x; umask 077\n",
        "set -eu; set +x; umask 077\nPATH=/usr/local/bin:/usr/bin:/bin\n",
        1,
    )
    assert doubled != func_text, "second-PATH mutation anchor missing (red)"
    dest_two = tmp_path / "vault-env-second-path"
    proc = _run_render(doubled, dest_two, "second-path")
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, "a second PATH= assignment in the body must be refused"
    assert "refusing" in combined and "PATH" in combined, combined
    assert not dest_two.exists(), "the refused destination must not be created"

    # r2: a destination that already exists as a symlink is refused the same
    # way, and the symlink's target is left untouched
    link_target = tmp_path / "symlink-target"
    link_target.write_text("SENTINEL-UNTOUCHED\n", encoding="utf-8")
    dest_link = tmp_path / "vault-env-symlinked"
    dest_link.symlink_to(link_target)
    proc = _run_render(func_text, dest_link, "symlinked")
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, "a symlink destination must be refused"
    assert "refusing" in combined and "symlink" in combined, combined
    assert dest_link.is_symlink(), "the refused symlink must be left in place"
    assert link_target.read_text(encoding="utf-8") == "SENTINEL-UNTOUCHED\n", (
        "the symlink target must be left untouched"
    )


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


# --- F2-r2: every external command absolute, proved statically ---------------

# F21: POSIX special built-ins (SUSv4 XCU 2.14) — commands sh itself
# implements, never external programs — plus the ordinary built-ins the
# helper may use.  This list catches external commands; it never dictates
# which shell built-ins the helper may use.
HELPER_ALLOWED_BUILTINS = {
    # POSIX special built-ins.
    "break", ":", "continue", "exit", "export", "readonly", "return",
    "set", "shift", "times", "trap", "unset",
    # Ordinary built-ins the helper is entitled to use.
    "cd", "echo", "false", "getopts", "hash", "pwd", "read",
    "test", "[", "type", "true", "umask", "wait", "printf", "local",
}
# Deliberately NOT allowed: `command`/`exec` run the word that follows them, and
# `.`/`eval` execute arbitrary text — each would let a planted utility or a
# vault-derived record through the scanner this file exists to be. The helper
# uses none of them.

# POSIX reserved words: grammar, never external commands, whatever they spell.
HELPER_SHELL_KEYWORDS = {
    "if", "then", "elif", "else", "fi", "do", "done", "case", "esac", "for",
    "while", "until", "in", "function", "select", "time", "!", "{", "}",
}
_HELPER_OPS = ";|&<>()"
_HELPER_REDIR = ("<", ">", ">>", "<<", ">&", "<&")
_HELPER_ASSIGN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


def _helper_dollar_paren(text: str, i: int) -> int:
    """Index of the ')' matching the '$(' at text[i:i+2], quote-aware."""
    j, n, depth = i + 2, len(text), 1
    while j < n and depth:
        c = text[j]
        if c == "'":
            k = text.find("'", j + 1)
            j = n if k == -1 else k + 1
        elif c == '"':
            j += 1
            while j < n and text[j] != '"':
                if text[j] == "\\":
                    j += 2
                elif text.startswith("$(", j):
                    j = _helper_dollar_paren(text, j) + 1
                else:
                    j += 1
            j += 1
        elif text.startswith("$(", j):
            j = _helper_dollar_paren(text, j) + 1
        elif c == "(":
            depth, j = depth + 1, j + 1
        elif c == ")":
            depth, j = depth - 1, j + 1
        else:
            j += 1
    return j - 1


def _helper_command_words(body: str) -> list[tuple[int, str]]:
    """Every word in command position in a POSIX sh stream, as (lineno, word).

    Sized to this helper's constructs: quotes, $(...) and $((...)),
    redirections (fd numbers, >&2), function definitions, case patterns, and
    for/while loops. Words inside $(...) are scanned as command streams of
    their own; case patterns and redirection targets are never commands."""
    words: list[tuple[int, str]] = []

    def scan(chunk: str, base: int) -> None:
        toks: list[tuple[str, str, int]] = []
        line, i, n = base, 0, len(chunk)
        while i < n:
            c = chunk[i]
            if c == "\n":
                toks.append(("op", "\n", line))
                line += 1
                i += 1
            elif c in " \t":
                i += 1
            elif c == "#":
                nl = chunk.find("\n", i)
                i = n if nl == -1 else nl
            elif c in _HELPER_OPS:
                two = chunk[i:i + 2]
                if two in (";;", "&&", "||", ">>", "<<", ">&", "<&"):
                    toks.append(("op", two, line))
                    i += 2
                else:
                    toks.append(("op", c, line))
                    i += 1
            else:
                wline, parts = line, []
                while i < n and chunk[i] not in _HELPER_OPS and chunk[i] not in " \t\n":
                    c = chunk[i]
                    if c == "'":
                        j = chunk.find("'", i + 1)
                        j = n if j == -1 else j
                        parts.append(chunk[i:j + 1])
                        i = j + 1
                    elif c == '"':
                        parts.append('"')
                        i += 1
                        while i < n and chunk[i] != '"':
                            if chunk[i] == "\\":
                                parts.append(chunk[i:i + 2])
                                i += 2
                            elif chunk.startswith("$(", i):
                                k = _helper_dollar_paren(chunk, i)
                                scan(chunk[i + 2:k], line)
                                parts.append("X")
                                i = k + 1
                            else:
                                if chunk[i] == "\n":
                                    line += 1
                                parts.append(chunk[i])
                                i += 1
                        parts.append('"')
                        i += 1
                    elif chunk.startswith("$(", i):
                        if chunk[i + 2:i + 3] == "(":
                            j, depth = i + 3, 1
                            while j < n and depth:
                                if chunk[j] == "(":
                                    depth += 1
                                elif chunk[j] == ")":
                                    depth -= 1
                                j += 1
                            k = chunk.find(")", j)
                            k = n if k == -1 else k
                            parts.append("X")
                            i = k + 1
                        else:
                            k = _helper_dollar_paren(chunk, i)
                            scan(chunk[i + 2:k], line)
                            parts.append("X")
                            i = k + 1
                    elif c == "\\":
                        parts.append(chunk[i:i + 2])
                        i += 2
                    else:
                        parts.append(chunk[i])
                        i += 1
                toks.append(("w", "".join(parts), wline))

        state, seen, case_depth, idx = "cmd", False, 0, 0
        while idx < len(toks):
            kind, val, ln = toks[idx]
            if kind == "op":
                if state == "casepat":
                    if val == ")":
                        state, seen = "cmd", False
                elif state == "forlist":
                    pass  # stay until the word `do`
                elif state == "target":
                    if val == "&":
                        pass  # >&2: the fd target follows
                    elif val in _HELPER_REDIR:
                        state = "target"
                    else:
                        state, seen = "cmd", False
                elif val in _HELPER_REDIR:
                    state = "target"
                elif val == ";;":
                    state = "casepat" if case_depth else "cmd"
                    seen = False
                else:
                    state, seen = "cmd", False
            elif state == "target":
                state = "arg" if seen else "cmd"
            elif state == "forlist":
                if val == "do":
                    state = "cmd"
            elif state == "casehead":
                if val == "in":
                    state, case_depth = "casepat", case_depth + 1
            elif state == "casepat":
                if val == "esac":
                    case_depth, state = max(0, case_depth - 1), "cmd"
            elif (state == "cmd" and val.isdigit() and idx + 1 < len(toks)
                  and toks[idx + 1][0] == "op" and toks[idx + 1][1] in _HELPER_REDIR):
                pass  # an fd prefix of a redirection, not a command
            elif state == "cmd":
                if (idx + 2 < len(toks) and toks[idx + 1][0] == "op"
                        and toks[idx + 1][1] == "(" and toks[idx + 2][0] == "op"
                        and toks[idx + 2][1] == ")"):
                    idx += 3  # function definition head; the body follows in cmd state
                    continue
                if val in HELPER_SHELL_KEYWORDS:
                    if val == "case":
                        state = "casehead"
                    elif val == "for":
                        state = "forlist"
                elif _HELPER_ASSIGN.match(val):
                    pass  # an assignment keeps command position
                else:
                    words.append((ln, val))
                    seen, state = True, "arg"
            idx += 1

    scan(body, 1)
    return words


def _helper_offenders(body: str) -> list[tuple[int, str, str]]:
    """The scanner's whole verdict for one sh body: every command-position
    word that is not an allowlisted builtin, a shell keyword, a function
    defined in the body, or an absolute /usr/bin|/bin path, as
    (lineno, word, source line)."""
    defined = set(re.findall(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)", body))
    lines = body.splitlines()
    return [
        (lineno, word, lines[lineno - 1].strip())
        for lineno, word in _helper_command_words(body)
        if (word not in HELPER_ALLOWED_BUILTINS and word not in HELPER_SHELL_KEYWORDS
            and word not in defined
            and not word.startswith("/usr/bin/") and not word.startswith("/bin/"))
    ]


def test_helper_commands_are_builtins_or_absolute():
    """r2 — every command word in the shipped helper is a shell keyword, an
    allowlisted builtin, a function defined in the body itself, or an
    absolute path under /usr/bin/ or /bin/. A bare chown, mv, mktemp,
    base64, sed, tr, rm, mkdir, chmod, dirname, stat — any bare word in
    command position — fails here with the offending line. Static, so it
    holds no matter what PATH the helper is run with. F21: the allowlist
    exists to catch external commands, never to dictate which shell built-ins
    the helper may use; if a built-in is missing, the list is wrong, not the
    code."""
    offenders = _helper_offenders(_helper_body())
    assert not offenders, (
        "bare command words in the helper body — every command must be a "
        "shell keyword, an allowlisted builtin, a function defined in the "
        f"body, or an absolute /usr/bin|/bin path: {offenders}"
    )


# --- F21: the allowlist is complete; the scanner never shapes the helper -----

F21_BUILTINS = (
    "export", "continue", "break", "shift", "cd", "unset", "wait",
    "readonly", "echo",
)


@pytest.mark.parametrize("utility", ["chown", "mv", "install"])
def test_helper_scanner_rejects_planted_bare_utilities(utility):
    """F21 — a complete allowlist is not a permissive one: a planted bare
    utility in command position is still a finding."""
    body = f"#!/bin/sh\n{utility} -f /run/svc/env\nprintf '%s\\n' ok\n"
    offenders = _helper_offenders(body)
    assert [word for _ln, word, _src in offenders] == [utility], offenders


def test_helper_scanner_ignores_commented_bare_utilities():
    """F21 — comments are stripped before scanning, so a commented-out bare
    utility is not a finding."""
    body = "#!/bin/sh\n# chown -R root:root /run/svc\nprintf '%s\\n' ok\n"
    assert _helper_offenders(body) == []


def test_helper_scanner_accepts_posix_builtins():
    """F21 — every POSIX special builtin and the ordinary builtins the
    helper may use are accepted by the scanner; rejecting them would be the
    scanner dictating the helper's code."""
    body = "#!/bin/sh\n" + "\n".join(F21_BUILTINS) + "\n"
    assert _helper_offenders(body) == []


def test_helper_scanner_builtin_and_keyword_sets_stay_disjoint():
    """F21 — grammar keywords live only in HELPER_SHELL_KEYWORDS: a word
    that is both a keyword and a builtin (`return`, `break`) is accepted
    once and never shadows the keyword set."""
    overlap = HELPER_ALLOWED_BUILTINS & HELPER_SHELL_KEYWORDS
    assert not overlap, (
        f"keywords duplicated into the builtin allowlist: {sorted(overlap)}"
    )
    for word in ("return", "break"):
        assert _helper_offenders(f"#!/bin/sh\n{word}\n") == []


# --- F21-r1-r2: execution builtins are out; the guard is the one under test --

EXECUTION_BUILTINS = ("command", "exec", ".", "eval")


@pytest.mark.parametrize("word, fixture", [
    ("command", "command chown 0:0 /run/svc/env"),
    ("exec", "exec mv /run/svc/env /run/svc/old"),
    (".", '. "$vault_record"'),
    ("eval", 'eval "$vault_record"'),
])
def test_helper_scanner_rejects_execution_builtins(word, fixture):
    """F21-r1-r2 — `command`/`exec` run the word that follows them and
    `.`/`eval` execute arbitrary text, so none may be allowlisted: each
    fixture line is a command-position finding. On the F21-r1 base all four
    are silently accepted (the red)."""
    offenders = _helper_offenders(f"#!/bin/sh\n{fixture}\n")
    assert [w for _ln, w, _src in offenders] == [word], offenders


def test_helper_body_uses_no_execution_builtins():
    """F21-r1-r2 — the shipped helper parses NAME=value records and never
    sources, evals or delegates to a planted utility. Pin the header's
    promise: no command/exec/./eval in command position, no leading `. `
    source call, and no `eval` word anywhere."""
    body = _helper_body()
    words = [w for _ln, w in _helper_command_words(body)]
    used = [w for w in words if w in EXECUTION_BUILTINS]
    assert not used, f"execution builtin in command position: {used}"
    assert not re.search(r"(?m)^\s*\.\s", body), "the helper must not source a file"
    assert not re.search(r"\beval\b", body), "the helper must not eval text"


def test_helper_scanner_admits_only_non_executing_introspection_builtins(tmp_path):
    """F21-r1-r2 — hash/type/getopts/times are admitted (the helper is
    entitled to them) but, unlike command/exec, none runs the word that
    follows: a planted `chown` named as their argument never executes, and
    each line still passes the scanner as a builtin in command position."""
    planted = tmp_path / "planted-bin"
    planted.mkdir()
    marker = tmp_path / "planted.marker"
    fake = planted / "chown"
    fake.write_text('#!/bin/sh\n: > "$F21_MARKER"\n', encoding="utf-8")
    fake.chmod(0o755)
    fixtures = {
        "hash": "hash chown",
        "type": "type chown",
        "getopts": 'getopts "c" chown',
        "times": "times",
    }
    for word, line in fixtures.items():
        proc = subprocess.run(
            ["/bin/sh", "-c", f"{line}\nexit 0"],
            env={"PATH": f"{planted}:/usr/bin:/bin", "F21_MARKER": str(marker)},
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (word, proc.stderr)
        assert not marker.exists(), f"{word} executed the word that follows it"
        assert _helper_offenders(f"#!/bin/sh\n{line}\n") == [], (
            f"{word} must stay accepted in command position"
        )


def test_helper_scanner_ignores_execution_words_as_data():
    """F21-r1-r2 edge — `command` in a comment, or as an assignment name, is
    data, not command position, and is not a finding."""
    body = (
        "#!/bin/sh\n"
        "# command chown -R root:root /run/svc/env\n"
        "command=chown\n"
        'printf "%s\\n" "$command"\n'
    )
    assert _helper_offenders(body) == []


def test_helper_scanner_reports_builtin_command_form():
    """F21-r1-r2 edge — a bash-only `builtin command` line is never silently
    allowed just because the word `command` appears in the file: `builtin`
    itself is not allowlisted, so the scanner still reports a finding."""
    offenders = _helper_offenders("#!/bin/sh\nbuiltin command chown 0:0 /x\n")
    assert offenders, "builtin command must not pass the scanner silently"


def test_helper_pin_line_exports_path(tmp_path):
    """F21-r1-r2 — the pin is the second line of the helper the installer
    actually renders, exactly as its prefix guard requires, and it is the
    only PATH= line in that helper."""
    func_text = _render_function_text()
    dest = tmp_path / "vault-env-rendered"
    proc = _run_render(func_text, dest, "rendered")
    assert proc.returncode == 0, proc.stderr
    rendered = dest.read_text(encoding="utf-8")
    lines = rendered.splitlines()
    assert lines[0] == "#!/bin/sh", lines[:2]
    assert lines[1] == PATH_PIN, (
        f"the pin must be the second line, got {lines[1]!r}"
    )
    pin_lines = [ln for ln in lines if ln.startswith("PATH=")]
    assert len(pin_lines) == 1, f"exactly one PATH= assignment, got {pin_lines}"


def test_installer_rejects_space_before_semicolon_in_pin(tmp_path):
    """F21-r1-r2 — the render guard compares the pin literally, so
    `PATH=/usr/bin:/bin ; export PATH` is refused. F21-r1 tested a regex the
    installer never used; this drives the real guard instead."""
    func_text = _render_function_text()
    spaced = func_text.replace(PATH_PIN, "PATH=/usr/bin:/bin ; export PATH", 1)
    assert spaced != func_text, "space-before-semicolon mutation anchor missing (red)"
    dest = tmp_path / "vault-env-spaced-pin"
    proc = _run_render(spaced, dest, "spaced-pin")
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, "the real guard accepts only the exact pin"
    assert "refusing" in combined and "PATH" in combined, combined
    assert not dest.exists(), "the refused destination must not be created"


# --- F22-r1: the installer's own commands + symlink-destination TOCTOU -------

INSTALLER_PATH_PIN = "PATH=/usr/bin:/bin; export PATH"
INSTALLER_PY = "/usr/bin/python3"
INSTALLER_VENV = "/opt/rhobear/rhobear-neo/.venv"


def _strip_quoted_heredoc(text: str, marker: str) -> str:
    """Remove one `<<'MARKER'` here-doc construct (token, body, delimiter)."""
    return re.sub(
        rf"<<'{re.escape(marker)}'\n.*?^{re.escape(marker)}[ \t]*$",
        "", text, flags=re.S | re.M,
    )


def _installer_scan_text() -> str:
    """The installer with its here-doc bodies removed (the helper and names
    bodies have their own tests) and the two deliberate command variables
    resolved to the paths the script itself pins."""
    text = _installer_text()
    for marker in (HELPER_MARKER, NAMES_MARKER):
        text = _strip_quoted_heredoc(text, marker)
    return (
        text.replace('"$PY"', INSTALLER_PY)
            .replace('"$VENV_DIR/bin/pip"', f"{INSTALLER_VENV}/bin/pip")
    )


def _installer_offenders(text: str) -> list[tuple[int, str, str]]:
    """F22-r1: the helper scanner's verdict for the installer — every
    command-position word must be a builtin/keyword, a function defined in
    the file, or an absolute path (the installer's own venv paths included)."""
    defined = set(re.findall(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)", text))
    lines = text.splitlines()
    return [
        (lineno, word, lines[lineno - 1].strip())
        for lineno, word in _helper_command_words(text)
        if (word not in HELPER_ALLOWED_BUILTINS and word not in HELPER_SHELL_KEYWORDS
            and word not in defined and not word.startswith("/"))
    ]


def test_installer_commands_are_builtins_or_absolute():
    """F22-r1 — every command word in deploy/install.sh is a shell
    builtin/keyword, a function defined in the file, or an absolute path
    under /usr/bin/ or /bin/. A bare python3, install, chmod, cat or
    systemctl fails here with the offending line."""
    text = _installer_text()
    offenders = _installer_offenders(_installer_scan_text())
    assert not offenders, (
        "bare command words in install.sh — every command must be a shell "
        f"builtin/keyword, a function, or an absolute path: {offenders}"
    )
    assert '"$PY"' in text, "the interpreter must be the resolved PY variable"
    assert '"$VENV_DIR/bin/pip"' in text, "pip must go through VENV_DIR"


@pytest.mark.parametrize("utility", ["python3", "install", "chmod", "cat", "systemctl"])
def test_installer_scanner_rejects_planted_bare_utilities(utility):
    """F22-r1 — a complete allowlist is not a permissive one: a planted bare
    utility in command position is still a finding for the installer scan."""
    found = [
        w for _ln, w, _src in
        _installer_offenders(f"#!/bin/bash\nset -euo pipefail\n{utility} x\n")
    ]
    assert found == [utility], found


def test_installer_resolves_interpreter_and_pins_path():
    """F22-r1 — the interpreter is resolved once, deliberately, the venv/pip
    calls go through PY/VENV_DIR, and the PATH pin precedes the first command
    that used to be resolved from PATH."""
    lines = [ln.strip() for ln in _installer_text().splitlines()]
    assert f"PY={INSTALLER_PY}" in lines, lines[:16]
    assert '[ -x "$PY" ] || fail "python3 not found at $PY"' in lines
    assert f"VENV_DIR={INSTALLER_VENV}" in lines
    assert '"$PY" -m venv "$VENV_DIR"' in lines
    assert '"$VENV_DIR/bin/pip" install -q --upgrade pip' in lines
    assert '"$VENV_DIR/bin/pip" install -q -r requirements.txt' in lines
    assert INSTALLER_PATH_PIN in lines, "the installer PATH pin is missing"
    pin_line = lines.index(INSTALLER_PATH_PIN)
    assert pin_line < lines.index('"$PY" -m venv "$VENV_DIR"'), (
        "the PATH pin must precede the first resolved command"
    )


def _installer_sandbox(tmp_path: Path, *, dry_run: bool, text: str | None = None,
                       dry_run_env: bool = False):
    """A runnable copy of the installer in a temp sandbox, `cd` neutered so
    the fixed /opt path is irrelevant, fakes for the commands the installer
    might resolve from PATH, and one shared marker."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "planted.marker"
    for name in ("python3", "cat", "install", "chmod", "mv", "mkdir",
                 "systemctl", "sleep", "ss"):
        fake = _write(fake_bin / name, f'#!/bin/sh\n: > "{marker}"\nexit 0\n')
        fake.chmod(0o755)
    script = _write(
        sandbox / "install-run.sh",
        "#!/bin/bash\ncd() { :; }\n" + (text if text is not None else _installer_text()),
    )
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    env["PLANTED_MARKER"] = str(marker)
    if dry_run_env:
        env["INSTALL_DRY_RUN"] = "1"
    argv = ["/bin/bash", str(script)] + (["--dry-run"] if dry_run else [])
    proc = subprocess.run(
        argv, cwd=sandbox, env=env,
        capture_output=True, text=True,
    )
    return proc, marker, sandbox


def test_installer_dry_run_ignores_planted_utilities(tmp_path):
    """F22-r1-r2 — the resolution/pre-flight section runs to completion (rc 0,
    via the --dry-run first argument) with a fake python3/cat/install first on
    PATH and never runs any of them: the PATH pin and the absolute paths make
    the planted commands inert."""
    proc, marker, _ = _installer_sandbox(tmp_path, dry_run=True)
    assert not marker.exists(), (
        "a planted PATH command executed during the installer pre-flight (red): "
        + proc.stdout + proc.stderr
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    assert proc.stdout == "", proc.stdout
    assert proc.stderr == "install.sh: dry run: no changes made\n", proc.stderr


def test_installer_dry_run_is_argument_not_environment_variable(tmp_path):
    """F22-r1-r2 — INSTALL_DRY_RUN in the environment is inert: an
    unprivileged run reaches the root check and fails nonzero instead of
    exiting 0 as a silent no-op, so a profile or CI variable can never turn a
    real install into a green nothing."""
    if os.geteuid() == 0:
        pytest.skip("non-root refusal is only observable unprivileged")
    proc, marker, _ = _installer_sandbox(tmp_path, dry_run=False, dry_run_env=True)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        "INSTALL_DRY_RUN must not turn an install into a green no-op"
    )
    assert "root" in combined.lower(), combined
    assert "no changes made" not in combined, combined
    assert not marker.exists(), combined
    assert "INSTALL_DRY_RUN" not in _installer_text(), (
        "the environment-variable dry-run escape hatch must be gone"
    )


def test_installer_refuses_non_root_before_any_command(tmp_path):
    """F22-r1 — without the dry-run seam, an unprivileged run fails closed at
    the root check before resolving or executing any command: the planted
    marker stays absent and no state is created."""
    if os.geteuid() == 0:
        pytest.skip("non-root refusal is only observable unprivileged")
    proc, marker, sandbox = _installer_sandbox(tmp_path, dry_run=False)
    combined = proc.stdout + proc.stderr
    assert not marker.exists(), f"a command ran before the root check: {combined}"
    assert proc.returncode != 0, "a non-root run must fail closed"
    assert "root" in combined.lower(), combined
    assert [p.name for p in sandbox.iterdir()] == ["install-run.sh"], "partial state"


def test_installer_missing_python_fails_named_without_state(tmp_path):
    """F22-r1 edge — python3 missing at /usr/bin/python3 fails with the path
    named and before any write; the anchor mutation stands in for a host
    without the interpreter."""
    missing = tmp_path / "no-such-python3"
    text = _installer_text().replace(f"PY={INSTALLER_PY}", f"PY={missing}", 1)
    assert text != _installer_text(), "the PY anchor is missing (red)"
    proc, marker, sandbox = _installer_sandbox(tmp_path, dry_run=True, text=text)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert str(missing) in combined, combined
    assert not marker.exists(), combined
    assert [p.name for p in sandbox.iterdir()] == ["install-run.sh"], "partial state"


def test_installer_helper_write_is_atomic_not_check_then_write():
    """F22-r1 (Apollo LOW neo#17) — the render function no longer writes
    through a checked-then-used $dest: it writes a temp name in the
    destination directory and mv -T -f's it into place."""
    func = _render_function_text()
    assert '> "$dest"' not in func, "check-then-write through $dest is back (red)"
    assert "/usr/bin/mktemp" in func, "the write must go through a temp name"
    assert "/usr/bin/mv -T -f" in func, "the temp name must be moved into place"


def test_installer_shebang_is_absolute():
    """F22-r1-r2 (Apollo HIGH) — the interpreter is named absolutely, so the
    kernel never looks `bash` up in the caller's PATH: /usr/bin/env resolves
    its argument through PATH, which made the F22-r1 shebang itself the one
    hijackable command."""
    text = _installer_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash", f"absolute shebang required, got {lines[0]!r}"
    assert not lines[0].startswith("#!/usr/bin/env"), (
        "/usr/bin/env resolves its argument through PATH; only an absolute "
        "interpreter removes that"
    )


def test_installer_shebang_ignores_planted_bash(tmp_path):
    """F22-r1-r2 — a hostile `bash` planted first on PATH is never the
    interpreter: the script is executed through its absolute shebang, runs to
    its --dry-run exit with rc 0, and the planted marker stays absent. On the
    F22-r1 base (#!/usr/bin/env bash) env resolves `bash` through PATH and the
    fake runs (red)."""
    sandbox = tmp_path / "shebang-sandbox"
    sandbox.mkdir()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "planted-bash.marker"
    fake = _write(fake_bin / "bash", f'#!/bin/sh\n: > "{marker}"\nexit 0\n')
    fake.chmod(0o755)
    script = _write(sandbox / "install.sh", _installer_text())
    script.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    proc = subprocess.run(
        [str(script), "--dry-run"], cwd=sandbox, env=env,
        capture_output=True, text=True,
    )
    combined = proc.stdout + proc.stderr
    assert not marker.exists(), (
        f"a planted bash executed as the interpreter (red): {combined}"
    )
    assert proc.returncode == 0, (proc.returncode, combined)
    assert "install.sh: dry run: no changes made" in proc.stderr, combined


def test_installer_names_write_is_atomic_not_check_then_write():
    """F22-r1-r2 — the .names files go through the same mktemp -> mv -T -f
    writer as the helper; one writer, both call sites, and no bare
    check-then-write through the destination."""
    text = _installer_text()
    assert '> "/etc/rhobear/vault-env.d/${svc}.names"' not in text, (
        "a bare check-then-write through the names destination is back (red)"
    )
    assert 'atomic_write "/etc/rhobear/vault-env.d/${svc}.names" 0644 0' in text
    assert text.count("atomic_write \"") == 2, "one writer, both call sites"


def test_installer_atomic_write_replaces_symlink_without_following(tmp_path):
    """F22-r1-r2 edge — atomic_write mv -T -f's onto the destination, so a
    pre-existing symlink is replaced by a regular file and its target is never
    written through."""
    writer = _function_text(WRITER_FN)
    sentinel = _write(tmp_path / "sentinel", "SENTINEL-UNTOUCHED\n")
    dest = tmp_path / "svc.names"
    dest.symlink_to(sentinel)
    script = _write(
        tmp_path / "writer-run.sh",
        f'TMP_FILES=""\n{writer}\natomic_write "{dest}" 0644 0 "NEW-NAMES"\n',
    )
    proc = subprocess.run(["/bin/bash", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert dest.is_file() and not dest.is_symlink(), "the symlink must be replaced"
    assert dest.read_text(encoding="utf-8") == "NEW-NAMES\n"
    assert stat.S_IMODE(dest.stat().st_mode) == 0o644
    assert sentinel.read_text(encoding="utf-8") == "SENTINEL-UNTOUCHED\n", (
        "the symlink target must be left untouched"
    )


def test_installer_cleanup_removes_registered_temp_files(tmp_path):
    """F22-r1-r2 — the installer traps EXIT INT TERM HUP; cleanup_tmp removes
    every registered .install-rhobear.* temp path and never touches an
    installed destination. The cleanup function is called directly: no signal
    is sent into a root-owned path."""
    text = _installer_text()
    assert "trap cleanup_tmp EXIT INT TERM HUP" in text, (
        "the installer must trap its temp-file cleanup"
    )
    writer = _function_text(WRITER_FN)
    cleanup = _function_text(CLEANUP_FN)
    temp = tmp_path / ".install-rhobear.ABC123"
    temp.write_text("partial-write", encoding="utf-8")
    dest = tmp_path / "installed.names"
    script = _write(
        tmp_path / "cleanup-run.sh",
        'TMP_FILES=""\n'
        f'{writer}\n{cleanup}\n'
        f'atomic_write "{dest}" 0644 0 "NEW-CONTENT"\n'
        f'TMP_FILES="$TMP_FILES {temp}"\n'
        "cleanup_tmp\n",
    )
    proc = subprocess.run(["/bin/bash", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert not temp.exists(), "the registered temp file must be removed"
    assert dest.read_text(encoding="utf-8") == "NEW-CONTENT\n", (
        "cleanup must never remove an installed destination"
    )
    assert list(tmp_path.glob(".install-rhobear.*")) == [], "no temp may survive"


def _run_install_vault_env(tmp_path: Path, *, bin_dir: Path, names_dir: Path):
    """Run the installer's atomic writer + helper render + names loop with the
    two fixed directories redirected into the sandbox."""
    text = _installer_text()
    assert f"{WRITER_FN}() {{" in text, f"{WRITER_FN}() absent from the installer (red)"
    functions = "\n".join(
        _function_text(name) for name in (WRITER_FN, RENDER_FN, INSTALL_FN)
    )
    functions = (
        functions
        .replace("/opt/rhobear/bin", str(bin_dir))
        .replace("/etc/rhobear/vault-env.d", str(names_dir))
    )
    script = _write(
        tmp_path / "run-vault-env.sh",
        'set -euo pipefail\nTMP_FILES=""\n' + functions + f"\n{INSTALL_FN}\n",
    )
    proc = subprocess.run(["/bin/bash", str(script)], capture_output=True, text=True)
    return proc, bin_dir / "vault-env", names_dir


def test_installer_two_names_writes_are_atomic_for_both_services(tmp_path):
    """F22-r1-r2 — both services in the loop get atomic 0644 names files via
    the shared writer, the helper is written by the same writer, and no temp
    file survives."""
    bin_dir = tmp_path / "bin"
    names_dir = tmp_path / "names"
    bin_dir.mkdir()
    names_dir.mkdir()
    proc, helper, names_dir = _run_install_vault_env(
        tmp_path, bin_dir=bin_dir, names_dir=names_dir
    )
    assert proc.returncode == 0, proc.stderr
    for svc in SERVICES:
        names = names_dir / f"{svc}.names"
        assert names.is_file() and not names.is_symlink()
        assert names.read_text(encoding="utf-8").splitlines() == NAMES_FILE_LINES
        assert stat.S_IMODE(names.stat().st_mode) == 0o644
    assert helper.read_text(encoding="utf-8").splitlines()[:2] == [
        "#!/bin/sh", PATH_PIN
    ]
    assert stat.S_IMODE(helper.stat().st_mode) == 0o755
    assert list(names_dir.glob(".install-rhobear.*")) == []
    assert list(bin_dir.glob(".install-rhobear.*")) == []


def test_installer_names_symlink_is_replaced_never_followed(tmp_path):
    """F22-r1-r2 edge — a pre-existing symlink at <svc>.names is replaced by
    the atomic move and its target is left untouched."""
    bin_dir = tmp_path / "bin"
    names_dir = tmp_path / "names"
    bin_dir.mkdir()
    names_dir.mkdir()
    sentinel = _write(tmp_path / "sentinel", "SENTINEL-UNTOUCHED\n")
    (names_dir / "rhobear-neo.names").symlink_to(sentinel)
    proc, _helper, names_dir = _run_install_vault_env(
        tmp_path, bin_dir=bin_dir, names_dir=names_dir
    )
    assert proc.returncode == 0, proc.stderr
    names = names_dir / "rhobear-neo.names"
    assert names.is_file() and not names.is_symlink()
    assert names.read_text(encoding="utf-8").splitlines() == NAMES_FILE_LINES
    assert sentinel.read_text(encoding="utf-8") == "SENTINEL-UNTOUCHED\n"


def test_installer_two_names_writes_second_failure_keeps_first(tmp_path):
    """F22-r1-r2 edge — when the second service's atomic write fails (its
    destination is a directory, so mv cannot replace it), the first service's
    file is already installed and intact, and the failed temp is removed."""
    bin_dir = tmp_path / "bin"
    names_dir = tmp_path / "names"
    bin_dir.mkdir()
    names_dir.mkdir()
    (names_dir / "rhobear-neo-builder.names").mkdir()
    proc, _helper, names_dir = _run_install_vault_env(
        tmp_path, bin_dir=bin_dir, names_dir=names_dir
    )
    assert proc.returncode != 0, proc.stdout
    assert "cannot install" in proc.stderr, proc.stderr
    first = names_dir / "rhobear-neo.names"
    assert first.read_text(encoding="utf-8").splitlines() == NAMES_FILE_LINES, (
        "a failure on the second write must leave the first intact"
    )
    assert list(names_dir.glob(".install-rhobear.*")) == [], (
        "the failed temp file must be removed"
    )


def test_installer_never_writes_through_a_swapped_symlink(tmp_path):
    """F22-r1 edge — a symlink that appears after the guard but before the
    move must not be written through: a fake /usr/bin/mv turns the
    destination into a symlink to a sentinel and then execs the real mv -T,
    so the move replaces the link and the sentinel bytes stay intact."""
    func = _render_function_text()
    sentinel = _write(tmp_path / "sentinel", "SENTINEL-INTACT\n")
    dest = tmp_path / "vault-env"
    fake_mv = _write(
        tmp_path / "mv-race",
        "#!/bin/bash\nset -eu\n"
        f'/usr/bin/ln -sfn "{sentinel}" "$2"\n'
        'exec "/usr/bin/mv" -T -f "$1" "$2"\n',
    )
    fake_mv.chmod(0o755)
    raced = func.replace('/usr/bin/mv -T -f "$tmp" "$dest"',
                        f'{fake_mv} "$tmp" "$dest"')
    assert raced != func, "the installer's atomic-move call is missing (red)"
    proc = _run_render(raced, dest, "race")
    assert proc.returncode == 0, proc.stderr
    assert sentinel.read_text(encoding="utf-8") == "SENTINEL-INTACT\n", (
        "the run wrote through the swapped symlink"
    )
    assert dest.is_file() and not dest.is_symlink()
    assert dest.read_text(encoding="utf-8").splitlines()[:2] == [
        "#!/bin/sh", PATH_PIN
    ]
