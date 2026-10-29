"""W1-03-r1 — vault-backed service environments without unversioned drop-ins.

Six tests over the installer-generated ``vault-env`` helper, the two owned
units and the secret ignore rules.  Every fixture is fabricated inside the
test: no real PEM, no real DSN, no real vault file, no network, no write to a
production path, and no systemd is started — the helper is exercised as a
plain POSIX ``sh`` script in pytest's tmp_path.  No product code is imported.
"""

from __future__ import annotations

import base64
import re
import stat
import subprocess
from pathlib import Path

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
    return match.group(2)


def _helper_body() -> str:
    return _heredoc(_installer_text(), HELPER_MARKER)


def _install_helper(tmp_path: Path) -> Path:
    """Extract the quoted helper heredoc from the installer to a temp dir."""
    helper = tmp_path / "vault-env"
    helper.write_text(_helper_body(), encoding="utf-8")
    helper.chmod(0o755)
    return helper


def _run_helper(helper: Path, service: str, *, vault: Path, run_dir: Path, names_dir: Path):
    """Run the helper as plain POSIX sh with only the documented overrides."""
    env = {
        "VAULT_FILE": str(vault),
        "RUN_DIR": str(run_dir),
        "NAMES_DIR": str(names_dir),
        "PATH": "/usr/bin:/bin",
    }
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
            f"NEO_GITHUB_APP_KEY_PATH={expected_pem}",
            f"RHOBEAR_NEO_WEBHOOK_SECRET={values['RHOBEAR_ZEUS_WEBHOOK_SECRET']}",
            f"DATABASE_URL={values['REVIEWS_DATABASE_URL']}",
            f"DEEPSEEK_API_KEY={values['DEEPSEEK_API_KEY']}",
            f"DEEPINFRA_API_KEY={values['DEEPINFRA_API_KEY']}",
        ], "the five service-specific aliases must be emitted exactly"
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

        # atomic delivery: the env file is installed via a same-dir temp + rename
        body = _helper_body()
        assert ".tmp." in body, "helper must stage output in a same-directory temp file"
        assert re.search(r"\bmv\b", body), "helper must rename temp output into place"

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

        assert lines.count(f"EnvironmentFile=/run/{service}/env") == 1, (
            f"{unit_name}: /run EnvironmentFile wiring absent (red)"
        )
        expected_pre = f"ExecStartPre=+/opt/rhobear/bin/vault-env {service}"
        assert lines.count(expected_pre) == 1, f"{unit_name}: exact {expected_pre} once"
        assert not any(ln.startswith("EnvironmentFile=/opt") for ln in lines), (
            f"{unit_name}: unversioned /opt EnvironmentFile must be gone"
        )
        assert ".env" not in text, f"{unit_name}: a repository .env claim remains"

        runtime_dirs = [ln for ln in lines if ln.startswith("RuntimeDirectory=")]
        assert runtime_dirs == [f"RuntimeDirectory={service} {service}-gh"], (
            f"{unit_name}: runtime directory wiring"
        )
        assert lines.count("RuntimeDirectoryMode=0700") == 1

        assert APP_ID_LINE in lines, f"{unit_name}: versioned App ID"
        assert "Environment=RHOBEAR_GH_APP_OWNER=deariencampbell1-sys" in lines
        assert f"Environment=RHOBEAR_GH_APP_CACHE=/run/{service}-gh" in lines
        path_lines = [ln for ln in lines if ln.startswith("Environment=PATH=")]
        assert len(path_lines) == 1, f"{unit_name}: exactly one PATH"
        path_value = path_lines[0].split("=", 2)[2]
        first = path_value.split(":")[0]
        assert first == "/opt/rhobear/gh-app", f"{unit_name}: gh-app wrapper must be first on PATH"

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
    assert not re.search(r"(^|[\s/=$}])\.env(\b|$)", text, re.M), (
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
