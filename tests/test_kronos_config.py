"""W1-08-r1 — Kronos configuration: vault names by name, loud when missing.

The unit's keys come from `vault-env rhobear-kronos`; the names list is the
brief's: both App keys and REVIEWS_DATABASE_URL. The two App private keys are
not in the vault yet (the manager files them before production moves) — so
the names are declared here and the missing-name path is LOUD, never silent.
"""
from __future__ import annotations

import pytest

from src.kronos.config import KronosConfig, REQUIRED_VAULT_NAMES


def test_required_vault_names_match_the_brief():
    assert set(REQUIRED_VAULT_NAMES) == {
        "RHOBEAR_REVIEWS_GH_APP_KEY_B64",
        "RHOBEAR_ZEUS_GH_APP_KEY_B64",
        "REVIEWS_DATABASE_URL",
    }


def test_defaults_match_the_brief_ports_and_units():
    cfg = KronosConfig()
    assert cfg.reviews_webhook_url == "http://127.0.0.1:8766"
    assert cfg.zeus_url == "http://127.0.0.1:8767"
    assert cfg.routegate_url == "http://127.0.0.1:8645"
    assert cfg.systemd_units == ("rhobear-reviews", "rhobear-reviews-dash",
                                 "rhobear-neo", "rhobear-neo-builder")
    assert cfg.pass_budget_secs == 240, "a 5-minute timer, a 4-minute budget"
    assert cfg.alert_repo == "deariencampbell1-sys/rhobear-reviews-app"
    assert cfg.alert_label == "kronos"


def test_missing_names_are_loud_with_their_vault_names(monkeypatch):
    for name in ("REVIEWS_DATABASE_URL", "RHOBEAR_KRONOS_ZEUS_APP_KEY",
                 "RHOBEAR_KRONOS_ZEUS_APP_KEY_PATH",
                 "RHOBEAR_KRONOS_ZEUS_APP_ID",
                 "RHOBEAR_KRONOS_REVIEWS_APP_ID",
                 "RHOBEAR_KRONOS_REVIEWS_APP_KEY",
                 "RHOBEAR_KRONOS_REVIEWS_APP_KEY_PATH"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        KronosConfig.required()
    msg = str(excinfo.value)
    # every missing vault NAME is named — never a value
    assert "RHOBEAR_ZEUS_GH_APP_KEY_B64" in msg
    assert "RHOBEAR_REVIEWS_GH_APP_KEY_B64" in msg
    assert "REVIEWS_DATABASE_URL" in msg


def test_key_is_read_from_path_or_text(monkeypatch, tmp_path):
    pem = tmp_path / "zeus.pem"
    pem.write_text("-----BEGIN FAKE TEST KEY-----\nnot-a-key\n", encoding="utf-8")
    monkeypatch.setenv("RHOBEAR_KRONOS_ZEUS_APP_KEY_PATH", str(pem))
    # the rest of the required set, as fake test values
    monkeypatch.setenv("REVIEWS_DATABASE_URL", "postgresql://fake/fake")
    monkeypatch.setenv("RHOBEAR_KRONOS_ZEUS_APP_ID", "1")
    monkeypatch.setenv("RHOBEAR_KRONOS_REVIEWS_APP_ID", "2")
    monkeypatch.setenv("RHOBEAR_KRONOS_REVIEWS_APP_KEY", "fake")
    cfg = KronosConfig.required()
    assert "FAKE TEST KEY" in cfg.zeus_app_key


def test_consumer_shape_matches_the_lane():
    cfg = KronosConfig()
    assert cfg.consumer_for("12345") == "reviews:kronos/12345"
    assert cfg.consumer_for("") == "reviews:kronos", \
        "never invent an install id"
    with pytest.raises(ValueError):
        cfg.consumer_for("bad id\nwith newline")


# ---------------------------------------------------------------------------
# HIGH 1 — the state lives somewhere durable, not in a RuntimeDirectory
# ---------------------------------------------------------------------------

def test_default_state_dir_is_durable_not_a_runtime_dir():
    """HIGH 1 — the oneshot's RuntimeDirectory is wiped when the service
    stops (and /run on a reboot): the alert dedupe, flap windows and judged
    markers would start empty EVERY five-minute pass. The default state dir
    must be the durable StateDirectory path."""
    import os

    def clean_env():
        return (os.environ.get("RHOBEAR_KRONOS_STATE_DIR") or "") == ""

    assert clean_env(), "the test needs RHOBEAR_KRONOS_STATE_DIR unset"
    cfg = KronosConfig()
    assert cfg.state_dir == "/var/lib/rhobear-kronos", cfg.state_dir


# ---------------------------------------------------------------------------
# HIGH 9 — the documented vault key PATH config is loaded from the file
# ---------------------------------------------------------------------------

def test_real_pass_requires_the_strict_config(monkeypatch, tmp_path):
    """HIGH 9 — a REAL pass must go through KronosConfig.required(): a
    missing key raises loudly naming the vault name, never a silent pass."""
    from src.kronos.__main__ import main
    for name in ("REVIEWS_DATABASE_URL", "RHOBEAR_KRONOS_ZEUS_APP_KEY",
                 "RHOBEAR_KRONOS_ZEUS_APP_KEY_PATH",
                 "RHOBEAR_KRONOS_ZEUS_APP_ID",
                 "RHOBEAR_KRONOS_REVIEWS_APP_ID",
                 "RHOBEAR_KRONOS_REVIEWS_APP_KEY",
                 "RHOBEAR_KRONOS_REVIEWS_APP_KEY_PATH"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        main(["--once", "--state-dir", str(tmp_path)],
              _services=FakeServicesLike())
    assert "RHOBEAR_ZEUS_GH_APP_KEY_B64" in str(excinfo.value)


def test_app_identity_gets_the_loaded_pem_never_the_path(monkeypatch,
                                                         tmp_path):
    """HIGH 9 — with ONLY a key PATH configured, every GitHubApp construction
    receives the PEM text loaded from that file, never the path string
    (GitHubApp signs with the string it is given; a path is not a key)."""
    import src.github_app as github_app
    from src.kronos import __main__ as km

    pem = tmp_path / "zeus.pem"
    pem.write_text("-----BEGIN FAKE TEST KEY-----\nnot-a-key\n",
                   encoding="utf-8")
    monkeypatch.setenv("RHOBEAR_KRONOS_ZEUS_APP_ID", "999")
    monkeypatch.setenv("RHOBEAR_KRONOS_ZEUS_APP_KEY_PATH", str(pem))
    monkeypatch.delenv("RHOBEAR_KRONOS_ZEUS_APP_KEY", raising=False)
    # the reviews App identity, as fake test values (required() wants both)
    monkeypatch.setenv("RHOBEAR_KRONOS_REVIEWS_APP_ID", "2")
    monkeypatch.setenv("RHOBEAR_KRONOS_REVIEWS_APP_KEY", "fake")
    monkeypatch.setenv("REVIEWS_DATABASE_URL", "postgresql://fake/fake")

    seen: list[tuple[str, str]] = []

    class RecordingApp:
        def __init__(self, app_id, key_pem, *a, **k):
            seen.append((str(app_id), str(key_pem)))

        def jwt(self):
            return "jwt"

        def token_for_repo(self, repo, **k):
            return "tok"

    monkeypatch.setattr(github_app, "GitHubApp", RecordingApp)
    cfg = KronosConfig.required()
    km._app_jwt(cfg, "zeus")
    km._readonly_token(cfg, "o/r", "42")
    alerter = km._alerter(cfg, _StateShim(), lambda: 0.0)
    # the alerter's token callable is lazy — invoke it so its GitHubApp is
    # constructed (a wiring defect there must not hide)
    alerter._client._token()
    km._pr_commenter(_SrvShim(), cfg)("o/r", 1, "body")
    assert seen, "no GitHubApp was constructed"
    for app_id, key in seen:
        assert "FAKE TEST KEY" in key, \
            f"a key PATH was passed instead of the PEM text: {key!r}"
        assert not key.endswith(".pem"), \
            f"GitHubApp got the PATH, not the PEM: {key!r}"


class FakeServicesLike:
    def runner(self, argv, *, timeout=30):  # pragma: no cover - never runs
        raise AssertionError


class _StateShim:
    def slot(self, name):
        return {}


class _SrvShim:
    def http(self, *a, **k):  # pragma: no cover - never runs
        return 201, "{}"
