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
