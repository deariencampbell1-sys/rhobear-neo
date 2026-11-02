"""Kronos configuration — env-driven, secrets by vault NAME only.

The unit delivers keys with `/opt/rhobear/bin/vault-env rhobear-kronos`
(W1-03's helper; if it is not installed yet the unit fails loudly — this
package never reads the vault itself and never ships its own key loader).
The names list is the brief's: both App private keys and the reviews
database URL. The two App keys are not in the vault yet (the manager files
them before production moves), so the names are declared and the
missing-name path raises LOUDLY naming the vault keys — never a silent skip.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# The vault names this service is provisioned with (M-11 files the App keys;
# the manager's names file maps the env vars below onto these names).
REQUIRED_VAULT_NAMES = (
    "RHOBEAR_REVIEWS_GH_APP_KEY_B64",
    "RHOBEAR_ZEUS_GH_APP_KEY_B64",
    "REVIEWS_DATABASE_URL",
)

# The consumer shape Kronos's model calls carry through RouteGate.
KRONOS_CONSUMER_LANE = "reviews:kronos"


def _get(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


@dataclass
class KronosConfig:
    # --- loopback services (all listening 2026-10-25) ---
    reviews_webhook_url: str = field(default_factory=lambda: _get(
        "KRONOS_REVIEWS_WEBHOOK_URL", "http://127.0.0.1:8766"))
    zeus_url: str = field(default_factory=lambda: _get(
        "KRONOS_ZEUS_URL", "http://127.0.0.1:8767"))
    routegate_url: str = field(default_factory=lambda: _get(
        "KRONOS_ROUTEGATE_URL", "http://127.0.0.1:8645"))

    # --- the four watched units (the restart allow-list) ---
    systemd_units: tuple[str, ...] = field(default_factory=lambda: tuple(
        u.strip() for u in _get(
            "KRONOS_UNITS",
            "rhobear-reviews,rhobear-reviews-dash,rhobear-neo,"
            "rhobear-neo-builder").split(",") if u.strip()))

    # --- state ---
    reviews_database_url: str = field(default_factory=lambda: _get(
        "REVIEWS_DATABASE_URL"))
    # HIGH 1 — the state dir is DURABLE: the r1 default (/run/rhobear-kronos)
    # is a RuntimeDirectory that systemd wipes when this oneshot stops (and
    # a reboot clears /run), so the alert dedupe, flap windows and judged
    # markers started empty EVERY five-minute pass. The unit mounts a
    # StateDirectory at /var/lib/rhobear-kronos and points this at it.
    state_dir: str = field(default_factory=lambda: _get(
        "RHOBEAR_KRONOS_STATE_DIR", "/var/lib/rhobear-kronos"))

    # --- GitHub identities: the two Apps ---
    # App ids are not secrets. Keys: PEM path (vault-env's decoded alias) or
    # PEM text — exactly the two shapes load_app_key() accepts upstream.
    reviews_app_id: str = field(default_factory=lambda: _get(
        "RHOBEAR_KRONOS_REVIEWS_APP_ID"))
    reviews_app_key_path: str = field(default_factory=lambda: _get(
        "RHOBEAR_KRONOS_REVIEWS_APP_KEY_PATH"))
    reviews_app_key: str = field(default_factory=lambda: _get(
        "RHOBEAR_KRONOS_REVIEWS_APP_KEY"))
    zeus_app_id: str = field(default_factory=lambda: _get(
        "RHOBEAR_KRONOS_ZEUS_APP_ID"))
    zeus_app_key_path: str = field(default_factory=lambda: _get(
        "RHOBEAR_KRONOS_ZEUS_APP_KEY_PATH"))
    zeus_app_key: str = field(default_factory=lambda: _get(
        "RHOBEAR_KRONOS_ZEUS_APP_KEY"))

    # --- alerts: one issue per problem, as the Zeus App ---
    alert_repo: str = field(default_factory=lambda: _get(
        "RHOBEAR_KRONOS_ALERT_REPO", "deariencampbell1-sys/rhobear-reviews-app"))
    alert_label: str = field(default_factory=lambda: _get(
        "RHOBEAR_KRONOS_ALERT_LABEL", "kronos"))
    # The owner workspace never meters (same convention as Zeus's config).
    owner_install_id: str = field(default_factory=lambda: _get(
        "NEO_OWNER_INSTALL_ID", "deariencampbell1-sys"))

    # --- cadence ---
    interval_secs: int = field(default_factory=lambda: _get_int(
        "KRONOS_INTERVAL_SECS", 300))            # timer: every 5 minutes
    pass_budget_secs: int = field(default_factory=lambda: _get_int(
        "KRONOS_PASS_BUDGET_SECS", 240))         # one pass: 4-minute budget
    remedy_window_secs: int = field(default_factory=lambda: _get_int(
        "KRONOS_REMEDY_WINDOW_SECS", 3600))      # remedy: once per hour
    alert_window_secs: int = field(default_factory=lambda: _get_int(
        "KRONOS_ALERT_WINDOW_SECS", 6 * 3600))   # alert: never twice in 6h
    stall_mins: int = field(default_factory=lambda: _get_int(
        "KRONOS_STALL_MINS", 30))                # delivery with no review
    stuck_mins: int = field(default_factory=lambda: _get_int(
        "KRONOS_STUCK_MINS", 30))                # job in a non-final phase
    flap_threshold: int = field(default_factory=lambda: _get_int(
        "KRONOS_FLAP_THRESHOLD", 5))             # restarts ...
    flap_window_secs: int = field(default_factory=lambda: _get_int(
        "KRONOS_FLAP_WINDOW_SECS", 600))         # ... per 10 minutes

    # --- the judge (Summit through RouteGate) ---
    # The route/model line is the owner's vocabulary (Summit); it is
    # configurable because the route table belongs to RouteGate, not here.
    judge_model: str = field(default_factory=lambda: _get(
        "KRONOS_JUDGE_MODEL", "summit"))
    judge_timeout_secs: int = field(default_factory=lambda: _get_int(
        "KRONOS_JUDGE_TIMEOUT_SECS", 300))

    def consumer_for(self, install_id: str) -> str:
        """`reviews:kronos/<install_id>` — the consumer header for every
        RouteGate call. An absent id degrades to the bare lane name (never
        invented); anything GitHub would never send is refused."""
        if not install_id:
            return KRONOS_CONSUMER_LANE
        if "\r" in install_id or "\n" in install_id or len(install_id) > 64:
            raise ValueError("invalid install id for the consumer header")
        return f"{KRONOS_CONSUMER_LANE}/{install_id}"

    # ------------------------------------------------------------------
    # Missing-name path: LOUD, with the vault names.
    # ------------------------------------------------------------------

    @staticmethod
    def required(state_dir: str | None = None) -> "KronosConfig":
        """A config fit for a real pass. Missing required env raises
        RuntimeError naming the VAULT key to provision — values are never
        printed, logged or stored. (HIGH 9: every real pass boots through
        this, and the documented vault key PATH files are loaded into the
        App keys HERE, once — call sites pass PEM text to GitHubApp, never
        a path string.)"""
        cfg = KronosConfig(state_dir=state_dir) if state_dir \
            else KronosConfig()
        missing: list[str] = []
        if not cfg.reviews_database_url:
            missing.append(
                "REVIEWS_DATABASE_URL (vault: REVIEWS_DATABASE_URL)")
        for app, vault in (("reviews", "RHOBEAR_REVIEWS_GH_APP_KEY_B64"),
                           ("zeus", "RHOBEAR_ZEUS_GH_APP_KEY_B64")):
            if not getattr(cfg, f"{app}_app_id"):
                missing.append(f"RHOBEAR_KRONOS_{app.upper()}_APP_ID")
            if not (getattr(cfg, f"{app}_app_key_path")
                    or getattr(cfg, f"{app}_app_key")):
                missing.append(
                    f"RHOBEAR_KRONOS_{app.upper()}_APP_KEY[_PATH] "
                    f"(vault: {vault})")
        if missing:
            raise RuntimeError(
                "rhobear-kronos missing required configuration: "
                + "; ".join(missing))
        # Key paths resolve to the PEM text at boot (the same shape
        # src.github_app.load_app_key consumes). Never logged, never printed.
        for app in ("reviews", "zeus"):
            path = getattr(cfg, f"{app}_app_key_path")
            if path and not getattr(cfg, f"{app}_app_key"):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        setattr(cfg, f"{app}_app_key", fh.read().strip())
                except OSError as exc:
                    raise RuntimeError(
                        f"cannot read {app} App private key at {path}: "
                        f"{exc}") from exc
        return cfg
