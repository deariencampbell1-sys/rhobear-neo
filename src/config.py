"""rhobear-neo configuration — env-driven, no secrets baked in.

Neo is the review-and-MERGE gate: it wakes when a trusted reviewer's verdict
lands on a PR head SHA, drives it to green (fix-forward trivial, dispatch a
builder for substantial), and merges on green behind a per-install auto-merge
toggle. Sibling service to rhobear-reviews on the same VPS.

The Hermes CLI (owner directive 2026-08-24: Hermes is the one dispatching
harness for this fabric — no separate Claude Code CLI dependency) replaces
the old Claude-Code-CLI/DeepSeek-Anthropic-wire engine. Neo now runs headless
via `hermes -z` against an approved Hermes provider profile, with full tool
access (gh, git, edit, test) in a per-run temp work directory.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field


def _get(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # --- webhook ingress (loopback; Caddy fronts the public endpoint) ---
    bind_host: str = field(default_factory=lambda: _get("NEO_BIND_HOST", "127.0.0.1"))
    bind_port: int = field(default_factory=lambda: _get_int("NEO_BIND_PORT", 8767))
    webhook_secret: str = field(default_factory=lambda: _get("RHOBEAR_NEO_WEBHOOK_SECRET"))

    # --- who Neo trusts as a "green" review signal (reviewer-agnostic) ---
    # A commit status context OR a check-run name from any of these counts as a
    # verdict Neo may gate on. rhobear-reviews is ours; others layer on.
    trusted_review_contexts: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            c.strip() for c in _get(
                "NEO_TRUSTED_REVIEW_CONTEXTS",
                "rhobear-reviews,CodeAnt,gemini-code-assist",
            ).split(",") if c.strip()
        )
    )

    # --- scope: which orgs/repos Neo is allowed to act on ---
    orgs: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in _get("RHOBEAR_NEO_ORGS", "rhobear-ai").split(",") if o.strip()
        )
    )
    # Optional hard repo allowlist ("owner/repo,owner/repo"). When set, Neo acts ONLY on
    # these repos regardless of how broadly the GitHub App is installed — the safety scope
    # for a first live run. Empty = act on any in-scope org repo.
    repo_allowlist: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            r.strip() for r in _get("NEO_REPO_ALLOWLIST", "").split(",") if r.strip()
        )
    )

    # --- engine: the Hermes CLI, one-shot headless mode ---
    # Neo has full tool access (gh, git, edit, test runner) through Hermes's
    # own tool loop. No credential is read here directly — Hermes resolves
    # each provider profile's key from its own config/env, so Neo never
    # handles a raw provider API key at all.
    hermes_bin: str = field(default_factory=lambda: _get(
        "NEO_HERMES_BIN", "/opt/rhobear-hermes/bin/hermes"))
    hermes_provider: str = field(default_factory=lambda: _get(
        "NEO_HERMES_PROVIDER", "rhobear-glm5"))
    hermes_model: str = field(default_factory=lambda: _get(
        "NEO_HERMES_MODEL", "bedrock/zai.glm-5"))
    deepseek_timeout: int = field(default_factory=lambda: _get_int(
        "NEO_TIMEOUT", 1800))

    # --- merge behaviour (the ONE per-install button) ---
    # Off  -> drive to green, fix-forward, dispatch builder, label neo:ready, STOP.
    # On   -> also run the squash-merge itself.
    auto_merge_default: bool = field(default_factory=lambda: _get_bool("NEO_AUTO_MERGE", False))

    # --- safety floors ---
    max_builder_rounds: int = field(default_factory=lambda: _get_int("NEO_MAX_BUILDER_ROUNDS", 2))

    # --- state (shared VPS Postgres; Neo uses its own tables) ---
    database_url: str = field(default_factory=lambda: _get("DATABASE_URL"))

    # --- github auth (App token preferred; gh-cli token fallback) ---
    gh_token: str = field(default_factory=lambda: _get("GH_TOKEN"))

    def require(self) -> "Config":
        missing = [n for n, v in {
            "RHOBEAR_NEO_WEBHOOK_SECRET": self.webhook_secret,
            "DATABASE_URL": self.database_url,
            "GH_TOKEN": self.gh_token,
        }.items() if not v]
        if missing:
            raise RuntimeError(f"rhobear-neo missing required env: {', '.join(missing)}")
        return self

    def validate(self) -> "Config":
        """Strict startup validation of the agent route — allowlist, no fallback.

        Neo is the highest-stakes agent in the system: it edits code, runs
        tests, and with NEO_AUTO_MERGE=true it lands the PR itself. Its
        route is therefore chosen for JUDGMENT QUALITY, not price — owner
        directive 2026-08-23. Hermes is the required dispatching harness for
        this fabric — owner directive 2026-08-24 — so there is exactly one
        engine, not a choice of CLIs.

        This is an allowlist of Hermes PROVIDER PROFILES, not a single
        hardcoded model. Pinning one vendor's one model meant that when that
        balance ran dry on 2026-08-23 the service stayed `active` while the
        gate was dead, with no legal way to move. An outage must cost a
        route, never the gate.

        Adding a route here is deliberate. Anything else is refused — no
        silent fallback, no stale env var quietly downgrading the reviewer.

        Raises ValueError (safe to log — no secrets) on any violation.
        """
        # provider profile -> models approved on that profile for driving
        # Neo's build/fix-forward/merge work. Each profile is a named route
        # in the VPS's own Hermes config (root's ~/.hermes/config.yaml) —
        # Hermes resolves the actual base_url and credential from there, so
        # this allowlist never needs to know either.
        APPROVED_ROUTES = {
            "rhobear-glm5": {"bedrock/zai.glm-5"},
            "rhobear-glm47": {"bedrock/zai.glm-4.7"},
        }

        if self.hermes_provider not in APPROVED_ROUTES:
            raise ValueError(
                f"neo hermes_provider {self.hermes_provider!r} is not an "
                f"approved agent route. Approved: {', '.join(sorted(APPROVED_ROUTES))}"
            )
        if self.hermes_model not in APPROVED_ROUTES[self.hermes_provider]:
            raise ValueError(
                f"neo hermes_model {self.hermes_model!r} is not approved on "
                f"{self.hermes_provider}. Approved there: "
                f"{', '.join(sorted(APPROVED_ROUTES[self.hermes_provider]))}"
            )
        # Check the hermes binary exists where practical (skip on Windows —
        # this config also loads there for local testing/dry-runs).
        if self.hermes_bin and os.name != "nt":
            resolved = shutil.which(self.hermes_bin)
            if resolved is None and not os.path.isfile(self.hermes_bin):
                raise ValueError(f"hermes_bin not found: {self.hermes_bin}")
        return self


def load() -> Config:
    return Config()
