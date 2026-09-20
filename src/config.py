"""rhobear-neo configuration — env-driven, no secrets baked in.

Neo is the review-and-MERGE gate: it wakes when a trusted reviewer's verdict
lands on a PR head SHA, drives it to green (fix-forward trivial, dispatch a
Claude Code agent for substantial), and merges on green behind a per-install
auto-merge toggle. Sibling service to rhobear-reviews on the same VPS.

Claude Code CLI replaces the old Pi/direct-DeepSeek and direct-HTTP-OpenRouter
paths. The agent now routes through DeepSeek Direct (Anthropic-compatible
endpoint) at the exact `deepseek-v4-flash` model and `max` reasoning effort. It
runs headless with isolated CLAUDE_CONFIG_DIR, a per-run temp work directory,
and full tool access (gh, git, edit, test).
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

    # --- engine: Claude Code CLI via DeepSeek Direct (Anthropic-compatible endpoint) ---
    # Claude Code runs headless with isolated config dir + per-run temp workdir.
    # The agent has full tool access (gh, git, edit, test runner).
    # DEEPSEEK_API_KEY is the single secret (rotated, owned by Neo).
    # Stale OpenRouter env vars (OPENROUTER_API_KEY / NEO_OPENROUTER_*) are
    # deliberately NOT read here — validate() rejects any non-direct base URL.
    deepseek_base_url: str = field(default_factory=lambda: _get(
        "NEO_DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic"))
    deepseek_key: str = field(default_factory=lambda: _get("DEEPSEEK_API_KEY"))
    deepseek_model: str = field(default_factory=lambda: _get(
        "NEO_DEEPSEEK_MODEL", "deepseek-v4-flash"))
    deepseek_reasoning_effort: str = field(default_factory=lambda: _get(
        "NEO_REASONING_EFFORT", "max"))
    deepseek_max_tokens: int = field(default_factory=lambda: _get_int(
        "NEO_MAX_TOKENS", 32000))
    deepseek_timeout: int = field(default_factory=lambda: _get_int(
        "NEO_TIMEOUT", 1800))

    # --- claude binary path (on the VPS: /usr/bin/claude) ---
    claude_bin: str = field(default_factory=lambda: _get("NEO_CLAUDE_BIN", "/usr/bin/claude"))

    # --- builder lane engine (Hermes) ---
    # Route through RHOBEAR RouteGate (127.0.0.1:8645), the loopback-only
    # capability router -- NOT a hardcoded model and NOT the Bedrock proxy.
    # Ask for a CAPABILITY ("builder_top"); RouteGate owns the provider keys
    # and the ordered failover chain, with Bedrock as the entitled floor.
    # Hardcoding a model here re-scatters keys into .env files and bypasses
    # failover, which is the thing RouteGate exists to prevent.
    # neo_builder.dispatch_one reads all three. Without them it raises
    # AttributeError at the moment it picks up real work, systemd restarts it,
    # and the lane crash-loops forever WITHOUT EVER DISPATCHING A FIX -- while
    # still logging harmless-looking "skip (closed/merged)" lines. Found
    # 2026-09-20 after the builder had silently dispatched nothing.
    hermes_bin: str = field(default_factory=lambda: _get("NEO_HERMES_BIN", "/opt/rhobear-hermes/bin/hermes"))
    hermes_provider: str = field(default_factory=lambda: _get("NEO_HERMES_PROVIDER", "routegate"))
    hermes_model: str = field(default_factory=lambda: _get("NEO_HERMES_MODEL", "builder_top"))

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
            "DEEPSEEK_API_KEY": self.deepseek_key,
            "DATABASE_URL": self.database_url,
            "GH_TOKEN": self.gh_token,
        }.items() if not v]
        if missing:
            raise RuntimeError(f"rhobear-neo missing required env: {', '.join(missing)}")
        return self

    def validate(self) -> "Config":
        """Strict startup validation — DeepSeek Direct only, no fallback.

        Verifies:
          - base_url exactly https://api.deepseek.com/anthropic (trailing slash
            tolerant) — any OpenRouter or bare api.deepseek.com value is rejected
          - model exactly deepseek-v4-flash (unprefixed direct ID; a stale
            deepseek/... or [...]-suffixed value is rejected)
          - effort exactly max
          - max_tokens >= 32000
          - claude_bin exists/executable (where practical)

        Raises ValueError (safe to log — no secrets) on any violation.
        """
        base = self.deepseek_base_url.rstrip("/")
        if base != "https://api.deepseek.com/anthropic":
            raise ValueError(
                f"deepseek_base_url must be https://api.deepseek.com/anthropic, "
                f"got {self.deepseek_base_url!r}"
            )
        if self.deepseek_model.strip() != "deepseek-v4-flash":
            raise ValueError(
                f"deepseek_model must be 'deepseek-v4-flash' (DeepSeek Direct "
                f"ID, no /deepseek prefix or [1m] suffix), "
                f"got {self.deepseek_model!r}"
            )
        if self.deepseek_reasoning_effort.strip().lower() != "max":
            raise ValueError(
                f"deepseek_reasoning_effort must be 'max', "
                f"got {self.deepseek_reasoning_effort!r}"
            )
        if self.deepseek_max_tokens < 32000:
            raise ValueError(
                f"deepseek_max_tokens must be >= 32000, "
                f"got {self.deepseek_max_tokens}"
            )
        # Check claude binary exists where practical (skip on Windows — it's a
        # remote VPS path like /usr/bin/claude).
        if self.claude_bin and os.name != "nt":
            resolved = shutil.which(self.claude_bin)
            if resolved is None:
                # Fallback: check exact path if not in PATH.
                if not os.path.isfile(self.claude_bin):
                    raise ValueError(
                        f"claude_bin not found: {self.claude_bin}"
                    )
        return self


def load() -> Config:
    return Config()