"""rhobear-neo configuration — env-driven, no secrets baked in.

Neo is the review-and-MERGE gate: it wakes when a trusted reviewer's verdict
lands on a PR head SHA, drives it to green (fix-forward trivial, dispatch a
Claude Code agent for substantial), and merges on green behind a per-install
auto-merge toggle. Sibling service to rhobear-reviews on the same VPS.

Claude Code CLI replaces the old Pi/direct-DeepSeek and direct-HTTP-OpenRouter
paths.  The agent runs headless with isolated CLAUDE_CONFIG_DIR, a per-run temp
work directory, and full tool access (gh, git, edit, test).
"""
from __future__ import annotations

import os
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

    # --- engine: Claude Code CLI via OpenRouter (Anthropic-compatible backend) ---
    # Claude Code runs headless with isolated config dir + per-run temp workdir.
    # The agent has full tool access (gh, git, edit, test runner).
    # Config shared with rhobear-reviews: OPENROUTER_API_KEY is the single secret.
    openrouter_base_url: str = field(default_factory=lambda: _get(
        "NEO_OPENROUTER_BASE_URL", "https://openrouter.ai/api"))
    openrouter_key: str = field(default_factory=lambda: _get("OPENROUTER_API_KEY"))
    openrouter_model: str = field(default_factory=lambda: _get(
        "NEO_OPENROUTER_MODEL", "deepseek/deepseek-v4-flash"))
    openrouter_reasoning_effort: str = field(default_factory=lambda: _get(
        "NEO_REASONING_EFFORT", "max"))
    openrouter_max_tokens: int = field(default_factory=lambda: _get_int(
        "NEO_MAX_TOKENS", 32000))
    openrouter_timeout: int = field(default_factory=lambda: _get_int(
        "NEO_TIMEOUT", 300))

    # --- claude binary path (on the VPS: /usr/bin/claude) ---
    claude_bin: str = field(default_factory=lambda: _get("NEO_CLAUDE_BIN", "/usr/bin/claude"))

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
            "OPENROUTER_API_KEY": self.openrouter_key,
            "DATABASE_URL": self.database_url,
            "GH_TOKEN": self.gh_token,
        }.items() if not v]
        if missing:
            raise RuntimeError(f"rhobear-neo missing required env: {', '.join(missing)}")
        return self


def load() -> Config:
    return Config()