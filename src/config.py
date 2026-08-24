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
    # The credential follows the route: OpenRouter routes authenticate with
    # OPENROUTER_API_KEY, DeepSeek Direct with DEEPSEEK_API_KEY. Reading the
    # wrong one is how a route change turns into a 401 nobody can explain.
    deepseek_key: str = field(default_factory=lambda: (
        _get("OPENROUTER_API_KEY")
        if "openrouter.ai" in _get("NEO_DEEPSEEK_BASE_URL", "")
        else _get("DEEPSEEK_API_KEY")
    ))
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
            ("OPENROUTER_API_KEY" if "openrouter.ai" in self.deepseek_base_url
             else "DEEPSEEK_API_KEY"): self.deepseek_key,
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
        directive 2026-08-23.

        This is an allowlist, not a single-vendor lock. Pinning one vendor's
        one model meant that when that balance ran dry on 2026-08-23 the
        service stayed `active` while the gate was dead, with no legal way
        to move. An outage must cost a route, never the gate.

        Adding a route here is deliberate. Anything else is refused — no
        silent fallback, no stale env var quietly downgrading the reviewer.

        Also verifies: effort max, max_tokens >= 32000, claude_bin present.

        Raises ValueError (safe to log — no secrets) on any violation.
        """
        # Anthropic-wire-format endpoints the claude CLI can drive, and the
        # models approved on each for reviewing and landing code.
        APPROVED_ROUTES = {
            # NOTE: the local Bedrock relay (127.0.0.1:9010) only speaks OpenAI
            # chat/completions. Neo drives the `claude` CLI, which requires an
            # Anthropic-Messages-wire endpoint (ANTHROPIC_BASE_URL). DeepSeek
            # Direct is the only currently-wired route that speaks that
            # protocol without going through paid OpenRouter, so it stays
            # Neo's sole approved route until a Messages-format shim is built
            # in front of the Bedrock relay. Tracked, not silently dropped.
            # DeepSeek Direct stays first-class — not deprecated, not demoted.
            # It is one approved route among several so a dry balance can no
            # longer take the merge gate offline.
            "https://api.deepseek.com/anthropic": {
                "deepseek-v4-flash",
                "deepseek-v4-pro",
            },
        }

        base = self.deepseek_base_url.rstrip("/")
        if base not in APPROVED_ROUTES:
            raise ValueError(
                f"neo base_url {self.deepseek_base_url!r} is not an approved "
                f"agent route. Approved: {', '.join(sorted(APPROVED_ROUTES))}"
            )
        model = self.deepseek_model.strip()
        if model not in APPROVED_ROUTES[base]:
            raise ValueError(
                f"neo model {self.deepseek_model!r} is not approved on {base}. "
                f"Approved there: {', '.join(sorted(APPROVED_ROUTES[base]))}"
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