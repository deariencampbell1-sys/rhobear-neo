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

import logging
import os
import re
import shutil
from dataclasses import dataclass, field

log = logging.getLogger("rhobear_neo.config")

# ── RouteGate (L2-platform, loopback) ────────────────────────────────────────
# Zeus's model calls are metered at the gate, one consumer header per lane:
#   X-RHOBEAR-Consumer: reviews:zeus/<install_id>
# The loopback root speaks the OpenAI shape (/v1/chat/completions,
# /v1/rerank); RouteGate does NOT expose an Anthropic Messages endpoint today
# (no /v1/messages in routegate.py, none documented in L2 CONTRACT.md), so
# NEO_ROUTEGATE_ANTHROPIC_URL stays EMPTY until L2 serves the wire — empty
# keeps Zeus's Claude-CLI triage on DeepSeek's Anthropic endpoint exactly as
# brief step 2's "if not" branch requires.  Fail-closed: only the loopback
# values below pass validate().
ROUTEGATE_PORT = 8645
ROUTEGATE_BASE_URL = f"http://127.0.0.1:{ROUTEGATE_PORT}/v1"
# The loopback BASE for the Anthropic wire (once L2 exposes it):
# ANTHROPIC_BASE_URL is a BASE and the Claude CLI appends /v1/messages
# itself — exactly how this repo already points it at
# deepseek_base_url = https://api.deepseek.com/anthropic.  This constant is
# the ALLOW-LIST value `routegate_anthropic_url` is checked against — NOT the
# field's default (the field defaults to empty: the gate stays off until L2
# serves the wire and the owner flips the switch).
ROUTEGATE_ANTHROPIC_URL = f"http://127.0.0.1:{ROUTEGATE_PORT}"
# The RouteGate capability that serves peak-hours/cost-aware routing
# (route-policy.yaml); also the route name recorded on gated attempts.
ROUTEGATE_MODEL = "peak"

# An install id may only be what GitHub sends: digits/word chars/dashes.
# Anything else never reaches a header (Config.zeus_consumer AND the agent's
# sink validation both consult this).
_INSTALL_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def install_id_is_valid(install_id: str) -> bool:
    """True only for what GitHub sends — the single shape check for an
    install id anywhere it might reach a header or an attribution row."""
    return bool(install_id) and bool(_INSTALL_ID_RE.fullmatch(install_id))


def zeus_consumer_for(install_id: str) -> str:
    """L0's consumer shape: `reviews:zeus/<install_id>`.

    Invalid or absent id → the bare lane default (a warning that never
    echoes the value).  Module-level so the store (`UsageStore.job_consumer`,
    `record_attempt`) and the agent's sink all validate through ONE shape.
    """
    if not install_id:
        return "reviews:zeus"
    if not install_id_is_valid(install_id):
        log.warning("invalid install id for consumer attribution — "
                    "using the bare lane default")
        return "reviews:zeus"
    return f"reviews:zeus/{install_id}"


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


#: Default install id whose usage is the owner workspace's own (never
#: metered).  Deployments override it with NEO_OWNER_INSTALL_ID — no install
#: id may be hard-coded in the worker.
DEFAULT_OWNER_INSTALL_ID = "deariencampbell1-sys"


@dataclass
class Config:
    # --- webhook ingress (loopback; Caddy fronts the public endpoint) ---
    bind_host: str = field(default_factory=lambda: _get("NEO_BIND_HOST", "127.0.0.1"))
    bind_port: int = field(default_factory=lambda: _get_int("NEO_BIND_PORT", 8767))
    webhook_secret: str = field(default_factory=lambda: _get("RHOBEAR_NEO_WEBHOOK_SECRET"))

    # --- metering: which install id is the owner workspace (never metered) ---
    # Deployment-specific (W1-13-r1 defect 7): configured, never hard-coded
    # in the worker.  Defaults to the legacy value for compatibility.
    owner_install_id: str = field(
        default_factory=lambda: _get("NEO_OWNER_INSTALL_ID", DEFAULT_OWNER_INSTALL_ID)
    )

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
    # Client credential for the agent. Direct DeepSeek uses the DeepSeek key;
    # when the local Claude-wire bridge is the base URL, this is the bridge's
    # shared handshake value and the bridge holds the real provider keys.
    # NOTE: the bridge serves DeepSeek/Bedrock only — it does NOT accept
    # forwarded provider credentials or serve GLM or other models.  Routes
    # with no Anthropic-wire endpoint (e.g. deepinfra) are skipped for the
    # Claude agent with one log line.
    deepseek_key: str = field(default_factory=lambda: (
        _get("NEO_DEEPSEEK_API_KEY") or _get("DEEPSEEK_API_KEY")))
    deepseek_model: str = field(default_factory=lambda: _get(
        "NEO_DEEPSEEK_MODEL", "deepseek-v4-flash"))
    deepseek_reasoning_effort: str = field(default_factory=lambda: _get(
        "NEO_REASONING_EFFORT", "max"))
    deepseek_max_tokens: int = field(default_factory=lambda: _get_int(
        "NEO_MAX_TOKENS", 32000))
    deepseek_timeout: int = field(default_factory=lambda: _get_int(
        "NEO_TIMEOUT", 1800))

    # --- RouteGate: Zeus's metered gate (L2-platform) ------------------------
    # OpenAI-shape root of the gate (its /v1/chat/completions ingress).  There
    # is no OpenAI-shape call inside the files this lane owns today — Zeus's
    # triage runs the Claude Code CLI on the Anthropic wire — so this is the
    # documented step-1 default, validated fail-closed to the loopback gate.
    routegate_base_url: str = field(default_factory=lambda: _get(
        "NEO_ROUTEGATE_BASE_URL", ROUTEGATE_BASE_URL))
    # Anthropic-wire entry of the gate.  Empty = gate not serving this wire
    # yet (the current state): the CLI keeps DeepSeek's Anthropic endpoint.
    # Set to the BASE http://127.0.0.1:8645 the moment L2 exposes it — the
    # Claude CLI appends /v1/messages itself (deepseek_base_url proves the
    # contract).  The historical /v1/messages endpoint form is ALSO accepted
    # and normalised to the base; any other path is discarded fail-closed.
    # Every triage attempt then runs through RouteGate with consumer
    # reviews:zeus (RouteGate owns failover — no provider fallback outside
    # the gate; the outage surface becomes the gate itself).
    routegate_anthropic_url: str = field(default_factory=lambda: _get(
        "NEO_ROUTEGATE_ANTHROPIC_URL", ""))
    # RouteGate capability name — also the route name metered on gated attempts.
    # OPERATIONAL NOTE: once NEO_ROUTEGATE_ANTHROPIC_URL is set, RouteGate is a
    # hard dependency for Zeus triage (one gated attempt; the gate owns
    # failover).  A provider fallback outside the gate would be drift per the
    # brief, so the owner accepts that dependency before flipping the env var.
    routegate_model: str = field(default_factory=lambda: _get(
        "NEO_ROUTEGATE_MODEL", ROUTEGATE_MODEL))

    # --- claude binary path (on the VPS: /usr/bin/claude) ---
    claude_bin: str = field(default_factory=lambda: _get("NEO_CLAUDE_BIN", "/usr/bin/claude"))

    # --- builder lane engine (Hermes) ---
    # neo_builder.dispatch_one reads all three; without them it raises
    # AttributeError and the service crash-loops without ever dispatching a fix.
    hermes_bin: str = field(default_factory=lambda: _get("NEO_HERMES_BIN", "/opt/rhobear-hermes/bin/hermes"))
    hermes_provider: str = field(default_factory=lambda: _get("NEO_HERMES_PROVIDER", "zai"))
    hermes_model: str = field(default_factory=lambda: _get("NEO_HERMES_MODEL", "glm-5.3-flash"))

    # --- merge behaviour (the ONE per-install button) ---
    # Off  -> drive to green, fix-forward, dispatch builder, label neo:ready, STOP.
    # On   -> also run the squash-merge itself.
    auto_merge_default: bool = field(default_factory=lambda: _get_bool("NEO_AUTO_MERGE", False))
    # Lane PRs (l<n>-<lane>/...) merge on Apollo's pass plus the lane architect's acceptance of the exact
    # head (lane_merge_authority), independent of the install switch above (L0, 2026-10-25).
    lane_merge: bool = field(default_factory=lambda: _get_bool("NEO_LANE_MERGE", False))
    # Lane PRs also need L0's drift check (status l0/drift) green on the exact head (SPRINT section 9).
    lane_require_drift: bool = field(default_factory=lambda: _get_bool("NEO_LANE_REQUIRE_DRIFT", False))

    # --- safety floors ---
    # Defaults to bounces_before_escalation (5): a smaller default would make
    # the 5-bounce escalation unreachable (the builder lane parks the round
    # after max_builder_rounds before the PR ever bounces 5 times).  A value
    # BELOW bounces_before_escalation is clamped up with a warning — the
    # previous release shipped NEO_MAX_BUILDER_ROUNDS=2 on the live box, and a
    # config the previous release shipped must never crash-loop production
    # (round-2 finding: the old validate() raised here and kept both services
    # down until a human edited .env).
    max_builder_rounds: int = field(default_factory=lambda: _get_int(
        "NEO_MAX_BUILDER_ROUNDS", 5))

    # --- bounce / escalation ---
    # Number of builder bounces before escalation to a frontier advisor.
    bounces_before_escalation: int = field(default_factory=lambda: _get_int(
        "NEO_BOUNCES_BEFORE_ESCALATION", 5))
    # How often Neo polls an escalating PR's advisor advice_id (seconds).
    escalation_poll_secs: int = field(default_factory=lambda: _get_int(
        "NEO_ESCALATION_POLL_SECS", 60))
    # Give up polling an escalating PR after this long and park it (seconds).
    escalation_timeout_secs: int = field(default_factory=lambda: _get_int(
        "NEO_ESCALATION_TIMEOUT_SECS", 1800))  # 30 minutes
    # Advisor service URL (tailnet, on the build box).  Fail closed if unset.
    advisor_url: str = field(default_factory=lambda: _get("NEO_ADVISOR_URL"))
    # Bearer token for the advisor service.
    advisor_token: str = field(default_factory=lambda: _get("NEO_ADVISOR_TOKEN"))

    # --- state (shared VPS Postgres; Neo uses its own tables) ---
    database_url: str = field(default_factory=lambda: _get("DATABASE_URL"))

    # --- github auth: Neo acts as the customer's GitHub App --------------
    # A paying customer's private repo is reached via the App installation on
    # that repo (installation token), never via a personal OAuth token. GH_TOKEN
    # is deliberately NOT an identity any more; if it is still set in the env
    # it is logged once at startup and ignored (see __main__).
    github_app_id: str = field(default_factory=lambda: _get("NEO_GITHUB_APP_ID"))
    # The App private key: either the path to the PEM file, or the PEM text
    # itself. Never logged.
    github_app_key_path: str = field(default_factory=lambda: _get("NEO_GITHUB_APP_KEY_PATH"))
    github_app_key: str = field(default_factory=lambda: _get("NEO_GITHUB_APP_KEY"))
    # Deprecated legacy field — kept so pre-App unit tests can pass a
    # placeholder token; production never uses it as an identity.
    gh_token: str = field(default_factory=lambda: _get("GH_TOKEN"))

    # --- trusted producer: the Reviews App's check runs -----------------
    # A merge is authorized only by a check run whose `app.id` equals this
    # GitHub App id AND whose `head_sha` equals the exact head SHA (full
    # 40-char equality). A review or status from any other account (including
    # the PR author) is ignored.  Unset = fail closed: no merge.
    trusted_reviewer_app_id: str = field(default_factory=lambda: _get(
        "NEO_TRUSTED_REVIEWER_APP_ID"))

    # --- advisor service (on the build box) ---
    advisor_bind_host: str = field(default_factory=lambda: _get(
        "NEO_ADVISOR_BIND_HOST", "100.111.118.24"))
    advisor_bind_port: int = field(default_factory=lambda: _get_int(
        "NEO_ADVISOR_BIND_PORT", 8787))
    advisor_routes: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            r.strip() for r in _get(
                "ADVISOR_ROUTES",
                "claude:opus,codex:gpt-6.1-sol:high,codex:gpt-6-sol:high",
            ).split(",") if r.strip()
        )
    )
    advisor_queue_max: int = field(default_factory=lambda: _get_int(
        "ADVISOR_QUEUE_MAX", 3))
    advisor_timeout: int = field(default_factory=lambda: _get_int(
        "ADVISOR_TIMEOUT", 1200))  # 20 minutes

    # --- route chain for Neo's triage agent and Hermes Builder -----------
    # Ordered `provider:model` list, default chain (same order as Reviews B4):
    #   GLM 5.3 Flash on DeepInfra -> Z.ai direct GLM 5.3 Flash ->
    #   DeepInfra V4 Flash -> V4 Flash 0731 -> DeepSeek direct Flash ->
    #   DeepInfra V4.1 Flash 1017, maximum reasoning
    # Keys come from env names per provider (DEEPINFRA_API_KEY, ZAI_API_KEY,
    # DEEPSEEK_API_KEY); missing key -> that route is skipped and logged once.
    # Validated at startup (unknown provider/model -> refuse to start).
    neo_routes: str = field(default_factory=lambda: _get(
        "NEO_ROUTES",
        "deepinfra:glm-5.3-flash,zai:glm-5.3-flash,"
        "deepinfra:deepseek-v4-flash,deepinfra:deepseek-v4-flash-0731,"
        "deepseek:deepseek-v4-flash,deepinfra:deepseek-v4.1-flash-1017",
    ))
    # Parsed routes (set by validate() during startup; None before that call).
    # Not a dataclass field — set as an attribute by validate().
    parsed_routes: list | None = None

    # Optional Jev review-of-the-reviewer call. This is a separate OpenRouter
    # credential from the Claude-wire DeepSeek route above.
    jev_api_key: str = field(default_factory=lambda: (
        _get("NEO_OPENROUTER_API_KEY") or _get("OPENROUTER_API_KEY")))

    def zeus_consumer(self, install_id: str = "") -> str:
        """The consumer attribution for Zeus's model calls (brief step 1):

            reviews:zeus/<install_id>   when the caller knows the install
            reviews:zeus                when it does not (never invented)

        The id is validated before it is embedded: it goes into a header env
        var, so a value with newlines/CRLF or anything GitHub would never
        send is dropped (bare lane default, warning without the value) rather
        than interpolated.  The shape logic is module-level
        (`zeus_consumer_for`) so the store and the agent's sink validate the
        same way.
        """
        return zeus_consumer_for((install_id or "").strip())

    def __post_init__(self) -> None:
        # Clamp the round limit up to the escalation floor at construction, so
        # every consumer (and a bare Config()) sees the effective value.
        self._clamp_rounds()

    def _clamp_rounds(self) -> None:
        """max_builder_rounds >= bounces_before_escalation, always.

        With a smaller max the builder lane parks the round after
        max_builder_rounds before the PR ever bounces
        bounces_before_escalation times, so escalation would be unreachable.
        Clamp with a warning instead of raising: a value the previous release
        shipped is not an error, it is an upgrade.  Invalid values (negative /
        non-integer) are NEVER clamped — validate() refuses those."""
        if (isinstance(self.max_builder_rounds, int)
                and isinstance(self.bounces_before_escalation, int)
                and self.max_builder_rounds >= 0
                and self.bounces_before_escalation >= 0
                and self.max_builder_rounds < self.bounces_before_escalation):
            log.warning(
                "NEO_MAX_BUILDER_ROUNDS (%s) < NEO_BOUNCES_BEFORE_ESCALATION "
                "(%s) — clamping max_builder_rounds up to %s so escalation "
                "stays reachable",
                self.max_builder_rounds, self.bounces_before_escalation,
                self.bounces_before_escalation,
            )
            self.max_builder_rounds = self.bounces_before_escalation

    def require(self) -> "Config":
        # GH_TOKEN is NOT required (and is not an identity). The GitHub App
        # identity is checked at startup (__main__ / neo_builder.main) so the
        # personal token can never become a silent fallback.
        missing = [n for n, v in {
            "RHOBEAR_NEO_WEBHOOK_SECRET": self.webhook_secret,
            "DEEPSEEK_API_KEY/NEO_DEEPSEEK_API_KEY": self.deepseek_key,
            "DATABASE_URL": self.database_url,
        }.items() if not v]
        if missing:
            raise RuntimeError(f"rhobear-neo missing required env: {', '.join(missing)}")
        return self

    def validate(self) -> "Config":
        """Strict startup validation — approved DeepSeek wire endpoints only.

        Two endpoints are approved:
          - https://api.deepseek.com/anthropic (DeepSeek Direct, native wire)
          - the local Claude-wire bridge (127.0.0.1:8790 or the tailnet
            100.64.112.103:8790), which serves DeepSeek primary with a
            Bedrock fallback when the balance runs out.

        Any other host (OpenRouter, bare api.deepseek.com, arbitrary URLs) is
        rejected so a stale env var can never silently select a route.

        Verifies:
          - base_url is one of the approved endpoints (trailing slash tolerant)
          - model exactly deepseek-v4-flash (unprefixed direct ID; a stale
            deepseek/... or [...]-suffixed value is rejected)
          - effort exactly max
          - max_tokens >= 32000
          - claude_bin exists/executable (where practical)

        Raises ValueError (safe to log — no secrets) on any violation.

        RouteGate endpoints: the Anthropic wire URL (which would carry
        briefs) is validated fail-closed by DISCARDING anything that is not
        the allow-listed loopback endpoint — the gate then stays off and the
        sanctioned direct chain runs, so a stale/foreign URL can neither point
        Zeus's briefs off-box nor crash-loop the services (the agent's parsed
        loopback guard is the second layer).  The other two RouteGate knobs warn-and-default (see below).
        """
        # --- RouteGate -------------------------------------------------
        # Brief containment is enforced at TWO layers: here a malformed
        # anthropic URL is discarded (never fatal, never used), and the agent
        # refuses any non-loopback URL it is handed at construction and again
        # at use.  `routegate_base_url` (brief-mandated config surface, read
        # here; reserved for the OpenAI-shape probe — no brief traffic ever
        # rides it) and the capability NAME warn-and-default, so a typo never
        # takes both services down (Apollo r3/r4 — burn history above).
        if self.routegate_base_url.rstrip("/") != ROUTEGATE_BASE_URL:
            log.warning("routegate_base_url %r is not the loopback gate — "
                        "defaulting to %s",
                        self.routegate_base_url, ROUTEGATE_BASE_URL)
            self.routegate_base_url = ROUTEGATE_BASE_URL
        rg_anthropic = self.routegate_anthropic_url.strip().rstrip("/")
        # BOTH documented forms turn the gate on (Apollo r6): the base is
        # canonical; the field comment's historical /v1/messages endpoint
        # form is normalised to it (the CLI appends the path itself), so
        # following either instruction can never silently disable the gate.
        if rg_anthropic == ROUTEGATE_ANTHROPIC_URL + "/v1/messages":
            log.info("routegate_anthropic_url: endpoint form normalised "
                     "to the base %s", ROUTEGATE_ANTHROPIC_URL)
            rg_anthropic = ROUTEGATE_ANTHROPIC_URL
        if not rg_anthropic:
            pass  # gate off stays off — the field default is empty
        elif rg_anthropic == ROUTEGATE_ANTHROPIC_URL:
            # Accept branch (Apollo r7): store the ONE canonical base, so
            # the post-validate field holds the same value from_config and
            # the agent guard will see, whatever spelling was read
            # (trailing slash / whitespace were only stripped for the
            # comparison before).
            self.routegate_anthropic_url = rg_anthropic
        else:
            # Not the allow-listed endpoint: DISCARD it (gate off, the direct
            # chain stands) instead of raising.  Raising would crash-loop both
            # services on a typo — the exact failure this codebase already
            # burned on — while containment never depended on the raise:
            # a discarded URL carries nothing, and the agent re-parses
            # whatever URL it is handed against the loopback guard at use
            # time anyway (defense in depth, fail-closed at the sink).
            log.warning("routegate_anthropic_url %r is not RouteGate's "
                        "Anthropic endpoint (%s) — disabling the gate, "
                        "running direct",
                        self.routegate_anthropic_url,
                        ROUTEGATE_ANTHROPIC_URL)
            self.routegate_anthropic_url = ""
        if self.routegate_model.strip() != ROUTEGATE_MODEL:
            log.warning("routegate_model %r is not the RouteGate capability "
                        "— defaulting to %r",
                        self.routegate_model, ROUTEGATE_MODEL)
            self.routegate_model = ROUTEGATE_MODEL

        base = self.deepseek_base_url.rstrip("/")
        if base not in {
            "https://api.deepseek.com/anthropic",
            "http://127.0.0.1:8790",
            "http://100.64.112.103:8790",
        }:
            raise ValueError(
                f"deepseek_base_url must be the DeepSeek direct wire or the "
                f"local Claude-wire bridge, got {self.deepseek_base_url!r}"
            )
        # Owner 2026-10-04: the loop runs on GLM 5.3 Flash (Z.ai direct), which
        # only the local Claude-wire bridge can speak to Claude Code.
        if self.deepseek_model.strip() == "glm-5.3-flash" and base != "https://api.deepseek.com/anthropic":
            pass
        elif self.deepseek_model.strip() != "deepseek-v4-flash":
            raise ValueError(
                f"deepseek_model must be 'deepseek-v4-flash' (DeepSeek Direct) or 'glm-5.3-flash' via the bridge "
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
        # --- Round limits: refuse only INVALID values -----------------------
        # max_builder_rounds < bounces_before_escalation is NOT invalid — the
        # previous release shipped NEO_MAX_BUILDER_ROUNDS=2 on the live box —
        # it is clamped up with a warning (_clamp_rounds) instead of raising.
        # What validate() refuses is a value that cannot work at all: a
        # negative limit or a non-integer (including in the raw env, where the
        # tolerant reader would otherwise silently substitute the default).
        for name, raw, value in (
            ("NEO_MAX_BUILDER_ROUNDS", os.environ.get("NEO_MAX_BUILDER_ROUNDS", ""),
             self.max_builder_rounds),
            ("NEO_BOUNCES_BEFORE_ESCALATION",
             os.environ.get("NEO_BOUNCES_BEFORE_ESCALATION", ""),
             self.bounces_before_escalation),
        ):
            if raw.strip() and not re.fullmatch(r"[+-]?\d+", raw.strip()):
                raise ValueError(f"{name} must be an integer, got {raw.strip()!r}")
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        self._clamp_rounds()
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

        # --- B4: validate NEO_ROUTES ---
        # Parse and validate the route chain at startup.  Unknown provider or
        # malformed entry raises ValueError (refuse to start).  Missing API
        # keys are handled later (route is skipped, logged once — never a
        # crash mid-run).
        from .usage import resolve_route_chain
        self.parsed_routes = resolve_route_chain(self.neo_routes)

        return self

    def claude_endpoint_for(self, provider: str) -> str | None:
        """Return the Anthropic-wire base URL for a provider, or None if
        the provider cannot be served by Claude Code."""
        return CLAUDE_WIRE_ENDPOINTS.get(provider)

    def hermes_provider_for(self, provider: str) -> str | None:
        """Return the Hermes --provider name for a provider, or None if
        the provider cannot be served by Hermes."""
        return HERMES_PROVIDER_MAP.get(provider)

    def api_key_for(self, provider: str) -> str:
        """Get the API key for a provider from the environment.

        Returns empty string if the key is not set.
        """
        from .usage import Route
        return _get(Route(provider, "").env_key_name)


# ── Provider→endpoint tables for B4 route chain ─────────────────────────────

# Anthropic-wire endpoints for Claude Code, keyed by provider name.
# A provider not in this dict cannot be served by Claude Code (skipped).
#
# The local Claude-wire bridge (127.0.0.1:8790) holds its own keys and serves
# DeepSeek primary with a Bedrock fallback — it does NOT accept forwarded
# provider credentials or serve arbitrary models.  Only providers with a real
# Anthropic-wire endpoint (DeepSeek, Z.ai) are listed here.  DeepInfra routes
# are skipped for the Claude agent with one log line — they are served by the
# Hermes Builder lane instead.
CLAUDE_WIRE_ENDPOINTS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/anthropic",
    "zai": "https://api.z.ai/api/anthropic",
}

# Providers that Claude Code CANNOT serve (no Anthropic-wire endpoint).
# A route with one of these providers is skipped for the Claude agent.
CLAUDE_WIRE_SKIP: frozenset[str] = frozenset()

# Map provider to Hermes --provider name.
# A provider not in this dict cannot be served by Hermes (skipped).
HERMES_PROVIDER_MAP: dict[str, str] = {
    "deepinfra": "deepinfra",
    "zai": "zai",
    "deepseek": "deepseek",
    "openrouter": "openrouter",
    "openai": "openai",
}

# Providers Hermes CANNOT serve (no Hermes provider profile).
HERMES_SKIP: frozenset[str] = frozenset()


def load() -> Config:
    return Config()
