"""Neo worker — turns a reviewer verdict into a Neo protocol run.

Flow per wake ({repo, sha, green, context[, pr_number]}):
  1. Resolve PR number from the head SHA (if not supplied).
  2. Loop/thrash guard: skip if we already acted on this SHA; read builder rounds.
  3. Entitlement/credits gate (per-install balance) — Neo actions only for a paid install.
  4. Build the Neo brief (neo_protocol) and run it headless via Claude Code CLI
     (deepseek-v4-flash at max reasoning effort via DeepSeek Direct, isolated
     config + temp workdir).  The brief itself dispatches a builder when a
     substantial fix is needed; the Claude Code agent handles gh, git, edit, and
     test runner tools.
  5. Record the action + credits (Gemini-baseline cost via the shared pricing) in state.

The heavy lifting (read findings, fix-forward, dispatch builder, merge) is done by the
LLM bound to the canon — this worker is the deterministic wrapper: guardrails,
credits, state. Merge authority + the auto-merge toggle live here.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess

from .config import Config
from .neo_state import NeoState
from .agent_claude import ClaudeAgent, AgentError
from . import neo_protocol

log = logging.getLogger("rhobear_neo.worker")

# Gemini-baseline credit model (see plans/neo-pricing-model.md). Credits charged =
# gemini_cost * 1.26 * 100_000. We meter AFTER the run from the agent's reported usage.
MARGIN = 1.26
CREDITS_PER_USD = 100_000
GEMINI = {  # $/token
    "flash_in": 0.30 / 1e6, "flash_out": 2.50 / 1e6,
    "pro_in": 2.00 / 1e6, "pro_out": 12.00 / 1e6, "pro_cache": (2.00 * 0.25) / 1e6,
}


def _gh(cfg: Config, *args: str, timeout: int = 30) -> tuple[int, str]:
    env = {**os.environ, "GH_TOKEN": cfg.gh_token}
    p = subprocess.run(["gh", *args], capture_output=True, text=True, env=env, timeout=timeout)
    return p.returncode, (p.stdout or p.stderr).strip()


def resolve_pr(cfg: Config, repo: str, sha: str, given: int | None) -> int | None:
    if given:
        return given
    rc, out = _gh(cfg, "api", f"repos/{repo}/commits/{sha}/pulls",
                  "-H", "Accept: application/vnd.github+json")
    if rc != 0:
        log.warning("resolve_pr gh failed repo=%s sha=%s: %s", repo, sha[:8], out[:120])
        return None
    try:
        prs = json.loads(out)
        opens = [p for p in prs if p.get("state") == "open"]
        return (opens or prs)[0]["number"] if prs else None
    except Exception:
        return None


def credits_for(usage: dict, builder: bool) -> int:
    """Gemini-baseline credits from the agent's reported DeepSeek usage."""
    fin = usage.get("input_tokens", 0) or 0
    fout = usage.get("output_tokens", 0) or 0
    cread = usage.get("cache_read_input_tokens", 0) or 0
    if builder:
        cost = fin * GEMINI["pro_in"] + cread * GEMINI["pro_cache"] + fout * GEMINI["pro_out"]
    else:
        cost = fin * GEMINI["flash_in"] + fout * GEMINI["flash_out"]
    return int(cost * MARGIN * CREDITS_PER_USD)


def run_neo(cfg: Config, state: NeoState, wake: dict) -> None:
    repo, sha = wake["repo"], wake["sha"]
    pr = resolve_pr(cfg, repo, sha, wake.get("pr_number"))
    if not pr:
        log.info("no open PR for %s@%s — skip", repo, sha[:8])
        return

    # --- loop / thrash guard ------------------------------------------------
    # Keyed on verdict COLOR, not sha alone. A green verdict arriving after a
    # red triage of the same head (the reviewer re-ran and passed it) is new
    # decision input and must not be swallowed by the old idempotency skip --
    # that skip is what left green-reviewed PRs open (green-stall) while the
    # red triage sat as their only record. Missing color on old rows counts
    # as red so a first green wake still re-triggers.
    green = bool(wake.get("green"))

    # Already merged: never re-open a head Neo merged. Checked before the claim
    # so we don't write a triage marker for a head that is already resolved.
    if state.already_acted(repo, pr, sha, "merged"):
        log.info("%s#%s@%s already merged by Neo — skip", repo, pr, sha[:8])
        return

    # ATOMIC CLAIM: already_acted() is a read, so two workers handling the same
    # wake concurrently could both pass it before either writes its triage
    # marker -- duplicate agent runs plus conflicting GitHub mutations for one
    # SHA. claim_action() does the check-and-insert under an advisory lock, so
    # exactly one caller wins; the loser skips. It also writes the triage
    # marker with the wake's context and color, replacing the separate
    # record("triage") call that used to follow the guard.
    if not state.claim_action(repo, pr, sha, "triage", green=green,
                              detail={"context": wake.get("context"),
                                      "green": wake.get("green")}):
        log.info("already triaged %s#%s@%s (green=%s) — skip (idempotent)",
                 repo, pr, sha[:8], green)
        return
    rounds = state.builder_rounds(repo, pr)

    # --- entitlement (per-install credit balance) ---------------------------
    org = repo.split("/", 1)[0]
    inst = state.install(org, org)          # dogfood: install keyed by org
    if inst.get("plan") and inst.get("credits_balance", 0) <= 0:
        state.record(repo, pr, sha, "escalated", verdict="ESCALATE",
                     detail={"reason": "out of credits"})
        _gh(cfg, "pr", "comment", str(pr), "-R", repo,
            "--body", "Neo: this install is out of credits — top up to resume auto-fix/merge.")
        log.info("%s#%s: install out of credits — gated", repo, pr)
        return
    auto_merge = bool(inst.get("auto_merge") or cfg.auto_merge_default)

    # --- run the Neo protocol headless via Claude Code CLI --------------------
    brief = neo_protocol.build_brief(
        repo=repo, pr=pr, head_sha=sha,
        reviewer_context=wake.get("context", "?"), reviewer_green=bool(wake.get("green")),
        auto_merge=auto_merge, builder_round=rounds,
        max_builder_rounds=cfg.max_builder_rounds, builder_model=cfg.deepseek_model,
    )
    agent = ClaudeAgent.from_config(cfg)
    usage, verdict = _run_agent(agent, brief)
    if not verdict:
        # The triage marker above is written BEFORE the agent runs, and
        # already_acted() treats it as done. So an agent that dies -- provider
        # 402, timeout, malformed stream -- used to strand this PR at this head
        # forever: every later wake for the same sha hit "already triaged --
        # skip". Clear it so the next wake retries. (2026-09-21: DeepSeek ran
        # out of credit mid-triage and capturd#40 stranded exactly this way.)
        state.clear(repo, pr, sha, "triage")
        log.warning("%s#%s@%s agent produced no verdict — triage marker cleared, "
                    "next wake retries", repo, pr, sha[:8])
        return
    is_builder = verdict.startswith("BOUNCE-BUILDER")
    credits = credits_for(usage, builder=is_builder)
    phase = {
        "ACCEPT-MERGED": "merged", "ACCEPT-READY": "ready", "FIX-FORWARD": "fix_forward",
        "BOUNCE-BUILDER": "builder", "ESCALATE": "escalated",
    }.get(verdict.split()[0] if verdict else "", "triage")
    state.record(repo, pr, sha, phase, verdict=verdict,
                 builder_round=(rounds + 1) if is_builder else rounds, credits=credits)
    if inst.get("plan"):
        state.debit(org, credits)
    log.info("%s#%s@%s verdict=%s credits=%d (auto_merge=%s round=%d)",
             repo, pr, sha[:8], verdict or "?", credits, auto_merge, rounds)


def _run_agent(agent: ClaudeAgent, brief: str) -> tuple[dict, str]:
    """Run the Neo brief headless via Claude Code CLI.

    Claude Code runs with isolated config dir + per-run temp workdir, giving
    the agent full tool access (gh, git, edit, test runner).  DeepSeek Direct
    provides the Anthropic-compatible endpoint.

    Returns (normalised usage, verdict line).  On any error both are empty
    so the caller skips merge and escalates."""
    try:
        usage, verdict = agent.run(brief)
        return usage, verdict
    except AgentError:
        log.exception("neo claude agent run failed")
        return {}, ""