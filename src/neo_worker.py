"""Neo worker — turns a reviewer verdict into a Neo protocol run.

Flow per wake ({repo, sha, green, context[, pr_number]}):
  1. Resolve PR number from the head SHA (if not supplied).
  2. Loop/thrash guard: skip if we already acted on this SHA; read builder rounds.
  3. Entitlement/credits gate (per-install balance) — Neo actions only for a paid install.
  4. Build the Neo brief (neo_protocol) and run it headless on DeepSeek (Flash triage;
     the brief itself dispatches a Pro builder when a substantial fix is needed).
  5. Record the action + credits (Gemini-baseline cost via the shared pricing) in state.

The heavy lifting (read findings, fix-forward, dispatch builder, merge) is done by the
DeepSeek agent bound to the canon — this worker is the deterministic wrapper: guardrails,
credits, state. Merge authority + the auto-merge toggle live here.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time

from .config import Config
from .neo_state import NeoState
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
    if state.already_acted(repo, pr, sha, "triage"):
        log.info("already triaged %s#%s@%s — skip (idempotent)", repo, pr, sha[:8])
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

    state.record(repo, pr, sha, "triage", detail={"context": wake.get("context"),
                                                   "green": wake.get("green")})

    # --- run the Neo protocol headless on DeepSeek --------------------------
    brief = neo_protocol.build_brief(
        repo=repo, pr=pr, head_sha=sha,
        reviewer_context=wake.get("context", "?"), reviewer_green=bool(wake.get("green")),
        auto_merge=auto_merge, builder_round=rounds,
        max_builder_rounds=cfg.max_builder_rounds, builder_model=cfg.model_builder,
    )
    usage, verdict = _run_agent(cfg, repo, brief)
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


def _run_agent(cfg: Config, repo: str, brief: str) -> tuple[dict, str]:
    """Run the Neo brief headless on **Pi** (pi.dev) pointed at DeepSeek.

    Pi over the Claude CLI (owner directive): leaner + cheaper, small system prompt (less token
    waste), auto-executes its bash/edit/write tools with no permission prompt (so no root/skip-
    permissions block), and it just WORKS the task — no sub-agent fan-out. The triage runs on Flash;
    the brief itself shells out to a Pro `pi` worker when a substantial fix is needed.

    Runs in a fresh per-PR work dir so the agent can `gh repo clone` + edit for a fix-forward without
    touching the service tree. Returns (normalized usage, verdict line)."""
    model = cfg.model_triage
    work = os.path.join("/tmp/neo-work", repo.replace("/", "__"))
    os.makedirs(work, exist_ok=True)
    env = {
        **os.environ,
        # Pi reads the DeepSeek provider key from env / ~/.pi/agent/auth.json.
        "DEEPSEEK_API_KEY": cfg.deepseek_key,
        "GH_TOKEN": cfg.gh_token,
    }
    try:
        p = subprocess.run(
            # --thinking off: deepseek-v4-flash is a reasoning model; on a complex brief its
            # reasoning eats the whole budget and it never ACTS (5-min hangs, empty output). Off
            # makes it decisively run tools + emit the verdict. (Builder=Pro can keep thinking.)
            ["pi", "--provider", "deepseek", "--model", model, "--thinking", "off", "-p", brief,
             "--mode", "json", "--no-context-files", "--no-session", "--no-skills"],
            capture_output=True, text=True, env=env, timeout=1800, cwd=work,
        )
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
        last_text = ""
        for line in (p.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = e.get("message") or {}
            if m.get("role") == "assistant":
                u = m.get("usage") or {}
                usage["input_tokens"] += u.get("input", 0) or 0
                usage["output_tokens"] += u.get("output", 0) or 0
                usage["cache_read_input_tokens"] += u.get("cacheRead", 0) or 0
                for b in (m.get("content") or []):
                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                        last_text = b["text"]
        verdict = ""
        for ln in last_text.splitlines():
            if ln.strip().upper().startswith("VERDICT:"):
                # models like to bold/quote the verdict (**FIX-FORWARD**, `ACCEPT`) — strip markdown.
                verdict = ln.split(":", 1)[1].replace("*", "").replace("`", "").strip()
                break
        if not verdict and p.returncode != 0:
            log.warning("pi rc=%d stderr=%s", p.returncode, (p.stderr or "")[-200:])
        return usage, verdict
    except Exception as e:
        log.exception("neo agent (pi) run failed: %s", e)
        return {}, ""
