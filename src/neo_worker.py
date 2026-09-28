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
from .jev_review import JevAssessment, assess as jev_assess

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


def _review_evidence(cfg: Config, repo: str, pr: int, sha: str) -> dict:
    """Machine-readable review evidence bound to ONE commit, not to the PR.

    Per the control-plane directive, a result produced for commit A may
    authorize only commit A. GitHub happily attaches a later review object to a
    moved PR head, so the binding is read from the review's own `commit_id`.
    """
    try:
        rc, raw = _gh(cfg, "pr", "view", str(pr), "-R", repo,
                      "--json", "headRefOid,reviews,statusCheckRollup", timeout=45)
    except Exception as exc:  # noqa: BLE001 — fail closed, never strand the wake
        return {"ok": False, "reason": f"evidence read failed ({type(exc).__name__})",
                "decided_sha": sha}
    if rc != 0:
        return {"ok": False, "reason": "pr view unavailable", "decided_sha": sha}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "pr view malformed", "decided_sha": sha}

    head = str(data.get("headRefOid") or "")
    reviews = data.get("reviews") or []
    review_sha = next((str(r.get("commit_id") or "") for r in reversed(reviews)
                       if r.get("body")), "")
    rollup = data.get("statusCheckRollup") or []
    failing = [c for c in rollup
               if str(c.get("conclusion") or "").upper() in {"FAILURE", "TIMED_OUT", "CANCELLED"}
               or str(c.get("state") or "").upper() in {"FAILURE", "ERROR"}]
    pending = [c for c in rollup
               if str(c.get("status") or "").upper() in {"IN_PROGRESS", "QUEUED", "PENDING", "WAITING"}
               or str(c.get("state") or "").upper() in {"PENDING", "EXPECTED"}]
    return {
        "ok": True,
        "decided_sha": sha,
        "head_sha": head,
        "review_sha": review_sha,
        "review_on_head": bool(review_sha) and (review_sha == sha or review_sha.startswith(sha)),
        "rollup_green": bool(rollup) and not failing and not pending,
        "failing_checks": len(failing),
        "pending_checks": len(pending),
    }


def merge_authority(install_auto_merge: bool, evidence: dict,
                    jev: JevAssessment) -> tuple[bool, str]:
    """Exact-SHA merge authority. Fails closed; never a veto by adjective."""
    if not install_auto_merge:
        return False, "auto-merge disabled for this install"
    if not evidence.get("ok"):
        return False, f"review evidence unavailable ({evidence.get('reason')})"
    if evidence.get("head_sha") != evidence.get("decided_sha"):
        return False, "head moved since the wake — stale evidence cannot authorize this SHA"
    if not evidence.get("review_on_head"):
        return False, "no review is bound to this exact head SHA"
    if not evidence.get("rollup_green"):
        return False, "required checks are not green on this head"
    blockers = jev.blockers
    if blockers:
        return False, "Jev blocking findings: " + ",".join(f.rule for f in blockers)
    return True, "exact-head evidence complete"


def _jev_for_pr(cfg: Config, repo: str, pr: int, sha: str = "") -> JevAssessment:
    """Read this PR's patch and review evidence for one cheap Jev decision."""
    if not isinstance(cfg.jev_api_key, str) or not cfg.jev_api_key:
        return JevAssessment(False, reason="no OpenRouter key")
    rc, diff = _gh(cfg, "pr", "diff", str(pr), "-R", repo, timeout=45)
    if rc != 0:
        return JevAssessment(False, reason="PR diff unavailable")
    rc, raw = _gh(cfg, "pr", "view", str(pr), "-R", repo,
                  "--json", "reviews,comments", timeout=45)
    if rc != 0:
        return JevAssessment(False, reason="review evidence unavailable")
    try:
        data = json.loads(raw)
        # GitHub status arrives after the review is published. The most recent
        # formal review is more relevant than concatenating old bot output.
        reviews = data.get("reviews") or []
        comments = data.get("comments") or []
        review = next((str(item.get("body")) for item in reversed(reviews)
                       if item.get("body")), "")
        if not review:
            review = next((str(item.get("body")) for item in reversed(comments)
                           if item.get("body")), "")
    except (TypeError, ValueError, AttributeError):
        return JevAssessment(False, reason="review evidence malformed")
    return jev_assess(diff, review, cfg.jev_api_key, head_sha=sha)


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
    if state.already_acted(repo, pr, sha, "triage", green=green):
        log.info("already triaged %s#%s@%s (green=%s) — skip (idempotent)",
                 repo, pr, sha[:8], green)
        return
    if state.already_acted(repo, pr, sha, "merged"):
        log.info("%s#%s@%s already merged by Neo — skip", repo, pr, sha[:8])
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
    install_auto_merge = bool(inst.get("auto_merge") or cfg.auto_merge_default)

    # --- exact-SHA evidence (Phase A) --------------------------------------
    # The wake carries the SHA the reviewer's status belongs to. If the PR head
    # has moved since, the old evidence may be shown as history but must not
    # open the merge gate: skip, and let the new head produce its own wake.
    evidence = _review_evidence(cfg, repo, pr, sha)
    if evidence.get("ok") and evidence.get("head_sha") and evidence["head_sha"] != sha:
        state.record(repo, pr, sha, "stale_head", verdict="STALE",
                     detail={"context": wake.get("context"),
                             "head_sha": evidence["head_sha"], "decided_sha": sha})
        log.info("%s#%s@%s head moved to %s — stale evidence, not mergeable (re-enter precheck)",
                 repo, pr, sha[:8], evidence["head_sha"][:8])
        return

    # Jev never supplies a green verdict. A *concrete* finding may remove
    # automatic merge authority; a risk adjective may not (Phase B).
    jev = _jev_for_pr(cfg, repo, pr, sha)
    auto_merge, authority_reason = merge_authority(install_auto_merge, evidence, jev)
    log.info("%s#%s@%s Jev available=%s risk=%s quality=%s partial=%s "
             "requires_full_review=%s blockers=%s",
             repo, pr, sha[:8], jev.available, jev.risk, jev.review_quality,
             jev.partial, jev.requires_full_review,
             ",".join(f.rule for f in jev.blockers) or "-")
    log.info("%s#%s@%s merge authority=%s (%s) head=%s review_sha=%s checks_green=%s",
             repo, pr, sha[:8], auto_merge, authority_reason,
             (evidence.get("head_sha") or "?")[:8], (evidence.get("review_sha") or "?")[:8],
             evidence.get("rollup_green"))

    state.record(repo, pr, sha, "triage", detail={"context": wake.get("context"),
                                                   "green": wake.get("green"),
                                                   "jev": jev.brief(),
                                                   "jev_attention": jev.requires_full_review,
                                                   "head_sha": evidence.get("head_sha"),
                                                   "review_sha": evidence.get("review_sha"),
                                                   "rollup_green": evidence.get("rollup_green"),
                                                   "jev_blockers": [f.rule for f in jev.blockers],
                                                   "auto_merge": auto_merge,
                                                   "reason": authority_reason})

    # --- run the Neo protocol headless via Claude Code CLI --------------------
    brief = neo_protocol.build_brief(
        repo=repo, pr=pr, head_sha=sha,
        reviewer_context=wake.get("context", "?"), reviewer_green=bool(wake.get("green")),
        auto_merge=auto_merge, builder_round=rounds,
        max_builder_rounds=cfg.max_builder_rounds, builder_model=cfg.deepseek_model,
        jev_context=jev.brief() + "\n" + evidence_brief(evidence, auto_merge, authority_reason),
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


def evidence_brief(evidence: dict, auto_merge: bool, reason: str) -> str:
    """Machine-readable evidence line handed to the Neo agent.

    The agent's prose must never outrank this: if it says merge authority is not
    granted, the agent may ACCEPT and label but must not merge.
    """
    if not evidence.get("ok"):
        return (f"MERGE_AUTHORITY: NOT_GRANTED — {reason}. Do not merge; continue non-merge work "
                "and record what blocked it.")
    head = (evidence.get("head_sha") or "?")[:12]
    review_sha = (evidence.get("review_sha") or "?")[:12]
    grant = "GRANTED" if auto_merge else "NOT_GRANTED"
    return (
        f"MERGE_AUTHORITY: {grant} ({reason}).\n"
        f"EVIDENCE head_sha={head} decided_sha={evidence.get('decided_sha', '?')[:12]} "
        f"review_sha={review_sha} review_on_head={evidence.get('review_on_head')} "
        f"checks_green={evidence.get('rollup_green')} "
        f"failing_checks={evidence.get('failing_checks')} pending_checks={evidence.get('pending_checks')}.\n"
        "Merge only when MERGE_AUTHORITY is GRANTED for this exact head SHA."
    )


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
