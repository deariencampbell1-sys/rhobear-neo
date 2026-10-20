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
import secrets
import subprocess

from .config import Config
from .jobs import Jobs
from .neo_state import NeoState
from .agent_claude import ClaudeAgent, AgentError
from .github_app import AppNotConfigured, GitHubAppError, NoInstallation, repo_token
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


def _gh(token: str, *args: str, timeout: int = 30) -> tuple[int, str]:
    """Run `gh` as the repo's installation token (never a personal token)."""
    env = {**os.environ, "GH_TOKEN": token}
    p = subprocess.run(["gh", *args], capture_output=True, text=True, env=env, timeout=timeout)
    return p.returncode, (p.stdout or p.stderr).strip()


def resolve_pr(token: str, repo: str, sha: str, given: int | None) -> int | None:
    if given:
        return given
    rc, out = _gh(token, "api", f"repos/{repo}/commits/{sha}/pulls",
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


def _compute_diff_stat(token: str, repo: str, pr: int) -> str:
    """Compute a diff stat for the PR from the pull request API.

    Returns a string like '3 files changed, +120/-40' or empty on failure.
    (gh has no `pr diff --stat`; the stat comes from GET /repos/{repo}/pulls/{pr}
    -> changed_files/additions/deletions.)
    """
    jq = '"\\(.changed_files) files changed, +\\(.additions)/-\\(.deletions)"'
    try:
        rc, out = _gh(token, "api", f"repos/{repo}/pulls/{pr}", "--jq", jq,
                      timeout=30)
        if rc == 0 and out:
            return out.strip()[:2000]
    except Exception:
        log.warning("diff_stat failed for %s#%s", repo, pr)
    return ""


def _trusted_contexts(cfg: Config) -> tuple[str, ...]:
    """Contexts whose status is the reviewer's own gate (config-driven)."""
    raw = getattr(cfg, "trusted_review_contexts", ()) or ()
    if isinstance(raw, str):
        raw = [c.strip() for c in raw.split(",") if c.strip()]
    return tuple(str(c) for c in raw)


def _is_trusted(context: str, trusted: tuple[str, ...]) -> bool:
    return any(context == c or context.startswith(c) for c in trusted)


def _review_evidence(token: str, cfg: Config, repo: str, pr: int, sha: str) -> dict:
    """Machine-readable review evidence bound to ONE commit, not to the PR.

    A result produced for commit A may authorize only commit A.  The trusted
    producer is the Reviews App's CHECK RUN on exactly `sha` (GET
    /repos/{repo}/commits/{sha}/check-runs): only a check run whose `app.id`
    equals NEO_TRUSTED_REVIEWER_APP_ID and whose `head_sha` equals `sha`
    (full 40-char equality) counts.  A review or status from any other
    account (including the PR author) is ignored.  Unresolved CRITICAL/HIGH
    and coverage come from that run's output.summary JSON — never hard-coded.
    """
    try:
        rc, raw = _gh(token, "pr", "view", str(pr), "-R", repo,
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

    # Trusted producer: check runs for EXACTLY this sha.  Only the run from
    # the configured Reviews App on this exact head authorizes a merge.
    try:
        rc, raw_checks = _gh(token, "api",
                             f"repos/{repo}/commits/{sha}/check-runs", timeout=30)
        if rc == 0:
            data["check_runs"] = (json.loads(raw_checks).get("check_runs") or [])
        else:
            data["check_runs"] = []
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"check-runs read failed ({type(exc).__name__})",
                "decided_sha": sha}

    reviews = data.get("reviews") or []
    review_sha = next((str(r.get("commit_id") or "") for r in reversed(reviews)
                       if r.get("body")), "")

    # The reviewer's gate, read straight off the commit (context-based legacy).
    trusted = _trusted_contexts(cfg)
    trusted_states: list[str] = []
    try:
        rc, raw_status = _gh(token, "api", f"repos/{repo}/commits/{sha}/status", timeout=30)
        if rc == 0:
            for item in (json.loads(raw_status).get("statuses") or []):
                if _is_trusted(str(item.get("context") or ""), trusted):
                    trusted_states.append(str(item.get("state") or "").lower())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"status read failed ({type(exc).__name__})",
                "decided_sha": sha}
    if not trusted:
        return {"ok": False, "reason": "no trusted review contexts configured",
                "decided_sha": sha}

    rollup = data.get("statusCheckRollup") or []
    failing = [c for c in rollup
               if str(c.get("conclusion") or "").upper() in {"FAILURE", "TIMED_OUT", "CANCELLED"}
               or str(c.get("state") or "").upper() in {"FAILURE", "ERROR"}]
    # Compute rollup_green from the rollup data (defect 5 fix).
    all_success = all(
        str(c.get("conclusion") or "").upper() == "SUCCESS"
        or str(c.get("state") or "").upper() == "SUCCESS"
        for c in rollup
    ) if rollup else None

    # Deterministic evidence (trusted App check run + output summary) plus
    # the legacy keys the rest of the worker still consumes.
    from . import review_evidence as rev_ev
    ev = rev_ev.build_evidence(cfg, data, sha)
    ev["review_sha"] = review_sha
    ev["review_bound_ok"] = ev.get("review_sha_match", True)
    ev["trusted_states"] = trusted_states
    ev["trusted_green"] = bool(trusted_states) and all(s == "success" for s in trusted_states)
    ev["failing_checks"] = ev.get("required_checks_failing", len(failing))
    ev["rollup_green"] = all_success
    return ev


def merge_authority(install_auto_merge: bool, evidence: dict,
                    jev: JevAssessment) -> tuple[bool, str]:
    """Exact-SHA merge authority. Fails closed; never a veto by adjective."""
    if not install_auto_merge:
        return False, "auto-merge disabled for this install"
    if not evidence.get("ok"):
        return False, f"review evidence unavailable ({evidence.get('reason')})"
    if evidence.get("head_sha") != evidence.get("decided_sha"):
        return False, "head moved since the wake — stale evidence cannot authorize this SHA"
    if not evidence.get("review_bound_ok", True):
        return False, "the reviewed commit is not this head SHA"
    if not evidence.get("trusted_green"):
        return False, "the trusted review status is not success on this exact head SHA"
    # Trusted producer = the Reviews App's check run on this exact head
    # (item 3).  A dict without the key (older patched evidence) keeps the
    # legacy behavior; an explicit False means no trusted run -> no merge.
    if evidence.get("trusted_producer_ok") is False:
        return False, "no trusted check run from the Reviews App on this exact head"
    blockers = jev.blockers
    if blockers:
        return False, "Jev blocking findings: " + ",".join(f.rule for f in blockers)
    return True, "exact-head evidence complete"


def _jev_for_pr(token: str, cfg: Config, repo: str, pr: int, sha: str = "") -> JevAssessment:
    """Read this PR's patch and review evidence for one cheap Jev decision."""
    if not isinstance(cfg.jev_api_key, str) or not cfg.jev_api_key:
        return JevAssessment(False, reason="no OpenRouter key")
    rc, diff = _gh(token, "pr", "diff", str(pr), "-R", repo, timeout=45)
    if rc != 0:
        return JevAssessment(False, reason="PR diff unavailable")
    rc, raw = _gh(token, "pr", "view", str(pr), "-R", repo,
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
        f"review_sha={review_sha} review_bound_ok={evidence.get('review_bound_ok')} "
        f"trusted_review_status={evidence.get('trusted_states')} "
        f"(other checks in this repo: {evidence.get('failing_checks')} failing, informational).\n"
        "Merge only when MERGE_AUTHORITY is GRANTED for this exact head SHA."
    )


def _run_agent(agent: ClaudeAgent, brief: str) -> tuple[dict, str]:
    """Run the Neo brief headless via Claude Code CLI.

    Claude Code runs with isolated config dir + per-run temp workdir, giving
    the agent full tool access (gh, git, edit, test runner).  DeepSeek Direct
    provides the Anthropic-compatible endpoint.

    Returns (normalised usage, verdict line).  AgentError is caught here (the
    caller retries the job); any OTHER exception from the agent spawn is
    allowed to propagate so the caller's broad handler releases the job too
    (defect 3: FileNotFoundError/OSError/UnicodeDecodeError must never strand
    the SHA).
    """
    try:
        usage, verdict = agent.run(brief)
        return usage, verdict
    except AgentError:
        log.exception("neo claude agent run failed")
        return {}, ""


def _wake_lease(cfg: Config) -> int:
    """Wake triage leases must cover the whole agent run (a 600 s default lease
    with a 30-minute agent would expire mid-run and let a second worker claim
    the same SHA)."""
    timeout = int(getattr(cfg, "deepseek_timeout", 1800) or 1800)
    return max(600, timeout + 120)


def run_neo(cfg: Config, state: NeoState, jobs: Jobs, wake: dict) -> None:
    repo, sha = wake["repo"], wake["sha"]
    green = bool(wake.get("green"))

    # --- durable wake job -------------------------------------------------
    # on_verdict() already persisted the job (keyed by repo + head sha +
    # verdict color, PR resolved later).  If we were called without a job row
    # (tests / manual replay), create it here — dedupe-on-outcome means a
    # terminal or expired job is re-armed, an active one is left alone.
    job_id = wake.get("job_id")
    if not job_id:
        job_id = jobs.create_wake(repo, sha, green,
                                  pr_number=wake.get("pr_number"), payload=wake)
        if job_id is None:
            log.info("deduped wake %s@%s — an active job already owns it", repo, sha[:8])
            return

    # --- GitHub identity for THIS repo (the customer's App installation) ---
    # Neo acts as the GitHub App: every GitHub call below uses this repo's
    # installation token, never a personal token. A repo with no App
    # installation is refused with a clear log line and the job resolves
    # `no_installation` — terminal, not retried (owner re-arms by installing
    # the App on the repo).
    try:
        token = repo_token(cfg, repo, wake.get("installation_id"))
    except (NoInstallation, AppNotConfigured) as exc:
        log.warning("refusing wake %s@%s: %s — no GitHub App installation "
                    "(no_installation)", repo, sha[:8], exc)
        wtoken = secrets.token_hex(16)
        if jobs.claim(job_id, pr_number=wake.get("pr_number"),
                      lease_secs=_wake_lease(cfg), worker_token=wtoken):
            jobs.resolve(job_id, "no_installation", worker_token=wtoken,
                         payload={"reason": f"no GitHub App installation for {repo}"})
        return
    except GitHubAppError as exc:
        # Transient GitHub API failure (5xx, timeout, token-mint error).
        # Before the claim, nothing else would release this job — its
        # submitted_at stamp hides it from the sweep.  Release it retryably.
        log.warning("transient GitHub error for %s@%s: %s — job released retryably",
                    repo, sha[:8], exc)
        wtoken = secrets.token_hex(16)
        if jobs.claim(job_id, pr_number=wake.get("pr_number"),
                      lease_secs=_wake_lease(cfg), worker_token=wtoken):
            jobs.retry(job_id, worker_token=wtoken,
                       reason=f"transient GitHub identity error ({type(exc).__name__})")
        return

    try:
        pr = resolve_pr(token, repo, sha, wake.get("pr_number"))
    except Exception as exc:
        # gh hung (TimeoutExpired) or is missing: before the claim, nothing else would
        # release this job — its submitted_at stamp hides it from the sweep. Release it
        # retryably, exactly like the no-PR branch below.
        log.warning("resolve_pr crashed (%s) for %s@%s — job released retryably",
                    type(exc).__name__, repo, sha[:8])
        wtoken = secrets.token_hex(16)
        if jobs.claim(job_id, pr_number=None, lease_secs=_wake_lease(cfg),
                      worker_token=wtoken):
            jobs.retry(job_id, worker_token=wtoken,
                       reason=f"pre-claim crash ({type(exc).__name__})")
        return
    if not pr:
        # Transient gh failure, or a commit status that fired before any PR
        # exists.  Release the job retryably (claim-then-retry: claim() marks
        # the run so the attempt cap bounds it, retry() returns it to queued +
        # outcome='failed').  The recovery loop re-submits it; after the cap the
        # job parks and a future wake re-arms it — the SHA is never permanently
        # poisoned (defect 5 / GLM 5).
        log.info("no open PR for %s@%s — job released retryably", repo, sha[:8])
        wtoken = secrets.token_hex(16)
        if jobs.claim(job_id, pr_number=None, lease_secs=_wake_lease(cfg),
                      worker_token=wtoken):
            jobs.retry(job_id, worker_token=wtoken, reason="no open PR for head sha")
        else:
            log.info("job %s already claimed for %s@%s — another worker resolves PR",
                     job_id, repo, sha[:8])
        return

    # --- claim the durable job (dedupe + triage in one transaction) -------
    # Atomic UPDATE ... WHERE state='queued' — if another worker already
    # claimed it (concurrent re-submit), we skip.  The worker_token is our
    # claimant identity: every resolve/retry/lease below must present it, so a
    # stale claimant (lease lost under a DB outage) can never mutate a
    # replacement run.
    wtoken = secrets.token_hex(16)
    if not jobs.claim(job_id, pr_number=pr, lease_secs=_wake_lease(cfg),
                      worker_token=wtoken):
        log.info("job already claimed %s job=%s — skip (deduped)", repo, job_id)
        return

    # --- per-SHA idempotency gate (integrator finding 2 / defect 2) -------
    # A duplicate delivery of the same verdict (GitHub redelivery, or a
    # reviewer re-publishing success on the same SHA — both normal) is re-armed
    # by create_wake (dedupe on OUTCOME, so the verdict is never dropped).  But
    # re-running the whole triage would re-debit credits, re-merge an
    # already-merged PR, and mint an extra builder round.  The color-aware
    # already_acted gate makes the re-arm idempotent.  The triage row is
    # recorded only AFTER a completed agent run (see below), so a retried job
    # whose agent crashed mid-run is NOT gated — it re-runs — while a verdict
    # that actually completed is treated as already-acted.
    if state.already_acted(repo, pr, sha, "triage", green=green) or \
            state.already_acted(repo, pr, sha, "merged"):
        jobs.resolve(job_id, "no_change", worker_token=wtoken,
                     payload={"reason": "already acted on this SHA/verdict"})
        log.info("%s#%s@%s already acted (triage green=%s or merged) — "
                 "duplicate delivery skipped", repo, pr, sha[:8], green)
        return
    # A previous run of this verdict did its work (phase row, builder job) and then died
    # before writing the triage marker. Re-running the agent would mint a second builder
    # round; finish the bookkeeping instead.
    for done_phase in ("builder", "ready", "fix_forward", "escalated",
                       "escalating", "builder_blocked", "blocked"):
        if state.already_acted(repo, pr, sha, done_phase, green=green):
            state.record(repo, pr, sha, "triage",
                         detail={"green": wake.get("green"), "recovered": done_phase})
            jobs.resolve(job_id, "done", worker_token=wtoken,
                         payload={"reason": f"{done_phase} already recorded; triage marker restored"})
            log.info("%s#%s@%s %s already recorded — triage marker restored, no re-run",
                     repo, pr, sha[:8], done_phase)
            return

    rounds = state.builder_rounds(repo, pr)

    # --- entitlement (per-install credit balance) ---------------------------
    org = repo.split("/", 1)[0]
    inst = state.install(org, org)          # dogfood: install keyed by org
    if inst.get("plan") and inst.get("credits_balance", 0) <= 0:
        state.record(repo, pr, sha, "parked", verdict="PARKED",
                     detail={"reason": "out of credits — owner must top up"})
        _gh(token, "pr", "comment", str(pr), "-R", repo,
            "--body", "Neo: this install is out of credits — top up to resume auto-fix/merge.")
        jobs.park(job_id, reason="out of credits — owner must top up")
        log.info("%s#%s: install out of credits — gated (parked)", repo, pr)
        return
    # P0 hotfix: nullable inheritance for auto-merge.
    inst_auto_merge = inst.get("auto_merge")
    if inst_auto_merge is None:
        install_auto_merge = bool(cfg.auto_merge_default)
    else:
        install_auto_merge = bool(inst_auto_merge)

    # --- exact-SHA evidence (Phase A) --------------------------------------
    evidence = _review_evidence(token, cfg, repo, pr, sha)
    if evidence.get("ok") and evidence.get("head_sha") and evidence["head_sha"] != sha:
        state.record(repo, pr, sha, "stale_head", verdict="STALE",
                     detail={"context": wake.get("context"),
                             "head_sha": evidence["head_sha"], "decided_sha": sha})
        jobs.resolve(job_id, "no_change", worker_token=wtoken,
                     payload={"reason": "stale head"})
        log.info("%s#%s@%s head moved to %s — stale evidence, not mergeable",
                 repo, pr, sha[:8], evidence["head_sha"][:8])
        return

    # Jev never supplies a green verdict.
    jev = _jev_for_pr(token, cfg, repo, pr, sha)
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

    # --- run the Neo protocol headless via Claude Code CLI --------------------
    # The triage row is recorded AFTER a completed run (below, with the phase
    # row), NOT here: a triage that crashes mid-agent must not leave a marker
    # that the per-SHA idempotency gate would then treat as already-acted on
    # the retry.  Recording it here is what stranded heads in the B0 baseline
    # (the old code had to clear() the marker on agent failure).  With the
    # durable job store the retry is the job's own lease cycle, so the marker
    # is only meaningful once a run actually completes.
    brief = neo_protocol.build_brief(
        repo=repo, pr=pr, head_sha=sha,
        reviewer_context=wake.get("context", "?"), reviewer_green=bool(wake.get("green")),
        auto_merge=auto_merge, builder_round=rounds,
        max_builder_rounds=cfg.max_builder_rounds, builder_model=cfg.deepseek_model,
        jev_context=jev.brief() + "\n" + evidence_brief(evidence, auto_merge, authority_reason),
    )
    # The agent gets the repo's installation token, never a personal one.
    agent = ClaudeAgent.from_config(cfg, gh_token=token)
    try:
        # Lease renewal: a static lease shorter than the agent run would let a
        # second worker claim the same SHA mid-run (GLM 6 applies to wakes too).
        with jobs.lease_renewal(job_id, lease_secs=_wake_lease(cfg),
                                worker_token=wtoken):
            usage, verdict = _run_agent(agent, brief)
    except Exception as exc:
        # Any exception from the agent spawn (FileNotFoundError/OSError/
        # UnicodeDecodeError included) releases the job for retry instead of
        # stranding the SHA (defect 3 / GLM 4).
        log.exception("%s#%s@%s agent crashed (%s) — job retried",
                      repo, pr, sha[:8], type(exc).__name__)
        jobs.retry(job_id, worker_token=wtoken,
                   reason=f"agent crashed ({type(exc).__name__})")
        return

    if not verdict:
        log.warning("%s#%s@%s agent produced no verdict — job retried", repo, pr, sha[:8])
        jobs.retry(job_id, worker_token=wtoken, reason="agent produced no verdict")
        return

    # --- B2: Merge executor (deterministic, not agent) --------------------
    # The agent NEVER holds merge credentials.  On ACCEPT/ACCEPT-MERGED we
    # call merge_executor.merge() which re-reads the head, checks evidence,
    # and merges with expected SHA.
    verdict_base = verdict.split()[0] if verdict else ""
    is_accept = verdict_base in ("ACCEPT", "ACCEPT-MERGED", "ACCEPT-READY")
    is_builder = verdict_base == "BOUNCE-BUILDER"
    advice_id = ""  # set when the escalation path submits to the advisor

    if is_accept:
        # Use the evidence already fetched by _review_evidence above.
        # Convert the old evidence dict to the new format for merge_authority.
        # Item 4: unresolved_critical_high and coverage_complete come from the
        # trusted check run's output.summary — nothing is hard-coded here;
        # missing -> merge denied.
        from . import review_evidence as rev_ev
        evidence_b2 = {
            "ok": evidence.get("ok", False),
            "reason": evidence.get("reason", ""),
            "head_sha": evidence.get("head_sha", ""),
            "decided_sha": evidence.get("decided_sha", sha),
            "trusted_producer_ok": evidence.get(
                "trusted_producer_ok", evidence.get("trusted_green", False)),
            "review_sha_match": evidence.get(
                "review_sha_match", evidence.get("review_bound_ok", False)),
            "required_checks_pass": (
                evidence.get("rollup_green") is not False
            ),
            "required_checks_failing": evidence.get("failing_checks", 0),
            # From the trusted check run's output — not hard-coded.  Missing
            # (None) is denied by merge_authority.
            "unresolved_critical_high": evidence.get("unresolved_critical_high"),
            "coverage_complete": evidence.get("coverage_complete", False),
            "auto_merge": install_auto_merge,
        }
        granted, reason = rev_ev.merge_authority(evidence_b2)

        if granted and auto_merge and verdict_base in ("ACCEPT", "ACCEPT-MERGED"):
            # Call the deterministic merge executor.
            from . import merge_executor
            try:
                merge_result = merge_executor.merge(
                    repo, pr, sha, evidence_b2, token,
                )
                phase = "merged"
                if merge_result.get("merged"):
                    log.info("%s#%s@%s merged by executor (sha=%s)",
                             repo, pr, sha[:8], merge_result.get("sha", "?")[:12])
            except merge_executor.HeadMoved as exc:
                log.warning("%s#%s@%s head moved before merge: %s",
                            repo, pr, sha[:8], exc)
                phase = "stale_head"
            except merge_executor.MergeDenied as exc:
                log.warning("%s#%s@%s merge denied: %s",
                            repo, pr, sha[:8], exc)
                phase = "ready"  # mark as ready but not merged
            except merge_executor.MergeError as exc:
                log.warning("%s#%s@%s merge failed: %s",
                            repo, pr, sha[:8], exc)
                phase = "ready"
        else:
            phase = "ready"
            if not auto_merge:
                log.info("%s#%s@%s ACCEPT but auto-merge disabled — ready",
                         repo, pr, sha[:8])
            else:
                log.info("%s#%s@%s ACCEPT but evidence blocks merge: %s",
                         repo, pr, sha[:8], reason)

    elif is_builder:
        # --- B2: Escalation check before dispatching builder ---------------
        rounds = state.builder_rounds(repo, pr)
        bounces_before = getattr(cfg, "bounces_before_escalation", 5)
        if state.already_escalated(repo, pr):
            # This PR already had its ONE escalated round; a further bounce
            # parks it for the owner — no second escalation, no more rounds.
            log.info("%s#%s@%s already escalated — bounce parks the PR",
                     repo, pr, sha[:8])
            phase = "parked"
            state.record(repo, pr, sha, "parked", verdict="PARKED",
                         detail={"reason": "escalated builder round bounced again"})
        elif rounds >= bounces_before:
            log.info("%s#%s@%s %d builder rounds — escalating to advisor",
                     repo, pr, sha[:8], rounds)
            from . import escalation
            try:
                round_records = [
                    escalation.RoundRecord(
                        round_no=r, sha=sha, outcome="bounced",
                    )
                    for r in range(1, rounds + 1)
                ]
                # Async: submit returns immediately (advisor runs on the build
                # box); the recovery loop polls and creates the ONE escalated
                # round when the work order is done.
                outcome = escalation.submit_escalation(
                    cfg, state, jobs, repo, pr, sha,
                    rounds=round_records,
                    diff_stat=_compute_diff_stat(token, repo, pr),
                    installation_id=wake.get("installation_id"),
                )
                if outcome["action"] == "escalating":
                    phase = "escalating"
                    advice_id = outcome.get("advice_id", "")
                else:
                    phase = "parked"
                    advice_id = ""
            except Exception as exc:
                log.warning("%s#%s@%s escalation failed: %s — parking",
                            repo, pr, sha[:8], exc)
                phase = "parked"
                state.record(repo, pr, sha, "parked", verdict="PARKED",
                             detail={"reason": f"escalation failed: {exc}"})
        else:
            # Normal builder dispatch.
            phase = "builder"
            jobs.create_builder(repo, pr, sha, rounds + 1, payload={
                "verdict": verdict, "wake_job": job_id,
                "head_sha": sha, "phase": "builder", "round": rounds + 1,
                "context": wake.get("context"),
                "installation_id": wake.get("installation_id"),
            })
    else:
        # FIXED/NO-CHANGE/BLOCKED from a builder-style agent, plus
        # FIX-FORWARD, ESCALATE, and unknown verdicts.
        # Builder contract: FIXED → wait for new head's review;
        # NO-CHANGE → counts as a bounce (ready); BLOCKED → park with reason.
        # Unknown verdicts → BLOCKED: unparseable, never silently dropped.
        phase = {
            "FIXED": "ready",
            "NO-CHANGE": "ready",
            # Not "parked": credits/advisor parks share that phase and are
            # transient — the recovery gate must not treat them as acted.
            "BLOCKED": "builder_blocked",
            "FIX-FORWARD": "fix_forward",
            "ESCALATE": "escalated",
        }.get(verdict_base, "blocked")

    credits = credits_for(usage, builder=is_builder)
    new_round = (rounds + 1) if is_builder else rounds

    # The phase row carries the verdict colour so a retry after a crash can tell this
    # verdict's work is already done (see the gate above).  An `escalating` row
    # carries the advisor advice_id (submit_escalation already recorded one with
    # the full detail; this row keeps the advice_id too so the recovery loop's
    # newest-first escalating_prs() always finds it) and the webhook
    # installation_id so the escalated round uses the id Neo already had instead
    # of a repo->installation lookup (defect 4).  It is recorded under the
    # per-PR advisory lock (record_escalating) so a concurrent take_escalation
    # can never clobber it (defect 1).
    if phase == "escalating":
        state.record_escalating(repo, pr, sha, verdict=verdict,
                                builder_round=new_round, credits=credits,
                                advice_id=advice_id,
                                installation_id=wake.get("installation_id"),
                                extra={"green": wake.get("green"),
                                       "wake_job": job_id})
    else:
        state.record(repo, pr, sha, phase, verdict=verdict,
                     builder_round=new_round, credits=credits,
                     detail={"green": wake.get("green"), "wake_job": job_id})
    # The triage marker goes LAST: it vouches that the phase row and any builder job above
    # exist. A crash before this line leaves the phase row, which the gate above detects.
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

    # Resolve the wake job with a terminal outcome.
    jobs.resolve(job_id, "done", worker_token=wtoken,
                 payload={"verdict": verdict, "phase": phase,
                          "credits": credits})
    if inst.get("plan"):
        state.debit(org, credits)
    log.info("%s#%s@%s verdict=%s phase=%s credits=%d (auto_merge=%s round=%d)",
             repo, pr, sha[:8], verdict or "?", phase, credits, auto_merge, rounds)



