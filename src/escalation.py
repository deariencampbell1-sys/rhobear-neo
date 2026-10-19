"""Escalation — 5-bounce escalation to a frontier advisor.

When a PR has bounced NEO_BOUNCES_BEFORE_ESCALATION (default 5) builder rounds
across any head SHAs, Neo escalates to a frontier advisor on the build box
instead of a 6th cheap round.

Flow:
  1. Count builder rounds per PR across head SHAs.
  2. When round >= bounces_before_escalation, the PR goes to `escalating`.
  3. Neo calls the advisor on the build box over the tailnet:
     POST http://100.111.118.24:8787/v1/advise
     Body: {repo, pr, head_sha, rounds: [...], diff_stat, read_token}
     where `read_token` is a new installation token minted with
     permissions: {contents: read, pull_requests: read}.
  4. The advisor returns a work order (JSON).
  5. Neo turns the work order into a 6th Builder round whose brief is the
     work order plus the round history, marked `escalated`.
  6. If that round also bounces, or the advisor fails/times out, the PR is
     parked for the owner with a one-line reason.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import Config

log = logging.getLogger("rhobear_neo.escalation")


class EscalationError(Exception):
    """Base for escalation failures."""


class AdvisorUnavailable(EscalationError):
    """The advisor service could not be reached."""


class AdvisorRefused(EscalationError):
    """The advisor refused the request (auth, queue full, etc.)."""


class AdvisorFailed(EscalationError):
    """The advisor returned an error or malformed response."""


@dataclass
class RoundRecord:
    """One builder round in the escalation context."""
    round_no: int
    sha: str
    findings: str = ""
    builder_summary: str = ""
    outcome: str = ""


@dataclass
class WorkOrder:
    """The advisor's work order — a structured fix plan."""
    route: str = ""
    summary: str = ""
    root_cause: str = ""
    steps: list[dict] = None
    tests: list[str] = None
    risks: str = ""

    def __post_init__(self):
        self.steps = self.steps or []
        self.tests = self.tests or []

    def to_builder_brief(self, repo: str, pr: int, head_sha: str) -> str:
        """Convert the work order into a Builder brief string."""
        lines = [
            f"ESCALATED ADVISOR WORK ORDER for {repo}#{pr} @ {head_sha[:12]}",
            f"Route: {self.route}",
            f"Summary: {self.summary}",
            f"Root cause: {self.root_cause}",
            "",
            "Steps:",
        ]
        for i, step in enumerate(self.steps, 1):
            file = step.get("file", "")
            change = step.get("change", "")
            why = step.get("why", "")
            lines.append(f"  {i}. {file} — {change}")
            if why:
                lines.append(f"     Why: {why}")
        if self.tests:
            lines.append("")
            lines.append("Tests to add:")
            for t in self.tests:
                lines.append(f"  - {t}")
        if self.risks:
            lines.append("")
            lines.append(f"Risks: {self.risks}")
        return "\n".join(lines)


def count_builder_rounds(state, repo: str, pr: int) -> int:
    """Count total builder rounds for a PR across all head SHAs."""
    return state.builder_rounds(repo, pr)


def should_escalate(round_count: int, max_rounds: int, bounces_before: int) -> bool:
    """Return True if the PR should escalate instead of another cheap round.

    Escalation fires when:
      - We've exhausted max_builder_rounds (the agent would refuse to dispatch)
      - OR we've hit bounces_before_escalation bounces across heads
    """
    return round_count >= max_rounds or round_count >= bounces_before


def _read_token(cfg, repo: str, installation_id: int | None = None) -> str:
    """Mint a READ-ONLY installation token for the escalation advisor.

    This token has permissions: {contents: read, pull_requests: read}
    and is NEVER the write token used by the merge executor or builder.
    """
    from .github_app import repo_token as _repo_token
    read_perms = {"contents": "read", "pull_requests": "read"}
    return _repo_token(
        cfg, repo, installation_id=installation_id,
        permissions=read_perms,
    )


def call_advisor(
    cfg: Config,
    repo: str,
    pr: int,
    head_sha: str,
    rounds: list[RoundRecord],
    diff_stat: str,
    read_token: str,
) -> WorkOrder:
    """Call the advisor on the build box.

    POST http://<advisor_url>/v1/advise with bearer auth.
    Fail closed if advisor_url or advisor_token is unset.
    """
    url = cfg.advisor_url
    token = cfg.advisor_token
    if not url or not token:
        raise AdvisorUnavailable("NEO_ADVISOR_URL or NEO_ADVISOR_TOKEN not configured")

    body = json.dumps({
        "repo": repo,
        "pr": pr,
        "head_sha": head_sha,
        "rounds": [
            {
                "round": r.round_no,
                "sha": r.sha,
                "findings": r.findings,
                "builder_summary": r.builder_summary,
                "outcome": r.outcome,
            }
            for r in rounds
        ],
        "diff_stat": diff_stat,
        "read_token": read_token,
    }).encode("utf-8")

    req = urllib.request.Request(
        url.rstrip("/") + "/v1/advise",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise AdvisorRefused("advisor queue full (429)")
        raise AdvisorRefused(
            f"advisor returned HTTP {exc.code}: {exc.read()[:200]}"
        )
    except urllib.error.URLError as exc:
        raise AdvisorUnavailable(f"advisor unreachable: {exc.reason}")

    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise AdvisorFailed(f"advisor returned malformed JSON: {exc}")

    return WorkOrder(
        route=str(data.get("route", "")),
        summary=str(data.get("summary", "")),
        root_cause=str(data.get("root_cause", "")),
        steps=data.get("steps") or [],
        tests=data.get("tests") or [],
        risks=str(data.get("risks", "")),
    )


def escalate_pr(
    cfg: Config,
    state,
    jobs,
    repo: str,
    pr: int,
    head_sha: str,
    rounds: list[RoundRecord],
    diff_stat: str,
    installation_id: int | None = None,
) -> dict:
    """Escalate a PR to the frontier advisor.

    Returns the outcome dict:
      {"action": "escalated" | "parked", "reason": "...", "work_order": ...}

    Flow:
      1. Mint a read-only token for the advisor.
      2. Call the advisor.
      3. If the advisor returns a work order, create an escalated builder round.
      4. If the advisor fails, park the PR with a one-line reason.
    """
    try:
        read_token = _read_token(cfg, repo, installation_id)
    except Exception as exc:
        reason = f"cannot mint read token: {type(exc).__name__}"
        log.warning("%s#%s@%s %s", repo, pr, head_sha[:8], reason)
        return _park_pr(state, jobs, repo, pr, head_sha, reason)

    try:
        work_order = call_advisor(
            cfg, repo, pr, head_sha, rounds, diff_stat, read_token,
        )
    except EscalationError as exc:
        reason = f"advisor unavailable: {exc}"
        log.warning("%s#%s@%s %s", repo, pr, head_sha[:8], reason)
        return _park_pr(state, jobs, repo, pr, head_sha, reason)

    if not work_order.steps:
        reason = "advisor returned empty work order (no steps)"
        return _park_pr(state, jobs, repo, pr, head_sha, reason)

    # Turn the work order into an escalated builder round.
    brief = work_order.to_builder_brief(repo, pr, head_sha)
    new_round = count_builder_rounds(state, repo, pr) + 1

    state.record(repo, pr, head_sha, "escalated", verdict="ESCALATE",
                 builder_round=new_round,
                 detail={
                     "reason": "escalated to frontier advisor",
                     "work_order": {
                         "route": work_order.route,
                         "summary": work_order.summary,
                         "root_cause": work_order.root_cause,
                         "steps_count": len(work_order.steps),
                     },
                     "advisor_round": new_round,
                 })

    # Create a builder job for the escalated round.
    if jobs:
        jobs.create_builder(repo, pr, head_sha, new_round, payload={
            "verdict": "BOUNCE-BUILDER",
            "head_sha": head_sha,
            "phase": "escalated",
            "round": new_round,
            "context": "neo:escalated",
            "escalated": True,
            "work_order": {
                "route": work_order.route,
                "summary": work_order.summary,
                "root_cause": work_order.root_cause,
                "steps": work_order.steps,
                "tests": work_order.tests,
                "risks": work_order.risks,
            },
            "brief": brief,
            "installation_id": installation_id,
        })

    log.info("escalated %s#%s@%s route=%s round=%d steps=%d",
             repo, pr, head_sha[:8], work_order.route, new_round, len(work_order.steps))

    return {
        "action": "escalated",
        "reason": "escalated to frontier advisor",
        "work_order": {
            "route": work_order.route,
            "summary": work_order.summary,
            "root_cause": work_order.root_cause,
            "steps": work_order.steps,
            "tests": work_order.tests,
            "risks": work_order.risks,
        },
        "round": new_round,
    }


def _park_pr(state, jobs, repo, pr, head_sha, reason: str) -> dict:
    """Park a PR with a one-line reason (no more rounds)."""
    state.record(repo, pr, head_sha, "parked", verdict="PARKED",
                 detail={"reason": reason})
    log.warning("parked %s#%s@%s: %s", repo, pr, head_sha[:8], reason)
    return {"action": "parked", "reason": reason}
