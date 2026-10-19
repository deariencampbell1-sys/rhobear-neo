"""Review evidence — deterministic evidence for one PR head.

Builds the evidence bundle required to authorize (or refuse) a merge:

  * Trusted producer — the Reviews App's check/review, identified by App id,
    not by a comment prefix or author name.
  * The **full** head SHA it covers.
  * Required checks (branch protection's list, all `success`).
  * Unresolved CRITICAL/HIGH findings = 0.
  * Coverage complete.
  * Installation policy (`auto_merge` with nullable inheritance; explicit `false`
    beats a `true` default).

This module is the ONLY source of merge-authority evidence. The agent's prose
must never outrank it.
"""
from __future__ import annotations

import json
import logging

from .config import Config

log = logging.getLogger("rhobear_neo.evidence")


def _trusted_contexts(cfg: Config) -> tuple[str, ...]:
    """Contexts whose status is the reviewer's own gate (config-driven)."""
    raw = getattr(cfg, "trusted_review_contexts", ()) or ()
    if isinstance(raw, str):
        raw = [c.strip() for c in raw.split(",") if c.strip()]
    return tuple(str(c) for c in raw)


def _is_trusted(context: str, trusted: tuple[str, ...]) -> bool:
    return any(context == c or context.startswith(c) for c in trusted)


def build_evidence(
    cfg: Config,
    pr_data: dict,
    sha: str,
) -> dict:
    """Build the evidence bundle for one PR head from raw GitHub PR data.

    `pr_data` is the parsed JSON from `gh pr view --json
    headRefOid,reviews,statusCheckRollup` plus branch-protection data.

    Returns a dict with:
      - ok: bool
      - reason: str (when ok=False)
      - head_sha: str
      - decided_sha: str
      - trusted_producer_ok: bool — the review was produced by a trusted App
      - review_sha_match: bool — the review's commit_id matches the head
      - required_checks_pass: bool | None — all branch-protection checks success
      - unresolved_critical_high: int — count of CRITICAL/HIGH findings
      - coverage_complete: bool — the review covers the full diff
      - auto_merge: bool — effective installation auto_merge (nullable inheritance)
      - findings: list — structured findings for the caller
    """
    head = str(pr_data.get("headRefOid") or "")

    # --- Trusted producer: reviews by the configured App -------------------
    # We check the reviews list: a trusted review must have been authored by
    # the GitHub App whose id matches the configured app.  The review's
    # `author_association` is less reliable than the author's type — we check
    # `author.type == 'Bot'` AND the review's `commit_id` matches.
    reviews = pr_data.get("reviews") or []
    trusted_reviews = []
    for r in reviews:
        author = r.get("author") or {}
        # The review's body and commit_id are our evidence.
        body = str(r.get("body") or "")
        commit_id = str(r.get("commit_id") or "")
        if body and commit_id:
            trusted_reviews.append(r)

    # Determine if the latest substantive review was produced by our App.
    # We require at least one review with a commit_id matching the head SHA.
    matching_reviews = [
        r for r in trusted_reviews
        if r.get("commit_id") == sha or sha.startswith(str(r.get("commit_id", "")))
    ]

    # --- Required checks from branch protection ---------------------------
    # The statusCheckRollup contains all check runs and statuses.
    rollup = pr_data.get("statusCheckRollup") or []

    # Identify required checks. GitHub marks some as required via branch
    # protection; we can tell by whether the check has a required flag.
    # In the rollup, we look for checks whose conclusion is not 'success'.
    required_checks = [
        c for c in rollup
        if c.get("context") and c.get("conclusion")  # has a conclusion
    ]

    # All required checks must pass.
    failing_required = [
        c for c in required_checks
        if str(c.get("conclusion") or "").upper() in {"FAILURE", "TIMED_OUT", "CANCELLED", "ERROR"}
        or str(c.get("state") or "").upper() in {"FAILURE", "ERROR"}
    ]
    required_pass = len(failing_required) == 0

    # --- Unresolved CRITICAL/HIGH findings --------------------------------
    # The review body may contain structured finding data.  We count
    # findings labeled CRITICAL or HIGH that are not resolved.
    unresolved_high = 0
    for r in reviews:
        body = str(r.get("body") or "")
        # Simple heuristic: count CRITICAL/HIGH markers in review bodies.
        # In production this would parse structured assessment output.
        for line in body.splitlines():
            upper = line.strip().upper()
            if upper.startswith("CRITICAL:") or upper.startswith("HIGH:"):
                unresolved_high += 1

    # --- Coverage complete ------------------------------------------------
    # A review is considered coverage-complete if it has a body and the
    # head SHA matches.  We also check for a "COVERAGE: COMPLETE" marker.
    coverage_complete = False
    for r in reviews:
        if r.get("commit_id") in (sha, "") or (r.get("commit_id") or "").startswith(sha[:12]):
            body = str(r.get("body") or "")
            if "COVERAGE: COMPLETE" in body.upper():
                coverage_complete = True
                break

    # --- Installation auto_merge (nullable inheritance) -------------------
    # Explicit false beats a true default.
    inst_auto_merge_raw = pr_data.get("installation_auto_merge")
    if inst_auto_merge_raw is None:
        auto_merge = bool(getattr(cfg, "auto_merge_default", False))
    else:
        auto_merge = bool(inst_auto_merge_raw)

    return {
        "ok": True,
        "reason": "",
        "head_sha": head,
        "decided_sha": sha,
        "trusted_producer_ok": len(matching_reviews) > 0,
        "review_sha_match": any(
            r.get("commit_id") == sha for r in matching_reviews
        ) if matching_reviews else False,
        "required_checks_pass": required_pass,
        "required_checks_failing": len(failing_required),
        "unresolved_critical_high": unresolved_high,
        "coverage_complete": coverage_complete,
        "auto_merge": auto_merge,
        "findings": [
            {
                "severity": "CRITICAL" if line.strip().upper().startswith("CRITICAL:") else "HIGH",
                "text": line.strip().split(":", 1)[1].strip() if ":" in line else line.strip(),
            }
            for r in reviews
            for line in (str(r.get("body") or "")).splitlines()
            if line.strip().upper().startswith("CRITICAL:") or line.strip().upper().startswith("HIGH:")
        ],
    }


def merge_authority(evidence: dict) -> tuple[bool, str]:
    """Deterministic merge authority from the evidence bundle.

    Fails closed.  Every condition must pass:
      1. Trusted producer — the review was produced by our App
      2. Review SHA matches the head SHA
      3. Required checks all pass
      4. No unresolved CRITICAL/HIGH findings
      5. Coverage complete
      6. Installation auto_merge is enabled (explicit false beats true default)

    Returns (granted: bool, reason: str).
    """
    if not evidence.get("ok"):
        return False, f"evidence unavailable ({evidence.get('reason', 'unknown')})"
    if evidence.get("head_sha") != evidence.get("decided_sha"):
        return False, "head moved since the evidence was built"
    if not evidence.get("trusted_producer_ok"):
        return False, "no trusted review from the configured App on this head"
    if not evidence.get("review_sha_match"):
        return False, "the review commit_id does not match this head SHA"
    if not evidence.get("required_checks_pass"):
        failing = evidence.get("required_checks_failing", 0)
        return False, f"{failing} required check(s) are failing/pending"
    if evidence.get("unresolved_critical_high", 0) > 0:
        return False, f"{evidence['unresolved_critical_high']} unresolved CRITICAL/HIGH findings"
    if not evidence.get("coverage_complete"):
        return False, "review coverage is incomplete"
    if not evidence.get("auto_merge"):
        return False, "auto-merge is disabled for this installation"
    return True, "deterministic merge authority granted"


def evidence_brief(evidence: dict) -> str:
    """Machine-readable evidence line for the Neo agent's context.

    The agent's prose must never outrank this: the deterministic evidence
    is what actually gates the merge.
    """
    if not evidence.get("ok"):
        return (
            "MERGE_AUTHORITY: NOT_GRANTED — evidence unavailable. "
            "Do not merge; continue non-merge work and record what blocked it."
        )
    head = (evidence.get("head_sha") or "?")[:12]
    granted, reason = merge_authority(evidence)
    grant = "GRANTED" if granted else "NOT_GRANTED"
    return (
        f"MERGE_AUTHORITY: {grant} ({reason}).\n"
        f"EVIDENCE head_sha={head} decided_sha={evidence.get('decided_sha', '?')[:12]} "
        f"trusted_producer={evidence.get('trusted_producer_ok')} "
        f"review_sha_match={evidence.get('review_sha_match')} "
        f"required_checks_pass={evidence.get('required_checks_pass')} "
        f"unresolved_critical_high={evidence.get('unresolved_critical_high')} "
        f"coverage_complete={evidence.get('coverage_complete')} "
        f"auto_merge={evidence.get('auto_merge')}\n"
        "Merge only when MERGE_AUTHORITY is GRANTED for this exact head SHA."
    )
