"""Review evidence — deterministic evidence for one PR head.

Builds the evidence bundle required to authorize (or refuse) a merge:

  * Trusted producer — the Reviews App's check run, identified by `app.id`
    (`NEO_TRUSTED_REVIEWER_APP_ID`), NOT by a comment prefix or author name.
    A review or status from any other account (including the PR author) is
    ignored.
  * The **full** head SHA it covers — `head_sha` full 40-char equality with
    the exact `sha` being decided.
  * Required checks (branch protection's list, all `success`).
  * Unresolved CRITICAL/HIGH findings and coverage come from the trusted check
    run's `output.summary` JSON (`{"unresolved_critical_high": n,
    "coverage": "complete|partial"}`); missing → merge denied.  Nothing is
    hard-coded.
  * Installation policy (`auto_merge` with nullable inheritance; explicit
    `false` beats a `true` default).

This module is the ONLY source of merge-authority evidence. The agent's prose
must never outrank it.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging

from .config import Config

log = logging.getLogger("rhobear_neo.evidence")


def _trusted_app_id(cfg: Config) -> str:
    """The Reviews App id whose check runs authorize a merge (fail closed)."""
    return str(getattr(cfg, "trusted_reviewer_app_id", "") or "").strip()


def _parse_evidence_summary(summary: str) -> dict | None:
    """Parse the trusted check run's output.summary JSON."""
    if not summary:
        return None
    try:
        data = json.loads(summary)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _check_run_completed(cr: dict) -> bool:
    """True only for a check run that actually COMPLETED.

    The production predicate is exact: `status == "completed"`.  A
    queued/in_progress/pending run is not evidence, and a payload carrying a
    `conclusion` but no `status` is NOT completed either (W1-13-r2 defect 4:
    GitHub's check-run schema always includes `status`; the pinned fixtures
    that omitted it were wrong and have been fixed — the predicate never
    bends to test data).
    """
    return str(cr.get("status") or "") == "completed"


def _run_id(cr: dict) -> int:
    """Coerce a check run id defensively (W1-13-r2 defect 10): a malformed
    id must SORT — it may never raise a ValueError on the evidence path."""
    try:
        return int(cr.get("id") or 0)
    except (TypeError, ValueError):
        return 0


def _completed_at(cr: dict) -> _dt.datetime:
    """Parse completed_at to an aware datetime before comparing (defect 10):
    mixed `Z` / `+00:00` strings sort wrongly as text.  Missing/unparseable
    sorts oldest (deterministic; the id tie-break still applies)."""
    raw = cr.get("completed_at")
    if isinstance(raw, str) and raw.strip():
        text = raw.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = _dt.datetime.fromisoformat(text)
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_dt.timezone.utc)
            return parsed
    return _dt.datetime.min.replace(tzinfo=_dt.timezone.utc)


def _trusted_check_run(check_runs: list, app_id: str, sha: str) -> dict | None:
    """The ONE check run that authorizes `sha`.

    Trusts only a check run whose `app.id` equals the configured Reviews App
    id AND whose `head_sha` equals `sha` (full 40-char equality) AND that
    COMPLETED successfully (conclusion in success/neutral).  When several
    runs match, the NEWEST (by completed_at, then by id) decides — never the
    first in the list, and never a fallback to an older run when the newest
    one is failing or still pending.
    """
    matches = []
    for cr in check_runs or []:
        cr_app = (cr.get("app") or {}) or {}
        if str(cr_app.get("id", "") or "") == str(app_id) \
                and str(cr.get("head_sha", "") or "") == sha:
            matches.append(cr)
    if not matches:
        return None
    cr = max(matches, key=lambda r: (_completed_at(r), _run_id(r)))
    # The newest matching run decides; if IT is not acceptable evidence,
    # there is no evidence (never fall back to an older run).
    if not _check_run_completed(cr):
        return None
    if str(cr.get("conclusion") or "").lower() not in ("success", "neutral"):
        return None
    return cr


def build_evidence(
    cfg: Config,
    pr_data: dict,
    sha: str,
) -> dict:
    """Build the evidence bundle for one PR head.

    `pr_data` is the parsed JSON from `gh pr view --json
    headRefOid,reviews,statusCheckRollup` plus `check_runs` (the list from
    `GET /repos/{repo}/commits/{sha}/check-runs` — fetched for exactly `sha`).

    Returns a dict with:
      - ok: bool (False when the trusted producer is not configured)
      - reason: str (when ok=False)
      - head_sha: str
      - decided_sha: str
      - trusted_producer_ok: bool — a check run from the Reviews App on this
        exact head SHA exists
      - review_sha_match: bool — full 40-char head equality
      - required_checks_pass: bool | None — all branch-protection checks
        success; None when there is no rollup data at all (unknown, denied)
      - unresolved_critical_high: int | None — from the trusted run's output
        (None when missing → denied)
      - coverage_complete: bool — output coverage == "complete"
      - auto_merge: bool — effective installation auto_merge (nullable
        inheritance)
      - findings: list — informational (structured findings)
    """
    head = str(pr_data.get("headRefOid") or "")

    # --- Trusted producer: fail closed when the App id is unset ------------
    trusted_app = _trusted_app_id(cfg)
    if not trusted_app:
        return {
            "ok": False,
            "reason": ("no trusted reviewer app id configured "
                       "(NEO_TRUSTED_REVIEWER_APP_ID unset)"),
            "head_sha": head,
            "decided_sha": sha,
        }

    # --- The trusted check run for EXACTLY this sha -----------------------
    check_runs = pr_data.get("check_runs") or []
    trusted = _trusted_check_run(check_runs, trusted_app, sha)
    trusted_producer_ok = trusted is not None

    # --- Unresolved CRITICAL/HIGH + coverage from that run's output --------
    unresolved_critical_high: int | None = None
    coverage_complete = False
    if trusted is not None:
        output = trusted.get("output") or {}
        summary = _parse_evidence_summary(str(output.get("summary") or ""))
        if summary is not None:
            raw_unresolved = summary.get("unresolved_critical_high")
            if isinstance(raw_unresolved, int) and not isinstance(raw_unresolved, bool):
                unresolved_critical_high = raw_unresolved
            coverage = str(summary.get("coverage") or "").lower()
            coverage_complete = coverage == "complete"

    # --- Required checks from branch protection (all success) --------------
    rollup = pr_data.get("statusCheckRollup") or []
    failing_required = [
        c for c in rollup
        if str(c.get("conclusion") or "").upper()
        in {"FAILURE", "TIMED_OUT", "CANCELLED", "ERROR"}
        or str(c.get("state") or "").upper() in {"FAILURE", "ERROR"}
    ]
    # Fail closed: an empty rollup is NO data (unknown -> None, matching
    # merge_authority's treatment of unknown findings), never a pass.
    required_pass = (len(failing_required) == 0) if rollup else None

    # --- Installation auto_merge (nullable inheritance) --------------------
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
        "trusted_producer_ok": trusted_producer_ok,
        "review_sha_match": trusted_producer_ok,   # full-head equality by construction
        "required_checks_pass": required_pass,
        "required_checks_failing": len(failing_required),
        "unresolved_critical_high": unresolved_critical_high,
        "coverage_complete": coverage_complete,
        "auto_merge": auto_merge,
        "findings": [],
    }


def merge_authority(evidence: dict) -> tuple[bool, str]:
    """Deterministic merge authority from the evidence bundle.

    Fails closed.  Every condition must pass:
      1. Evidence available (trusted producer configured)
      2. Trusted producer — a check run from the Reviews App on this head
      3. Review SHA matches the head SHA (full 40-char equality)
      4. Required checks all pass
      5. Unresolved CRITICAL/HIGH present in the output AND == 0
      6. Coverage complete
      7. Installation auto_merge is enabled (explicit false beats true default)

    Returns (granted: bool, reason: str).
    """
    if not evidence.get("ok"):
        return False, f"evidence unavailable ({evidence.get('reason', 'unknown')})"
    if evidence.get("head_sha") != evidence.get("decided_sha"):
        return False, "head moved since the evidence was built"
    if not evidence.get("trusted_producer_ok"):
        return False, "no trusted check run from the Reviews App on this exact head"
    if not evidence.get("review_sha_match"):
        return False, "the trusted check run head_sha does not match this head SHA"
    if not evidence.get("required_checks_pass"):
        failing = evidence.get("required_checks_failing", 0)
        return False, f"{failing} required check(s) are failing/pending"
    if evidence.get("unresolved_critical_high") is None:
        return False, "trusted check run output has no unresolved-findings summary"
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
