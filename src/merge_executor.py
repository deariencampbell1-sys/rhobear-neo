"""Deterministic merge executor — the only merge path in the Neo service.

When the agent returns ACCEPT, the worker calls `merge()` here, which:

  1. Re-reads the PR head immediately before merging.
  2. Refuses if the head moved (expected_sha mismatch).
  3. Calls the merge API with `sha=expected_sha` (GitHub rejects if head
     changed).
  4. Uses the installation token (B1g's github_app).
  5. Records `merged` with the SHA.

`merge()` is the ONLY code path in the Neo service that calls the merge API.
No agent prompt, no other function, no other subprocess call in Neo may invoke
`gh pr merge`.

The triage/fix-forward agent is instructed never to merge and never receives a
personal token, but it does hold a repo-scoped installation token (it needs
Contents: write to push fix-forward commits).  GitHub's "Merge a pull request"
endpoint requires only Contents: write, so that token is technically capable
of merging.  The agent-side guarantee is therefore procedural, not a hard
credential boundary.  The enforced boundary against agent-initiated merges is
branch protection on the protected branch: required pull-request review (the
reviewer's check) + required status checks.  See DEPLOY.md.
"""
from __future__ import annotations

import json
import logging
import subprocess

log = logging.getLogger("rhobear_neo.merge")


class MergeError(Exception):
    """Base for merge failures — fail closed."""


class HeadMoved(MergeError):
    """The PR head changed between evidence and merge."""


class MergeDenied(MergeError):
    """GitHub refused the merge (e.g. branch protection, required checks)."""


class MergeFailed(MergeError):
    """The merge API call failed for another reason."""


def _gh(token: str, *args: str, timeout: int = 30) -> tuple[int, str]:
    """Run `gh` as the repo's installation token (never a personal token)."""
    import os
    env = {**os.environ, "GH_TOKEN": token}
    p = subprocess.run(
        ["gh", *args], capture_output=True, text=True, env=env, timeout=timeout
    )
    return p.returncode, (p.stdout or p.stderr).strip()


#: W1-07-r1: the two greens + the trusted red (the merge rule (b)-(d)).
_RULE_FIELDS = ("green1_ok", "green2_ok", "trusted_red")


def two_green_gate(evidence: dict) -> tuple[bool, str]:
    """The merge rule (b)-(d) on the exact head SHA. Fails closed.

    (b) green 1 — Apollo's review state is ACCEPT on this head SHA: a commit
    status or check-run success is never enough, COMMENTED is not approval;
    (c) green 2 — the architect's approval bound to this head SHA, or an
    enabled trusted reviewer's success on it (either counts); (d) no enabled
    trusted reviewer is failing or requesting changes on it.  Every fact is
    bound to the head SHA, so any new push resets all three.
    """
    problems: list[str] = []
    if not evidence.get("green1_ok"):
        problems.append("green 1 missing: Apollo's review is not ACCEPT on this "
                        "exact head SHA (%s)" % (evidence.get("green1_reason")
                                                 or "no review by Apollo on this head"))
    red = [str(x) for x in (evidence.get("trusted_red") or [])]
    if red:
        problems.append("a trusted reviewer is red on this exact head: " + ", ".join(red))
    if not evidence.get("green2_ok"):
        problems.append("green 2 missing: " + (evidence.get("green2_reason")
                        or "no architect approval and no enabled trusted reviewer "
                           "green on this exact head"))
    return (not problems, "; ".join(problems) or "two greens on this exact head")


def has_rule_fields(evidence: dict) -> bool:
    """One predicate for a bundle carrying any two-green rule fields."""
    return any(k in evidence for k in _RULE_FIELDS)


def pre_merge_belts(evidence: dict) -> tuple[bool, str]:
    """The evidence belts EVERY merge path runs immediately before merging."""
    if not evidence.get("ok"):
        return False, f"evidence unavailable ({evidence.get('reason') or 'unknown'})"
    if has_rule_fields(evidence):
        ok, why = two_green_gate(evidence)
        if not ok:
            return False, why
    if evidence.get("trusted_producer_ok") is False:
        return False, "no trusted check run from the Reviews App on this exact head"
    return True, "two greens on this exact head"


def post_merge_comment(token: str, repo: str, pr: int, evidence: dict) -> None:
    """Post the one merge comment AFTER a confirmed merge; never raises."""
    if not has_rule_fields(evidence):
        return
    try:
        rc, out = _gh(token, "pr", "comment", str(pr), "-R", repo,
                      "--body", two_green_comment(evidence), timeout=30)
    except Exception as exc:  # noqa: BLE001 — the merge stands regardless
        log.warning("merge comment failed after the merge (%s) — the merge stands",
                    type(exc).__name__)
        return
    if rc != 0:
        log.warning("merge comment refused by GitHub after the merge: %s", out[:200])


def two_green_comment(evidence: dict) -> str:
    """The one merge comment (board `verdicts` 02): the two greens with links
    and the CI summary — "Two greens on a91d2e0 / 1: Apollo ACCEPT / 2: Zeus
    ACCEPT / CI: 6 of 6 checks passed" — posted right AFTER a successful
    merge (W1-07-r2 defect 9)."""
    head = str(evidence.get("head_sha") or evidence.get("decided_sha") or "")

    def _line(n: int, label: str, url: str, at: str) -> str:
        text = f"[{label}]({url})" if url else str(label)
        when = f" · {str(at)[11:16]}" if at else ""
        return f"{n}: {text}{when}"

    return "\n".join([
        f"Two greens on {head[:7]}",
        _line(1, evidence.get("green1_label") or "Apollo ACCEPT",
              str(evidence.get("green1_url") or ""), str(evidence.get("green1_at") or "")),
        _line(2, evidence.get("green2_label") or "architect ACCEPT",
              str(evidence.get("green2_url") or ""), str(evidence.get("green2_at") or "")),
        f"CI: {evidence.get('ci_green', 0)} of {evidence.get('ci_total', 0)} checks passed",
    ])


def merge(
    repo: str,
    pr: int,
    expected_sha: str,
    evidence: dict,
    token: str,
) -> dict:
    """Merge `pr` in `repo` at `expected_sha` via the GitHub REST merge API.

    Steps:
      1. Re-read the PR head (headRefOid + headRefName) from GitHub.
      2. If it does not match `expected_sha`, raise HeadMoved.
      3. Call the merge API
         (PUT /repos/{repo}/pulls/{pr}/merge) with `sha=expected_sha` —
         GitHub rejects the merge if the head changed between our re-read
         and the merge call.  (`delete_branch` is NOT a merge-API parameter;
         the head branch is deleted explicitly in step 5.)
      4. Record the merge result.
      5. Delete the PR's head branch explicitly
         (DELETE /repos/{repo}/git/refs/heads/<headRefName>), tolerating
         404/422 when GitHub already auto-deleted it.

    Returns `{"merged": True, "sha": <head_sha>}` on success.

    Raises MergeError subclasses on failure.  This is the only code path in
    the Neo service that merges; no agent prompt runs `gh pr merge`.  (The
    agent's repo-scoped installation token is technically capable of merging
    because GitHub requires Contents: write for merge — branch protection is
    the enforced boundary.  See DEPLOY.md.)
    """
    # --- Step 1: Re-read the PR head ---------------------------------------
    try:
        rc, raw = _gh(
            token,
            "pr", "view", str(pr), "-R", repo,
            "--json", "headRefOid,state,mergeable,headRefName,isCrossRepository",
            timeout=45,
        )
    except subprocess.TimeoutExpired as exc:
        # A hung `gh` must become a MergeError, not escape run_neo's handlers
        # (the wake job would otherwise stay running and lease recovery would
        # re-run triage).
        raise MergeFailed(f"timed out reading PR {repo}#{pr} head") from exc
    if rc != 0:
        raise MergeFailed(f"cannot read PR {repo}#{pr}: {raw[:200]}")

    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise MergeFailed(f"malformed PR view response: {exc}") from exc

    head = str(data.get("headRefOid") or "")
    head_ref = str(data.get("headRefName") or "")
    cross_repo = bool(data.get("isCrossRepository"))
    state = str(data.get("state") or "").lower()

    # --- Step 2: Verify head has not moved ---------------------------------
    if not head:
        raise MergeFailed("PR has no headRefOid")
    if head != expected_sha:
        raise HeadMoved(
            f"PR head changed: expected {expected_sha[:12]}, "
            f"got {head[:12]}"
        )
    if state != "open":
        raise MergeDenied(f"PR is not open (state={state})")

    # --- Step 3: Verify evidence supports the merge ------------------------
    # Safety net (W1-13-r1 defect 6): run the SAME merge_authority gate the
    # rest of Neo uses, so the worker's gate and merge() can never disagree —
    # any denial surfaces as MergeDenied carrying the gate's reason.  Every
    # production evidence bundle (build_evidence) carries the gate's decision
    # fields and is gated in full here.  Evidence carrying NONE of them (the
    # legacy {"ok", shas} shape contract-pinned in tests/test_merge_authority
    # for the mechanics of this function) has no findings to consult at this
    # layer and falls back to the original ok check.
    from . import review_evidence as _review_ev
    _gate_fields = ("trusted_producer_ok", "review_sha_match",
                    "required_checks_pass", "unresolved_critical_high",
                    "coverage_complete", "auto_merge")
    if any(k in evidence for k in _gate_fields):
        granted, reason = _review_ev.merge_authority(evidence)
        # Belt (W1-13 defect 1): `ok` is checked HERE, in front of the gate —
        # `not ok or not granted` denies.  merge_authority subsumes `ok`
        # (its first condition denies an ok=False bundle), so the gate branch
        # is transitively closed today; the belt stays because `ok` encodes
        # evidence completeness the six gate fields do not ("evidence
        # retrieval partially failed but the fetched fields look green"), and
        # the merge path must fail closed on it even if the gate's answer ever
        # drifts.  A gate that grants must never override ok=False.
        if not evidence.get("ok"):
            raise MergeDenied(f"evidence unavailable ({evidence.get('reason', 'unknown')})")
        if not granted:
            raise MergeDenied(f"merge authority denied: {reason}")
    elif not evidence.get("ok"):
        raise MergeDenied(f"evidence unavailable ({evidence.get('reason', 'unknown')})")

    # W1-07-r1 steps 3/5: the two-green merge rule runs AGAIN immediately
    # before the merge API call (a new push since the wake resets both greens
    # and is denied here).  The one merge comment (the two greens with links +
    # the CI summary) is posted AFTER a successful merge (W1-07-r2 defect 9:
    # posting it first left a stale "Two greens on <sha>" comment on every
    # refused or moved-head merge, and a retry posted a duplicate).  A bundle
    # without the rule's fields is the legacy {"ok", shas} mechanics shape
    # contract-pinned in tests/test_merge_authority: nothing to consult here,
    # so the W1-13 belt above applies unchanged.
    granted, why = pre_merge_belts(evidence)
    if not granted:
        raise MergeDenied(f"merge authority denied: {why}")

    # --- Step 4: Execute the merge -----------------------------------------
    # REST merge API with sha=expected_sha: GitHub rejects with 422 if the
    # head no longer matches, closing the race between re-read and merge.
    # `delete_branch` is NOT a parameter of this API (unknown fields are
    # ignored), so the head branch is deleted explicitly in step 5.
    merge_args = [
        "api", "-X", "PUT",
        f"repos/{repo}/pulls/{pr}/merge",
        "-f", "merge_method=squash",
        "-f", f"sha={expected_sha}",
    ]
    try:
        rc, out = _gh(token, *merge_args, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise MergeFailed(f"merge API call timed out for {repo}#{pr}") from exc

    if rc != 0:
        error_msg = out[:300] if out else "unknown error"
        low = error_msg.lower()
        if "required" in low or "protection" in low:
            raise MergeDenied(f"GitHub refused merge: {error_msg}")
        if "sha" in low and ("does not match" in low or "not the head" in low
                             or "provided sha" in low or "head" in low):
            raise HeadMoved(f"GitHub rejected merge — head moved: {error_msg}")
        raise MergeFailed(f"merge API call failed (exit={rc}): {error_msg}")

    try:
        result = json.loads(out)
    except (TypeError, ValueError):
        result = {}
    if not result.get("merged"):
        raise MergeFailed(f"GitHub returned no merge confirmation: {out[:200]}")

    log.info("merged %s#%s at %s", repo, pr, expected_sha[:12])

    # --- the one merge comment, post-merge (defect 9) ---------------------
    # Only now, so a refused or moved-head merge leaves no stale comment.
    # The merge is already confirmed: a refused comment is logged, never a
    # failed merge.
    post_merge_comment(token, repo, pr, evidence)

    # --- Step 5: delete the head branch -----------------------------------
    # The REST merge API never deletes the branch; gh's --delete-branch did it
    # client-side.  Best-effort only: the merge is already confirmed, so nothing
    # here may raise, and a shared branch is never deleted.
    _delete_head_branch(token, repo, pr, head_ref, cross_repo)

    return {
        "merged": True,
        "sha": expected_sha,
        "detail": out[:500],
    }


def _delete_head_branch(token: str, repo: str, pr: int, head_ref: str, cross_repo: bool) -> None:
    """Delete a merged PR's own head branch; never raises.

    Skipped for a fork PR (the branch is not in this repo), for the repo's
    default branch, and when another open PR still uses the same head branch
    (a shared `develop` or release branch must survive a squash merge).
    """
    if not head_ref or cross_repo:
        return
    try:
        rc, default = _gh(token, "api", f"repos/{repo}", "--jq", ".default_branch", timeout=30)
        if rc != 0 or not default.strip():
            log.warning("head branch %s for %s#%s kept: default branch unknown", head_ref, repo, pr)
            return
        if head_ref == default.strip():
            return
        rc, others = _gh(token, "pr", "list", "-R", repo, "--head", head_ref, "--state", "open",
                         "--limit", "1000", "--json", "number", "--jq", ".[].number", timeout=30)
        if rc != 0:
            log.warning("head branch %s for %s#%s kept: open-PR check failed", head_ref, repo, pr)
            return
        if any(line.strip() and line.strip() != str(pr) for line in others.splitlines()):
            log.info("head branch %s for %s#%s kept: another open PR uses it", head_ref, repo, pr)
            return
        rc_del, out_del = _gh(token, "api", "-X", "DELETE",
                              f"repos/{repo}/git/refs/heads/{head_ref}", timeout=60)
        low = (out_del or "").lower()
        if rc_del == 0:
            log.info("deleted head branch %s for %s#%s", head_ref, repo, pr)
        elif "404" in low or "422" in low:
            log.info("head branch %s for %s#%s already gone", head_ref, repo, pr)
        else:
            log.warning("could not delete head branch %s for %s#%s: %s", head_ref, repo, pr, (out_del or "")[:200])
    except Exception as exc:  # noqa: BLE001 — the merge is confirmed; deletion is best-effort
        log.warning("head branch delete for %s#%s errored (merge already confirmed): %s", repo, pr, exc)
