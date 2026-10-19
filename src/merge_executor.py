"""Deterministic merge executor — the ONLY code path that merges a PR.

The agent never holds merge credentials or runs `gh pr merge`.  When the agent
returns ACCEPT, the worker calls `merge()` here, which:

  1. Re-reads the PR head immediately before merging.
  2. Refuses if the head moved (expected_sha mismatch).
  3. Calls the merge API with `sha=expected_sha` (GitHub rejects if head
     changed).
  4. Uses the installation token (B1g's github_app).
  5. Records `merged` with the SHA.

This is the ONLY code path that can merge a PR.  No agent prompt, no other
function, no other subprocess call may invoke `gh pr merge`.
"""
from __future__ import annotations

import json
import logging

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
    import subprocess
    import os
    env = {**os.environ, "GH_TOKEN": token}
    # Remove any stale personal token from the process env.
    env.pop("GH_TOKEN", None)
    env["GH_TOKEN"] = token
    p = subprocess.run(
        ["gh", *args], capture_output=True, text=True, env=env, timeout=timeout
    )
    return p.returncode, (p.stdout or p.stderr).strip()


def merge(
    repo: str,
    pr: int,
    expected_sha: str,
    evidence: dict,
    token: str,
) -> dict:
    """Merge `pr` in `repo` at `expected_sha` via the GitHub REST merge API.

    Steps:
      1. Re-read the PR head (headRefOid) from GitHub.
      2. If it does not match `expected_sha`, raise HeadMoved.
      3. Call the merge API
         (PUT /repos/{repo}/pulls/{pr}/merge) with `sha=expected_sha` —
         GitHub rejects the merge if the head changed between our re-read
         and the merge call.
      4. Record the merge result and return it.

    Returns `{"merged": True, "sha": <head_sha>}` on success.

    Raises MergeError subclasses on failure.  This is the ONLY code path that
    merges; no agent prompt runs `gh pr merge`.
    """
    # --- Step 1: Re-read the PR head ---------------------------------------
    rc, raw = _gh(
        token,
        "pr", "view", str(pr), "-R", repo,
        "--json", "headRefOid,state,mergeable",
        timeout=45,
    )
    if rc != 0:
        raise MergeFailed(f"cannot read PR {repo}#{pr}: {raw[:200]}")

    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise MergeFailed(f"malformed PR view response: {exc}") from exc

    head = str(data.get("headRefOid") or "")
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
    # The caller should have already checked evidence.  We double-check here
    # as a safety net.
    if not evidence.get("ok"):
        raise MergeDenied(f"evidence unavailable ({evidence.get('reason', 'unknown')})")

    # --- Step 4: Execute the merge -----------------------------------------
    # REST merge API with sha=expected_sha: GitHub rejects with 422 if the
    # head no longer matches, closing the race between re-read and merge.
    merge_args = [
        "api", "-X", "PUT",
        f"repos/{repo}/pulls/{pr}/merge",
        "-f", "merge_method=squash",
        "-f", f"sha={expected_sha}",
        "-f", "delete_branch=true",
    ]
    rc, out = _gh(token, *merge_args, timeout=60)

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

    return {
        "merged": True,
        "sha": expected_sha,
        "detail": out[:500],
    }
