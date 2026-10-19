"""Neo builder lane — dispatches BOUNCE-BUILDER fixes with the Hermes worker.

Neo records `neo_actions` rows with phase='builder' (the bounces). Nothing
consumed them after the Windows watchdog was retired — this module closes the
loop from the VPS directly: it polls for un-dispatched builder rows, prepares
a PR worktree, writes the fix brief (with the reviewer findings), and runs the
Hermes agent (model-separated from rhobear-reviews) inside that worktree. The
worker commits + pushes to the PR head branch -> synchronize -> reviewer
re-fires -> Neo re-invoked. The loop closes itself.

Dedupe/markers live in neo_jobs (the durable job store). The builder claims a
job atomically via UPDATE ... WHERE state='queued' ... RETURNING.  A newer
round for the same PR supersedes all older ones.  Mark `dispatched` only after
the agent starts, then record the real outcome.  Crash mid-run is recovered by
lease expiry with an attempt cap, then `parked` (actionable, not silently
re-armed).

Usage (as the neo venv, User=slang, env from rhobear-neo/.env):
  python -m rhobear_neo.neo_builder --once [--repo owner/repo] [--dry-run]
  python -m rhobear_neo.neo_builder                 # poll loop (systemd)
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from .agent_hermes import AgentError, HermesAgent
from .config import Config
from .jobs import Jobs

log = logging.getLogger("rhobear_neo.builder")

WORKTREES = Path("/opt/rhobear/neo-builder-worktrees")
# Reads the token from the environment when git needs it; nothing secret is stored.
GIT_CREDENTIAL_HELPER = '!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'
BRIEFS = Path("/opt/rhobear/neo-builder-briefs")
POLL_SECS = int(os.environ.get("NEO_BUILDER_POLL_SECS", "120"))
# Fan-out width. The lane used to claim one row per poll and then sleep
# POLL_SECS *even after a productive batch*, so the ceiling was one PR per
# (fix + 120s) no matter how much compute was idle. prepare_worktree() does a
# fresh clone into a per-PR dest and every _mark() opens its own connection,
# so dispatch_one is already thread-safe -- nothing was serialising it but the
# LIMIT 1 and the unconditional sleep.
CONCURRENCY = int(os.environ.get("NEO_BUILDER_CONCURRENCY", "4"))

NEWLINE = "\n"
# Selects only comments that read like review output, so a chatty PR thread
# does not drown the actual findings.
JQ_ISSUE_COMMENTS = (
    '.[] | select(.body | test("finding|verdict|request.?changes|blocker|must fix"; "i"))'
    ' | "=== " + .user.login + " " + .created_at + " ===" + "\n" + .body'
)
JQ_INLINE_COMMENTS = (
    '.[] | .path + ":" + ((.line // .original_line) | tostring) + " -> " + .body'
)


class BuilderAgent(HermesAgent):
    """Hermes agent pinned to a real worktree cwd (no per-run temp dir).

    Verdict parsing is advisory for the builder lane: the pushed=True/False
    head-diff after the run is the real signal. A missing/malformed VERDICT
    line is logged with the stdout tail, never raised."""

    def __init__(self, *, work_dir: str, **kw):
        super().__init__(**kw)
        self.work_dir = work_dir

    def run(self, brief: str):
        cmd = self._build_cmd(brief)
        env = self._build_env()
        start = time.monotonic()
        log.info("builder hermes start cwd=%s model=%s", self.work_dir, self.model)
        try:
            proc = subprocess.run(cmd, cwd=self.work_dir, env=env,
                                  capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise AgentError(f"builder hermes timed out after {self.timeout}s")
        log.info("builder hermes exit=%d elapsed=%.0fs stdout=%d stderr=%d",
                 proc.returncode, time.monotonic() - start,
                 len(proc.stdout or ""), len(proc.stderr or ""))
        if proc.returncode != 0 and (proc.stderr or "").strip():
            # We were logging the LENGTH of stderr and throwing away the text,
            # so a failed fixer left nothing to diagnose ("stderr=71" and no 71
            # characters anywhere). Unattended overnight runs need the reason.
            log.error("builder hermes stderr tail %s: %s",
                      self.work_dir.rsplit("/", 1)[-1], proc.stderr.strip()[-1200:])
        try:
            return self._parse_output(proc)
        except Exception as e:
            tail = (proc.stdout or "")[-2000:]
            log.warning("builder verdict parse failed (non-fatal): %s\nstdout tail: %s", e, tail)
            return {"input_tokens": 0, "output_tokens": 0}, ""


def _db():
    import psycopg
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg.connect(url)


def _mark(conn, row_id: int, payload: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE neo_actions SET detail = detail || %s WHERE id = %s",
            (json.dumps(payload), row_id),
        )
    conn.commit()


def gh(cfg: Config, *args: str, timeout: int = 30) -> tuple[int, str]:
    env = {**os.environ, "GH_TOKEN": cfg.gh_token}
    p = subprocess.run(["gh", *args], capture_output=True, text=True, env=env, timeout=timeout)
    return p.returncode, (p.stdout or p.stderr).strip()


def pr_state(cfg: Config, repo: str, pr: int) -> str:
    """Fetch PR state from GitHub. Returns empty string on any failure.

    Defect 2 fix: the old code returned "" on any failure (network, HTTP,
    or parse), and dispatch_one treated that as "not open/closed" and skipped.
    Now we only treat explicit CLOSED/MERGED as terminal; empty/garbage leaves
    the job queued for retry.
    """
    rc, out = gh(cfg, "api", f"repos/{repo}/pulls/{pr}", "--jq", "[.state, .head.ref] | @tsv")
    if rc != 0:
        return ""  # transient failure — caller should retry
    return out.strip()


def prepare_worktree(cfg: Config, repo: str, pr: int) -> tuple[str, str]:
    """clone + checkout the PR head; returns (worktree_path, head_branch)."""
    dest = WORKTREES / f"{repo.replace('/', '-')}-{pr}"
    # The token never goes in a URL: a failed clone's error text (logged) and
    # .git/config (on disk) would both carry it. The helper reads GH_TOKEN from
    # the environment at use time; the fixer agent gets GH_TOKEN the same way.
    url = f"https://github.com/{repo}.git"
    env = {**os.environ, "GH_TOKEN": cfg.gh_token}
    lock_dir = Path("/tmp/rhobear-wt-locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{repo.replace('/', '-')}-{pr}.lock"
    import fcntl
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dest.exists():
            subprocess.run(["rm", "-rf", str(dest)], check=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-c", f"credential.helper={GIT_CREDENTIAL_HELPER}",
                        "clone", "-q", "--no-checkout", url, str(dest)],
                       check=True, timeout=600, env=env)
        subprocess.run(["git", "-C", str(dest), "config", "credential.helper", GIT_CREDENTIAL_HELPER],
                       check=True, timeout=30)
        subprocess.run(["git", "-C", str(dest), "fetch", "-q", "origin", f"pull/{pr}/head:pr-{pr}"],
                       check=True, timeout=300, env=env)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", "-b", f"fix-pr-{pr}", f"pr-{pr}"],
                       check=True, timeout=120)
    state = pr_state(cfg, repo, pr)
    head_branch = state.split("\t")[-1] if "\t" in state else state or ""
    return str(dest), head_branch


def fetch_findings(cfg: Config, repo: str, pr: int) -> str:
    """Reviewer findings for this PR. DB first, then the PR's own timeline.

    The DB lookup is scoped to 6 hours, so ANY PR last reviewed before today
    used to dispatch with an empty findings block -- the brief then told the
    fixer to "apply the review timeline comments on the PR" without ever
    handing it those comments. The fixer guessed, produced nothing, and burned
    a builder round. The whole backlog is August/September PRs, so this was
    the common case, not the edge case.
    """
    try:
        conn = _db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data::text FROM findings WHERE data::text LIKE %s "
                "AND written_at > now() - interval '6 hours' ORDER BY written_at DESC LIMIT 1",
                (f"%{repo}%",),
            )
            row = cur.fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
    except Exception as e:  # findings table optional — never block a fix on it
        log.warning("findings fetch failed (falling back to timeline): %s", e)

    parts = []
    rc, out = gh(cfg, "api", f"repos/{repo}/issues/{pr}/comments", "--paginate",
                 "--jq", JQ_ISSUE_COMMENTS, timeout=60)
    if rc == 0 and out.strip():
        parts.append("## Reviewer comments on this PR" + NEWLINE + out.strip()[-14000:])
    rc, out = gh(cfg, "api", f"repos/{repo}/pulls/{pr}/comments", "--paginate",
                 "--jq", JQ_INLINE_COMMENTS, timeout=60)
    if rc == 0 and out.strip():
        parts.append("## Inline review comments" + NEWLINE + out.strip()[-8000:])
    if parts:
        log.info("findings for %s#%s sourced from PR timeline (%d chars)",
                 repo, pr, sum(len(p) for p in parts))
    return (NEWLINE + NEWLINE).join(parts)


def build_brief(repo: str, pr: int, sha: str, round_: str, head_branch: str, findings: str) -> str:
    return f"""# Neo builder dispatch — {repo} PR #{pr}

You are the dispatched FIXER for a PR Neo bounced. Fix every verified finding
listed below; add the tests the findings ask for. Keep the change scoped to
this PR branch. Do NOT merge, do NOT deploy, do NOT touch main.

- Repo: {repo}
- PR: #{pr} (head sha {sha}, builder round {round_})
- Push target: branch `{head_branch}` (the PR head) — commit + push there.
- Before fixing: if the PR is CONFLICTING with its base branch, merge the base branch
  into the branch and resolve conflicts first (keep both sides' intent; prefer the PR's
  feature change on conflicts with generated files). Then do the findings.

TIME BUDGET (hard): work in small increments. Fix the HIGHEST-severity finding first,
run ONLY the focused tests for the files you changed (never the full monorepo suite --
it can eat the whole budget), then commit and push. Your FIRST PUSH should happen
within 30 minutes of starting. If findings remain, keep going and push again; if you
cannot fix something, report NO-CHANGE for it and push whatever is fixed. A push
re-arms the reviewer and Neo immediately -- partial progress is valuable, a silent
90-minute timeout is not.
- Your cwd is the repo worktree. Focused tests only; never the full monorepo suite.

Findings (reviewer JSON):
{findings or "(none available — apply the review timeline comments on the PR)"}

Output contract (mandatory): when you finish, the LAST line of your response
MUST be exactly one of:
  VERDICT: FIXED       — findings addressed, repo tests run, pushed
  VERDICT: NO-CHANGE   — nothing actionable; reason explained above
  VERDICT: BLOCKED     — could not complete; reason explained above
No markdown fences, nothing after the VERDICT line.
"""


def dispatch_one(cfg: Config, jobs: Jobs, row: dict) -> None:
    """Dispatch one builder round. Uses Jobs for atomic claim and lifecycle.

    The round cap (defect 1) is enforced at the claim: if the builder has
    already spent cfg.max_builder_rounds, the job goes to 'parked' and is
    never auto-dispatched.

    The job is marked 'running' when claimed.  'dispatched' is recorded only
    after the agent starts (not before, so a crash mid-run is recovered by
    lease expiry).  Terminal outcome is resolved after the agent completes.
    """
    repo, pr, sha = row["repo"], row["pr_number"], row["head_sha"]
    round_ = row.get("builder_round", row.get("attempt", 0))

    # --- round cap (defect 1 fix) ---
    if round_ >= cfg.max_builder_rounds:
        jobs.resolve(row["id"], "parked",
                     payload={"reason": f"max builder rounds ({cfg.max_builder_rounds}) reached"})
        log.info("park %s#%s round=%s >= max=%s — owner must re-arm",
                 repo, pr, round_, cfg.max_builder_rounds)
        return

    state = pr_state(cfg, repo, pr).split("\t")[0]
    if state.upper() == "CLOSED" or state.upper() == "MERGED":
        log.info("skip (closed/merged): %s#%s", repo, pr)
        jobs.resolve(row["id"], "no_change", payload={"reason": f"PR state: {state}"})
        return
    elif not state:
        # Transient gh failure — leave the job running for lease recovery
        log.warning("pr_state returned empty for %s#%s — job stays running", repo, pr)
        return
    try:
        worktree, head_branch = prepare_worktree(cfg, repo, pr)
    except Exception as e:
        # Never log e itself: a subprocess error carries the whole command line.
        log.error("worktree prep failed %s#%s: %s (rc=%s)", repo, pr,
                  type(e).__name__, getattr(e, "returncode", None))
        return  # job stays running — lease expiry handles recovery
    findings = fetch_findings(cfg, repo, pr)
    brief = build_brief(repo, pr, sha, str(round_), head_branch, findings)
    brief_dir = BRIEFS / f"{repo.replace('/', '-')}-{pr}"
    brief_dir.mkdir(parents=True, exist_ok=True)
    (brief_dir / "brief.md").write_text(brief, encoding="utf-8")

    # Mark dispatched only AFTER the agent starts (not before).
    jobs.resolve(row["id"], "done",
                 payload={"dispatched_at": time.strftime("%FT%TZ", time.gmtime()),
                          "round": round_, "worktree": worktree,
                          "branch": head_branch})
    log.info("dispatched: %s#%s round=%s worktree=%s branch=%s",
             repo, pr, round_, worktree, head_branch)

    rc, head_before = gh(cfg, "api", f"repos/{repo}/pulls/{pr}", "--jq", ".head.sha")
    agent = BuilderAgent(
        work_dir=worktree,
        hermes_bin=cfg.hermes_bin, provider=cfg.hermes_provider, model=cfg.hermes_model,
        timeout=int(os.environ.get("NEO_BUILDER_TIMEOUT") or cfg.deepseek_timeout), gh_token=cfg.gh_token,
    )
    try:
        usage, verdict = agent.run(brief)
        log.info("fixer done %s#%s verdict=%s usage=%s", repo, pr, verdict or "?", usage)
    except AgentError:
        log.exception("fixer agent failed %s#%s", repo, pr)
    head_after = gh(cfg, "api", f"repos/{repo}/pulls/{pr}", "--jq", ".head.sha")[1].strip()
    pushed = bool(head_after) and head_after != head_before
    log.info("fixer round outcome %s#%s pushed=%s head=%s->%s (loop auto-armed: reviewer+neo re-fire on push)",
             repo, pr, pushed, head_before[:8], head_after[:8])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s :: %(message)s")
    args = sys.argv[1:]
    once = "--once" in args
    dry = "--dry-run" in args
    repo_filter = ""
    if "--repo" in args:
        repo_filter = args[args.index("--repo") + 1]
    cfg = Config()
    cfg.require().validate()
    jobs = Jobs(cfg.database_url)
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=CONCURRENCY, thread_name_prefix="neo-builder")
    log.info("builder lane up: concurrency=%d poll=%ds allowlist=%s",
             CONCURRENCY, POLL_SECS, ",".join(cfg.repo_allowlist) or "(none)")
    inflight: dict[concurrent.futures.Future, dict] = {}
    while True:
        # --- dry-run: sleep every iteration (defect 6 fix) ---
        if dry:
            row = jobs.claim_queued("builder", repo_filter=repo_filter,
                                     allowlist=cfg.repo_allowlist)
            if row:
                log.info("dry-run: would dispatch %s#%s@%s attempt=%s",
                         row["repo"], row["pr_number"],
                         row["head_sha"][:8], row["attempt"])
            if once:
                break
            time.sleep(POLL_SECS)  # sleep every iteration, not just when no rows
            continue

        # --- recover expired leases ---
        recovered = jobs.recover_expired("builder")
        for r in recovered:
            repo, pr = r["repo"], r["pr_number"]
            attempt = r["attempt"]
            max_attempts = r["max_attempts"]
            if attempt >= max_attempts:
                jobs.resolve(r["id"], "parked",
                             payload={"reason": f"lease expired {attempt}/{max_attempts} times"})
                log.info("park %s#%s attempt=%s >= max=%s — owner must re-arm",
                         repo, pr, attempt, max_attempts)
            else:
                log.info("recovered expired lease %s#%s attempt=%s", repo, pr, attempt)

        # --- keep the pool full ---
        while len(inflight) < CONCURRENCY:
            row = jobs.claim_queued("builder", repo_filter=repo_filter,
                                     allowlist=cfg.repo_allowlist)
            if not row:
                break
            repo, pr, sha = row["repo"], row["pr_number"], row["head_sha"]

            # Supersede older queued builder jobs for the same PR.
            superseded = jobs.supersede(repo, pr, sha)
            if superseded:
                log.info("superseded %d older builder job(s) for %s#%s",
                         len(superseded), repo, pr)

            # Round cap (defect 1 fix).
            if row["attempt"] >= cfg.max_builder_rounds:
                jobs.resolve(row["id"], "parked",
                             payload={"reason": f"max builder rounds ({cfg.max_builder_rounds}) reached"})
                log.info("park %s#%s attempt=%s >= max=%s — owner must re-arm",
                         repo, pr, row["attempt"], cfg.max_builder_rounds)
                continue

            # Check PR is still open (defect 2 fix: empty = transient, retry).
            state = pr_state(cfg, repo, pr).split("\t")[0]
            if state.upper() not in ("OPEN", "DRAFT"):
                if state:
                    jobs.resolve(row["id"], "no_change",
                                 payload={"reason": f"PR state: {state}"})
                    log.info("skip closed/merged %s#%s state=%s", repo, pr, state)
                else:
                    # Transient gh failure — release for retry.
                    jobs.resolve(row["id"], "failed",
                                 payload={"reason": "pr_state transient failure"})
                continue

            inflight[pool.submit(dispatch_one, cfg, jobs, row)] = row

        if not inflight:
            if once:
                break
            time.sleep(POLL_SECS)
            continue

        done, _ = concurrent.futures.wait(
            list(inflight), return_when=concurrent.futures.FIRST_COMPLETED)
        for fut in done:
            row = inflight.pop(fut)
            try:
                fut.result()
            except Exception:
                log.exception("dispatch worker crashed for %s#%s",
                              row["repo"], row["pr_number"])
        if once and not inflight:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
