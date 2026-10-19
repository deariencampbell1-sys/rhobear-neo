"""Neo builder lane — dispatches BOUNCE-BUILDER fixes with the Hermes worker.

Neo's worker creates a durable `kind='builder'` job (keyed by repo + PR +
round) in the same place the bounce is recorded, so a BOUNCE verdict always
reaches Hermes.  This module polls for queued builder jobs, prepares a PR
worktree, writes the fix brief (with the reviewer findings), and runs the
Hermes agent (model-separated from rhobear-reviews) inside that worktree. The
worker commits + pushes to the PR head branch -> synchronize -> reviewer
re-fires -> Neo re-invoked. The loop closes itself.

Dedupe/markers live in neo_jobs (the durable job store).

* Claim is atomic (UPDATE ... WHERE state='queued' ... RETURNING), NEWEST
  round first; older queued rounds for the same PR are superseded, so a stale
  round never runs after a newer one exists.
* The round cap (builder_round <= cfg.max_builder_rounds) is enforced at the
  claim — beyond it the job is `parked` (visible, owner-actionable), never
  auto-dispatched.
* `dispatched` is recorded only when the agent is about to start; the real
  outcome (pushed / no-change / retry) is recorded after the run.  A crash
  mid-run is recovered by lease expiry (renewed during the run) with an
  attempt cap, then `parked`.
* Transient gh failures never consume a round: only a parsed CLOSED/MERGED
  resolves to no_change; empty pr_state leaves the job running for lease
  recovery.
* --dry-run is read-only (peek, no claim) so an operator can inspect the
  queue without mutating durable job state.

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
# Builder agent runs are long (up to deepseek_timeout, often 30+ minutes).  The
# lease must cover the whole run and be renewed while the agent works, or a
# second builder would claim the same round and two agents would share one
# worktree (fcntl only serializes prep, not the run).
BUILDER_LEASE_SECS = int(os.environ.get("NEO_BUILDER_LEASE_SECS", "3600"))
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

    The job is already 'running' (claimed by main).  'dispatched' is recorded
    only when the agent is about to start (not before, so a crash mid-run is
    recovered by lease expiry).  The real outcome is resolved after the agent
    completes: pushed -> done, NO-CHANGE -> no_change, BLOCKED/empty -> retry
    (bounded by the attempt cap, then parked).
    """
    repo, pr, sha = row["repo"], row["pr_number"], row["head_sha"]
    round_ = int(row.get("builder_round") or row.get("attempt") or 0)

    # --- round cap (defect 1 fix) ---
    if round_ > cfg.max_builder_rounds:
        jobs.park(row["id"],
                  reason=f"max builder rounds ({cfg.max_builder_rounds}) reached (round {round_})")
        log.info("park %s#%s round=%s > max=%s — owner must re-arm",
                 repo, pr, round_, cfg.max_builder_rounds)
        return

    state = pr_state(cfg, repo, pr).split("\t")[0]
    if state.upper() == "CLOSED" or state.upper() == "MERGED":
        log.info("skip (closed/merged): %s#%s", repo, pr)
        jobs.resolve(row["id"], "no_change", payload={"reason": f"PR state: {state}"})
        return
    elif not state:
        # Transient gh failure — leave the job running for lease recovery.
        # Never resolve a transient failure as terminal (GLM 8).
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

    # Mark dispatched only as the agent is about to start (not before — a
    # crash at this point is recovered by lease expiry, not consumed forever).
    jobs.note(row["id"], {"dispatched_at": time.strftime("%FT%TZ", time.gmtime()),
                           "round": round_, "worktree": worktree,
                           "branch": head_branch})
    log.info("dispatched: %s#%s round=%s worktree=%s branch=%s",
             repo, pr, round_, worktree, head_branch)

    rc, head_before = gh(cfg, "api", f"repos/{repo}/pulls/{pr}", "--jq", ".head.sha")
    head_before = (head_before or "").strip()
    agent = BuilderAgent(
        work_dir=worktree,
        hermes_bin=cfg.hermes_bin, provider=cfg.hermes_provider, model=cfg.hermes_model,
        timeout=int(os.environ.get("NEO_BUILDER_TIMEOUT") or cfg.deepseek_timeout), gh_token=cfg.gh_token,
    )
    try:
        # Lease renewal: the agent run can outlive a static lease — without it
        # a second builder would claim the same round and two agents would work
        # one worktree (GLM 6).  BUILDER_LEASE_SECS covers the whole run.
        with jobs.lease_renewal(row["id"], lease_secs=BUILDER_LEASE_SECS):
            usage, verdict = agent.run(brief)
        log.info("fixer done %s#%s verdict=%s usage=%s", repo, pr, verdict or "?", usage)
    except AgentError:
        log.exception("fixer agent failed %s#%s", repo, pr)
        jobs.retry(row["id"], reason="builder agent failed")
        return
    except Exception as exc:
        log.exception("fixer agent crashed (%s) %s#%s", type(exc).__name__, repo, pr)
        jobs.retry(row["id"], reason=f"builder agent crashed ({type(exc).__name__})")
        return

    head_after = gh(cfg, "api", f"repos/{repo}/pulls/{pr}", "--jq", ".head.sha")[1].strip()
    # A push is the real outcome signal (more reliable than the advisory
    # verdict); it re-arms reviewer + Neo immediately.
    if head_before and head_after:
        pushed = head_after != head_before
    else:
        # Couldn't read the PR head (transient gh failure) — can't confirm.
        log.warning("head read failed for %s#%s before=%r after=%r", repo, pr, head_before, head_after)
        pushed = False
    log.info("fixer round outcome %s#%s pushed=%s head=%s->%s (loop auto-armed: reviewer+neo re-fire on push)",
             repo, pr, pushed, head_before[:8], head_after[:8])
    if pushed:
        jobs.resolve(row["id"], "done", payload={"outcome": "pushed",
                                                   "verdict": verdict or "",
                                                   "head_before": head_before,
                                                   "head_after": head_after})
    elif verdict == "NO-CHANGE":
        jobs.resolve(row["id"], "no_change", payload={"outcome": "no-change",
                                                       "verdict": verdict})
    elif verdict == "FIXED":
        # Claimed a fix but the head didn't move — record it; the loop re-arms
        # via the new head when it lands.
        jobs.resolve(row["id"], "done", payload={"outcome": "fixed-no-push",
                                                   "verdict": verdict})
    else:
        # BLOCKED / missing verdict / unverifiable head — retryable; bounded by
        # the attempt cap, then parked (owner-actionable).
        log.warning("%s#%s round=%s no push and no actionable verdict (%r) — retrying",
                    repo, pr, round_, verdict)
        jobs.retry(row["id"], reason=f"no push, verdict={verdict or 'missing'}")


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
    jobs.ensure_schema()
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=CONCURRENCY, thread_name_prefix="neo-builder")
    log.info("builder lane up: concurrency=%d poll=%ds lease=%ds allowlist=%s",
             CONCURRENCY, POLL_SECS, BUILDER_LEASE_SECS,
             ",".join(cfg.repo_allowlist) or "(none)")
    inflight: dict[concurrent.futures.Future, dict] = {}
    while True:
        # --- dry-run: read-only peek (defect 6 fix + GLM 9: never mutate) ---
        if dry:
            rows = jobs.peek_queued("builder", repo_filter=repo_filter,
                                    allowlist=cfg.repo_allowlist)
            for row in rows:
                log.info("dry-run: would dispatch %s#%s round=%s attempt=%s",
                         row["repo"], row["pr_number"], row["builder_round"], row["attempt"])
            if once:
                break
            time.sleep(POLL_SECS)  # sleep every iteration, not just when empty
            continue

        # --- recover expired builder leases (park at the attempt cap) -------
        recovered = jobs.recover_expired("builder")
        for r in recovered:
            repo, pr = r["repo"], r["pr_number"]
            attempt, max_attempts = r["attempt"], r["max_attempts"]
            if attempt >= max_attempts:
                jobs.park(r["id"],
                          reason=f"lease expired {attempt}/{max_attempts} times")
                log.info("park %s#%s attempt=%s >= max=%s — owner must re-arm",
                         repo, pr, attempt, max_attempts)
            else:
                log.info("recovered expired lease %s#%s round=%s attempt=%s (re-queued)",
                         repo, pr, r["builder_round"], attempt)

        # --- keep the pool full ---------------------------------------------
        while len(inflight) < CONCURRENCY:
            row = jobs.claim_queued("builder", lease_secs=BUILDER_LEASE_SECS,
                                    repo_filter=repo_filter,
                                    allowlist=cfg.repo_allowlist)
            if not row:
                break
            repo, pr, sha = row["repo"], row["pr_number"], row["head_sha"]
            round_ = int(row["builder_round"] or 0)

            # Newest round wins: supersede OLDER queued builder jobs for the
            # same PR (GLM 7 — the old queued() claimed newest-first; older
            # rounds are stale by definition).
            superseded = jobs.supersede_older(repo, pr, round_)
            if superseded:
                log.info("superseded %d older builder job(s) for %s#%s",
                         len(superseded), repo, pr)

            # Round cap (defect 1 fix): beyond max -> parked, never dispatched.
            if round_ > cfg.max_builder_rounds:
                jobs.park(row["id"],
                          reason=f"max builder rounds ({cfg.max_builder_rounds}) reached (round {round_})")
                log.info("park %s#%s round=%s > max=%s — owner must re-arm",
                         repo, pr, round_, cfg.max_builder_rounds)
                continue

            # Check PR is still open (defect 2 fix: empty = transient, retry;
            # GLM 8: never resolve transient failure as terminal 'failed').
            state = pr_state(cfg, repo, pr).split("\t")[0]
            if state.upper() in ("CLOSED", "MERGED"):
                jobs.resolve(row["id"], "no_change",
                             payload={"reason": f"PR state: {state}"})
                log.info("skip closed/merged %s#%s state=%s", repo, pr, state)
                continue
            if not state:
                # Transient gh failure — job stays running; lease expiry
                # re-queues it.
                log.warning("pr_state empty for %s#%s — job stays running (lease recovery)",
                            repo, pr)
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
