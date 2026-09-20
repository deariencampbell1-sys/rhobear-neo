"""Neo builder lane — dispatches BOUNCE-BUILDER fixes with the Hermes worker.

Neo records `neo_actions` rows with phase='builder' (the bounces). Nothing
consumed them after the Windows watchdog was retired — this module closes the
loop from the VPS directly: it polls for un-dispatched builder rows, prepares
a PR worktree, writes the fix brief (with the reviewer findings), and runs the
Hermes agent (model-separated from rhobear-reviews) inside that worktree. The
worker commits + pushes to the PR head branch -> synchronize -> reviewer
re-fires -> Neo re-invoked. The loop closes itself.

Dedupe/markers live in neo_actions.detail (jsonb): `brief_dispatched` is set
BEFORE the dispatch (a crash re-arms by clearing it by hand — never double-run
a fixer), `skipped_merged` marks stale rows for closed/merged PRs.

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

log = logging.getLogger("rhobear_neo.builder")

WORKTREES = Path("/opt/rhobear/neo-builder-worktrees")
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
    rc, out = gh(cfg, "api", f"repos/{repo}/pulls/{pr}", "--jq", "[.state, .head.ref] | @tsv")
    return out.strip() if rc == 0 else ""


def prepare_worktree(cfg: Config, repo: str, pr: int) -> tuple[str, str]:
    """clone + checkout the PR head; returns (worktree_path, head_branch)."""
    dest = WORKTREES / f"{repo.replace('/', '-')}-{pr}"
    url = f"https://x-access-token:{cfg.gh_token}@github.com/{repo}.git"
    if dest.exists():
        subprocess.run(["rm", "-rf", str(dest)], check=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", "--no-checkout", url, str(dest)],
                   check=True, timeout=600)
    subprocess.run(["git", "-C", str(dest), "fetch", "-q", "origin", f"pull/{pr}/head:pr-{pr}"],
                   check=True, timeout=300)
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
- Your cwd is the repo worktree; run the repo's own test suite before pushing.

Findings (reviewer JSON):
{findings or "(none available — apply the review timeline comments on the PR)"}

Output contract (mandatory): when you finish, the LAST line of your response
MUST be exactly one of:
  VERDICT: FIXED       — findings addressed, repo tests run, pushed
  VERDICT: NO-CHANGE   — nothing actionable; reason explained above
  VERDICT: BLOCKED     — could not complete; reason explained above
No markdown fences, nothing after the VERDICT line.
"""


def dispatch_one(cfg: Config, row: dict) -> None:
    repo, pr, sha, round_ = row["repo"], row["pr_number"], row["head_sha"], row["builder_round"]
    state = pr_state(cfg, repo, pr).split("\t")[0]
    if state.upper() not in ("OPEN", "DRAFT"):
        log.info("skip (closed/merged): %s#%s", repo, pr)
        # brief_dispatched is what queued() filters on. Marking only
        # skipped_merged leaves the row claimable, and since queued() is
        # ORDER BY created_at DESC LIMIT 1, a single closed PR at the head
        # is re-picked every poll and starves the entire queue behind it.
        _mark(_db(), row["id"], {"skipped_merged": True, "brief_dispatched": True})
        return
    try:
        worktree, head_branch = prepare_worktree(cfg, repo, pr)
    except Exception as e:
        log.error("worktree prep failed %s#%s: %s", repo, pr, e)
        return
    findings = fetch_findings(cfg, repo, pr)
    brief = build_brief(repo, pr, sha, round_, head_branch, findings)
    brief_dir = BRIEFS / f"{repo.replace('/', '-')}-{pr}"
    brief_dir.mkdir(parents=True, exist_ok=True)
    (brief_dir / "brief.md").write_text(brief, encoding="utf-8")

    _mark(_db(), row["id"], {"brief_dispatched": True,
                             "dispatched_at": time.strftime("%FT%TZ", time.gmtime())})
    log.info("dispatched: %s#%s round=%s worktree=%s branch=%s",
             repo, pr, round_, worktree, head_branch)

    rc, head_before = gh(cfg, "api", f"repos/{repo}/pulls/{pr}", "--jq", ".head.sha")
    agent = BuilderAgent(
        work_dir=worktree,
        hermes_bin=cfg.hermes_bin, provider=cfg.hermes_provider, model=cfg.hermes_model,
        timeout=cfg.deepseek_timeout, gh_token=cfg.gh_token,
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


def queued(conn, repo_filter: str = "", limit: int = 1,
           allowlist: tuple[str, ...] = ()) -> list[dict]:
    # DISTINCT ON collapses multiple rounds of the SAME PR to one row.
    # prepare_worktree() keys its dest on repo+pr, so claiming round 1 and
    # round 2 of one PR in the same batch made two threads rm -rf and clone
    # the same directory -- git aborts with
    # "BUG: initial ref transaction called with existing refs" (SIGABRT).
    # Serial LIMIT 1 never hit this. Newest round wins; the older round is
    # stale by definition.
    sql = ("SELECT id, repo, pr_number, head_sha, builder_round FROM ("
           "SELECT DISTINCT ON (repo, pr_number) "
           "id, repo, pr_number, head_sha, builder_round, created_at "
           "FROM neo_actions "
           "WHERE phase = 'builder' AND NOT (detail ? 'brief_dispatched')")
    params: list = []
    if repo_filter:
        sql += " AND repo = %s"
        params.append(repo_filter)
    elif allowlist:
        # Scope the lane to repos that are actually alive. Without this the
        # builder happily burns fix rounds on archived and renamed repos
        # (rhobear-app -> rhobear-app-legacy/ARCHIVED, rhobear-hub-web ->
        # rhobear-builds-web), which can never merge.
        sql += " AND repo = ANY(%s)"
        params.append(list(allowlist))
    sql += (" ORDER BY repo, pr_number, builder_round DESC, created_at DESC"
            ") q ORDER BY created_at DESC LIMIT %s")
    params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s :: %(message)s")
    args = sys.argv[1:]
    once = "--once" in args
    dry = "--dry-run" in args
    repo_filter = ""
    if "--repo" in args:
        repo_filter = args[args.index("--repo") + 1]
    cfg = Config()
    conn = _db()
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=CONCURRENCY, thread_name_prefix="neo-builder")
    log.info("builder lane up: concurrency=%d poll=%ds allowlist=%s",
             CONCURRENCY, POLL_SECS, ",".join(cfg.repo_allowlist) or "(none)")
    last_ids: set[int] = set()
    while True:
        rows = queued(conn, repo_filter, limit=CONCURRENCY,
                      allowlist=cfg.repo_allowlist)
        ids = {r["id"] for r in rows}
        if rows and dry:
            for row in rows:
                log.info("dry-run: would dispatch %s#%s@%s round=%s",
                         row["repo"], row["pr_number"], row["head_sha"][:8],
                         row["builder_round"])
        elif rows:
            log.info("claiming %d row(s) across %d repo(s)",
                     len(rows), len({r["repo"] for r in rows}))
            futs = [pool.submit(dispatch_one, cfg, row) for row in rows]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    fut.result()
                except Exception:
                    log.exception("dispatch worker crashed")
        if once:
            break
        # Sleep only when there is nothing to do, or when the batch made no
        # progress. dispatch_one returns WITHOUT marking the row if worktree
        # prep fails, so an unfixable row would otherwise spin a tight loop;
        # an identical id set two passes running is exactly that condition.
        if not rows or ids == last_ids:
            time.sleep(POLL_SECS)
        last_ids = ids
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
