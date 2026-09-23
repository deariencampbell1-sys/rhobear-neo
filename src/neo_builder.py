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

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .agent_claude import AgentError, ClaudeAgent
from .config import Config

log = logging.getLogger("rhobear_neo.builder")

WORKTREES = Path("/opt/rhobear/neo-builder-worktrees")
BRIEFS = Path("/opt/rhobear/neo-builder-briefs")
POLL_SECS = int(os.environ.get("NEO_BUILDER_POLL_SECS", "120"))


class BuilderAgent(ClaudeAgent):
    """Claude agent pinned to a real worktree cwd (no per-run temp dir).

    Verdict parsing is advisory for the builder lane: the pushed=True/False
    head-diff after the run is the real signal. A missing/malformed VERDICT
    line is logged with the stdout tail, never raised."""

    def __init__(self, *, work_dir: str, **kw):
        super().__init__(**kw)
        self.work_dir = work_dir

    def run(self, brief: str):
        # Create a minimal temp config dir for CLAUDE_CONFIG_DIR (never populated)
        with tempfile.TemporaryDirectory(prefix="neo-builder-config-") as config_dir:
            return self._run_in(brief, self.work_dir, config_dir)


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


def fetch_findings(repo: str) -> str:
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
        return row[0] if row else ""
    except Exception as e:  # findings table optional — never block a fix on it
        log.warning("findings fetch failed (proceeding): %s", e)
        return ""


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
        _mark(_db(), row["id"], {"skipped_merged": True})
        return
    try:
        worktree, head_branch = prepare_worktree(cfg, repo, pr)
    except Exception as e:
        log.error("worktree prep failed %s#%s: %s", repo, pr, e)
        return
    findings = fetch_findings(repo)
    brief = build_brief(repo, pr, sha, round_, head_branch, findings)
    brief_dir = BRIEFS / f"{repo.replace('/', '-')}-{pr}"
    brief_dir.mkdir(parents=True, exist_ok=True)
    (brief_dir / "brief.md").write_text(brief, encoding="utf-8")

    _mark(_db(), row["id"], {"brief_dispatched": True,
                             "dispatched_at": time.strftime("%FT%TZ", time.gmtime())})
    log.info("dispatched: %s#%s round=%s worktree=%s branch=%s",
             repo, pr, round_, worktree, head_branch)

    rc, head_before = gh(cfg, "api", f"repos/{repo}/pulls/{pr}", "--jq", ".head.sha")
    if rc != 0:
        log.error("failed to get PR head SHA %s#%s: %s", repo, pr, head_before[:200])
        return
    agent = BuilderAgent(
        work_dir=worktree,
        claude_bin=cfg.claude_bin, api_key=cfg.deepseek_key, base_url=cfg.deepseek_base_url,
        model=cfg.deepseek_model, effort=cfg.deepseek_reasoning_effort,
        max_tokens=cfg.deepseek_max_tokens, timeout=cfg.deepseek_timeout, gh_token=cfg.gh_token,
    )
    try:
        usage, verdict = agent.run(brief)
        log.info("fixer done %s#%s verdict=%s usage=%s", repo, pr, verdict or "?", usage)
    except AgentError:
        log.exception("fixer agent failed %s#%s", repo, pr)
    rc2, head_after = gh(cfg, "api", f"repos/{repo}/pulls/{pr}", "--jq", ".head.sha")
    if rc2 != 0:
        log.error("failed to get PR head SHA after run %s#%s: %s", repo, pr, head_after[:200])
        head_after = ""
    else:
        head_after = head_after.strip()
    pushed = bool(head_after) and head_after != head_before
    log.info("fixer round outcome %s#%s pushed=%s head=%s->%s (loop auto-armed: reviewer+neo re-fire on push)",
             repo, pr, pushed, head_before[:8], head_after[:8])


def queued(conn, repo_filter: str = "", limit: int = 1) -> list[dict]:
    sql = ("SELECT id, repo, pr_number, head_sha, builder_round FROM neo_actions "
           "WHERE phase = 'builder' AND NOT (detail ? 'brief_dispatched')")
    params: list = []
    if repo_filter:
        sql += " AND repo = %s"
        params.append(repo_filter)
    sql += " ORDER BY created_at DESC LIMIT %s"
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
    while True:
        rows = queued(conn, repo_filter, limit=1)
        if rows:
            for row in rows:
                if dry:
                    log.info("dry-run: would dispatch %s#%s@%s round=%s",
                             row["repo"], row["pr_number"], row["head_sha"][:8],
                             row["builder_round"])
                else:
                    dispatch_one(cfg, row)
        if once:
            break
        time.sleep(POLL_SECS)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
