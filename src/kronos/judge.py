"""Kronos judge — the escalator: judges bad fix loops, never a 7th round.

When a PR reaches 5 Zeus fix rounds without two greens, Kronos sends the
round history (verdicts, Apollo's findings, the diffs' stats — bounded, see
build_report) to Summit through RouteGate, consumer
`reviews:kronos/<install_id>`, carrying a READ-ONLY installation token
({contents: read, pull_requests: read} — never the write token).

The answer is JSON {"progress": "yes"|"no", "plan": "...", "reason": "..."}:

  yes          -> ONE escalated Zeus round whose brief is the plan
                  (a `neo_jobs` builder round 6 with `escalated: true` plus
                  the `escalated` marker row Zeus already consumes);
  no / bad / timeout
               -> Zeus stops on that PR: Kronos writes the stop marker Zeus
                  checks on every dispatch (already_escalated), posts ONE
                  comment with the reason on the PR, and alerts the owner.

Never a 7th round: 5 normal rounds + at most 1 escalated round = 6, and the
stop marker parks every further bounce.

Deconfliction with Zeus's own advisor escalation (src/escalation.py, W1-10's
file — read, never touched): while a phase='escalating' marker exists, or an
escalated round is already recorded, Kronos defers — Zeus's path owns the
PR. Kronos judges only when the loop is at 5+ rounds and Zeus has not taken
it, and never twice for the same PR (the judged marker in Kronos's state).

The stop marker: a phase='escalated' row with detail.advisor_round set and
detail.kronos_stop explaining the judgment. Zeus's neo_worker checks
`already_escalated()` (detail->'advisor_round' IS NOT NULL) BEFORE any
further round, so the PR can never reach round 7 — that marker is the one
piece of shared state the judge writes, and its detail says exactly why.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit

log = logging.getLogger("rhobear_neo.kronos.judge")

JUDGE_ROUNDS_TRIGGER = 5
#: The report budget (edge 4): an unbounded report would paste whole journals.
REPORT_BUDGET_CHARS = 8000

READONLY_PERMISSIONS = {"contents": "read", "pull_requests": "read"}


class JudgeTimeout(Exception):
    """Summit did not answer inside the budget — a stop, not a retry."""


@dataclass
class JudgeOutcome:
    action: str          # escalated | stopped | skip
    reason: str
    plan: str = ""


# ---------------------------------------------------------------------------
# round history (read-only SQL over the shared DSN)
# ---------------------------------------------------------------------------

def round_history(db, repo: str, pr: int) -> list[dict]:
    """The PR's fix rounds, oldest first, with verdicts and (when the worker
    recorded it) the tree hash. Findings are attached per head from the
    assessments table (Apollo's findings), bounded."""
    rounds = db.rows(
        "SELECT builder_round AS round, head_sha, verdict, "
        "COALESCE(detail->>'tree_sha', '') AS tree_sha, "
        "COALESCE(detail->>'diff_stat', '') AS diff_stat, created_at "
        "FROM neo_actions "
        "WHERE repo = %s AND pr_number = %s AND phase = 'builder' "
        "ORDER BY id ASC LIMIT 20",
        (repo, pr),
    )
    out = []
    for r in rounds:
        row = {
            "round": int(r.get("round") or 0),
            "head_sha": str(r.get("head_sha") or ""),
            "verdict": str(r.get("verdict") or ""),
            "tree_sha": str(r.get("tree_sha") or ""),
            "diff_stat": str(r.get("diff_stat") or ""),
            "findings": "",
        }
        if row["head_sha"]:
            row["findings"] = findings_summary(db, repo, row["head_sha"])
        out.append(row)
    return out


def findings_summary(db, repo: str, head_sha: str, *, max_chars: int = 600) -> str:
    """Apollo's findings for this head, bounded (the report budget applies
    upstream; this keeps one head's findings from eating it whole)."""
    try:
        rows = db.rows(
            "SELECT data FROM assessments "
            "WHERE repo = %s AND pr_number = %s AND head_sha = %s "
            "ORDER BY updated_at DESC LIMIT 1",
            (repo, _pr_of(db, repo, head_sha), head_sha),
        )
    except Exception as exc:  # noqa: BLE001 — findings are best-effort
        log.debug("assessments unreadable: %s", type(exc).__name__)
        return ""
    if not rows:
        return ""
    data = rows[0].get("data") or {}
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (TypeError, ValueError):
            data = {}
    findings = data.get("findings") or []
    lines = []
    for f in findings[:10]:
        if isinstance(f, dict):
            sev = str(f.get("severity") or "").upper()
            title = str(f.get("title") or f.get("message") or "").strip()
            if title:
                lines.append(f"{sev}: {title}" if sev else title)
    text = "; ".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + f"… [dropped {len(text) - max_chars} chars of findings]"
    return text


def _pr_of(db, repo: str, head_sha: str) -> int:
    rows = db.rows(
        "SELECT pr_number FROM assessments "
        "WHERE repo = %s AND head_sha = %s LIMIT 1", (repo, head_sha))
    return int(rows[0]["pr_number"]) if rows else 0


# ---------------------------------------------------------------------------
# loop detection (edge 3) and the bounded report (edge 4)
# ---------------------------------------------------------------------------

def same_commit_loop(rounds: list[dict]) -> list[tuple[int, int, str]]:
    """A fix loop that repeats itself: same commit twice, or two different
    commits with the SAME tree (an empty "fix"). Detected by tree hash when
    the rows carry one, else by head SHA. Returns (round_a, round_b, key)."""
    loops: list[tuple[int, int, str]] = []
    by_tree: dict[str, int] = {}
    by_sha: dict[str, int] = {}
    for r in rounds:
        rnd = int(r.get("round") or 0)
        tree = str(r.get("tree_sha") or "")
        sha = str(r.get("head_sha") or "")
        if tree:
            if tree in by_tree:
                loops.append((by_tree[tree], rnd, tree))
            else:
                by_tree[tree] = rnd
        elif sha:
            if sha in by_sha:
                loops.append((by_sha[sha], rnd, sha))
            else:
                by_sha[sha] = rnd
    return loops


def loop_note(loops: list[tuple[int, int, str]],
              rounds: list[dict] | None = None) -> str:
    """Both offending SHAs travel in the note (the brief: detected by tree
    hash, reported with both SHAs)."""
    rounds = rounds or []
    sha_of = {int(r.get("round") or 0): str(r.get("head_sha") or "")
              for r in rounds}
    parts = []
    for a, b, key in loops:
        detail = f" (same tree {key})" if key not in (
            sha_of.get(a), sha_of.get(b)) else ""
        parts.append(f"rounds {a} and {b} repeat "
                     f"{sha_of.get(a, '?')} == {sha_of.get(b, '?')}"
                     f"{detail}")
    return "Loop: " + "; ".join(parts)


def build_report(history: list[dict], *, budget: int = REPORT_BUDGET_CHARS,
                 loops: list[tuple[int, int, str]] | None = None) -> str:
    """The judge's report: verdicts, findings, diff stats — HARD-bounded to
    `budget` chars with an explicit "[dropped ...]" note (edge 4: never
    silently truncated, never a whole journal)."""
    sections: list[str] = []
    if loops:
        sections.append(loop_note(loops, history))
    for r in history:
        lines = [f"round {r.get('round')}: verdict {r.get('verdict') or '?'}"
                 f" head {str(r.get('head_sha') or '')[:12]}"]
        if r.get("diff_stat"):
            lines.append(f"  diff: {r['diff_stat']}")
        findings = str(r.get("findings") or "")
        if findings:
            lines.append(f"  findings: {findings}")
        sections.append("\n".join(lines))
    report = "\n".join(sections)
    if len(report) <= budget:
        return report
    cut = report[:budget]
    # Keep whole lines so the cut never lands mid-fact, and reserve room for
    # the dropped-note itself so the bounded report never exceeds the budget.
    cut = cut[:cut.rfind("\n")] if "\n" in cut else cut
    dropped = len(report) - len(cut)
    note = (f"\n[dropped {dropped} chars of round history: "
            f"kept {len(cut)} of {len(report)}]")
    while len(cut) + len(note) > budget and "\n" in cut:
        cut = cut[:cut.rfind("\n")]
        dropped = len(report) - len(cut)
        note = (f"\n[dropped {dropped} chars of round history: "
                f"kept {len(cut)} of {len(report)}]")
    return cut + note


# ---------------------------------------------------------------------------
# the Summit call (RouteGate, OpenAI shape, consumer reviews:kronos/<id>)
# ---------------------------------------------------------------------------

def ask_summit(routegate_url: str, model: str, report: str, *,
               install_id: str, read_token: str, http,
               timeout: int = 300) -> dict:
    """One chat completion asking for the JSON verdict. Raises JudgeTimeout
    on a timeout; returns the parsed answer dict (validated) or raises
    ValueError on a bad shape."""
    consumer = f"reviews:kronos/{install_id}" if install_id \
        else "reviews:kronos"
    body = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You judge whether an automated fix loop is making "
                        "progress. Answer ONLY with JSON: "
                        '{"progress": "yes"|"no", "plan": "...", '
                        '"reason": "..."}. "yes" means one more escalated '
                        "round with your plan could plausibly reach green; "
                        '"no" means stop and hand it to a person.'},
            {"role": "user",
             "content": f"Read-only installation token (contents:read, "
                        f"pull_requests:read): {read_token}\n\n"
                        f"Fix-loop history for this PR:\n\n{report}"},
        ],
        "max_tokens": 2000,
    }
    headers = {"Content-Type": "application/json",
               "X-RHOBEAR-Consumer": consumer}
    try:
        status, raw = http("POST",
                           routegate_url.rstrip("/") + "/v1/chat/completions",
                           headers=headers, body=json.dumps(body).encode(),
                           timeout=timeout)
    except TimeoutError as exc:
        raise JudgeTimeout("summit did not answer in time") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise JudgeTimeout(f"routegate unreachable: "
                           f"{type(exc).__name__}") from exc
    if status != 200:
        raise ValueError(f"routegate answered HTTP {status}")
    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"routegate answer malformed: {exc}") from exc
    answer = parse_judge_answer(content)
    if answer is None:
        raise ValueError("judge answer is not the agreed JSON shape")
    return answer


def parse_judge_answer(content: str) -> dict | None:
    """{"progress": "yes"|"no", "plan": "...", "reason": "..."} — a `yes`
    without a plan is a bad answer; anything else unparseable is None."""
    try:
        data = json.loads(str(content or "").strip().strip("`"))
        if isinstance(data, str):
            data = json.loads(data)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    progress = data.get("progress")
    plan = str(data.get("plan") or "")
    reason = str(data.get("reason") or "")
    if progress not in ("yes", "no") or not reason:
        return None
    if progress == "yes" and not plan.strip():
        return None
    return {"progress": progress, "plan": plan, "reason": reason}


# ---------------------------------------------------------------------------
# the flow
# ---------------------------------------------------------------------------

def judge_pr(cfg, db, *, comments, alerter, state, repo: str, pr: int,
             head_sha: str, install_id: str, read_token: str, http,
             now_fn, dry_run: bool = False) -> JudgeOutcome:
    """Judge one PR (or defer). Writes only the two state shapes Zeus
    already consumes; posts ONE PR comment; alerts on a stop."""
    marker_slot = state.slot("judged")
    marker = f"{repo}#{pr}"
    rounds = round_history(db, repo, pr)
    if len(rounds) < JUDGE_ROUNDS_TRIGGER:
        return JudgeOutcome("skip", f"{len(rounds)} rounds < "
                                    f"{JUDGE_ROUNDS_TRIGGER}")
    if marker in marker_slot:
        return JudgeOutcome("skip", "already judged by kronos")
    # Deconfliction with Zeus's own advisor path (read-only checks):
    if db.rows("SELECT 1 FROM neo_actions WHERE phase='escalating' "
               "AND repo = %s AND pr_number = %s LIMIT 1", (repo, pr)):
        return JudgeOutcome("skip", "Zeus advisor escalation is in flight")
    if db.rows("SELECT 1 FROM neo_actions WHERE phase='escalated' "
               "AND detail->'advisor_round' IS NOT NULL "
               "AND repo = %s AND pr_number = %s LIMIT 1", (repo, pr)):
        return JudgeOutcome("skip", "escalated round already recorded")

    loops = same_commit_loop(rounds)
    report = build_report(rounds, loops=loops)
    if dry_run:
        return JudgeOutcome("skip", "dry-run")

    try:
        _validate_loopback(cfg.routegate_url)
        answer = ask_summit(cfg.routegate_url, cfg.judge_model, report,
                            install_id=install_id, read_token=read_token,
                            http=http, timeout=cfg.judge_timeout_secs)
    except JudgeTimeout as exc:
        return _stop(db, comments, alerter, state, marker, repo, pr,
                     rounds, reason=f"judge timeout: {exc}", dry_run=dry_run)
    except ValueError as exc:
        return _stop(db, comments, alerter, state, marker, repo, pr,
                     rounds, reason=f"bad judge answer: {exc}", dry_run=dry_run)

    if answer["progress"] == "yes":
        return _escalate(cfg, db, comments, state, marker, repo, pr,
                         head_sha, rounds, answer, install_id=install_id)
    return _stop(db, comments, alerter, state, marker, repo, pr, rounds,
                 reason=answer["reason"], dry_run=dry_run)


def _escalate(cfg, db, comments, state, marker, repo, pr, head_sha,
              rounds, answer, *, install_id: str) -> JudgeOutcome:
    """`yes` -> the ONE escalated Zeus round whose brief is the plan."""
    new_round = len(rounds) + 1
    plan = answer["plan"]
    redacted_plan = _redact(plan)
    work_order = {
        "route": cfg.judge_model,
        "summary": redacted_plan,
        "root_cause": answer["reason"],
        "steps": [{"file": "plan", "change": redacted_plan}],
        "tests": [],
        "risks": "",
    }
    db.record(repo, pr, head_sha, "escalated", verdict="ESCALATE",
              builder_round=new_round,
              detail={
                  "reason": "kronos judge: one escalated round with "
                            "Summit's plan",
                  "advisor_round": new_round,
                  "advice_id": f"kronos:{repo}#{pr}",
                  "installation_id": install_id,
                  "kronos": True,
                  "work_order": {
                      "route": work_order["route"],
                      "summary": redacted_plan,
                      "steps_count": len(work_order["steps"]),
                  },
              })
    db.create_builder(repo, pr, head_sha, new_round, payload={
        "verdict": "BOUNCE-BUILDER",
        "head_sha": head_sha,
        "phase": "escalated",
        "round": new_round,
        "context": "kronos:escalated",
        "escalated": True,
        "work_order": work_order,
        "installation_id": install_id,
    })
    state.slot("judged")[marker] = now_iso()
    comment = (
        "**Kronos · judged this loop**\n\n"
        f"{len(rounds)} fix rounds without two greens; asking whether "
        "iterating further helps.\n\n"
        f"Reason: {answer['reason']}\n\n"
        f"Proposal (one escalated round, never a 7th):\n\n{redacted_plan}"
    )
    comments(repo, pr, comment)
    log.info("kronos judged %s#%s: escalated round %d", repo, pr, new_round)
    return JudgeOutcome("escalated", answer["reason"], plan=plan)


def _stop(db, comments, alerter, state, marker, repo, pr, rounds, *,
          reason: str, dry_run: bool = False) -> JudgeOutcome:
    """`no` / bad answer / timeout -> Zeus stops on that PR, one comment,
    the owner alerted."""
    head_sha = str(rounds[-1]["head_sha"]) if rounds else ""
    new_round = len(rounds) + 1
    if not dry_run:
        db.record(repo, pr, head_sha, "escalated", verdict="PARKED",
                  builder_round=0,
                  detail={
                      "reason": f"kronos judge stopped the loop: {reason}",
                      "advisor_round": new_round,
                      "advice_id": f"kronos-stop:{repo}#{pr}",
                      "kronos_stop": True,
                      "kronos": True,
                  })
        state.slot("judged")[marker] = now_iso()
        comments(repo, pr, (
            "**Kronos · stopped: 5 rounds but still red**\n\n"
            f"{len(rounds)} fix rounds without two greens; nothing will "
            "improve by iterating in the same direction.\n\n"
            f"Reason: {reason}\n\n"
            "This needs a decision from a person. Zeus is off this PR — "
            "never a 7th round."
        ))
        alerter.alert(
            f"loop:{repo}#{pr}",
            f"Kronos stopped Zeus on #{pr} — needs you",
            f"{repo}#{pr}: {reason}")
    log.warning("kronos stopped %s#%s: %s", repo, pr, reason)
    return JudgeOutcome("stopped", reason)


def _redact(text: str) -> str:
    """Summit's plan is model output that may echo secrets — the same
    redaction Zeus applies to advisor work orders, reused read-only."""
    try:
        from ..redact import redact_data
        out = redact_data({"plan": text})
        if isinstance(out, dict):
            return str(out.get("plan") or text)
        return text
    except Exception:  # noqa: BLE001 — redaction must never break the flow
        return text


def now_iso() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _validate_loopback(url: str) -> None:
    """The judge's RouteGate URL is loopback, full stop (a brief or a PR
    body must never redirect the read token off-box)."""
    parts = urlsplit(url)
    if parts.scheme != "http" or (parts.hostname or "") not in (
            "127.0.0.1", "localhost"):
        raise ValueError("routegate_url must be loopback http")
