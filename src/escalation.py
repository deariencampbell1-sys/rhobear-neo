"""Escalation — 5-bounce escalation to a frontier advisor (async).

When a PR has bounced NEO_BOUNCES_BEFORE_ESCALATION (default 5) builder rounds
across any head SHAs, Neo escalates to a frontier advisor on the build box
instead of a 6th cheap round.

Flow (asynchronous — Neo never waits on the advisor):
  1. Count builder rounds per PR across head SHAs.
  2. When round >= bounces_before_escalation, the PR goes to `escalating`:
     `submit_escalation` mints a read-only installation token
     (permissions: {contents: read, pull_requests: read} — never the write
     token), POSTs to the advisor, stores the `advice_id` on the PR's state
     row (phase='escalating') and returns immediately.
  3. The recovery loop (`__main__.py`) polls each `escalating` PR every
     `escalation_poll_secs` s (give up after `escalation_timeout_secs` →
     park with a one-line reason) and on `done` calls `escalate_pr` to
     create the ONE escalated builder round whose brief is the work order
     plus the round history, marked `escalated`.
  4. If that round also bounces, or the advisor fails/times out, the PR is
     parked for the owner with a one-line reason (no more rounds).
"""
from __future__ import annotations

import ipaddress
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit

from .config import Config
from .redact import redact_data

log = logging.getLogger("rhobear_neo.escalation")


class EscalationError(Exception):
    """Base for escalation failures."""


class AdvisorUnavailable(EscalationError):
    """The advisor service could not be reached."""


class AdvisorRefused(EscalationError):
    """The advisor refused the request (auth, queue full, etc.)."""


class AdvisorFailed(EscalationError):
    """The advisor returned an error or malformed response."""


class AdvisorNotFound(AdvisorFailed):
    """The advisor has no record for a known advice_id (e.g. it restarted
    and lost its in-memory results).  Fail-closed: park immediately instead
    of waiting out the timeout."""


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

    @classmethod
    def from_dict(cls, data: dict | None) -> "WorkOrder":
        """Build a WorkOrder from a raw advisor JSON dict (poll response)."""
        data = data or {}
        return cls(
            route=str(data.get("route", "") or ""),
            summary=str(data.get("summary", "") or ""),
            root_cause=str(data.get("root_cause", "") or ""),
            steps=data.get("steps") or [],
            tests=data.get("tests") or [],
            risks=str(data.get("risks", "") or ""),
        )


@dataclass
class AdviceStatus:
    """The advisor's poll result for one advice_id."""
    status: str = ""                    # queued|running|done|failed
    work_order: dict | None = None      # when done
    error: str = ""                     # when failed


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


_TAILNET = ipaddress.ip_network("100.64.0.0/10")


def _validate_advisor_url(url: str) -> None:
    """Refuse to send the GitHub read token to a plaintext/arbitrary URL.

    The advisor receives a read-scoped installation token in the POST body.
    Only HTTPS (any host — the tailnet listener has a cert), or plain HTTP to
    loopback/tailnet (127.0.0.1, localhost, 100.64.0.0/10), is allowed.
    The error message never includes the token (there is none in the URL).
    """
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError as exc:
        raise AdvisorUnavailable("advisor URL is invalid") from exc
    if parts.scheme == "https":
        if not host:
            raise AdvisorUnavailable("advisor URL has no host")
        return
    if parts.scheme != "http":
        raise AdvisorUnavailable(
            "advisor URL must use https (or http to loopback/tailnet)"
        )
    if host in ("127.0.0.1", "localhost"):
        return
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        raise AdvisorUnavailable(
            "advisor URL must use https unless the host is loopback/tailnet"
        )
    if addr.version == 4 and addr in _TAILNET:
        return
    raise AdvisorUnavailable(
        "advisor URL must use https unless the host is loopback/tailnet"
    )


def _headers(cfg: Config, token: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _post_advise(
    cfg: Config,
    repo: str,
    pr: int,
    head_sha: str,
    rounds: list[RoundRecord],
    diff_stat: str,
    read_token: str,
) -> str:
    """POST /v1/advise and return the advice_id.

    The advisor responds 202 immediately (the agent runs in the background);
    this never waits for the work order.  Fail closed if url/token unset.
    """
    url = cfg.advisor_url
    token = cfg.advisor_token
    if not url or not token:
        raise AdvisorUnavailable("NEO_ADVISOR_URL or NEO_ADVISOR_TOKEN not configured")
    _validate_advisor_url(url)

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
        headers=_headers(cfg, token),
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
    advice_id = str(data.get("advice_id", "") or "")
    if not advice_id:
        raise AdvisorFailed("advisor returned no advice_id")
    return advice_id


def poll_advice(cfg: Config, advice_id: str) -> AdviceStatus:
    """GET /v1/advise/<advice_id> — the advisor's async status.

    Returns {queued|running|done|failed} with the work_order when done or the
    error when failed.  Raises EscalationError on transport/auth failure.
    """
    url = cfg.advisor_url
    token = cfg.advisor_token
    if not url or not token:
        raise AdvisorUnavailable("NEO_ADVISOR_URL or NEO_ADVISOR_TOKEN not configured")
    _validate_advisor_url(url)

    req = urllib.request.Request(
        url.rstrip("/") + "/v1/advise/" + advice_id,
        method="GET",
        headers=_headers(cfg, token),
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # The advisor restarted and lost its in-memory results: a known
            # advice_id is now unknown.  Fail closed — park immediately, do
            # not wait out the timeout (Extra A).
            raise AdvisorNotFound(
                f"advisor has no record for advice_id {advice_id}"
            ) from exc
        raise AdvisorRefused(f"advisor poll returned HTTP {exc.code}")
    except urllib.error.URLError as exc:
        raise AdvisorUnavailable(f"advisor unreachable: {exc.reason}")

    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise AdvisorFailed(f"advisor poll returned malformed JSON: {exc}")

    return AdviceStatus(
        status=str(data.get("status", "") or ""),
        work_order=data.get("work_order"),
        error=str(data.get("error", "") or ""),
    )


def submit_escalation(
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
    """Submit an escalation to the advisor and record it on the PR state row.

    Mints a READ-ONLY token, POSTs (which returns 202 immediately), stores the
    advice_id on the PR's state row (phase='escalating'), and returns — never
    waits for the work order.

    Returns:
      {"action": "escalating", "advice_id": ...} on success, or
      {"action": "parked", "reason": ...} when the advisor cannot be reached
      (fail closed).
    """
    try:
        read_token = _read_token(cfg, repo, installation_id)
    except Exception as exc:  # noqa: BLE001
        reason = f"cannot mint read token: {type(exc).__name__}"
        log.warning("%s#%s@%s %s", repo, pr, head_sha[:8], reason)
        return _park_pr(state, jobs, repo, pr, head_sha, reason)

    try:
        advice_id = _post_advise(cfg, repo, pr, head_sha, rounds, diff_stat,
                                 read_token)
    except EscalationError as exc:
        reason = f"advisor unavailable: {exc}"
        log.warning("%s#%s@%s %s", repo, pr, head_sha[:8], reason)
        return _park_pr(state, jobs, repo, pr, head_sha, reason)

    state.record_escalating(repo, pr, head_sha, verdict="ESCALATE",
                            advice_id=advice_id,
                            installation_id=installation_id,
                            extra={"advisor_url": cfg.advisor_url or ""})
    log.info("escalating %s#%s@%s advice_id=%s (async — poll for done)",
             repo, pr, head_sha[:8], advice_id)
    return {"action": "escalating", "advice_id": advice_id}


def escalate_pr(
    cfg: Config,
    state,
    jobs,
    repo: str,
    pr: int,
    head_sha: str,
    work_order: WorkOrder,
    installation_id: int | None = None,
    advice_id: str | None = None,
) -> dict:
    """Create the ONE escalated builder round from a completed work order.

    The round is marked `escalated`, carries the work order in its payload
    (and in its brief), and is exempt from the max-builder-rounds cap.

    Idempotent per `advice_id` (defect 2): if the escalated round for this
    advice was already recorded (a retry after a transient failure), it is
    reused and only the builder job is ensured — never a second round.

    Returns {"action": "escalated", ...} or {"action": "parked", ...}.
    """
    if not work_order or not work_order.steps:
        reason = "advisor returned empty work order (no steps)"
        return _park_pr(state, jobs, repo, pr, head_sha, reason)

    # HIGH 1 (W1-10-r1): the advisor's fields are attacker-controlled output
    # that MAY echo host secrets.  Redact BEFORE anything is persisted — the
    # raw work order plus a pre-rendered brief used to be written to
    # neo_jobs.payload, leaving every echoed secret at rest in the database
    # and its backups (a clean egress render does not help at rest).  The
    # rendered brief is derived data and is no longer stored at all: the
    # builder renders it from the redacted work_order at dispatch time.
    redacted = redact_data({
        "route": work_order.route,
        "summary": work_order.summary,
        "root_cause": work_order.root_cause,
        "steps": work_order.steps,
        "tests": work_order.tests,
        "risks": work_order.risks,
    })
    new_round = count_builder_rounds(state, repo, pr) + 1

    existing_round = (state.advisor_escalation_for(repo, pr, advice_id)
                      if advice_id else None)
    if existing_round is None:
        # r4 MEDIUM 2: the DECISION carries the (redacted) plan, so the
        # builder job of a committed decision can be created LATER — the
        # recovery repairs a decision whose job insert failed instead of
        # parking a successful advice and losing its round.
        stored_plan = dict(redacted)
        stored_plan["steps_count"] = len(work_order.steps)
        state.record(repo, pr, head_sha, "escalated", verdict="ESCALATE",
                     builder_round=new_round,
                     detail={
                         "reason": "escalated to frontier advisor",
                         "advice_id": str(advice_id or ""),
                         "work_order": stored_plan,
                         "installation_id": installation_id,
                         "advisor_round": new_round,
                     })
    else:
        # A retry after a partial failure: reuse the recorded round number so
        # the builder job (ensured below) matches the recorded escalated row.
        new_round = existing_round

    # Create a builder job for the escalated round.
    if jobs:
        jobs.create_builder(repo, pr, head_sha, new_round, payload={
            "verdict": "BOUNCE-BUILDER",
            "head_sha": head_sha,
            "phase": "escalated",
            "round": new_round,
            "context": "neo:escalated",
            "escalated": True,
            # Redacted at rest (HIGH 1): never the raw advisor output, and
            # never the derived brief (it would double the exposure).
            "work_order": redacted,
            "installation_id": installation_id,
        })

    log.info("escalated %s#%s@%s route=%s round=%d steps=%d",
             repo, pr, head_sha[:8], work_order.route, new_round,
             len(work_order.steps))
    return {
        "action": "escalated",
        "reason": "escalated to frontier advisor",
        "work_order": redacted,
        "round": new_round,
    }


def _committed_decision(state, repo: str, pr: int, advice_id: str):
    """The COMMITTED escalation decision of this advice, or None.

    r4 MEDIUM 2 — the decision (the 'escalated' row) is written BEFORE the
    builder job is created.  When the job insert fails the decision exists
    without its effect: the recovery must REPAIR that partial outcome (never
    park it, never write a second decision)."""
    if not advice_id:
        return None
    getter = getattr(state, "advisor_escalation_detail", None)
    if getter is None:
        return None
    try:
        decision = getter(repo, pr, advice_id)
    except Exception:  # noqa: BLE001 — a read failure is not a decision
        return None
    # only a real decision record counts (a stubbed state returns anything)
    if not isinstance(decision, dict):
        return None
    return decision


def _recorded_plan(decision: dict | None) -> dict:
    """The (redacted) plan recorded with a committed decision — empty when
    the decision predates the recorded plan (then the builder job cannot be
    rebuilt from the row alone)."""
    if not isinstance(decision, dict):
        return {}
    plan = (decision.get("detail") or {}).get("work_order")
    return plan if isinstance(plan, dict) else {}


def _repair_escalated_round(state, jobs, repo: str, pr: int, head_sha: str,
                             decision: dict) -> dict:
    """Create the builder job of a COMMITTED escalation decision when it is
    missing (the job insert failed after the decision committed).

    Idempotent: the job of a round is unique — an existing job is never
    re-armed and never duplicated, and the decision itself is never written
    again (ONE advice, ONE decision, ONE round)."""
    plan = _recorded_plan(decision)
    if not plan.get("steps"):
        raise EscalationError(
            "the committed escalation carries no recorded plan — the "
            "builder job cannot be rebuilt")
    try:
        round_no = int(decision.get("round") or 0)
    except (TypeError, ValueError):
        round_no = 0
    if not round_no:
        raise EscalationError(
            "the committed escalation carries no round — the builder job "
            "cannot be rebuilt")
    getter = getattr(state, "builder_job_state", None)
    job_state = getter(repo, pr, round_no) if getter is not None else None
    if isinstance(job_state, str):
        return {"action": "escalated",
                "reason": "the builder job of the committed round exists"}
    if jobs is None:
        raise EscalationError(
            "no jobs writer: the builder job cannot be created")
    decision_sha = decision.get("head_sha") or head_sha
    jobs.create_builder(repo, pr, decision_sha, round_no, payload={
        "verdict": "BOUNCE-BUILDER",
        "head_sha": decision_sha,
        "phase": "escalated",
        "round": round_no,
        "context": "neo:escalated",
        "escalated": True,
        "work_order": plan,
        "installation_id": (decision.get("detail") or {}).get(
            "installation_id"),
    })
    log.info("repaired the builder job of the committed escalation %s#%s "
             "(round %d)", repo, pr, round_no)
    return {"action": "escalated",
            "reason": "the builder job of the committed round was created",
            "round": round_no}


def _never_park(reason: str):
    """Never park a COMMITTED escalation decision — the advice stays pending
    so its round can be repaired (the decision is the successful outcome)."""
    raise EscalationError(reason)


def _consume_and_act(state, repo: str, pr: int, head_sha: str,
                     advice_id: str, installation_id, act,
                     started_at: float | None = None,
                     detail: dict | None = None) -> dict | None:
    """Consume this advice's escalating marker (only if it is still the NEWEST
    escalation for the PR), then run `act()`.  Returns:

      - None        -> the advice was superseded by a newer escalation; the
                       caller drops it without acting (a round is never built
                       from a stale head).
      - act() dict  -> the terminal action's result (parked / escalated ...).
      - retry dict  -> `act()` raised (transient DB error) after the marker
                       was consumed: the marker is re-recorded under the
                       advisory lock so the next poll retries.  The PR is
                       NEVER silently stranded (defect 2); idempotent
                       escalate_pr prevents a duplicate round.
    """
    if not state.take_escalation(repo, pr, advice_id):
        return None
    try:
        out = act()
    except Exception as exc:  # noqa: BLE001 — fail closed, never strand
        log.warning("escalation act failed for %s#%s advice %s: %s — marker "
                    "re-recorded, retry next poll", repo, pr, advice_id, exc)
        # Keep the marker's own detail (green / wake_job feed the worker's
        # color-aware crash gate) and the escalation's original start.
        extra = {k: v for k, v in (detail or {}).items()
                 if k not in ("advice_id", "installation_id")}
        if started_at:
            extra["escalation_started"] = started_at
        state.record_escalating(
            repo, pr, head_sha, advice_id=advice_id,
            installation_id=installation_id, extra=extra or None)
        return {"action": "retry", "reason": f"transient failure: {exc}"}
    # r4 (HIGH C): the outcome is recorded — the in-flight marker the take
    # left behind is removed now, and ONLY now. A failure here is retried by
    # the stale reclaim (a crashed caller cannot strand the marker).
    try:
        state.clear_inflight_markers(repo, pr, advice_id)
    except Exception as exc:  # noqa: BLE001 — the stale reclaim covers it
        log.warning("could not clear the in-flight marker for %s#%s "
                    "(the stale reclaim will): %s", repo, pr, exc)
    return out


def recover_escalations(
    cfg: Config,
    state,
    jobs,
    timeout_secs: int = 1800,
) -> list[dict]:
    """Poll every escalating PR; advance it one step (called by the recovery
    loop in __main__).

    For each `escalating` PR:
      - waited too long  -> park with a one-line reason
      - advisor `done`   -> escalate_pr creates the ONE escalated builder round
      - advisor `failed` -> park with the advisor's error
      - advisor 404 for a known advice_id (restart) -> park immediately
      - queued/running   -> keep waiting (next poll)
      - transient poll error -> keep waiting (next poll; bounded by timeout)
      - superseded       -> drop the stale advice, never act on it (defect 3)

    Every terminal action consumes the escalating marker atomically under the
    per-PR advisory lock (`state.take_escalation`): it verifies the advice is
    still the NEWEST escalation for the PR and consumes only that advice's
    rows. r4 (HIGH C): the consumed rows become a durable 'escalation-inflight'
    marker (never a deletion), removed when the outcome records and reclaimed
    after ESCALATION_INFLIGHT_STALE_SECS — the consume-then-act gap is closed.
    A newer escalation (a new head bounced while this advice was queued)
    supersedes it, so no round is created for a stale head and the newer
    advice's marker is never deleted.  If the action itself raises after the
    consume, the marker is re-recorded and the PR retries next poll (defect 2).

    Returns the list of handled rows for the caller/tests.
    """
    handled: list[dict] = []
    # r4 (HIGH C) — reclaim stale in-flight markers FIRST: a crashed Zeus's
    # consumed-but-never-acted advice re-enters this pipeline (or is dropped
    # when a newer escalation took over), so it cannot block a PR forever.
    # Fresh markers are left alone — Zeus may still be mid-act.
    try:
        for rec in state.reclaim_stale_inflight():
            log.info("escalation in-flight marker reclaimed for %s#%s: %s",
                     rec.get("repo"), rec.get("pr"), rec.get("action"))
    except Exception as exc:  # noqa: BLE001 — the poll loop still runs
        log.warning("stale in-flight reclaim failed: %s", exc)
    for row in state.escalating_prs():
        repo = row["repo"]
        pr = row["pr"]
        head_sha = row["head_sha"]
        advice_id = row.get("advice_id", "")
        installation_id = row.get("installation_id")
        started_at = row.get("started_at") or row["created_at"].timestamp()
        waited = time.time() - started_at

        # r4 MEDIUM 2 — a COMMITTED decision is never parked: its builder
        # JOB is repaired instead (idempotently), so the round is not lost
        # and no contradictory PARKED row is appended to a successful
        # advice.  The repair needs no advisor — the decision carries the
        # (redacted) plan the job is built from.
        decision = _committed_decision(state, repo, pr, advice_id)
        if decision is not None and _recorded_plan(decision).get("steps"):
            out = _consume_and_act(
                state, repo, pr, head_sha, advice_id, installation_id,
                lambda: _repair_escalated_round(state, jobs, repo, pr,
                                                head_sha, decision),
                started_at=started_at, detail=row.get("detail"),
            )
            if out is None:
                handled.append({"repo": repo, "pr": pr, "action": "dropped",
                                "reason": "superseded by a newer escalation"})
            else:
                handled.append({"repo": repo, "pr": pr, **out})
            continue
        # A decision whose plan was not recorded (written before the plan
        # was stored) cannot be rebuilt from the row: it is still never
        # parked — the advice stays pending and the normal poll may still
        # complete the round.
        decided = decision is not None

        if waited > timeout_secs and not decided:
            if not advice_id:
                # A marker without an advice_id can never be polled: consume it
                # under the per-PR lock ONLY if it is still the newest marker (a
                # newer advice submitted meanwhile is never deleted), then park.
                reason = f"escalation marker without advice_id waited > {timeout_secs}s"
                out = _consume_and_act(
                    state, repo, pr, head_sha, "", installation_id,
                    lambda: _park_pr(state, jobs, repo, pr, head_sha, reason),
                    started_at=started_at, detail=row.get("detail"),
                )
                if out is None:
                    handled.append({"repo": repo, "pr": pr, "action": "dropped",
                                    "reason": "superseded by a newer escalation"})
                else:
                    handled.append({"repo": repo, "pr": pr, **out})
                continue
            out = _consume_and_act(
                state, repo, pr, head_sha, advice_id, installation_id,
                lambda: _park_pr(
                    state, jobs, repo, pr, head_sha,
                    f"advisor did not finish within {timeout_secs}s",
                ),
                started_at=started_at, detail=row.get("detail"),
            )
            if out is None:
                handled.append({"repo": repo, "pr": pr, "action": "dropped",
                                "reason": "superseded by a newer escalation"})
            else:
                handled.append({"repo": repo, "pr": pr, **out})
            continue
        if not advice_id:
            # No advice_id recorded yet — transient; next poll (timeout bounds).
            continue

        try:
            status = poll_advice(cfg, advice_id)
        except AdvisorNotFound as exc:
            if decided:
                # r4 MEDIUM 2: the decision is committed — the advisor lost
                # ITS record, the decision is not.  The advice stays pending
                # (never parked): a successful advice is not a failure.
                out = _consume_and_act(
                    state, repo, pr, head_sha, advice_id, installation_id,
                    lambda: _never_park(
                        f"advisor lost the record for {advice_id}: {exc}"),
                    started_at=started_at, detail=row.get("detail"),
                )
                if out is None:
                    handled.append({"repo": repo, "pr": pr,
                                    "action": "dropped",
                                    "reason": "superseded by a newer "
                                              "escalation"})
                else:
                    handled.append({"repo": repo, "pr": pr, **out})
                continue
            # Extra A: the advisor restarted and lost the record.  Park
            # immediately with the reason — no 30-minute wait.
            lost_reason = f"advisor lost the record for {advice_id}: {exc}"
            out = _consume_and_act(
                state, repo, pr, head_sha, advice_id, installation_id,
                lambda: _park_pr(state, jobs, repo, pr, head_sha, lost_reason),
                started_at=started_at, detail=row.get("detail"),
            )
            if out is None:
                handled.append({"repo": repo, "pr": pr, "action": "dropped",
                                "reason": "superseded by a newer escalation"})
            else:
                handled.append({"repo": repo, "pr": pr, **out})
            continue
        except EscalationError as exc:
            log.warning("poll escalation %s#%s: %s (retry next poll)",
                        repo, pr, exc)
            continue

        if status.status == "done":
            # Re-check inside the handling transaction that this advice is
            # still the newest for the PR; a newer escalation supersedes it.
            work_order = WorkOrder.from_dict(status.work_order)
            out = _consume_and_act(
                state, repo, pr, head_sha, advice_id, installation_id,
                lambda: escalate_pr(
                    cfg, state, jobs, repo, pr, head_sha, work_order,
                    installation_id=installation_id,
                    advice_id=advice_id,
                ),
                started_at=started_at, detail=row.get("detail"),
            )
            if out is None:
                log.info("escalation %s#%s advice %s superseded by a newer "
                         "escalation — dropped, no round", repo, pr, advice_id)
                handled.append({"repo": repo, "pr": pr, "action": "dropped",
                                "reason": "superseded by a newer escalation"})
            else:
                handled.append({"repo": repo, "pr": pr, **out})
        elif status.status == "failed":
            if decided:
                # r4 MEDIUM 2: the decision is committed — the advice is not
                # parked (never a contradictory PARKED row next to a
                # successful escalation): it stays pending.
                out = _consume_and_act(
                    state, repo, pr, head_sha, advice_id, installation_id,
                    lambda: _never_park(
                        f"advisor failed: {status.error or 'unknown'}"),
                    started_at=started_at, detail=row.get("detail"),
                )
                if out is None:
                    handled.append({"repo": repo, "pr": pr,
                                    "action": "dropped",
                                    "reason": "superseded by a newer "
                                              "escalation"})
                else:
                    handled.append({"repo": repo, "pr": pr, **out})
                continue
            out = _consume_and_act(
                state, repo, pr, head_sha, advice_id, installation_id,
                lambda: _park_pr(
                    state, jobs, repo, pr, head_sha,
                    f"advisor failed: {status.error or 'unknown'}",
                ),
                started_at=started_at, detail=row.get("detail"),
            )
            if out is None:
                handled.append({"repo": repo, "pr": pr, "action": "dropped",
                                "reason": "superseded by a newer escalation"})
            else:
                handled.append({"repo": repo, "pr": pr, **out})
        else:
            # queued / running — keep waiting.
            log.debug("escalation %s#%s@%s status=%s (waiting)",
                      repo, pr, head_sha[:8], status.status)
    return handled


def _park_pr(state, jobs, repo, pr, head_sha, reason: str) -> dict:
    """Park a PR with a one-line reason (no more rounds)."""
    state.record(repo, pr, head_sha, "parked", verdict="PARKED",
                 detail={"reason": reason})
    log.warning("parked %s#%s@%s: %s", repo, pr, head_sha[:8], reason)
    return {"action": "parked", "reason": reason}
