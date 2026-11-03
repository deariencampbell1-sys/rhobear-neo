"""Kronos checks — each returns ok / warn / broken with a one-line reason.

Nothing here mutates anything. External touches (systemctl, journalctl,
HTTP, Postgres) are injected so tests build fakes inside themselves:

  * ``runner(argv) -> result``  subprocess shape (rc, stdout, stderr);
  * ``http(method, url, ...) -> (status, body)``;
  * ``db.rows(sql, params) -> list[dict]``  (read-only SQL).

The live journal needs root, which the unit will not have: the tail comes
from `journalctl` when it is readable and falls back to `systemctl status`
(systemd prints the unit's recent lines itself, unprivileged) — what you can
see without root, never more.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit

OK, WARN, BROKEN = "ok", "warn", "broken"


@dataclass
class CheckResult:
    name: str
    status: str          # OK | WARN | BROKEN
    reason: str          # one line, safe to post

    def line(self) -> str:
        return f"kronos check {self.name}: {self.status} {self.reason}"


def _iso_ts(value) -> float:
    """ISO-8601 (GitHub `delivered_at`, Postgres timestr) -> epoch, 0 on junk."""
    if value is None:
        return 0.0
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except (TypeError, ValueError):
            return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# HTTP health — the three loopback services answer POST-only webhooks, so
# "answering an HTTP request at all" is the health signal (r2, MEDIUM 2: a
# real bounded HTTP request — TCP-connect-only called a service that binds
# but never answers healthy). Connection refused / timeout / no response is
# broken; ANY status (even 404 on a GET against a POST-only webhook) is ok.
# ---------------------------------------------------------------------------

def _default_http_request(url: str, timeout: float) -> tuple[int, str]:
    """One bounded GET. Returns (status, body); raises on no-HTTP."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(4096).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(4096).decode("utf-8", "replace")


def http_health(name: str, url: str, *, timeout: float = 5.0,
                request=None) -> CheckResult:
    request = request or _default_http_request
    try:
        status, _body = request(url, timeout)
    except Exception as exc:  # noqa: BLE001 — one line, no body dumped
        return CheckResult(name, BROKEN,
                           f"no HTTP answer from {url}: "
                           f"{type(exc).__name__}")
    return CheckResult(name, OK,
                       f"answering HTTP {status} on {url}")


# ---------------------------------------------------------------------------
# Unit health
# ---------------------------------------------------------------------------

def unit_active(name: str, unit: str, *, runner) -> CheckResult:
    """`systemctl is-active <unit>` — active is OK, everything else broken
    with the exit status of the last run."""
    res = runner(["systemctl", "is-active", unit])
    state = (getattr(res, "stdout", "") or "").strip().lower()
    if state == "active":
        return CheckResult(name, OK, f"{unit} active")
    detail = ""
    show = runner(["systemctl", "show", unit, "-p", "ExecMainStatus",
                   "-p", "ActiveState", "-p", "SubState", "--value"])
    vals = [v.strip() for v in
            (getattr(show, "stdout", "") or "").splitlines() if v.strip()]
    if len(vals) >= 3:
        detail = f" (exit status {vals[0]}, {vals[1]}/{vals[2]})"
    return CheckResult(name, BROKEN, f"{unit} {state or 'unknown'}{detail}")


def journal_tail(unit: str, *, runner, lines: int = 20) -> str:
    """The unit's last `lines` journal lines — via journalctl when readable,
    else via `systemctl status` (unprivileged, systemd prints them).

    r2 (MEDIUM 2): reads subprocess' REAL field (returncode) — the r1 code
    looked for an `rc` attribute a CompletedProcess does not carry.
    """
    res = runner(["journalctl", "-u", unit, "-n", str(lines),
                  "--no-pager", "-o", "short-iso"])
    out = (getattr(res, "stdout", "") or "").strip()
    if getattr(res, "returncode", 1) == 0 and out:
        return out
    status = runner(["systemctl", "status", unit, "-n", str(lines),
                     "--no-pager", "-l"])
    fallback = ((getattr(status, "stdout", "") or "").strip())
    return fallback or (
        f"journal unreadable without root "
        f"(returncode={getattr(res, 'returncode', '?')})")


def nrestarts(unit: str, *, runner) -> int:
    """`systemctl show <unit> -p NRestarts` — restarts since unit load.

    r3 (MEDIUM 3): a failed or malformed probe RAISES into the caller's
    'check:flap-detect' alert branch — the r2 code silently read a denied
    probe (returncode != 0) or garbage output as zero restarts, leaving a
    BLIND flap detector that looked healthy."""
    res = runner(["systemctl", "show", unit, "-p", "NRestarts", "--value"])
    rc = getattr(res, "returncode", 1)
    out = (getattr(res, "stdout", "") or "").strip()
    if rc != 0:
        raise RuntimeError(
            f"systemctl show {unit} NRestarts failed (rc={rc}) — flap "
            f"detection is blind")
    try:
        return int(out)
    except ValueError:
        raise RuntimeError(
            f"systemctl show {unit} NRestarts returned malformed output — "
            f"flap detection is blind") from None


def flap_status(state: dict, unit: str, *, nrestarts: int, now: float,
                threshold: int = 5, window_secs: int = 600):
    """Edge 2 — a unit that restarts five times in ten minutes is reported
    ONCE, with the count.

    Pure: `state` is the flap slot (per unit: base count, window start);
    returns (should_report, new_state). After a report the window re-arms
    from the current count; a window older than `window_secs` resets first.

    r3 (MEDIUM 2): the slot keeps `reported_at` — the time the LAST report
    was raised — and PRESERVES it across window resets, so the resolution
    pass can close the alert after a quiet window even though the detection
    window re-armed (r2 reset window_start on aging, which kept a healthy
    unit's flap alert open forever)."""
    slot = state.setdefault(unit, {"base": nrestarts, "window_start": now})
    if now - slot["window_start"] > window_secs:
        prev_reported = slot.get("reported_at")
        slot = state[unit] = {"base": nrestarts, "window_start": now}
        if prev_reported:
            slot["reported_at"] = prev_reported
    climbed = nrestarts - slot["base"]
    if climbed >= threshold:
        state[unit] = {"base": nrestarts, "window_start": now,
                       "reported_at": now}
        return True, state
    return False, state


def flap_report_body(unit: str, *, count: int, window_secs: int) -> str:
    return (f"{unit} restarted {count} times in "
            f"{window_secs // 60} minutes.")


# ---------------------------------------------------------------------------
# Database checks (read-only SQL over the shared reviews DSN)
# ---------------------------------------------------------------------------

def postgres_reachable(name: str, dsn: str, *, db, timeout: float = 5.0) -> CheckResult:
    if not dsn:
        return CheckResult(name, WARN,
                           "REVIEWS_DATABASE_URL not configured — loud, "
                           "never silent")
    try:
        db.rows("SELECT 1 AS ok", ())
    except Exception as exc:  # noqa: BLE001 — the reason is one line, no DSN
        return CheckResult(name, BROKEN,
                           f"postgres unreachable: {type(exc).__name__}")
    return CheckResult(name, OK, "postgres reachable")


_FAILED_DELIVERY_SQL_WHYS = ("status_code>=400 or status not OK")


def failed_deliveries(name: str, *, app_name: str, list_deliveries,
                      hours: float = 1.0, now: float = 0.0) -> tuple:
    """Failed webhook deliveries in the last hour for ONE App.

    `list_deliveries()` returns the delivery dicts (GET /app/hook/deliveries
    with the App's JWT). A delivery is failed when its status_code is
    400-599 or its status says anything but OK. Returns (results, failed)
    where failed is the list for the remedy.
    """
    try:
        deliveries = list_deliveries()
    except Exception as exc:  # noqa: BLE001
        return (CheckResult(name, BROKEN,
                            f"{app_name}: deliveries unreadable: "
                            f"{type(exc).__name__}"), [])
    failed = []
    for d in deliveries or []:
        ts = _iso_ts(d.get("delivered_at"))
        if now and ts and now - ts > hours * 3600:
            continue
        code = d.get("status_code") or 0
        status = str(d.get("status") or "")
        if (isinstance(code, int) and code >= 400) or \
                (status and status.upper() not in ("OK",)):
            failed.append(d)
    if failed:
        ids = ", ".join(str(d.get("id")) for d in failed[:5])
        return (CheckResult(name, BROKEN,
                            f"{app_name}: {len(failed)} failed delivery/ies "
                            f"in the last hour (ids {ids})"), failed)
    return (CheckResult(name, OK, f"{app_name}: no failed deliveries "
                                  f"in the last hour"), [])


def stalled_reviews(name: str, *, db, mins: float = 30.0) -> CheckResult:
    """PRs with a webhook delivery but no Apollo review after `mins`.

    review_jobs rows exist only because a webhook delivery created them
    (delivery_id); a row still non-terminal after 30 minutes means the
    delivery arrived and no review was published for it.
    """
    try:
        rows = db.rows(
            "SELECT repo, pr_number, head_sha, state, created_at "
            "FROM review_jobs "
            "WHERE delivery_id IS NOT NULL "
            "AND state IN ('queued','leased','inferred','publishing','failed') "
            "AND created_at < now() - make_interval(mins => %s) "
            "ORDER BY created_at LIMIT 10",
            (mins,),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, BROKEN,
                           f"review_jobs unreadable: {type(exc).__name__}")
    if not rows:
        return CheckResult(name, OK, "no stalled reviews")
    first = rows[0]
    return CheckResult(name, BROKEN,
                       f"{len(rows)}+ PRs with a delivery but no review "
                       f"after {mins:g}m (e.g. {first['repo']}#"
                       f"{first['pr_number']} {first['state']})")


def stuck_jobs(name: str, *, db, mins: float = 30.0) -> CheckResult:
    """Zeus jobs in a non-final phase for over `mins` (neo_jobs: a builder
    round queued or running for 30+ minutes is stuck)."""
    try:
        rows = db.rows(
            "SELECT id, repo, pr_number, builder_round, state, updated_at "
            "FROM neo_jobs "
            "WHERE kind = 'builder' "
            "AND state IN ('queued','running') "
            "AND updated_at < now() - make_interval(mins => %s) "
            "ORDER BY updated_at LIMIT 10",
            (mins,),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, BROKEN,
                           f"neo_jobs unreadable: {type(exc).__name__}")
    if not rows:
        return CheckResult(name, OK, "no stuck Zeus jobs")
    first = rows[0]
    return CheckResult(name, BROKEN,
                       f"{len(rows)}+ Zeus jobs non-final for {mins:g}m+ "
                       f"(e.g. job {first['id']} {first['repo']}#"
                       f"{first['pr_number']} r{first['builder_round']} "
                       f"{first['state']})")


def credit_gated_installs(name: str, *, db, owner_install_id: str) -> CheckResult:
    """Installs gated for credits — alert, never silent.

    The exact gate Zeus applies (meters_credits): a paying install (a plan
    set) that is not the owner workspace, with balance <= 0.
    """
    try:
        rows = db.rows(
            "SELECT install_id, org, plan, credits_balance "
            "FROM neo_installs "
            "WHERE COALESCE(plan, '') <> '' "
            "AND install_id <> %s "
            "AND credits_balance <= 0",
            (owner_install_id,),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, BROKEN,
                           f"neo_installs unreadable: {type(exc).__name__}")
    if not rows:
        return CheckResult(name, OK, "no credit-gated installs")
    ids = ", ".join(str(r["install_id"]) for r in rows[:5])
    return CheckResult(name, BROKEN,
                       f"{len(rows)} install(s) gated for credits: {ids}")


# ---------------------------------------------------------------------------
# r2 alert-resolution queries (read-only) — every alert kind can CLOSE
# ---------------------------------------------------------------------------

def loop_resolved(db, repo: str, pr: int) -> bool:
    """The judged-loop alert (`loop:o/r#52`) closes on Zeus's REAL terminal
    shapes (r3 MEDIUM 2): a merged row, or a ready row (the worker ACCEPTed)
    bound to the NEWEST builder head — the same-head evidence the merge rule
    produces. The r2 signal (two DISTINCT ACCEPT verdicts on builder rows)
    never occurs: accepted work is recorded as phase='ready'/'merged'."""
    if db is None:
        return False
    merged = db.rows(
        "SELECT 1 FROM neo_actions WHERE repo = %s AND pr_number = %s "
        "AND phase = 'merged' LIMIT 1", (repo, pr))
    if merged:
        return True
    accepted = db.rows(
        "SELECT 1 FROM neo_actions WHERE repo = %s AND pr_number = %s "
        "AND phase = 'ready' "
        "AND head_sha = (SELECT head_sha FROM neo_actions "
        "                WHERE repo = %s AND pr_number = %s "
        "                AND phase = 'builder' ORDER BY id DESC LIMIT 1) "
        "LIMIT 1", (repo, pr, repo, pr))
    return bool(accepted)


def list_deliveries_via_app(http_opener=None):
    """Factory: GET /app/hook/deliveries with the given App JWT (page 1).
    (Moved here r2 — it is a CHECK input, not a remedy; the remedies module
    is retired by the read-only ruling.)

    r3 (MEDIUM 1): the GET's timeout is a parameter — the caller clamps it
    to the remaining pass budget instead of a free-standing 30 s."""
    from urllib.request import Request, build_opener

    def _list(jwt: str, *, per_page: int = 100, timeout: float = 30):
        opener = http_opener or build_opener()
        req = Request(
            f"{API_GITHUB}/app/hook/deliveries?per_page={per_page}",
            method="GET", headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {jwt}",
                "X-GitHub-Api-Version": "2022-11-28",
            })
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    return _list


API_GITHUB = "https://api.github.com"
