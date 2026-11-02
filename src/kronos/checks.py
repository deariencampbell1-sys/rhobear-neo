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

import socket
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
# "listening and answering" is the health signal (connection refused/timeout
# is broken; any HTTP response means the process is alive).
# ---------------------------------------------------------------------------

def http_health(name: str, url: str, *, timeout: float = 5.0,
                connect=None) -> CheckResult:
    parts = urlsplit(url)
    host, port = parts.hostname or "127.0.0.1", parts.port or 80
    connect = connect or socket.create_connection
    try:
        with connect((host, port), timeout=timeout):
            return CheckResult(name, OK, f"listening on {host}:{port}")
    except OSError as exc:
        return CheckResult(name, BROKEN,
                           f"not listening on {host}:{port}: "
                           f"{type(exc).__name__}")


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
    else via `systemctl status` (unprivileged, systemd prints them)."""
    res = runner(["journalctl", "-u", unit, "-n", str(lines),
                  "--no-pager", "-o", "short-iso"])
    out = (getattr(res, "stdout", "") or "").strip()
    if getattr(res, "rc", 0) == 0 and out:
        return out
    status = runner(["systemctl", "status", unit, "-n", str(lines),
                     "--no-pager", "-l"])
    return ((getattr(status, "stdout", "") or "")
            or f"journal unreadable without root "
               f"(rc={getattr(res, 'rc', '?')})")


def nrestarts(unit: str, *, runner) -> int:
    """`systemctl show <unit> -p NRestarts` — restarts since unit load."""
    res = runner(["systemctl", "show", unit, "-p", "NRestarts", "--value"])
    try:
        return int((getattr(res, "stdout", "") or "0").strip() or 0)
    except ValueError:
        return 0


def flap_status(state: dict, unit: str, *, nrestarts: int, now: float,
                threshold: int = 5, window_secs: int = 600):
    """Edge 2 — a unit that restarts five times in ten minutes is reported
    ONCE, with the count.

    Pure: `state` is the flap slot (per unit: base count, window start);
    returns (should_report, new_state). After a report the window re-arms
    from the current count; a window older than `window_secs` resets first.
    """
    slot = state.setdefault(unit, {"base": nrestarts, "window_start": now})
    if now - slot["window_start"] > window_secs:
        slot = state[unit] = {"base": nrestarts, "window_start": now}
    climbed = nrestarts - slot["base"]
    if climbed >= threshold:
        state[unit] = {"base": nrestarts, "window_start": now,
                       "reported": True}
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
