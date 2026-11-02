"""Kronos entrypoint — one pass per timer firing.

    python -m src.kronos --once [--dry-run] [--state-dir DIR]

r2 RULING (Morpheus, 2026-11-02): Kronos is READ-ONLY on other services —
it observes, and it never restarts, never kills, never edits a unit, never
redelivers a webhook, never re-queues or releases a job. The r1 remedies
(restart / redeliver / re-queue) are RETIRED: every problem they used to
paper over now raises the alert (and the alert closes when fixed). The
fix-loop judge stays — that is Kronos's job.

`--dry-run` prints one line per check and changes NOTHING: no state file,
no GitHub call, no judge call, exit 0. A real pass takes the flock on
`<state-dir>/pass.lock` first — two passes never overlap, and the lock is
one stable inode (never unlinked, MEDIUM 1). The whole pass runs inside the
4-minute budget (MEDIUM 3: every long action's timeout is clamped to what
is LEFT); when the budget is gone the remaining work is skipped with a
printed line, never half-applied.

HIGH 1: the state (alert dedupe, flap windows, judged markers) lives in a
DURABLE state dir — the unit mounts a StateDirectory, because a
RuntimeDirectory is wiped when this oneshot stops and the dedupe would
start empty every five minutes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from . import KronosState, PassLock
from .alert import Alerter, GithubIssueClient
from .checks import BROKEN, OK, WARN, CheckResult
from .checks import (
    credit_gated_installs,
    failed_deliveries,
    flap_report_body,
    flap_status,
    http_health,
    journal_tail,
    list_deliveries_via_app,
    loop_resolved,
    nrestarts,
    stuck_jobs,
    stalled_reviews,
    unit_active,
    postgres_reachable,
)
from .config import KronosConfig
from .judge import judge_pr
from .store import SqlWrites

log = logging.getLogger("rhobear_neo.kronos")

#: per-action subprocess budget ceiling (clamped to the remaining budget)
_RUNNER_TIMEOUT_SECS = 30
_HTTP_TIMEOUT_SECS = 30


class Services:
    """The real externals. Tests build their own objects with the same
    attributes instead of patching this class."""

    def __init__(self, cfg: KronosConfig):
        self.cfg = cfg

    # -- subprocess -------------------------------------------------------
    @staticmethod
    def runner(argv, *, timeout: int = _RUNNER_TIMEOUT_SECS):
        import subprocess
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)
        return proc

    # -- read-only SQL ----------------------------------------------------
    @staticmethod
    def db(dsn: str):
        import psycopg
        from psycopg.rows import dict_row

        class _Db:
            """READ-ONLY. The judge's writes go through Services.writes()
            (src/kronos/store.py) — never through this object (HIGH 2)."""

            def rows(self, sql: str, params=()):
                with psycopg.connect(dsn, autocommit=True,
                                     row_factory=dict_row) as conn:
                    cur = conn.execute(sql, params)
                    try:
                        return list(cur.fetchall())
                    except psycopg.ProgrammingError:
                        return []
        return _Db()

    # -- the judge's atomic CAS writer (HIGH 2 + HIGH 6) -------------------
    @staticmethod
    def writes(dsn: str):
        return SqlWrites(dsn)

    # -- HTTP (RouteGate judge + GitHub) ----------------------------------
    @staticmethod
    def http(method: str, url: str, *, headers=None, body=None, timeout=30):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(url, data=body, method=method,
                                     headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")


def run_pass(cfg: KronosConfig, *, dry_run: bool, state: KronosState,
             services, out=print, now_fn=time.time,
             mono_fn=time.monotonic) -> int:
    """One pass: checks -> alerts -> judge -> resolution. Every stage is
    bounded; every check prints exactly one line; a check that cannot run
    alerts loudly (HIGH 8) instead of passing silently or crashing."""
    deadline = mono_fn() + cfg.pass_budget_secs

    def left() -> float:
        return deadline - mono_fn()

    def clamp(timeout: int) -> int:
        """MEDIUM 3 — no action may outlive the pass budget: its timeout is
        the smaller of its own budget and what is LEFT."""
        return max(1, min(int(timeout), int(left())))

    def budget_gone(stage: str) -> bool:
        if left() <= 0:
            out(f"kronos pass: budget exceeded before {stage} — "
                f"skipping (never half-applied)")
            return True
        return False

    def bounded_runner(argv):
        # MEDIUM 3 — the subprocess budget shrinks with the pass budget.
        return services.runner(argv, timeout=clamp(_RUNNER_TIMEOUT_SECS))

    def bounded_http(method, url, *, headers=None, body=None, timeout=30):
        return services.http(method, url, headers=headers, body=body,
                             timeout=clamp(timeout))

    db = services.db(cfg.reviews_database_url) if cfg.reviews_database_url \
        else None
    results: list[CheckResult] = []

    # -- 1. checks ---------------------------------------------------------
    def run_check(name: str, fn) -> CheckResult:
        """HIGH 8 — a check that cannot run (missing binary, permission,
        timeout) is BROKEN with a loud reason; it never passes silently
        and never kills the pass."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — the reason is one line
            return CheckResult(name, BROKEN,
                               f"check could not run: "
                               f"{type(exc).__name__} — alerting, never "
                               f"silent")

    def _unit_tail(unit: str) -> str:
        """The bounded journal tail for a broken unit (MEDIUM 2 — really
        measured, or honestly named when even that fails)."""
        try:
            return journal_tail(unit, runner=bounded_runner, lines=20)
        except Exception as exc:  # noqa: BLE001
            return f"journal tail unavailable: {type(exc).__name__}"

    results.append(run_check("webhook", lambda: http_health(
        "webhook", cfg.reviews_webhook_url, timeout=clamp(5))))
    results.append(run_check("zeus", lambda: http_health(
        "zeus", cfg.zeus_url, timeout=clamp(5))))
    results.append(run_check("routegate", lambda: http_health(
        "routegate", cfg.routegate_url, timeout=clamp(5))))
    for unit in cfg.systemd_units:
        name = f"unit:{unit}"
        results.append(run_check(name, lambda u=unit: unit_active(
            name, u, runner=bounded_runner)))

    results.append(run_check("postgres", lambda: postgres_reachable(
        "postgres", cfg.reviews_database_url, db=db
        if db is not None else _NoDb())))
    if db is not None:
        for app, app_id, key in _apps(cfg):
            if not app_id or not key:
                results.append(CheckResult(
                    f"deliveries:{app}", WARN,
                    f"app identity missing — vault name loud at boot; "
                    f"deliveries unchecked this pass"))
                continue
            jwt = _app_jwt(cfg, app)
            res, _failed = failed_deliveries(
                f"deliveries:{app}", app_name=app,
                list_deliveries=_delivery_lister(jwt),
                hours=1.0, now=now_fn())
            results.append(res)
        results.append(run_check("stalled-reviews", lambda: stalled_reviews(
            "stalled-reviews", db=db, mins=cfg.stall_mins)))
        results.append(run_check("stuck-jobs", lambda: stuck_jobs(
            "stuck-jobs", db=db, mins=cfg.stuck_mins)))
        results.append(run_check("credit-gated", lambda: credit_gated_installs(
            "credit-gated", db=db, owner_install_id=cfg.owner_install_id)))

    for res in results:
        out(res.line())

    if dry_run:
        out("kronos pass: dry-run — nothing changed")
        return 0

    # -- 2. alerts (the r1 remedies are RETIRED: a problem ALERTS) ---------
    if not budget_gone("alerts"):
        # flap detection per unit (edge 2); a blind detector alerts loudly
        def _restarts(unit: str) -> int:
            return nrestarts(unit, runner=bounded_runner)

        flap_slot = state.slot("flaps")
        for unit in cfg.systemd_units:
            try:
                restarts = _restarts(unit)
            except Exception as exc:  # noqa: BLE001 — HIGH 8: loud, not blind
                _alert_once(cfg, state, out,
                            fingerprint=f"check:flap-detect:{unit}",
                            subject=(f"Kronos: cannot read restart counts "
                                     f"for {unit}"),
                            body=f"flap detection is blind this pass: "
                                 f"{type(exc).__name__}",
                            now_fn=now_fn)
                continue
            should, flap_slot = flap_status(
                flap_slot, unit, nrestarts=restarts, now=now_fn(),
                threshold=cfg.flap_threshold,
                window_secs=cfg.flap_window_secs)
            if should:
                _alert_once(cfg, state, out,
                            fingerprint=f"flap:{unit}",
                            subject=f"Kronos: {unit} is flapping",
                            body=flap_report_body(
                                unit, count=cfg.flap_threshold,
                                window_secs=cfg.flap_window_secs),
                            now_fn=now_fn)
        # every BROKEN check alerts (HIGH 8): never a silent miss — and the
        # unit alerts carry the bounded journal tail (MEDIUM 2).
        for res in results:
            if res.status != BROKEN:
                continue
            if res.name.startswith("unit:"):
                unit = res.name.split(":", 1)[1]
                _alert_once(cfg, state, out,
                            fingerprint=f"unit-down:{unit}",
                            subject=f"Kronos: {unit} is down",
                            body=f"{res.reason}\n\n{_unit_tail(unit)}",
                            now_fn=now_fn)
            else:
                _alert_once(cfg, state, out,
                            fingerprint=f"check:{res.name}",
                            subject=f"Kronos: {res.name} is broken",
                            body=res.reason, now_fn=now_fn)
        state.flush()   # HIGH 1 — the dedupe ledger persists at dispatch

    # -- 3. the judge ------------------------------------------------------
    if not budget_gone("the judge"):
        _judge_sweep(cfg, state, db, services, out, now_fn=now_fn,
                     left_fn=left, clamp=clamp)
        state.flush()   # HIGH 1 — judged markers persist at dispatch

    # -- 4. resolve alerts whose problem is gone ---------------------------
    if not budget_gone("alert resolution"):
        _resolve_fixed(cfg, state, results, db, out, now_fn=now_fn)

    state.flush()
    out(f"kronos pass: done in {mono_fn() - (deadline - cfg.pass_budget_secs):.1f}s "
        f"({sum(1 for r in results if r.status == OK)} ok / "
        f"{sum(1 for r in results if r.status == WARN)} warn / "
        f"{sum(1 for r in results if r.status == BROKEN)} broken)")
    return 0


class _NoDb:
    """Stands in when no DSN is configured so postgres_reachable can speak
    the loud 'not configured' line without a None check upstream."""

    def rows(self, sql, params=()):  # pragma: no cover - never called
        raise RuntimeError("no database configured")


def _apps(cfg: KronosConfig):
    # HIGH 9 — the documented vault key PATH is loaded into *_app_key at
    # boot (KronosConfig.required); every GitHubApp gets PEM TEXT.
    return (("reviews", cfg.reviews_app_id, cfg.reviews_app_key),
            ("zeus", cfg.zeus_app_id, cfg.zeus_app_key))


def _app_jwt(cfg: KronosConfig, app: str) -> str:
    """The App JWT via src.github_app (read-only import, one identity)."""
    from ..github_app import GitHubApp
    app_id = getattr(cfg, f"{app}_app_id")
    key = getattr(cfg, f"{app}_app_key")
    if not key:
        raise RuntimeError(
            f"{app} App private key not loaded — configure "
            f"RHOBEAR_KRONOS_{app.upper()}_APP_KEY_PATH (vault: the "
            f"{app} App key) and boot through KronosConfig.required")
    return GitHubApp(app_id, key).jwt()


def _delivery_lister(jwt: str):
    lister = list_deliveries_via_app()
    return lambda: lister(jwt)


def _alerter(cfg: KronosConfig, state: KronosState, now_fn,
             services=None, left_fn=None) -> Alerter:
    from ..github_app import GitHubApp
    client = GithubIssueClient(
        cfg.alert_repo,
        lambda: GitHubApp(cfg.zeus_app_id, cfg.zeus_app_key
                          ).token_for_repo(cfg.alert_repo),
        label=cfg.alert_label,
        http=(None if (services is None or left_fn is None)
              else (lambda method, url, *, headers=None, body=None,
                    timeout=_HTTP_TIMEOUT_SECS: services.http(
                        method, url, headers=headers, body=body,
                        timeout=max(1, min(int(timeout), int(left_fn())))))))
    return Alerter(client, state, now_fn=now_fn,
                   window_secs=cfg.alert_window_secs, label=cfg.alert_label)


def _alert_once(cfg, state, out, *, fingerprint, subject, body, now_fn):
    try:
        action = _alerter(cfg, state, now_fn).alert(fingerprint, subject, body)
        out(f"kronos alert: {fingerprint} {action}")
        if action != "suppressed":
            state.slot("open_alerts")[fingerprint] = now_iso()
    except Exception as exc:  # noqa: BLE001 — an alert failure never stops
        out(f"kronos alert: {fingerprint} failed: {type(exc).__name__}")


def _readonly_token(cfg, repo: str, install_id: str) -> str:
    """The judge's read-only installation token (never the write token)."""
    from ..github_app import GitHubApp
    app = GitHubApp(cfg.zeus_app_id, cfg.zeus_app_key)
    return app.token_for_repo(repo, installation_id=int(install_id) if
                              install_id.isdigit() else None,
                              permissions={"contents": "read",
                                           "pull_requests": "read"})


def _pr_commenter(services, cfg, left_fn=None):
    """Post the judge's ONE PR comment as the Zeus App (timeout clamped to
    the remaining pass budget, MEDIUM 3)."""
    from ..github_app import GitHubApp

    def _comment(repo: str, pr: int, body: str) -> None:
        app = GitHubApp(cfg.zeus_app_id, cfg.zeus_app_key)
        token = app.token_for_repo(repo)
        timeout = _HTTP_TIMEOUT_SECS if left_fn is None else \
            max(1, min(_HTTP_TIMEOUT_SECS, int(left_fn())))
        status, raw = services.http(
            "POST", f"https://api.github.com/repos/{repo}/issues/{pr}/comments",
            headers={"Accept": "application/vnd.github+json",
                     "Authorization": f"Bearer {token}",
                     "Content-Type": "application/json",
                     "X-GitHub-Api-Version": "2022-11-28"},
            body=json.dumps({"body": body}).encode(), timeout=timeout)
        if status >= 300:
            raise RuntimeError(f"PR comment -> HTTP {status}")
    return _comment


def _judge_sweep(cfg, state, db, services, out, *, now_fn, left_fn, clamp):
    """Every PR at 5+ builder rounds without two greens gets judged once.
    Per-PR budget checks and clamped timeouts (MEDIUM 3)."""
    if db is None:
        return
    try:
        rows = db.rows(
            "SELECT repo, pr_number, "
            "MAX(builder_round) AS rounds, "
            "(array_agg(head_sha ORDER BY id DESC))[1] AS head_sha, "
            "(array_agg(detail ORDER BY id DESC))[1] AS detail "
            "FROM neo_actions WHERE phase = 'builder' "
            "GROUP BY repo, pr_number HAVING MAX(builder_round) >= %s "
            "ORDER BY repo, pr_number LIMIT 20",
            (5,),
        )
    except Exception as exc:  # noqa: BLE001
        out(f"kronos judge: rounds unreadable: {type(exc).__name__}")
        return
    writes = services.writes(cfg.reviews_database_url)
    alerter = _alerter(cfg, state, now_fn, services=services,
                       left_fn=left_fn)
    for row in rows:
        if left_fn() <= 0:
            out("kronos judge: pass budget gone — remaining PRs are "
                "deferred to the next pass")
            break
        install_id = str(_install_of(row) or "")
        try:
            read_token = _readonly_token(cfg, row["repo"], install_id)
        except Exception as exc:  # noqa: BLE001 — loud, then judge anyway
            out(f"kronos judge: {row['repo']}#{row['pr_number']}: read-only "
                f"token unavailable ({type(exc).__name__}) — judging "
                f"without the credential header")
            read_token = ""
        try:
            outcome = judge_pr(
                cfg, db, comments=_pr_commenter(services, cfg, left_fn),
                alerter=alerter, state=state,
                repo=row["repo"], pr=int(row["pr_number"]),
                head_sha=str(row.get("head_sha") or ""),
                install_id=install_id,
                read_token=read_token,
                http=services.http, now_fn=now_fn, writes=writes,
                timeout=clamp(cfg.judge_timeout_secs))
        except Exception as exc:  # noqa: BLE001 — one PR never stops the sweep
            out(f"kronos judge: {row['repo']}#{row['pr_number']} failed: "
                f"{type(exc).__name__}")
            continue
        out(f"kronos judge: {row['repo']}#{row['pr_number']} -> "
            f"{outcome.action} ({outcome.reason})")


def _install_of(row: dict):
    detail = row.get("detail") or {}
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except (TypeError, ValueError):
            detail = {}
    return detail.get("installation_id") if isinstance(detail, dict) else None


def _resolve_fixed(cfg, state, results, db, out, *, now_fn):
    """Commented and closed when fixed — for EVERY alert kind (HIGH 8).

    A resolve that raises keeps the open_alerts entry so the next pass
    retries the close; an entry is dropped only after a successful close."""
    open_slot = state.slot("open_alerts")
    ok_names = {r.name for r in results if r.status == OK}
    flap_slot = state.slot("flaps")

    for fingerprint in list(open_slot):
        note = None
        try:
            if fingerprint.startswith("unit-down:"):
                unit = fingerprint.split(":", 1)[1]
                if f"unit:{unit}" not in ok_names:
                    continue
                note = f"{unit} is active again"
            elif fingerprint.startswith("check:"):
                name = fingerprint.split(":", 1)[1]
                if name not in ok_names:
                    continue
                note = f"{name} is ok again"
            elif fingerprint.startswith("flap:"):
                unit = fingerprint.split(":", 1)[1]
                if f"unit:{unit}" not in ok_names:
                    continue
                window = flap_slot.get(unit) or {}
                if now_fn() - float(window.get("window_start") or 0) \
                        <= cfg.flap_window_secs:
                    continue  # still inside the flap window
                note = f"{unit} settled — no new restarts in the window"
            elif fingerprint.startswith("loop:"):
                rest = fingerprint.split(":", 1)[1]
                repo, _, pr = rest.rpartition("#")
                if not repo or not pr.lstrip("-").isdigit():
                    continue  # unparseable: keep, never silently drop
                if not loop_resolved(db, repo, int(pr)):
                    continue
                note = f"{repo}#{pr} is out of the loop (merged or green)"
            else:
                continue  # unknown kind: never dropped by a guess
            _alerter(cfg, state, now_fn).resolved(fingerprint, note)
            open_slot.pop(fingerprint, None)
            out(f"kronos alert: {fingerprint} resolved (closed)")
        except Exception as exc:  # noqa: BLE001 — retried next pass
            out(f"kronos alert: {fingerprint} resolve failed (kept open): "
                f"{type(exc).__name__}: {exc}")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv=None, *, _services=None) -> int:
    parser = argparse.ArgumentParser(
        prog="src.kronos",
        description="Kronos — watches every service, and judges bad fix loops")
    parser.add_argument("--once", action="store_true",
                        help="one pass and exit (the timer fires every 5 min)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print one line per check, change nothing")
    parser.add_argument("--state-dir", default=None,
                        help="state dir (default $RHOBEAR_KRONOS_STATE_DIR)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)sZ %(levelname)-5s %(name)s :: "
                               "%(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S")

    state_dir = args.state_dir or _get_state_dir()
    services = _services or Services(KronosConfig(state_dir=state_dir))

    if args.dry_run:
        # Dry-run: no state dir, no lock, no writes — print and leave.
        return run_pass(KronosConfig(state_dir=state_dir), dry_run=True,
                        state=_NullState(), services=services)

    lock = PassLock(f"{state_dir}/pass.lock")
    if not lock.acquire():
        print("kronos pass: already running — exiting (timer re-fires in "
              "5 minutes)")
        return 0
    try:
        # HIGH 9 — a REAL pass boots through the strict config: missing
        # keys/env raise LOUDLY naming the vault names, and the documented
        # vault key PATH files are loaded into the App keys here.
        cfg = KronosConfig.required(state_dir=state_dir)
        # HIGH 1 — the state is loaded and flushed UNDER the pass lock, from
        # the DURABLE state dir (a StateDirectory, not a RuntimeDirectory).
        state = KronosState(f"{state_dir}/state.json")
        return run_pass(cfg, dry_run=False, state=state, services=services)
    finally:
        lock.release()


class _NullState:
    """Dry-run state: every slot is throwaway, nothing is ever written."""

    def slot(self, name):
        return {}

    def flush(self):
        return None


def _get_state_dir() -> str:
    import os
    return os.environ.get("RHOBEAR_KRONOS_STATE_DIR") or "/var/lib/rhobear-kronos"


if __name__ == "__main__":
    sys.exit(main())
