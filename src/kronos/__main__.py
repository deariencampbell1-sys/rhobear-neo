"""Kronos entrypoint — one pass per timer firing.

    python -m src.kronos --once [--dry-run] [--state-dir DIR]

`--dry-run` prints one line per check and changes NOTHING: no state file, no
remedy, no GitHub call, no judge call, exit 0. A real pass takes the flock
on `<state-dir>/pass.lock` first — two passes never overlap; a locked pass
exits immediately and quietly (the timer fires again in 5 minutes). The
whole pass runs inside the 4-minute budget; when the budget is gone the
remaining work is skipped with a printed line, never half-applied.
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
    nrestarts,
    stuck_jobs,
    stalled_reviews,
    unit_active,
    postgres_reachable,
)
from .config import KronosConfig
from .judge import judge_pr
from .remedies import (
    ALLOWED_UNITS,
    RemedyLedger,
    redeliver_delivery,
    restart_unit,
    requeue_stuck_job,
)

log = logging.getLogger("rhobear_neo.kronos")


def _git_timezone_agnostic() -> None:  # pragma: no cover - documentation
    """No-op placeholder: timestamps come from time.time() only."""
    return None


class Services:
    """The real externals. Tests build their own objects with the same
    attributes instead of patching this class."""

    def __init__(self, cfg: KronosConfig):
        self.cfg = cfg

    # -- subprocess -------------------------------------------------------
    @staticmethod
    def runner(argv):
        import subprocess
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=30)
        return proc

    # -- read-only SQL ----------------------------------------------------
    @staticmethod
    def db(dsn: str):
        import psycopg
        from psycopg.rows import dict_row

        class _Db:
            def rows(self, sql: str, params=()):
                with psycopg.connect(dsn, autocommit=True,
                                     row_factory=dict_row) as conn:
                    cur = conn.execute(sql, params)
                    try:
                        return list(cur.fetchall())
                    except psycopg.ProgrammingError:
                        return []
        return _Db()

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
             services, out=print, now_fn=time.time) -> int:
    """One pass: checks -> remedies -> judge -> alerts. Every stage is
    bounded; every check prints exactly one line."""
    started = now_fn()
    deadline = started + cfg.pass_budget_secs

    def left() -> float:
        return deadline - now_fn()

    def budget_gone(stage: str) -> bool:
        if left() <= 0:
            out(f"kronos pass: budget exceeded before {stage} — "
                f"skipping (never half-applied)")
            return True
        return False

    db = services.db(cfg.reviews_database_url) if cfg.reviews_database_url \
        else None
    results: list[CheckResult] = []

    # -- 1. checks ---------------------------------------------------------
    results.append(http_health("webhook", cfg.reviews_webhook_url))
    results.append(http_health("zeus", cfg.zeus_url))
    results.append(http_health("routegate", cfg.routegate_url))
    for unit in cfg.systemd_units:
        name = f"unit:{unit}"
        if unit not in ALLOWED_UNITS:
            results.append(CheckResult(name, WARN, "outside the allow-list"))
            continue
        results.append(unit_active(name, unit, runner=services.runner))

    results.append(postgres_reachable(
        "postgres", cfg.reviews_database_url, db=db
        if db is not None else _NoDb()))
    if db is not None:
        for app, app_id, key in _apps(cfg):
            if not app_id or not key:
                results.append(CheckResult(
                    f"deliveries:{app}", WARN,
                    f"app identity missing — vault name loud at boot; "
                    f"deliveries unchecked this pass"))
                continue
            jwt = _app_jwt(cfg, app)
            res, failed = failed_deliveries(
                f"deliveries:{app}", app_name=app,
                list_deliveries=_delivery_lister(services, jwt),
                hours=1.0, now=now_fn())
            results.append(res)
            if failed and not dry_run and left() > 0:
                _remedy_deliveries(cfg, state, app, jwt, failed, out,
                                   now_fn=now_fn)
        results.append(stalled_reviews("stalled-reviews", db=db,
                                       mins=cfg.stall_mins))
        res = stuck_jobs("stuck-jobs", db=db, mins=cfg.stuck_mins)
        results.append(res)
        results.append(credit_gated_installs(
            "credit-gated", db=db, owner_install_id=cfg.owner_install_id))

    # flap detection per unit (edge 2)
    if not dry_run:
        flap_slot = state.slot("flaps")
        for unit in cfg.systemd_units:
            count = nrestarts(unit, runner=services.runner)
            should, flap_slot = flap_status(
                flap_slot, unit, nrestarts=count, now=now_fn(),
                threshold=cfg.flap_threshold, window_secs=cfg.flap_window_secs)
            if should:
                _alert_once(cfg, state, out,
                            fingerprint=f"flap:{unit}",
                            subject=f"Kronos: {unit} is flapping",
                            body=flap_report_body(
                                unit, count=cfg.flap_threshold,
                                window_secs=cfg.flap_window_secs),
                            now_fn=now_fn)

    for res in results:
        out(res.line())

    if dry_run:
        out("kronos pass: dry-run — nothing changed")
        return 0

    # -- 2. remedies -------------------------------------------------------
    if not budget_gone("remedies"):
        ledger = RemedyLedger(state.slot("remedies"), now_fn=now_fn)
        for res in results:
            if res.status != BROKEN:
                continue
            if res.name.startswith("unit:"):
                unit = res.name.split(":", 1)[1]
                if ledger.may(f"unit:{unit}"):
                    ok, note = restart_unit(services.runner, unit)
                    out(f"kronos remedy: {note}")
                    ledger.did(f"unit:{unit}")
                    _alert_once(cfg, state, out,
                                fingerprint=f"unit-down:{unit}",
                                subject=f"Kronos: {unit} is down",
                                body=f"{res.reason}. {note}",
                                now_fn=now_fn)
                else:
                    _alert_once(cfg, state, out,
                                fingerprint=f"unit-down:{unit}",
                                subject=f"Kronos: {unit} is down",
                                body=res.reason, now_fn=now_fn)
            elif res.name.startswith("stuck-jobs"):
                _remedy_stuck_jobs(cfg, state, db, out, now_fn=now_fn)

    # -- 3. the judge ------------------------------------------------------
    if not budget_gone("the judge"):
        _judge_sweep(cfg, state, db, services, out, now_fn=now_fn)

    # -- 4. resolve alerts whose problem is gone ---------------------------
    if not budget_gone("alert resolution"):
        _resolve_fixed(cfg, state, results, out)

    state.flush()
    out(f"kronos pass: done in {now_fn() - started:.1f}s "
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
    return (("reviews", cfg.reviews_app_id, cfg.reviews_app_key
             or cfg.reviews_app_key_path),
            ("zeus", cfg.zeus_app_id, cfg.zeus_app_key
             or cfg.zeus_app_key_path))


def _app_jwt(cfg: KronosConfig, app: str) -> str:
    """The App JWT via src.github_app (read-only import, one identity)."""
    from ..github_app import GitHubApp
    app_id = getattr(cfg, f"{app}_app_id")
    key = (getattr(cfg, f"{app}_app_key_path")
           or getattr(cfg, f"{app}_app_key"))
    return GitHubApp(app_id, key).jwt()


def _delivery_lister(services, jwt: str):
    from .remedies import list_deliveries_via_app
    lister = list_deliveries_via_app()
    return lambda: lister(jwt)


def _alerter(cfg: KronosConfig, state: KronosState, now_fn) -> Alerter:
    from ..github_app import GitHubApp
    client = GithubIssueClient(
        cfg.alert_repo,
        lambda: GitHubApp(cfg.zeus_app_id,
                          cfg.zeus_app_key_path or cfg.zeus_app_key
                          ).token_for_repo(cfg.alert_repo),
        label=cfg.alert_label)
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


def _remedy_deliveries(cfg, state, app, jwt, failed, out, *, now_fn):
    ledger = RemedyLedger(state.slot("remedies"), now_fn=now_fn)
    for d in failed:
        target = f"delivery:{app}:{d.get('id')}"
        if not ledger.may(target):
            continue
        ok, note = redeliver_delivery(None, app, jwt, d.get("id"))
        out(f"kronos remedy: {note}")
        ledger.did(target)
        if not ok:
            _alert_once(cfg, state, out,
                        fingerprint=f"delivery-failed:{app}",
                        subject=f"Kronos: {app} webhook deliveries failing",
                        body=note, now_fn=now_fn)


def _remedy_stuck_jobs(cfg, state, db, out, *, now_fn):
    if db is None:
        return
    try:
        rows = db.rows(
            "SELECT id, repo, pr_number, builder_round, state, updated_at "
            "FROM neo_jobs WHERE kind = 'builder' "
            "AND state IN ('queued','running') "
            "AND updated_at < now() - make_interval(mins => %s) "
            "ORDER BY updated_at LIMIT 10", (cfg.stuck_mins,))
    except Exception as exc:  # noqa: BLE001
        out(f"kronos remedy: stuck jobs unreadable: {type(exc).__name__}")
        return
    from ..jobs import Jobs
    store = Jobs(cfg.reviews_database_url)
    ledger = RemedyLedger(state.slot("remedies"), now_fn=now_fn)
    for row in rows:
        target = f"job:{row['id']}"
        if not ledger.may(target):
            continue
        ok, note = requeue_stuck_job(store, row)
        out(f"kronos remedy: {note}")
        ledger.did(target)
        if not ok:
            _alert_once(cfg, state, out,
                        fingerprint=f"stuck-job:{row['repo']}#{row['pr_number']}",
                        subject="Kronos: a Zeus job is stuck",
                        body=note, now_fn=now_fn)


def _judge_sweep(cfg, state, db, services, out, *, now_fn):
    """Every PR at 5+ builder rounds without two greens gets judged once."""
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
    alerter = _alerter(cfg, state, now_fn)
    for row in rows:
        install_id = str(_install_of(row) or "")
        read_token = _readonly_token(cfg, row["repo"], install_id)
        try:
            outcome = judge_pr(
                cfg, db, comments=_pr_commenter(services, cfg),
                alerter=alerter, state=state,
                repo=row["repo"], pr=int(row["pr_number"]),
                head_sha=str(row.get("head_sha") or ""),
                install_id=install_id, read_token=read_token,
                http=services.http, now_fn=now_fn)
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


def _readonly_token(cfg, repo: str, install_id: str) -> str:
    """The judge's read-only installation token (never the write token)."""
    from ..github_app import GitHubApp
    app = GitHubApp(cfg.zeus_app_id,
                    cfg.zeus_app_key_path or cfg.zeus_app_key)
    return app.token_for_repo(repo, installation_id=int(install_id) if
                              install_id.isdigit() else None,
                              permissions={"contents": "read",
                                           "pull_requests": "read"})


def _pr_commenter(services, cfg):
    """Post the judge's ONE PR comment as the Zeus App."""
    from ..github_app import GitHubApp

    def _comment(repo: str, pr: int, body: str) -> None:
        app = GitHubApp(cfg.zeus_app_id,
                        cfg.zeus_app_key_path or cfg.zeus_app_key)
        token = app.token_for_repo(repo)
        status, raw = services.http(
            "POST", f"https://api.github.com/repos/{repo}/issues/{pr}/comments",
            headers={"Accept": "application/vnd.github+json",
                     "Authorization": f"Bearer {token}",
                     "Content-Type": "application/json",
                     "X-GitHub-Api-Version": "2022-11-28"},
            body=json.dumps({"body": body}).encode(), timeout=30)
        if status >= 300:
            raise RuntimeError(f"PR comment -> HTTP {status}")
    return _comment


def _resolve_fixed(cfg, state, results, out):
    """Commented and closed when fixed."""
    open_slot = state.slot("open_alerts")
    ok_names = {r.name for r in results if r.status == OK}
    for fingerprint in list(open_slot):
        unit = fingerprint.removeprefix("unit-down:")
        if unit and f"unit:{unit}" in ok_names:
            try:
                _alerter(cfg, state, time.time).resolved(
                    fingerprint, f"{unit} is active again")
                out(f"kronos alert: {fingerprint} resolved (closed)")
            except Exception as exc:  # noqa: BLE001
                out(f"kronos alert: {fingerprint} resolve failed: "
                    f"{type(exc).__name__}")
            open_slot.pop(fingerprint, None)


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
    cfg = KronosConfig(state_dir=state_dir)
    services = _services or Services(cfg)

    if args.dry_run:
        # Dry-run: no state dir, no lock, no writes — print and leave.
        return run_pass(cfg, dry_run=True, state=_NullState(),
                        services=services)

    lock = PassLock(f"{state_dir}/pass.lock")
    if not lock.acquire():
        print("kronos pass: already running — exiting (timer re-fires in "
              "5 minutes)")
        return 0
    try:
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
    return os.environ.get("RHOBEAR_KRONOS_STATE_DIR") or "/run/rhobear-kronos"


if __name__ == "__main__":
    sys.exit(main())
