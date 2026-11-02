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

r3 changes (ASTRA re-review, issuecomment-6578963903):

- HIGH 3: the judge sweep's SQL itself excludes terminal (merged/ready) and
  already-judged (escalated) rows and PRs Zeus is still moving, and pages
  FAIRLY through a persisted keyset cursor — the fixed `ORDER BY ... LIMIT
  20` starved every PR behind the first 20 historical rows forever.
- HIGH 5: the App JWT construction and the delivery probe run INSIDE
  run_check — a JWT failure is a BROKEN deliveries:<app> check that alerts
  with a non-secret reason, never a crashed pass.
- HIGH 6: the judge's installation identity is read from the AUTHORITATIVE
  rows Zeus writes (neo_jobs.payload, or a recorded identity on an action
  row) — never from the builder action detail Zeus does not write. An
  unproven identity defers loudly (no bare consumer, no repo-lookup token).
- HIGH 7: every committed decision's notifications go through the durable
  outbox (src/kronos/judge.py), drained at the start of every judge sweep —
  a failed PR comment can no longer swallow the owner alert.
- MEDIUM 1: Postgres connects with bounded connect/statement timeouts, and
  the delivery probe, every alerter (dispatch AND resolution) and the PR
  commenter all clamp their HTTP timeouts to the remaining budget.
- MEDIUM 2: flap and flap-detect alerts now register and CLOSE (quiet
  window / probe recovered), and loop alerts close on Zeus's real terminal
  shapes.
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
from .judge import (drain_notify_outbox, judge_pr,
                    reconstruct_pending_notifications)
from .store import ROUND_CEILING, SqlWrites

log = logging.getLogger("rhobear_neo.kronos")

#: per-action subprocess budget ceiling (clamped to the remaining budget)
_RUNNER_TIMEOUT_SECS = 30
_HTTP_TIMEOUT_SECS = 30
_DB_TIMEOUT_SECS = 15


class BudgetSpent(RuntimeError):
    """r4 MEDIUM E — the pass budget is gone: no NEW call may start. The
    work is deferred to the next pass instead of overrunning the deadline."""


def _mint_timeout(left_fn) -> int:
    """r4 MEDIUM E — a token mint's HTTP timeout is bounded by what is LEFT
    of the pass budget (the App client's own default is a free-standing
    30 s)."""
    if left_fn is None:
        return _HTTP_TIMEOUT_SECS
    return max(1, min(_HTTP_TIMEOUT_SECS, int(left_fn())))

#: The judge sweep's page (HIGH 3, r3): filtered, quiet, undecided PRs only,
#: keyset-paged past the persisted cursor. The NOT EXISTS clause drops
#: terminal rows (merged/ready) and decided rows (any escalated marker with
#: an advisor_round — Kronos's stop, Kronos's escalation, Zeus's own).
#: r4 (HIGH B): the NOT EXISTS is correlated on BOTH columns of the OUTER
#: row — `other.repo = neo_actions.repo` (the unqualified `repo` used to
#: bind to the INNER neo_actions row, so a same-numbered PR in ANOTHER
#: repository suppressed this PR from every page forever).
_JUDGE_SWEEP_SQL = (
    "SELECT repo, pr_number, "
    "MAX(builder_round) AS rounds, "
    "(array_agg(head_sha ORDER BY id DESC))[1] AS head_sha "
    "FROM neo_actions "
    "WHERE phase = 'builder' "
    "GROUP BY repo, pr_number "
    "HAVING MAX(builder_round) >= %s "
    "AND MAX(builder_round) <= %s "
    "AND MAX(created_at) < now() - make_interval(secs => %s) "
    "AND NOT EXISTS ("
    "  SELECT 1 FROM neo_actions AS other "
    "  WHERE other.repo = neo_actions.repo "
    "    AND other.pr_number = neo_actions.pr_number "
    "    AND (other.phase IN ('merged', 'ready') "
    "         OR (other.phase = 'escalated' "
    "             AND other.detail->'advisor_round' IS NOT NULL))) "
    "AND (repo, pr_number) > (%s, %s) "
    "ORDER BY repo, pr_number "
    "LIMIT %s"
)


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
    def db(dsn: str, *, connect_timeout: int | None = None,
           statement_timeout_ms: int | None = None):
        import psycopg
        from psycopg.rows import dict_row

        kwargs = {}
        if connect_timeout:
            kwargs["connect_timeout"] = max(1, int(connect_timeout))

        class _Db:
            """READ-ONLY. The judge's writes go through Services.writes()
            (src/kronos/store.py) — never through this object (HIGH 2).

            MEDIUM 1 (r3): the connection is bounded — a connect timeout
            and a per-statement timeout, both inside the pass budget."""

            _st_ms = statement_timeout_ms

            def rows(self, sql: str, params=()):
                with psycopg.connect(dsn, autocommit=True,
                                     row_factory=dict_row, **kwargs) as conn:
                    if self._st_ms:
                        # SET cannot take a bound parameter: inline the
                        # validated int.
                        conn.execute(f"SET statement_timeout = "
                                     f"{max(1, int(self._st_ms))}")
                    cur = conn.execute(sql, params)
                    try:
                        return list(cur.fetchall())
                    except psycopg.ProgrammingError:
                        return []
        return _Db()

    # -- the judge's atomic CAS writer (HIGH 2 + HIGH 6) -------------------
    @staticmethod
    def writes(dsn: str, *, quiet_secs: int | None = None,
               connect_timeout: int | None = None,
               statement_timeout_ms: int | None = None):
        kw = {}
        if quiet_secs is not None:
            kw["quiet_secs"] = quiet_secs
        if connect_timeout is not None:
            kw["connect_timeout"] = connect_timeout
        if statement_timeout_ms is not None:
            kw["statement_timeout_ms"] = statement_timeout_ms
        return SqlWrites(dsn, **kw)

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

    db = services.db(cfg.reviews_database_url,
                     connect_timeout=clamp(_DB_TIMEOUT_SECS),
                     statement_timeout_ms=clamp(_DB_TIMEOUT_SECS) * 1000) \
        if cfg.reviews_database_url else None
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
            # r3 HIGH 5 — the JWT construction AND the delivery probe are
            # INSIDE the check: a JWT failure is a BROKEN result that
            # alerts with a non-secret reason, never a crashed pass.
            results.append(run_check(
                f"deliveries:{app}",
                lambda a=app: _deliveries_result(cfg, a, now_fn=now_fn,
                                                 lister_timeout=clamp(
                                                     _HTTP_TIMEOUT_SECS))))
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
    flap_blind: set[str] = set()   # MEDIUM 2 — units whose probe failed
    if not budget_gone("alerts"):
        # flap detection per unit (edge 2); a blind detector alerts loudly
        def _restarts(unit: str) -> int:
            # MEDIUM 3 (r3) — a failed probe RAISES (never a silent zero).
            return nrestarts(unit, runner=bounded_runner)

        flap_slot = state.slot("flaps")
        for unit in cfg.systemd_units:
            try:
                restarts = _restarts(unit)
            except Exception as exc:  # noqa: BLE001 — HIGH 8: loud, not blind
                flap_blind.add(unit)
                _alert_once(cfg, state, out,
                            fingerprint=f"check:flap-detect:{unit}",
                            subject=(f"Kronos: cannot read restart counts "
                                     f"for {unit}"),
                            body=f"flap detection is blind this pass: "
                                 f"{type(exc).__name__}",
                            now_fn=now_fn, services=services, left_fn=left)
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
                            now_fn=now_fn, services=services, left_fn=left)
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
                            now_fn=now_fn, services=services, left_fn=left)
            else:
                _alert_once(cfg, state, out,
                            fingerprint=f"check:{res.name}",
                            subject=f"Kronos: {res.name} is broken",
                            body=res.reason, now_fn=now_fn,
                            services=services, left_fn=left)
        state.flush()   # HIGH 1 — the dedupe ledger persists at dispatch

    # -- 3. the judge ------------------------------------------------------
    if not budget_gone("the judge"):
        _judge_sweep(cfg, state, db, services, out, now_fn=now_fn,
                     left_fn=left, clamp=clamp)
        state.flush()   # HIGH 1 — judged markers persist at dispatch

    # -- 4. resolve alerts whose problem is gone ---------------------------
    if not budget_gone("alert resolution"):
        _resolve_fixed(cfg, state, results, db, out, now_fn=now_fn,
                       services=services, left_fn=left,
                       flap_blind=flap_blind)

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


def _deliveries_result(cfg: KronosConfig, app: str, *, now_fn,
                       lister_timeout: int) -> CheckResult:
    """The deliveries check for ONE app — r3 HIGH 5: the App JWT is
    constructed HERE (inside run_check's try). A JWT failure is a BROKEN
    result with a stable, NON-SECRET reason (the exception class only)."""
    name = f"deliveries:{app}"
    try:
        jwt = _app_jwt(cfg, app)
    except Exception as exc:  # noqa: BLE001 — one line, nothing secret
        return CheckResult(
            name, BROKEN,
            f"{app} App JWT could not be minted: {type(exc).__name__} — "
            f"deliveries unchecked, alerting")
    res, _failed = failed_deliveries(
        name, app_name=app,
        list_deliveries=_delivery_lister(jwt, timeout=lister_timeout),
        hours=1.0, now=now_fn())
    return res


def _delivery_lister(jwt: str, *, timeout: int = 30):
    lister = list_deliveries_via_app()
    return lambda: lister(jwt, timeout=timeout)


def _alerter(cfg: KronosConfig, state: KronosState, now_fn,
             services=None, left_fn=None) -> Alerter:
    from ..github_app import GitHubApp
    client = GithubIssueClient(
        cfg.alert_repo,
        # r4 MEDIUM E — the token MINT is bounded by the budget too: the
        # App client's HTTP timeout is what is LEFT, never a free 30 s.
        lambda: GitHubApp(cfg.zeus_app_id, cfg.zeus_app_key,
                          http_timeout=_mint_timeout(left_fn)
                          ).token_for_repo(cfg.alert_repo),
        label=cfg.alert_label,
        http=(None if (services is None or left_fn is None)
              else (lambda method, url, *, headers=None, body=None,
                    timeout=_HTTP_TIMEOUT_SECS: services.http(
                        method, url, headers=headers, body=body,
                        timeout=max(1, min(int(timeout), int(left_fn())))))))
    return Alerter(client, state, now_fn=now_fn,
                   window_secs=cfg.alert_window_secs, label=cfg.alert_label)


def _alert_once(cfg, state, out, *, fingerprint, subject, body, now_fn,
                services=None, left_fn=None):
    """One alert dispatch — r3 MEDIUM 1: the alerter's HTTP is built WITH
    the pass budget (its issue-list/create/comment calls clamp to what is
    left, never a free-standing 30 s)."""
    try:
        action = _alerter(cfg, state, now_fn, services=services,
                          left_fn=left_fn).alert(fingerprint, subject, body)
        out(f"kronos alert: {fingerprint} {action}")
        if action != "suppressed":
            state.slot("open_alerts")[fingerprint] = now_iso()
    except Exception as exc:  # noqa: BLE001 — an alert failure never stops
        out(f"kronos alert: {fingerprint} failed: {type(exc).__name__}")


def _readonly_token(cfg, repo: str, install_id: str,
                    *, left_fn=None) -> str:
    """The judge's read-only installation token (never the write token).
    r4 MEDIUM E — the mint never starts past the deadline (BudgetSpent) and
    its HTTP timeout is bounded by what is LEFT of the pass budget."""
    from ..github_app import GitHubApp
    if left_fn is not None and left_fn() <= 0:
        raise BudgetSpent(
            "pass budget exhausted before the read-only token mint")
    app = GitHubApp(cfg.zeus_app_id, cfg.zeus_app_key,
                    http_timeout=_mint_timeout(left_fn))
    return app.token_for_repo(repo, installation_id=int(install_id) if
                              install_id.isdigit() else None,
                              permissions={"contents": "read",
                                           "pull_requests": "read"})


def _pr_commenter(services, cfg, left_fn=None):
    """Post the judge's ONE PR comment as the Zeus App (timeout clamped to
    the remaining pass budget, MEDIUM 3). r4 MEDIUM E — the token mint is
    bounded and never starts past the deadline: a BudgetSpent leaves the
    comment pending in the outbox for the next pass."""
    from ..github_app import GitHubApp

    def _comment(repo: str, pr: int, body: str) -> None:
        if left_fn is not None and left_fn() <= 0:
            raise BudgetSpent(
                "pass budget exhausted before the PR comment")
        app = GitHubApp(cfg.zeus_app_id, cfg.zeus_app_key,
                        http_timeout=_mint_timeout(left_fn))
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


def install_id_for(db, repo: str, pr: int) -> str:
    """r3 HIGH 6 — the PR's installation identity from the AUTHORITATIVE
    rows Zeus writes: neo_jobs.payload (the job's payload carries the
    webhook installation id), else a recorded identity on an action row.
    Zeus's builder ACTION rows do NOT carry it (their detail is green /
    wake_job / route metadata), so the r2 code read the wrong place and
    always degraded to the bare consumer. Returns '' = unproven."""
    try:
        rows = db.rows(
            "SELECT payload FROM neo_jobs WHERE kind='builder' "
            "AND repo=%s AND pr_number=%s ORDER BY id DESC LIMIT 1",
            (repo, pr))
    except Exception as exc:  # noqa: BLE001 — read failures defer below
        log.info("kronos judge: neo_jobs unreadable for %s#%s: %s",
                 repo, pr, type(exc).__name__)
        rows = []
    for r in rows:
        val = _json_field(r.get("payload"), "installation_id")
        if val not in (None, ""):
            return str(val)
    try:
        rows = db.rows(
            "SELECT detail FROM neo_actions WHERE repo=%s AND pr_number=%s "
            "AND detail->>'installation_id' IS NOT NULL "
            "ORDER BY id DESC LIMIT 1", (repo, pr))
    except Exception as exc:  # noqa: BLE001
        log.info("kronos judge: neo_actions unreadable for %s#%s: %s",
                 repo, pr, type(exc).__name__)
        rows = []
    for r in rows:
        val = _json_field(r.get("detail"), "installation_id")
        if val not in (None, ""):
            return str(val)
    return ""


def _json_field(value, key: str):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        return value.get(key)
    return None


def _judge_sweep(cfg, state, db, services, out, *, now_fn, left_fn, clamp):
    """Every PR at 5+ builder rounds without two greens gets judged once.

    r3 HIGH 3 — the page is FILTERED (terminal/judged/quiet rows are
    dropped in SQL, see _JUDGE_SWEEP_SQL) and FAIR: a persisted keyset
    cursor walks the qualifying set across passes, so no PR starves behind
    a fixed LIMIT. Per-PR budget checks and clamped timeouts (MEDIUM 3)."""
    if db is None:
        return
    writes = services.writes(cfg.reviews_database_url,
                             quiet_secs=cfg.zeus_quiet_secs,
                             connect_timeout=clamp(_DB_TIMEOUT_SECS),
                             statement_timeout_ms=clamp(_DB_TIMEOUT_SECS)
                             * 1000)
    alerter = _alerter(cfg, state, now_fn, services=services,
                       left_fn=left_fn)
    comments = _pr_commenter(services, cfg, left_fn)

    def _on_notify_delivered(entry: dict) -> None:
        """r4 HIGH D — the durable delivered-flag on the decision's marker
        row: the reconstruction never re-creates a notified entry."""
        try:
            writes.mark_notified(str(entry.get("repo")),
                                 int(entry.get("pr") or 0))
        except Exception as exc:  # noqa: BLE001 — at-least-once alerting
            log.info("kronos notify: delivered-flag write failed for "
                     "%s#%s (%s) — the reconstruction may notify once more",
                     entry.get("repo"), entry.get("pr"), type(exc).__name__)

    # r4 HIGH D — a committed decision's notifications survive a process
    # death between the DB COMMIT and the state flush: rebuild what the DB
    # says is un-notified BEFORE draining.
    try:
        recreated = reconstruct_pending_notifications(db, state)
        if recreated:
            out(f"kronos judge: reconstructed {recreated} pending "
                f"notification(s) from committed decisions")
    except Exception as exc:  # noqa: BLE001 — the sweep still runs
        out(f"kronos judge: notification reconstruction failed: "
            f"{type(exc).__name__}")
    # r3 HIGH 7 — deliver what a previous pass still owes (comment/alert).
    # r4 MEDIUM E — the drain respects the same global deadline.
    drain_notify_outbox(state, comments, alerter, budget_left=left_fn,
                        on_delivered=_on_notify_delivered)
    cursor_slot = state.slot("judge_cursor")
    cur_repo = str(cursor_slot.get("repo") or "")
    cur_pr = int(cursor_slot.get("pr") or 0)
    try:
        rows = db.rows(
            _JUDGE_SWEEP_SQL,
            (5, 6, cfg.zeus_quiet_secs, cur_repo, cur_pr,
             cfg.judge_page_size),
        )
    except Exception as exc:  # noqa: BLE001
        out(f"kronos judge: rounds unreadable: {type(exc).__name__}")
        return
    last_processed: tuple[str, int] | None = None
    for row in rows:
        if left_fn() <= 0:
            out("kronos judge: pass budget gone — remaining PRs are "
                "deferred to the next pass (the cursor keeps the rest)")
            break
        repo, pr = str(row["repo"]), int(row["pr_number"])
        install_id = install_id_for(db, repo, pr)
        try:
            # r4 MEDIUM E — the clock is checked AGAIN right before the
            # mint: a token mint never starts past the deadline.
            read_token = _readonly_token(cfg, repo, install_id,
                                         left_fn=left_fn)
        except BudgetSpent:
            out(f"kronos judge: {repo}#{pr}: pass budget exhausted before "
                f"the token mint — deferred to the next pass")
            break
        except Exception as exc:  # noqa: BLE001 — loud, then judge anyway
            out(f"kronos judge: {repo}#{pr}: read-only "
                f"token unavailable ({type(exc).__name__}) — judging "
                f"without the credential header")
            read_token = ""
        try:
            outcome = judge_pr(
                cfg, db, comments=comments,
                alerter=alerter, state=state,
                repo=repo, pr=pr,
                head_sha=str(row.get("head_sha") or ""),
                install_id=install_id,
                read_token=read_token,
                http=services.http, now_fn=now_fn, writes=writes,
                github=services.http, github_timeout=clamp(15),
                apollo_login=cfg.apollo_login,
                timeout=clamp(cfg.judge_timeout_secs))
        except Exception as exc:  # noqa: BLE001 — one PR never stops the sweep
            out(f"kronos judge: {repo}#{pr} failed: "
                f"{type(exc).__name__}")
            last_processed = (repo, pr)
            continue
        out(f"kronos judge: {repo}#{pr} -> "
            f"{outcome.action} ({outcome.reason})")
        last_processed = (repo, pr)
    # r3 HIGH 3 — persist the cursor: past the last PROCESSED row (a budget
    # break never skips unprocessed rows), and wrap to the top when the
    # qualifying set is exhausted.
    if not rows or len(rows) < cfg.judge_page_size:
        cursor_slot.clear()
    elif last_processed is not None:
        cursor_slot["repo"], cursor_slot["pr"] = last_processed
    state.flush()


def _resolve_fixed(cfg, state, results, db, out, *, now_fn,
                   services=None, left_fn=None, flap_blind=None):
    """Commented and closed when fixed — for EVERY alert kind (HIGH 8).

    A resolve that raises keeps the open_alerts entry so the next pass
    retries the close; an entry is dropped only after a successful close.

    r3 MEDIUM 2 — the kinds that could never close now do:
      * `loop:` closes on Zeus's real terminal shapes (checks.loop_resolved);
      * `flap:` closes after a quiet window measured from the REPORT (the
        detection window's reset no longer hides it);
      * `check:flap-detect:` closes when the unit is ok AND the restart
        probe succeeded again;
      * `job-conflict:` closes when the blocking job is gone or the PR left
        the loop."""
    open_slot = state.slot("open_alerts")
    ok_names = {r.name for r in results if r.status == OK}
    flap_slot = state.slot("flaps")
    flap_blind = flap_blind or set()

    for fingerprint in list(open_slot):
        note = None
        try:
            if fingerprint.startswith("unit-down:"):
                unit = fingerprint.split(":", 1)[1]
                if f"unit:{unit}" not in ok_names:
                    continue
                note = f"{unit} is active again"
            elif fingerprint.startswith("check:flap-detect:"):
                # the fingerprint carries TWO colons before the unit name
                unit = fingerprint.split(":", 2)[2]
                if f"unit:{unit}" not in ok_names or unit in flap_blind:
                    continue
                note = (f"{unit}'s restart probe succeeded — flap "
                        f"detection can see again")
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
                reported_at = float(window.get("reported_at") or 0)
                if not reported_at or \
                        now_fn() - reported_at <= cfg.flap_window_secs:
                    continue  # still inside the quiet window since the report
                note = f"{unit} settled — no new restarts since the report"
            elif fingerprint.startswith("loop:"):
                rest = fingerprint.split(":", 1)[1]
                repo, _, pr = rest.rpartition("#")
                if not repo or not pr.lstrip("-").isdigit():
                    continue  # unparseable: keep, never silently drop
                if not loop_resolved(db, repo, int(pr)):
                    continue
                note = f"{repo}#{pr} is out of the loop (merged or accepted)"
            elif fingerprint.startswith("job-conflict:"):
                rest = fingerprint.split(":", 1)[1]
                repo, _, pr = rest.rpartition("#")
                if not repo or not pr.lstrip("-").isdigit():
                    continue
                if loop_resolved(db, repo, int(pr)):
                    note = f"{repo}#{pr} left the loop"
                elif not _round_job_exists(db, repo, int(pr),
                                           _conflict_round(state, repo,
                                                           int(pr))):
                    note = "the blocking job is gone"
                else:
                    continue
            else:
                continue  # unknown kind: never dropped by a guess
            _alerter(cfg, state, now_fn, services=services,
                     left_fn=left_fn).resolved(fingerprint, note)
            open_slot.pop(fingerprint, None)
            out(f"kronos alert: {fingerprint} resolved (closed)")
        except Exception as exc:  # noqa: BLE001 — retried next pass
            out(f"kronos alert: {fingerprint} resolve failed (kept open): "
                f"{type(exc).__name__}: {exc}")


def _conflict_round(state, repo: str, pr: int) -> int:
    """r4 MEDIUM F — the round whose fresh-only insert conflicted (recorded
    when the job-conflict alert fired). Unknown -> the round-six ceiling:
    the judge can only ever conflict on round 6 (it refuses at 6 stored
    rounds, so its new round is always 6)."""
    try:
        value = (state.slot("job_conflicts") or {}).get(f"{repo}#{pr}")
        return int(value) if value is not None else ROUND_CEILING
    except (TypeError, ValueError):
        return ROUND_CEILING


def _round_job_exists(db, repo: str, pr: int, builder_round: int) -> bool:
    """r4 MEDIUM F — True while the CONFLICTING round's builder job still
    exists in ANY state (queued, running, parked, done — the unique key
    blocks the fresh-only insert regardless). The r3 check matched only
    queued/running, so a PARKED or DONE blocking job closed the alert while
    the row still prevented every future write."""
    if db is None:
        return True  # cannot prove it is gone: stay open
    rows = db.rows(
        "SELECT 1 FROM neo_jobs WHERE kind='builder' AND repo=%s "
        "AND pr_number=%s AND builder_round=%s LIMIT 1",
        (repo, pr, int(builder_round)))
    return bool(rows)


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
