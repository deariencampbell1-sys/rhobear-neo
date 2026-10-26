"""rhobear-neo entrypoint: webhook server + parallel triage workers.

The webhook handler must return 200 fast (GitHub retries slow hooks), but a Neo run
takes minutes. So the dispatcher submits normalized verdict wakes to a
ThreadPoolExecutor (NEO_MAX_WORKERS, default 8) and returns immediately.

Durable jobs: every wake is persisted to neo_jobs BEFORE the worker picks it up
(keyed by repo + head SHA + verdict color — the PR number is resolved later and
may be NULL at create time, which is how `status` events, which carry no PR, are
never dropped).  If the executor is saturated the job stays queued — nothing is
ever dropped.

A lease-recovery loop (the only background loop in this process, and it touches
only the local job store — no external polling, no notifications) re-queues
expired wake leases and re-submits orphaned queued wakes, so a crashed worker or
a restarted process cannot strand a head SHA.
"""
from __future__ import annotations

import logging
import os
import threading as _threading
import time
import types
import concurrent.futures

# W1-13-r2 defect 7 — module-local threading alias: this module builds its
# wake-recovery thread through THIS `threading` namespace, so tests patch
# `src.__main__.threading.Thread` without patching the global `threading`
# module (a global patch would hijack every thread any other fixture or
# library starts while such a test runs).  Add members here only as this
# module starts using them.
threading = types.SimpleNamespace(Thread=_threading.Thread)

from . import config as _config
from .github_app import github_app_for
from .jobs import Jobs
from .neo_state import NeoState
from .usage import UsageStore
from .neo_worker import run_neo
from .webhook_server import NeoDispatcher, serve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ %(levelname)-5s %(name)s :: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("rhobear_neo")

# How often the wake lease-recovery loop sweeps expired/orphaned wake jobs.
RECOVERY_SECS = int(os.environ.get("NEO_WAKE_RECOVERY_SECS", "30"))


def _submit_wake(executor, jobs: Jobs, wake: dict, runner) -> bool:
    """Hand a persisted wake to the executor, then stamp the hand-off.

    Order matters: ``stale_queued()`` selects ``submitted_at IS NULL``.  If
    the executor rejects the task (RuntimeError while shutting down) the row
    must stay unstamped, or the recovery sweep would never see it again until
    a restart.  Returns True when the task was accepted.
    """
    try:
        executor.submit(runner, wake)
    except RuntimeError:
        # create_wake() already stamped submitted_at; clear it so the row is
        # visible to stale_queued() and the recovery sweep re-submits it.
        job_id = wake.get("job_id")
        if job_id is not None:
            try:
                jobs.clear_submitted(job_id)
            except Exception:  # noqa: BLE001
                log.exception("clear_submitted failed for job %s", job_id)
        log.warning("neo executor shutting down — wake stays queued for %s",
                    wake.get("repo"))
        return False
    job_id = wake.get("job_id")
    if job_id is not None:
        try:
            jobs.mark_submitted(job_id)
        except Exception:  # noqa: BLE001
            log.exception("mark_submitted failed for job %s", job_id)
    return True


def main() -> None:
    cfg = _config.load().require().validate()

    # Neo acts as the customer's GitHub App — a personal GH_TOKEN is never an
    # identity. Refuse to serve without the App identity (fail closed) and log
    # once if a stale GH_TOKEN is still in the env (it is ignored).
    if cfg.gh_token:
        log.warning("GH_TOKEN is set in the environment but IGNORED — Neo "
                    "authenticates as the GitHub App (NEO_GITHUB_APP_ID + key)")
    if github_app_for(cfg) is None:
        raise RuntimeError(
            "rhobear-neo missing GitHub App identity — set NEO_GITHUB_APP_ID "
            "and NEO_GITHUB_APP_KEY_PATH (or NEO_GITHUB_APP_KEY); a personal "
            "GH_TOKEN is never used as an identity"
        )

    state = NeoState(cfg.database_url)
    state.ensure_schema()

    # Durable job store for wake and builder jobs (one schema definition).
    jobs = Jobs(cfg.database_url)
    jobs.ensure_schema()

    # Per-attempt usage store (B4).
    usage_store = UsageStore(cfg.database_url)
    usage_store.ensure_schema()

    # Parallel triage: up to NEO_MAX_WORKERS agent runs concurrent. run_neo is
    # thread-safe — subprocess calls (gh, hermes) + independent Postgres
    # connections, no shared mutable Python state.
    max_workers = max(1, int(os.environ.get("NEO_MAX_WORKERS", "8")))
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="neo-worker",
    )

    def _submit(wake: dict) -> None:
        # Submit BEFORE stamping submitted_at: a RuntimeError here (executor
        # shutting down) must leave the row unstamped so stale_queued()
        # (submitted_at IS NULL) re-submits it on the next recovery pass.
        _submit_wake(executor, jobs, wake, _run_wake)

    def on_verdict(wake: dict) -> None:
        # Persist the wake as a durable job BEFORE the executor picks it up.
        # Status events carry no pr_number (resolved later via the head SHA), so
        # the job key is (kind, repo, head_sha, verdict color) — pr may be NULL.
        repo = wake.get("repo", "")
        sha = wake.get("sha", "")
        green = bool(wake.get("green"))
        if not (repo and sha):
            log.warning("incomplete wake: %s", wake)
            return
        job_id = jobs.create_wake(repo, sha, green,
                                  pr_number=wake.get("pr_number"),
                                  payload=wake)
        if job_id is None:
            log.debug("duplicate wake %s@%s — an active job already owns it",
                      repo, sha[:8])
            return  # deduped — another queued/running job handles this SHA
        wake["job_id"] = job_id
        log.debug("wake queued %s@%s job=%s", repo, sha[:8], job_id)
        _submit(wake)

    def _run_wake(wake: dict) -> None:
        try:
            run_neo(cfg, state, jobs, wake, usage_store=usage_store)
        except Exception:
            log.exception("run_neo crashed on %s", wake.get("repo"))

    def _recovery_loop() -> None:
        """Re-queue expired wake leases and orphaned queued wakes, re-submitting
        them to the executor, AND poll escalating PRs (async advisor).

        This is lease recovery for the durable job store (a wake the old code
        handled is never dropped), not an external poller.  The escalation
        pass touches only PRs Neo itself marked `escalating` (it does not
        watch the outside world).

        Startup: after a restart every executor task died, so any queued wake
        is an orphan — clear the submitted stamps once so the first
        stale_queued sweep re-submits them all.  Every submission is stamped
        again (mark_submitted), so in steady state a wake waiting in the
        executor is never re-submitted each pass."""
        from . import escalation as _escalation
        try:
            jobs.reset_queued_submissions("wake")
        except Exception:  # noqa: BLE001
            log.exception("startup wake submission reset failed")
        last_esc_poll = 0.0
        while True:
            time.sleep(RECOVERY_SECS)
            try:
                for r in jobs.recover_expired("wake"):
                    repo, sha = r["repo"], r["head_sha"]
                    if r["attempt"] >= r["max_attempts"]:
                        jobs.park(r["id"],
                                  reason=f"lease expired {r['attempt']}/{r['max_attempts']} times")
                        log.info("park wake %s@%s attempt=%s >= max=%s — owner must re-arm",
                                 repo, sha[:8], r["attempt"], r["max_attempts"])
                    else:
                        log.info("recovered expired wake lease %s@%s attempt=%s",
                                 repo, sha[:8], r["attempt"])
                        wake = dict(r.get("payload") or {})
                        wake["job_id"] = r["id"]
                        if wake.get("repo") and wake.get("sha"):
                            _submit(wake)
                # After a restart, executor tasks for queued wakes are lost;
                # re-submit any wake that was NEVER submitted (submitted_at IS
                # NULL — a wake that IS submitted and merely waiting is never
                # re-submitted while it waits).
                for r in jobs.stale_queued("wake", older_than_secs=RECOVERY_SECS):
                    log.info("re-submitting orphaned queued wake %s@%s",
                             r["repo"], r["head_sha"][:8])
                    wake = dict(r.get("payload") or {})
                    wake["job_id"] = r["id"]
                    if wake.get("repo") and wake.get("sha"):
                        _submit(wake)
                # --- Escalation poll (async advisor) --------------------
                # Poll each `escalating` PR every escalation_poll_secs; on
                # done create the ONE escalated builder round, on failure/
                # timeout park with a one-line reason.
                now = time.time()
                if now - last_esc_poll >= cfg.escalation_poll_secs:
                    last_esc_poll = now
                    try:
                        _escalation.recover_escalations(
                            cfg, state, jobs,
                            timeout_secs=cfg.escalation_timeout_secs,
                        )
                    except Exception:  # noqa: BLE001
                        log.exception("escalation poll failed")
            except Exception:
                log.exception("wake lease-recovery loop failed")

    threading.Thread(target=_recovery_loop, daemon=True,
                     name="neo-wake-recovery").start()

    log.info("rhobear-neo up — github_app_id=%s orgs=%s auto_merge_default=%s max_workers=%s",
             cfg.github_app_id, ",".join(cfg.orgs),
             cfg.auto_merge_default, max_workers)
    serve(cfg, NeoDispatcher(cfg, on_verdict))


if __name__ == "__main__":
    main()
