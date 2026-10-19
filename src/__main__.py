"""rhobear-neo entrypoint: webhook server + parallel triage workers.

The webhook handler must return 200 fast (GitHub retries slow hooks), but a Neo run
takes minutes. So the dispatcher submits normalized verdict wakes to a
ThreadPoolExecutor (NEO_MAX_WORKERS, default 8) and returns immediately.

Durable jobs: wakes are persisted to neo_jobs BEFORE the worker picks them up.
If the executor is saturated, the job stays queued — nothing is ever dropped.
The existing recovery loop handles expired leases.
"""
from __future__ import annotations

import logging
import os
import concurrent.futures

from . import config as _config
from .jobs import Jobs
from .neo_state import NeoState
from .neo_worker import run_neo
from .webhook_server import NeoDispatcher, serve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ %(levelname)-5s %(name)s :: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("rhobear_neo")


def main() -> None:
    cfg = _config.load().require().validate()
    state = NeoState(cfg.database_url)
    state.ensure_schema()

    # Durable job store for wake and builder jobs.
    jobs = Jobs(cfg.database_url)

    # Parallel triage: up to NEO_MAX_WORKERS agent runs concurrent. run_neo is
    # thread-safe — subprocess calls (gh, hermes) + independent Postgres
    # connections, no shared mutable Python state.
    max_workers = max(1, int(os.environ.get("NEO_MAX_WORKERS", "8")))
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="neo-worker",
    )

    def on_verdict(wake: dict) -> None:
        # Persist the wake as a durable job BEFORE the executor picks it up.
        # The job is keyed by (repo, pr_number, head_sha) — UNIQUE constraint
        # deduplicates concurrent duplicate wakes.  If the executor is
        # saturated the job stays queued and will be picked up when a slot
        # frees; nothing is ever dropped.
        repo = wake.get("repo", "")
        pr = wake.get("pr_number") or wake.get("pr")
        sha = wake.get("sha", "")
        if repo and pr and sha:
            job_id = jobs.create(repo, pr, sha, payload=wake)
            if job_id is None:
                log.debug("duplicate wake %s#%s@%s — already queued", repo, pr, sha[:8])
                return  # duplicate — nothing to do
            log.debug("wake queued %s#%s@%s job=%s", repo, pr, sha[:8], job_id)
        else:
            log.warning("incomplete wake: %s", wake)

        def _task():
            try:
                _run_wake(wake)
            except Exception:
                log.exception("run_neo crashed on %s", wake.get("repo"))

        try:
            executor.submit(_task)
        except RuntimeError:
            log.warning("neo executor shutting down — wake stays queued for %s", wake.get("repo"))

    def _run_wake(wake: dict) -> None:
        try:
            run_neo(cfg, state, jobs, wake)
        except Exception:
            log.exception("run_neo crashed on %s", wake.get("repo"))

    log.info("rhobear-neo up — app_id=%s orgs=%s auto_merge_default=%s max_workers=%s",
             cfg.__dict__.get("_app_id", "4347986"), ",".join(cfg.orgs), cfg.auto_merge_default, max_workers)
    serve(cfg, NeoDispatcher(cfg, on_verdict))


if __name__ == "__main__":
    main()
