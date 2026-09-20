"""rhobear-neo entrypoint: webhook server + parallel triage workers.

The webhook handler must return 200 fast (GitHub retries slow hooks), but a Neo run
takes minutes. So the dispatcher submits normalized verdict wakes to a
ThreadPoolExecutor (NEO_MAX_WORKERS, default 8) and returns immediately.
"""
from __future__ import annotations

import logging
import os
import concurrent.futures

from . import config as _config
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

    # Parallel triage: up to NEO_MAX_WORKERS agent runs concurrent. run_neo is
    # thread-safe — subprocess calls (gh, hermes) + independent Postgres
    # connections, no shared mutable Python state. The idempotency guard
    # (already_acted, color-aware) dedups on (repo, pr, sha, green), so the
    # rare concurrent duplicate for the same sha is a wasted agent run, not a
    # correctness issue.
    max_workers = max(1, int(os.environ.get("NEO_MAX_WORKERS", "8")))
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="neo-worker",
    )

    def on_verdict(wake: dict) -> None:
        try:
            executor.submit(_run_wake, wake)
        except RuntimeError:
            log.warning("neo executor shutting down — dropping %s", wake.get("repo"))

    def _run_wake(wake: dict) -> None:
        try:
            run_neo(cfg, state, wake)
        except Exception:
            log.exception("run_neo crashed on %s", wake.get("repo"))

    log.info("rhobear-neo up — app_id=%s orgs=%s auto_merge_default=%s max_workers=%s",
             cfg.__dict__.get("_app_id", "4347986"), ",".join(cfg.orgs), cfg.auto_merge_default, max_workers)
    serve(cfg, NeoDispatcher(cfg, on_verdict))


if __name__ == "__main__":
    main()
