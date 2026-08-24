"""rhobear-neo entrypoint: webhook server + a background worker queue.

The webhook handler must return 200 fast (GitHub retries slow hooks), but a Neo run
takes minutes. So the dispatcher just enqueues normalized verdict wakes; a worker
thread drains the queue and runs the protocol one PR at a time.
"""
from __future__ import annotations

import logging
import queue
import threading

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

    q: "queue.Queue[dict]" = queue.Queue(maxsize=1000)

    def on_verdict(wake: dict) -> None:
        try:
            q.put_nowait(wake)
        except queue.Full:
            log.warning("verdict queue full — dropping %s", wake.get("repo"))

    def worker() -> None:
        while True:
            wake = q.get()
            try:
                run_neo(cfg, state, wake)
            except Exception:
                log.exception("run_neo crashed on %s", wake.get("repo"))
            finally:
                q.task_done()

    threading.Thread(target=worker, name="neo-worker", daemon=True).start()
    log.info("rhobear-neo up — app_id=%s orgs=%s auto_merge_default=%s",
             cfg.__dict__.get("_app_id", "4347986"), ",".join(cfg.orgs), cfg.auto_merge_default)
    serve(cfg, NeoDispatcher(cfg, on_verdict))


if __name__ == "__main__":
    main()
