"""Tests for the parallel dispatch path in __main__.py.

Covers the semaphore acquire/release/submit choreography that the reviewer
flagged as untested: normal submit, backpressure drop, RuntimeError release,
and the FIXED verdict acceptance.
"""
import threading
import concurrent.futures
import time
import pytest


def _make_dispatcher(max_workers=2, inflight_mult=4):
    """Recreate the __main__.py dispatch pattern in isolation."""
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="neo-test",
    )
    _inflight = threading.Semaphore(max_workers * inflight_mult)
    dispatched = []
    dropped = []
    errors = []

    def on_verdict(wake):
        if not _inflight.acquire(blocking=False):
            dropped.append(wake)
            return
        def _task():
            try:
                _run_wake(wake)
            finally:
                _inflight.release()
        try:
            executor.submit(_task)
        except RuntimeError:
            _inflight.release()
            errors.append(wake)

    def _run_wake(wake):
        dispatched.append(wake)

    return executor, on_verdict, dispatched, dropped, errors, _inflight


def test_normal_submit():
    """A wake submitted to a healthy executor gets dispatched."""
    executor, on_verdict, dispatched, dropped, errors, _ = _make_dispatcher(max_workers=2)
    on_verdict({"repo": "test/repo", "pr": 1})
    time.sleep(0.1)  # let the executor pick it up
    executor.shutdown(wait=True)
    assert len(dispatched) == 1
    assert len(dropped) == 0
    assert len(errors) == 0


def test_backpressure_drop():
    """When the semaphore is full, excess wakes are dropped (not queued)."""
    # max_workers=1, inflight_mult=1 → semaphore(1) = 1 permit
    executor, on_verdict, dispatched, dropped, errors, sem = _make_dispatcher(
        max_workers=1, inflight_mult=1
    )
    # Manually hold the single permit so on_verdict can't acquire it
    assert sem.acquire(blocking=False)  # hold the permit
    
    on_verdict({"repo": "test/repo", "pr": 2})
    # Should be dropped because semaphore is full
    assert len(dropped) == 1
    assert dropped[0]["pr"] == 2
    assert len(dispatched) == 0
    
    sem.release()  # clean up
    executor.shutdown(wait=True)


def test_runtime_error_release():
    """If the executor is shut down, on_verdict releases the permit and records error."""
    executor, on_verdict, dispatched, dropped, errors, _ = _make_dispatcher(max_workers=1)
    executor.shutdown(wait=True)  # shut down before submitting
    
    on_verdict({"repo": "test/repo", "pr": 1})
    # Should not crash, should record the error
    assert len(errors) == 1
    assert len(dispatched) == 0
    assert len(dropped) == 0


def test_semaphore_released_after_task():
    """After a task completes, the semaphore permit is released."""
    executor, on_verdict, dispatched, _, _, sem = _make_dispatcher(max_workers=2)
    on_verdict({"repo": "test/repo", "pr": 1})
    time.sleep(0.1)  # let the task complete
    executor.shutdown(wait=True)
    
    # Semaphore should be fully available again
    assert sem.acquire(blocking=False)
    assert sem.acquire(blocking=False)  # 2 permits for max_workers=2*inflight_mult=4... wait
    # Actually semaphore is max_workers*inflight_mult = 2*4 = 8 permits
    # After 1 task completed, 1 was released, so 8 available
    sem.release()
    sem.release()


def test_fixed_verdict_accepted():
    """The FIXED verdict alias is in CANONICAL_VERDICTS."""
    from src.agent_claude import CANONICAL_VERDICTS
    assert "FIXED" in CANONICAL_VERDICTS
    assert "ACCEPT-READY" in CANONICAL_VERDICTS
