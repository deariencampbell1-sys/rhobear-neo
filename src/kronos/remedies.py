"""Kronos remedies — knock heads loose, at most once per hour per target.

In order: restart that one unit (allow-list ONLY), redeliver each failed
delivery, re-queue a stuck job — then alert. The RemedyLedger (state-backed)
enforces "at most once per hour per target"; the allow-list below is the
whole of what Kronos may ever restart, matching the sudoers line the manager
installs (M-11):

    rhobear-kronos ALL=(root) NOPASSWD: \\
        /usr/bin/systemctl restart rhobear-reviews.service, \\
        /usr/bin/systemctl restart rhobear-reviews-dash.service, \\
        /usr/bin/systemctl restart rhobear-neo.service, \\
        /usr/bin/systemctl restart rhobear-neo-builder.service

`sudo` (not plain systemctl) is what the brief locks: the unit user has no
other privilege, and every call is `sudo -n` — non-interactive, so a missing
sudoers line fails loudly instead of hanging the pass.
"""
from __future__ import annotations

from urllib.request import build_opener

API_BASE = "https://api.github.com"
RESTART_BIN = "/usr/bin/systemctl"
SUDO_BIN = "/usr/bin/sudo"

# The restart allow-list — the four units from the brief, and nothing else.
ALLOWED_UNITS = (
    "rhobear-reviews",
    "rhobear-reviews-dash",
    "rhobear-neo",
    "rhobear-neo-builder",
)

REMEDY_WINDOW_SECS = 3600


class RemedyLedger:
    """Once per hour per target, backed by a KronosState slot so the window
    survives across passes (and across a restart of the unit itself)."""

    def __init__(self, dict_state: dict, *, now_fn, window_secs: int = REMEDY_WINDOW_SECS):
        self._state = dict_state
        self._now = now_fn
        self._window = window_secs

    def may(self, target: str) -> bool:
        last = self._state.get(target)
        if last is None:
            return True
        try:
            return (self._now() - float(last)) >= self._window
        except (TypeError, ValueError):
            return True

    def did(self, target: str) -> None:
        self._state[target] = self._now()

    def last(self, target: str) -> float | None:
        last = self._state.get(target)
        try:
            return None if last is None else float(last)
        except (TypeError, ValueError):
            return None


def restart_unit(runner, unit: str, *, dry_run: bool = False):
    """Restart ONE allow-listed unit via `sudo -n /usr/bin/systemctl restart`.

    Returns (ok, one-line note). A unit outside ALLOWED_UNITS is refused
    without running anything — the allow-list is enforced here in code, not
    only in the sudoers file.
    """
    if unit not in ALLOWED_UNITS:
        return False, f"{unit} is outside the restart allow-list — refused"
    argv = [SUDO_BIN, "-n", RESTART_BIN, "restart", unit]
    if dry_run:
        return True, f"dry-run: would run {' '.join(argv)}"
    try:
        res = runner(argv)
    except Exception as exc:  # noqa: BLE001
        return False, f"{unit} restart failed: {type(exc).__name__}"
    rc = getattr(res, "rc", 1)
    if rc == 0:
        return True, f"{unit} restarted"
    err = (getattr(res, "stderr", "") or "").strip().splitlines()
    return False, f"{unit} restart failed (rc={rc}): {err[0] if err else 'no error line'}"


def redeliver_delivery(http, app_name: str, jwt: str, delivery_id: int, *,
                       dry_run: bool = False):
    """POST /app/hook/deliveries/{id}/attempts with the App's JWT (202 ok)."""
    url = f"{API_BASE}/app/hook/deliveries/{int(delivery_id)}/attempts"
    if dry_run:
        return True, f"dry-run: would redeliver {app_name} delivery {delivery_id}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {jwt}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        status, _body = http("POST", url, headers=headers, body=b"",
                             timeout=30)
    except Exception as exc:  # noqa: BLE001
        return False, (f"redelivery of {app_name} delivery {delivery_id} "
                       f"failed: {type(exc).__name__}")
    if status == 202:
        return True, f"redelivered {app_name} delivery {delivery_id}"
    return False, (f"redelivery of {app_name} delivery {delivery_id} "
                   f"got HTTP {status}")


def requeue_stuck_job(jobs_store, row: dict, *, dry_run: bool = False):
    """Re-queue a stuck RUNNING job: release() returns it to `queued`
    without burning an attempt and clears the dead claimant. A QUEUED row
    has no claimant — there is nothing to re-queue (alert covers it)."""
    job_id = row.get("id")
    if job_id is None:
        return False, "job row has no id — nothing to re-queue"
    if row.get("state") != "running":
        return False, (f"job {job_id} is {row.get('state')}, not running — "
                       f"nothing to re-queue (alerted instead)")
    if dry_run:
        return True, f"dry-run: would re-queue job {job_id}"
    try:
        ok = jobs_store.release(int(job_id))
    except Exception as exc:  # noqa: BLE001
        return False, f"re-queue of job {job_id} failed: {type(exc).__name__}"
    if ok:
        return True, f"job {job_id} re-queued (was stuck running)"
    return False, f"job {job_id} could not be re-queued (held by a live worker?)"


def list_deliveries_via_app(http_opener=None):
    """Factory: GET /app/hook/deliveries with the given App JWT (page 1)."""
    def _list(jwt: str, *, per_page: int = 100):
        opener = http_opener or build_opener()
        req = _request(f"{API_BASE}/app/hook/deliveries?per_page={per_page}",
                       jwt)
        with opener.open(req, timeout=30) as resp:
            import json as _json
            return _json.loads(resp.read().decode("utf-8"))
    return _list


def _request(url: str, jwt: str):
    from urllib.request import Request
    return Request(url, method="GET", headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {jwt}",
        "X-GitHub-Api-Version": "2022-11-28",
    })
