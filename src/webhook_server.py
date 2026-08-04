"""rhobear-neo webhook receiver.

Stdlib HTTP server (mirrors rhobear-reviews — no FastAPI weight for one HMAC'd
POST route). Bound to loopback; Caddy fronts the public https endpoint.

Neo does NOT subscribe to `pull_request`. It wakes on a REVIEWER'S VERDICT
landing on a head SHA:

  * `status`            — a commit-status context (rhobear-reviews uses this) goes
                          success/failure. GitHub delivers `state` + `context` + `sha`.
  * `check_run`         — action=completed for a Checks-API reviewer (CodeAnt etc.).

Either way we map the event to (repo, pr_number, head_sha, verdict) and hand it to
the worker. The pull_request → review → verdict ordering is enforced by GitHub's
own event graph, so Neo never races the reviewer.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .config import Config

log = logging.getLogger("rhobear_neo.webhook")


def verify_github_signature(body: bytes, header_sig: str, secret: str) -> bool:
    if not header_sig or not header_sig.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header_sig)


@dataclass
class DispatchOutcome:
    code: int
    detail: str


class NeoDispatcher:
    """Turns a raw GitHub event into a normalized wake for the worker."""

    def __init__(self, cfg: Config, on_verdict: Callable[[dict], None]):
        self.cfg = cfg
        self.on_verdict = on_verdict

    def _in_scope(self, repo_full: str) -> bool:
        if self.cfg.repo_allowlist:               # hard scope for a first live run
            return repo_full in self.cfg.repo_allowlist
        org = repo_full.split("/", 1)[0] if "/" in repo_full else ""
        return org in self.cfg.orgs

    def _trusted(self, context: str) -> bool:
        return any(context == c or context.startswith(c) for c in self.cfg.trusted_review_contexts)

    def dispatch(self, event: str, payload: dict[str, Any], delivery_id: str = "") -> DispatchOutcome:
        repo_full = (payload.get("repository") or {}).get("full_name", "")
        if not repo_full or not self._in_scope(repo_full):
            return DispatchOutcome(204, f"out of scope repo={repo_full!r}")

        if event == "status":
            context = payload.get("context") or ""
            state = payload.get("state") or ""          # pending|success|failure|error
            sha = payload.get("sha") or ""
            if state == "pending":
                return DispatchOutcome(204, f"status pending ({context})")
            if not self._trusted(context):
                return DispatchOutcome(204, f"untrusted status context={context!r}")
            # A commit status carries no PR number — resolve via the head SHA.
            self.on_verdict({
                "kind": "status", "repo": repo_full, "sha": sha,
                "context": context, "green": state == "success",
                "delivery": delivery_id,
            })
            return DispatchOutcome(200, f"verdict via status {context}={state} sha={sha[:8]}")

        if event == "check_run":
            if payload.get("action") != "completed":
                return DispatchOutcome(204, f"check_run action={payload.get('action')!r}")
            cr = payload.get("check_run") or {}
            name = cr.get("name") or ""
            if not self._trusted(name):
                return DispatchOutcome(204, f"untrusted check name={name!r}")
            concl = cr.get("conclusion") or ""          # success|failure|neutral|...
            sha = cr.get("head_sha") or ""
            prs = cr.get("pull_requests") or []
            pr_number = prs[0].get("number") if prs else None
            self.on_verdict({
                "kind": "check_run", "repo": repo_full, "sha": sha,
                "context": name, "green": concl == "success",
                "pr_number": pr_number, "delivery": delivery_id,
            })
            return DispatchOutcome(200, f"verdict via check_run {name}={concl} sha={sha[:8]}")

        return DispatchOutcome(204, f"ignored event={event!r}")


def make_handler(cfg: Config, dispatcher: NeoDispatcher):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence default stderr spam
            pass

        def _reply(self, code: int, detail: str):
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(detail.encode())

        def do_POST(self):
            if self.path != "/webhook":
                return self._reply(404, "not found")
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            sig = self.headers.get("X-Hub-Signature-256", "")
            if not verify_github_signature(body, sig, cfg.webhook_secret):
                return self._reply(401, "bad signature")
            event = self.headers.get("X-GitHub-Event", "")
            delivery = self.headers.get("X-GitHub-Delivery", "")
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                return self._reply(400, "bad json")
            try:
                out = dispatcher.dispatch(event, payload, delivery)
            except Exception:
                log.exception("dispatch failed delivery=%s", delivery)
                return self._reply(500, "dispatch error")
            ip = self.client_address[0]
            log.info("webhook: ip=%s delivery=%s event=%s -> %d (%s)",
                     ip, delivery[:8], event, out.code, out.detail)
            return self._reply(out.code, out.detail)

    return Handler


def serve(cfg: Config, dispatcher: NeoDispatcher) -> None:
    handler = make_handler(cfg, dispatcher)
    srv = ThreadingHTTPServer((cfg.bind_host, cfg.bind_port), handler)
    log.info("rhobear-neo webhook listening on %s:%d", cfg.bind_host, cfg.bind_port)
    srv.serve_forever()
