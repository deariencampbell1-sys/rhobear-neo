"""HIGH 2.1 (W1-10-r1) — the advisor must be fail-closed by default.

The advisor reads attacker-controlled PR content and, until W1-03 installs
the hardened `rhobear-neo-advisor` unit as a dedicated unprivileged user, it
runs as the service user with the deploy dir (`.env`, `app-key.pem`) in
reach.  Its output reaches a `--yolo` fixer with a push token: enabling it is
a live secret-exfiltration path.

So the advisor path is OFF unless explicitly enabled (`NEO_ADVISOR_ENABLED=1`
read from `os.environ` in `advisor/server.py`).  With the variable unset the
service must log a single clear line naming the variable and continue
serving — but the advise path is never called.

Edge case covered: with the enable variable unset the advisor path is never
called and the service still serves.
"""
from __future__ import annotations

import io
import json
import logging

import advisor.server as advisor_server
from advisor.server import ADVISOR_ENABLE_VAR, AdvisorRequestHandler

TOKEN = "advisor-token"
SHA = "a" * 40
VALID = {
    "repo": "acme/app",
    "pr": 7,
    "head_sha": SHA,
    "read_token": "ghs_read_token",
}


class _RecordingServer:
    """Stand-in for ThreadingHTTPServer — captures the configured server so
    the real handler can be driven against exactly what serve() produced."""

    instances: list = []

    def __init__(self, addr, handler_cls):
        self.addr = addr
        self.handler_cls = handler_cls
        self.serve_forever_called = False
        _RecordingServer.instances.append(self)

    def serve_forever(self):
        self.serve_forever_called = True


class _RecordingService:
    """Stand-in for AdvisorService — records any submit, i.e. any attempt to
    enter the advisor path."""

    submits: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def submit(self, payload):
        _RecordingService.submits.append(payload)
        return "advice-1"


class _FakeHandler(AdvisorRequestHandler):
    """Drive do_POST without BaseHTTPRequestHandler's socket machinery."""

    def __init__(self, server, payload):
        self.server = server
        body = json.dumps(payload).encode()
        self.headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Length": str(len(body)),
        }
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.path = "/v1/advise"
        self.replies = []

    def _reply(self, code, body):
        self.replies.append((code, body))


def _drive_main(monkeypatch):
    """Run the real advisor main() with fake server machinery."""
    _RecordingServer.instances.clear()
    _RecordingService.submits.clear()
    monkeypatch.setenv("NEO_ADVISOR_TOKEN", TOKEN)
    monkeypatch.setenv("NEO_ADVISOR_BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("NEO_ADVISOR_BIND_PORT", "8787")
    monkeypatch.setattr(advisor_server, "ThreadingHTTPServer", _RecordingServer)
    monkeypatch.setattr(advisor_server, "AdvisorService", _RecordingService)
    advisor_server.main()
    assert len(_RecordingServer.instances) == 1, "serve() must start the server"
    return _RecordingServer.instances[0]


def test_advisor_disabled_by_default(monkeypatch, caplog):
    """With NEO_ADVISOR_ENABLED unset the advisor path is never called and
    the service still serves."""
    monkeypatch.delenv(ADVISOR_ENABLE_VAR, raising=False)

    with caplog.at_level(logging.INFO):
        srv = _drive_main(monkeypatch)

    # The service still serves.
    assert srv.serve_forever_called, "the service must continue serving"

    # ...and it logged a single clear line naming the enable variable.
    mentions = [r for r in caplog.records
                if ADVISOR_ENABLE_VAR in r.getMessage()]
    assert len(mentions) == 1, (
        f"exactly one log line must name {ADVISOR_ENABLE_VAR}: {caplog.text}"
    )
    assert "NEO_ADVISOR_TOKEN not set" not in caplog.text, (
        "the disabled path must not fail on the token check — it serves closed"
    )

    # The advisor path is never called: a valid, authorized request must not
    # reach the service/agent machinery.
    handler = _FakeHandler(srv, VALID)
    handler.do_POST()
    assert _RecordingService.submits == [], (
        "the advisor path must not be called while the enable variable is unset"
    )
    assert handler.replies, "the request must get an explicit refusal"
    code, body = handler.replies[0]
    assert code != 202, "the advise path must be refused, not accepted"
    assert ADVISOR_ENABLE_VAR in json.dumps(body), (
        "the refusal must name the variable an operator has to set"
    )


def test_advisor_enabled_explicitly_runs_the_path(monkeypatch):
    """The flip side: NEO_ADVISOR_ENABLED=1 is the only way in — with it set
    the same request reaches the advisor service (nothing else changed)."""
    monkeypatch.setenv(ADVISOR_ENABLE_VAR, "1")

    srv = _drive_main(monkeypatch)

    handler = _FakeHandler(srv, VALID)
    handler.do_POST()
    assert handler.replies == [(202, {"advice_id": "advice-1"})]
    assert _RecordingService.submits == [VALID]


def test_advisor_disabled_defense_in_depth(monkeypatch):
    """Disabled mode must leave the server with no service wired at all —
    even a handler bug cannot start an agent run."""
    monkeypatch.delenv(ADVISOR_ENABLE_VAR, raising=False)

    srv = _drive_main(monkeypatch)

    assert getattr(srv, "advisor_enabled", None) is False
    assert getattr(srv, "advisor_service", "unset") is None
