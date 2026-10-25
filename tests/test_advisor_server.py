"""Advisor HTTP handler hardening — MEDIUM-1/2/3.

No network: the handler is driven directly with fake headers/rfile/wfile and
a fake service.  Covers request-field validation (values reach `git fetch`
argv), constant-time bearer comparison, and Content-Length bounds.
"""
from __future__ import annotations

import io
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from advisor.server import AdvisorRequestHandler, MAX_BODY_BYTES, _run_agent

TOKEN = "advisor-token"
SHA = "a" * 40
VALID = {
    "repo": "acme/app",
    "pr": 7,
    "head_sha": SHA,
    "read_token": "ghs_read_token",
}


class _FakeHandler(AdvisorRequestHandler):
    """Drive do_POST without BaseHTTPRequestHandler's socket machinery."""

    def __init__(self, headers=None, body=b"", token=TOKEN, service=None,
                 rfile=None):
        self.headers = headers or {}
        self.rfile = rfile if rfile is not None else io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.path = "/v1/advise"
        self.server = SimpleNamespace(
            advisor_token=token, advisor_service=service,
        )
        self.replies = []

    def _reply(self, code, body):
        self.replies.append((code, body))


def _handler(payload, *, headers=None, service=None, rfile=None):
    body = json.dumps(payload).encode() if payload is not None else b""
    hdrs = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Length": str(len(body)),
    }
    if headers:
        hdrs.update(headers)
    return _FakeHandler(headers=hdrs, body=body, service=service, rfile=rfile)


# ---------------------------------------------------------------------------
# MEDIUM-1 — request-body values reach git argv / GitHub URL
# ---------------------------------------------------------------------------

def test_accepts_valid_request():
    service = MagicMock()
    service.submit.return_value = "advice-1"
    h = _handler(VALID, service=service)
    h.do_POST()
    assert h.replies == [(202, {"advice_id": "advice-1"})]
    service.submit.assert_called_once_with(VALID)


@pytest.mark.parametrize("field,value", [
    ("head_sha", "a" * 39),
    ("head_sha", "A" * 40),
    ("head_sha", "--upload-pack=touch /tmp/pwned"),
    ("head_sha", "a" * 40 + "; rm -rf /"),
    ("repo", "acme"),
    ("repo", "acme/app;rm -rf /"),
    ("repo", None),
    ("pr", 0),
    ("pr", -3),
    ("pr", True),
    ("pr", "7"),
    ("pr", None),
    ("read_token", ""),
    ("read_token", "   "),
    ("read_token", None),
    ("read_token", 123),
])
def test_rejects_invalid_field(field, value):
    service = MagicMock()
    payload = dict(VALID)
    payload[field] = value
    h = _handler(payload, service=service)
    h.do_POST()
    assert h.replies and h.replies[0][0] == 400
    service.submit.assert_not_called()


def test_rejects_missing_field():
    service = MagicMock()
    payload = dict(VALID)
    del payload["read_token"]
    h = _handler(payload, service=service)
    h.do_POST()
    assert h.replies == [(400, {"error": "invalid required fields"})]
    service.submit.assert_not_called()


# ---------------------------------------------------------------------------
# MEDIUM-2 — constant-time bearer comparison
# ---------------------------------------------------------------------------

def test_authorized_equal_token():
    h = _FakeHandler(headers={"Authorization": f"Bearer {TOKEN}"})
    assert h._authorized() is True


def test_authorized_unequal_token():
    h = _FakeHandler(headers={"Authorization": "Bearer wrong-token"})
    assert h._authorized() is False


def test_authorized_missing_header():
    h = _FakeHandler(headers={})
    assert h._authorized() is False


def test_authorized_requires_expected_token():
    h = _FakeHandler(headers={"Authorization": "Bearer wrong-token"}, token="")
    assert h._authorized() is False


def test_authorized_uses_compare_digest():
    h = _FakeHandler(headers={"Authorization": f"Bearer {TOKEN}"})
    with patch("advisor.server.hmac.compare_digest", return_value=True) as cmp:
        assert h._authorized() is True
    cmp.assert_called_once_with(TOKEN, TOKEN)


# ---------------------------------------------------------------------------
# MEDIUM-3 — Content-Length parsing and bound
# ---------------------------------------------------------------------------

class _UnreadableRFile:
    def read(self, *a):
        raise AssertionError("body must not be read")


@pytest.mark.parametrize("raw_length", ["abc", "12x", "1.5"])
def test_non_numeric_content_length_is_400(raw_length):
    h = _handler(VALID, headers={"Content-Length": raw_length},
                 service=MagicMock(), rfile=_UnreadableRFile())
    h.do_POST()
    assert h.replies == [(400, {"error": "bad content-length"})]


def test_oversized_content_length_is_400_before_read():
    h = _handler(VALID,
                 headers={"Content-Length": str(MAX_BODY_BYTES + 1)},
                 service=MagicMock(), rfile=_UnreadableRFile())
    h.do_POST()
    assert h.replies == [(400, {"error": "request body too large"})]


def test_negative_content_length_is_400():
    h = _handler(VALID, headers={"Content-Length": "-1"},
                 service=MagicMock(), rfile=_UnreadableRFile())
    h.do_POST()
    assert h.replies == [(400, {"error": "request body too large"})]


def test_body_at_limit_is_read_and_parsed():
    service = MagicMock()
    service.submit.return_value = "advice-limit"
    body = json.dumps(VALID).encode()
    h = _handler(VALID, headers={"Content-Length": str(len(body))},
                 service=service)
    h.do_POST()
    assert h.replies == [(202, {"advice_id": "advice-limit"})]


# ---------------------------------------------------------------------------
# HIGH — the frontier agent must not inherit service secrets
# ---------------------------------------------------------------------------

_ALLOWED_AGENT_ENV = {"PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR"}
_FORBIDDEN_AGENT_ENV = (
    "NEO_ADVISOR_TOKEN", "NEO_GITHUB_APP_KEY", "GH_TOKEN", "GITHUB_TOKEN",
    "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
)


def test_run_agent_env_is_scrubbed(monkeypatch):
    """The child env is an explicit allowlist: a prompt-injected brief cannot
    make the agent read service secrets out of /proc/self/environ."""
    for name in _FORBIDDEN_AGENT_ENV:
        monkeypatch.setenv(name, "secret-value")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"summary": "s", "root_cause": "r", "steps": []}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    raw = _run_agent("claude", {"model": "opus"}, "PR-derived brief")

    assert raw.startswith('{"summary"')
    assert calls, "the agent subprocess must run"
    env = calls[0][1]["env"]
    assert env["PATH"]
    assert env["HOME"]
    assert set(env) == _ALLOWED_AGENT_ENV
    for forbidden in _FORBIDDEN_AGENT_ENV:
        assert forbidden not in env
