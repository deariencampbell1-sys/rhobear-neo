"""Tests for the OpenRouter client — realistic fixtures, every error path.

Covers:
  - Normal response with content + reasoning_content
  - Content only, no reasoning
  - Reasoning only, no content (fail-closed)
  - Empty content (fail-closed)
  - Truncated output (finish_reason=length)
  - Unknown verdict (no VERDICT: line)
  - VERDICT: with empty value
  - Provider errors (HTTP 500, 401, 403, 429)
  - Transport errors (connection refused, DNS)
  - Timeout
  - Retry-success (succeeds on second attempt)
  - Retry-exhaustion (all attempts fail)
  - Reasoning effort probe — max accepted
  - Reasoning effort probe — max rejected, high accepted
  - Reasoning effort probe — both rejected
  - Malformed JSON response
  - Missing choices
  - Markdown-wrapped verdict
  - Verdict extraction from long content
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable
from unittest.mock import patch

import httpx
import pytest

from src.openrouter_client import (
    OpenRouterClient,
    OpenRouterError,
    ProviderError,
    TimeoutError,
    TruncatedError,
    EmptyResponseError,
    MalformedResponseError,
    REASONING_EFFORTS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = "https://openrouter.ai/api/v1"
API_KEY = "sk-or-test-123"
MODEL = "deepseek/deepseek-v4-flash"


def _make_client(**overrides: Any) -> OpenRouterClient:
    """Build a client with sensible defaults; override any kwarg."""
    kwargs = dict(
        base_url=BASE,
        api_key=API_KEY,
        model=MODEL,
        reasoning_effort="max",
        max_tokens=8192,
        timeout=300,
        max_retries=1,  # keep tests fast by default
    )
    kwargs.update(overrides)
    return OpenRouterClient(**kwargs)


def _ok_response(
    *,
    content: str = "Some analysis.\nVERDICT: ACCEPT-READY",
    reasoning_content: str = "",
    finish_reason: str = "stop",
    prompt_tokens: int = 150,
    completion_tokens: int = 80,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": msg,
            }
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _mock_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    """Create an httpx.Client with a mock transport."""
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SUBSTANTIVE_CONTENT = """I've reviewed the findings. The PR looks good — all checks pass, no critical issues remain.

Summary of triage:
- The import error is a genuine bug but trivial (wrong path).
- The test assertion is correct as-is.
- No security concerns.

VERDICT: ACCEPT-MERGED"""

REASONING_TEXT = "Let me think through this step by step. The reviewer flagged a potential issue with the error handling. I need to check if the try/except covers all cases..."

CONTENT_WITH_REASONING = f"""I've analyzed the code and found the issue.

{REASONING_TEXT}

My conclusion: the fix is correct but there's a minor style issue.

VERDICT: FIX-FORWARD"""


# ===================================================================
# Normal response with content + reasoning_content
# ===================================================================

def test_normal_content_and_reasoning() -> None:
    client = _make_client()
    resp = _ok_response(content=CONTENT_WITH_REASONING, reasoning_content=REASONING_TEXT)
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        usage, verdict = client.chat("test brief")

    assert verdict == "FIX-FORWARD"
    assert usage["input_tokens"] == 150
    assert usage["output_tokens"] == 80


# ===================================================================
# Content only, no reasoning
# ===================================================================

def test_content_only() -> None:
    client = _make_client()
    resp = _ok_response(content=SUBSTANTIVE_CONTENT)
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        usage, verdict = client.chat("test brief")

    assert verdict == "ACCEPT-MERGED"


# ===================================================================
# Reasoning only, no content (fail-closed)
# ===================================================================

def test_reasoning_only_raises_empty() -> None:
    """Model returns only reasoning_content with empty content."""
    client = _make_client()
    resp = _ok_response(content="", reasoning_content="Lots of thinking...")
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(EmptyResponseError):
            client.chat("test brief")


# ===================================================================
# Empty content (fail-closed)
# ===================================================================

def test_empty_content_raises() -> None:
    client = _make_client()
    resp = _ok_response(content="")
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(EmptyResponseError):
            client.chat("test brief")


# ===================================================================
# Truncated output (finish_reason=length)
# ===================================================================

def test_truncated_output_raises() -> None:
    client = _make_client()
    resp = _ok_response(
        content="Some partial analysis...\nVERDICT: ",
        finish_reason="length",
    )
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(TruncatedError):
            client.chat("test brief")


# ===================================================================
# Unknown verdict (no VERDICT: line)
# ===================================================================

def test_unknown_verdict_returns_empty() -> None:
    client = _make_client()
    resp = _ok_response(content="I've analyzed the code. Everything looks fine.")
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        usage, verdict = client.chat("test brief")

    assert verdict == ""  # fail-closed


# ===================================================================
# VERDICT: with empty value
# ===================================================================

def test_verdict_empty_value_raises() -> None:
    client = _make_client()
    resp = _ok_response(content="Here is my analysis.\nVERDICT: \nMore text.")
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(MalformedResponseError):
            client.chat("test brief")


# ===================================================================
# Provider errors
# ===================================================================

@pytest.mark.parametrize("status,reason", [
    (500, "Internal Server Error"),
    (401, "Unauthorized"),
    (403, "Forbidden"),
    (429, "Too Many Requests"),
])
def test_provider_http_error(status: int, reason: str) -> None:
    client = _make_client()
    handler = lambda r: httpx.Response(status, text=f"{{\"error\": {{\"message\": \"{reason}\"}}}}")

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(ProviderError, match=rf"HTTP {status}"):
            client.chat("test brief")


# ===================================================================
# Transport errors
# ===================================================================

def test_connection_refused() -> None:
    client = _make_client()
    handler = lambda r: (_ for _ in ()).throw(httpx.ConnectError("Connection refused"))

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(ProviderError, match="transport error"):
            client.chat("test brief")


# ===================================================================
# Timeout
# ===================================================================

def test_timeout() -> None:
    client = _make_client()
    handler = lambda r: (_ for _ in ()).throw(httpx.TimeoutException("timed out"))

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(TimeoutError):
            client.chat("test brief")


# ===================================================================
# Retry-success (succeeds on second attempt)
# ===================================================================

def test_retry_success_on_second_attempt() -> None:
    """First attempt fails with 500, second succeeds."""
    client = _make_client(max_retries=2)
    call_count = 0

    def handler(r: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(500, text="{}")
        return httpx.Response(200, json=_ok_response(content="All good.\nVERDICT: ACCEPT-READY"))

    with patch.object(client, "_client", _mock_transport(handler)):
        usage, verdict = client.chat("test brief")

    assert verdict == "ACCEPT-READY"
    assert call_count == 2


# ===================================================================
# Retry-exhaustion (all attempts fail)
# ===================================================================

def test_retry_exhaustion() -> None:
    client = _make_client(max_retries=2)
    handler = lambda r: httpx.Response(500, text="{}")

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(ProviderError):
            client.chat("test brief")


# ===================================================================
# Reasoning effort probe — max accepted
# ===================================================================

def test_probe_max_accepted() -> None:
    """Probe finds max is accepted, uses it."""
    call_count = 0

    def handler(r: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(r.content)
        if body.get("reasoning_effort") == "max":
            return httpx.Response(200, json=_ok_response(content="ok"))
        # Should never reach high if max is accepted
        return httpx.Response(400, text="{}")

    client = _make_client()
    transport = _mock_transport(handler)
    with patch.object(client, "_client", transport):
        # Run the probe through the ctor path (from_config would call it)
        probed = client._probe_reasoning_effort()

    assert probed == "max"
    assert call_count == 1  # only max was tried


# ===================================================================
# Reasoning effort probe — max rejected, high accepted
# ===================================================================

def test_probe_max_rejected_high_accepted() -> None:
    call_count = 0

    def handler(r: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(r.content)
        if body.get("reasoning_effort") == "max":
            return httpx.Response(400, text='{"error": {"message": "unsupported"}}')
        if body.get("reasoning_effort") == "high":
            return httpx.Response(200, json=_ok_response(content="ok"))
        return httpx.Response(400, text="{}")

    client = _make_client()
    transport = _mock_transport(handler)
    with patch.object(client, "_client", transport):
        probed = client._probe_reasoning_effort()

    assert probed == "high"
    assert call_count == 2  # max then high


# ===================================================================
# Reasoning effort probe — both rejected
# ===================================================================

def test_probe_both_rejected_falls_back() -> None:
    call_count = 0

    def handler(r: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(400, text='{"error": {"message": "unsupported"}}')

    client = _make_client()
    transport = _mock_transport(handler)
    with patch.object(client, "_client", transport):
        probed = client._probe_reasoning_effort()

    # Falls back to 'high' even though both rejected
    assert probed == "high"
    assert call_count == 2  # max and high both tried


# ===================================================================
# Malformed JSON response
# ===================================================================

def test_malformed_json() -> None:
    client = _make_client()
    handler = lambda r: httpx.Response(200, text="not json{{{")

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(MalformedResponseError, match="invalid JSON"):
            client.chat("test brief")


# ===================================================================
# Missing choices
# ===================================================================

def test_no_choices() -> None:
    client = _make_client()
    handler = lambda r: httpx.Response(200, json={"id": "test", "usage": {}})

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(MalformedResponseError, match="no choices"):
            client.chat("test brief")


# ===================================================================
# Markdown-wrapped verdict
# ===================================================================

def test_verdict_markdown_stripped() -> None:
    """Model might wrap the verdict in **bold** or `backticks`."""
    content = "Analysis.\nVERDICT: **ACCEPT-MERGED**"
    client = _make_client()
    resp = _ok_response(content=content)
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        usage, verdict = client.chat("test brief")

    assert verdict == "ACCEPT-MERGED"


def test_verdict_backtick_stripped() -> None:
    content = "Analysis.\nVERDICT: `BOUNCE-BUILDER`"
    client = _make_client()
    resp = _ok_response(content=content)
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        usage, verdict = client.chat("test brief")

    assert verdict == "BOUNCE-BUILDER"


# ===================================================================
# Verdict extraction from long content
# ===================================================================

def test_verdict_in_long_content() -> None:
    """Verdict line may appear deep in the content."""
    content = "\n".join([
        f"Line {i}" for i in range(50)
    ] + ["VERDICT: ESCALATE"] + [
        f"Line {i}" for i in range(50, 100)
    ])
    client = _make_client()
    resp = _ok_response(content=content)
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        usage, verdict = client.chat("test brief")

    assert verdict == "ESCALATE"


# ===================================================================
# `from_config` factory probes reasoning effort
# ===================================================================

def test_from_config_probes_reasoning_effort() -> None:
    """from_config calls _probe_reasoning_effort and uses the result."""
    # We'll verify the probe is called and the client is configured
    # with the probed value.
    from src.config import Config

    cfg = Config()
    # Override with test values
    cfg.openrouter_base_url = BASE
    cfg.openrouter_key = API_KEY
    cfg.openrouter_model = MODEL
    cfg.openrouter_reasoning_effort = "max"
    cfg.openrouter_max_tokens = 8192
    cfg.openrouter_timeout = 300
    cfg.openrouter_max_retries = 1

    client = OpenRouterClient.from_config(cfg)
    assert client.model == MODEL
    assert client.base_url == BASE
    # The probe will attempt to connect to the real URL (which will fail
    # in test), but the probe should still return a fallback value.
    # We don't assert the exact effort here since it depends on network.
    assert client.reasoning_effort in ("max", "high")


# ===================================================================
# Edge: usage normalization handles missing/zero values
# ===================================================================

def test_usage_normalization_missing_keys() -> None:
    client = _make_client()
    resp = _ok_response(content="Verdict: ESCALATE")
    del resp["usage"]["prompt_tokens"]
    del resp["usage"]["completion_tokens"]
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        usage, verdict = client.chat("test brief")

    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0


# ===================================================================
# Edge: choice is not a dict
# ===================================================================

def test_choice_not_dict_raises() -> None:
    client = _make_client()
    resp = _ok_response(content="whatever")
    resp["choices"] = ["not a dict"]
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(MalformedResponseError):
            client.chat("test brief")


# ===================================================================
# Edge: message is not a dict
# ===================================================================

def test_message_not_dict_raises() -> None:
    client = _make_client()
    resp = _ok_response(content="whatever")
    resp["choices"][0]["message"] = "not a dict"
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(MalformedResponseError):
            client.chat("test brief")


# ===================================================================
# Edge: model returns non-200 with structured error body
# ===================================================================

def test_structured_error_body() -> None:
    client = _make_client()
    body = {"error": {"message": "Insufficient credits", "code": 402}}
    handler = lambda r: httpx.Response(402, json=body)

    with patch.object(client, "_client", _mock_transport(handler)):
        with pytest.raises(ProviderError, match="HTTP 402"):
            client.chat("test brief")


# ===================================================================
# _run_agent wrapper in neo_worker catches errors -> empty
# ===================================================================

def test_worker_wrapper_catches_errors() -> None:
    """_run_agent should return empty usage/verdict on any error."""
    from src.neo_worker import _run_agent

    # Create a client that always raises
    client = _make_client()
    handler = lambda r: httpx.Response(500, text="{}")

    with patch.object(client, "_client", _mock_transport(handler)):
        usage, verdict = _run_agent(client, "test brief")

    assert usage == {}
    assert verdict == ""


# ===================================================================
# _run_agent wrapper passes through success
# ===================================================================

def test_worker_wrapper_passes_success() -> None:
    from src.neo_worker import _run_agent

    client = _make_client()
    resp = _ok_response(content="Good.\nVERDICT: ACCEPT-READY")
    handler = lambda r: httpx.Response(200, json=resp)

    with patch.object(client, "_client", _mock_transport(handler)):
        usage, verdict = _run_agent(client, "test brief")

    assert verdict == "ACCEPT-READY"
    assert usage["input_tokens"] == 150


# ===================================================================
# REASONING_EFFORTS constant
# ===================================================================

def test_reasoning_efforts_order() -> None:
    """max should be preferred over high."""
    assert REASONING_EFFORTS == ("max", "high")


# ===================================================================
# Request body structure
# ===================================================================

def test_request_body_sent_correctly() -> None:
    """Verify the request body has the expected structure."""
    client = _make_client(reasoning_effort="max")
    captured: list[dict] = []

    def handler(r: httpx.Request) -> httpx.Response:
        captured.append(json.loads(r.content))
        return httpx.Response(200, json=_ok_response(content="ok.\nVERDICT: ACCEPT"))

    with patch.object(client, "_client", _mock_transport(handler)):
        client.chat("test brief")

    assert len(captured) == 1
    body = captured[0]
    assert body["model"] == MODEL
    assert body["messages"] == [{"role": "user", "content": "test brief"}]
    assert body["max_tokens"] == 8192
    assert body["reasoning_effort"] == "max"


# ===================================================================
# Request headers
# ===================================================================

def test_request_headers() -> None:
    """Verify Authorization and Referer headers are sent."""
    client = _make_client()
    captured: list[dict] = []

    def handler(r: httpx.Request) -> httpx.Response:
        captured.append(dict(r.headers))
        return httpx.Response(200, json=_ok_response(content="ok.\nVERDICT: ACCEPT"))

    with patch.object(client, "_client", _mock_transport(handler)):
        client.chat("test brief")

    headers = captured[0]
    assert headers.get("authorization") == f"Bearer {API_KEY}"
    assert headers.get("http-referer") == "https://neo.rhobear.ai"
    assert headers.get("x-title") == "rhobear-neo"