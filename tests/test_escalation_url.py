"""MEDIUM-8 — never POST the GitHub read token to a plaintext/arbitrary URL.

Only https:// (any host) or http:// to loopback/tailnet is accepted; the
rejection happens before any network call.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.escalation import (
    AdvisorUnavailable,
    _post_advise,
    _validate_advisor_url,
    poll_advice,
)

SHA = "a" * 40
READ_TOKEN = "ghs_super_secret_read_token"


@pytest.mark.parametrize("url", [
    "https://advisor.example.com",
    "https://advisor.example.com:8787",
    "https://100.111.118.24:8787",
    "http://127.0.0.1:8787",
    "http://localhost:8787",
    "http://100.64.0.1:8787",
    "http://100.127.255.254:8787",
])
def test_allowed_advisor_urls(url):
    _validate_advisor_url(url)  # must not raise


@pytest.mark.parametrize("url", [
    "http://evil.example",
    "http://10.0.0.5:8787",
    "http://192.168.1.5:8787",
    "http://100.128.0.1:8787",      # just outside the tailnet /10
    "http://100.63.255.255:8787",   # just below the tailnet /10
    "ftp://advisor.example.com",
    "file:///etc/passwd",
    "",
    "not-a-url",
])
def test_rejected_advisor_urls(url):
    with pytest.raises(AdvisorUnavailable):
        _validate_advisor_url(url)


def test_post_advise_rejects_plain_http_before_network():
    cfg = SimpleNamespace(advisor_url="http://evil.example",
                          advisor_token="advisor-token")
    with patch("src.escalation.urllib.request.urlopen",
               side_effect=AssertionError("network must not be touched")) as urlopen:
        with pytest.raises(AdvisorUnavailable) as exc:
            _post_advise(cfg, "acme/a", 1, SHA, [], "", READ_TOKEN)
    urlopen.assert_not_called()
    assert READ_TOKEN not in str(exc.value)


def test_poll_advice_rejects_plain_http_before_network():
    cfg = SimpleNamespace(advisor_url="http://evil.example",
                          advisor_token="advisor-token")
    with patch("src.escalation.urllib.request.urlopen",
               side_effect=AssertionError("network must not be touched")) as urlopen:
        with pytest.raises(AdvisorUnavailable) as exc:
            poll_advice(cfg, "advice-1")
    urlopen.assert_not_called()
    assert READ_TOKEN not in str(exc.value)


def test_https_advisor_is_not_rejected_by_the_guard():
    """The https path reaches the transport (no early AdvisorUnavailable from
    the URL guard)."""
    cfg = SimpleNamespace(advisor_url="https://advisor.example.com",
                          advisor_token="advisor-token")
    with patch("src.escalation.urllib.request.urlopen",
               side_effect=AssertionError("transport reached")) as urlopen:
        with pytest.raises(AssertionError, match="transport reached"):
            _post_advise(cfg, "acme/a", 1, SHA, [], "", READ_TOKEN)
    urlopen.assert_called_once()
