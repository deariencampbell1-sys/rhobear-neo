"""OpenRouter client for Neo — direct HTTP, no CLI shelling out.

Replaces the old Pi/direct-DeepSeek _run_agent subprocess with a proper
Python API client. Probes reasoning.effort at startup, handles content
and reasoning_content, and fails closed on every error path.

Usage:
    client = OpenRouterClient.from_config(cfg)
    usage, verdict = client.chat(brief)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from .config import Config

log = logging.getLogger("rhobear_neo.openrouter")

# Reasoning effort levels to probe, in preference order. max is
# DeepSeek-specific; high is the universal fallback.
REASONING_EFFORTS = ("max", "high")


class OpenRouterError(Exception):
    """Base for all OpenRouter failures — fail-closed root."""


class ProviderError(OpenRouterError):
    """Provider returned a non-200 or transport error."""


class TimeoutError(OpenRouterError):
    """Request timed out."""


class TruncatedError(OpenRouterError):
    """Output was truncated (finish_reason=length)."""


class EmptyResponseError(OpenRouterError):
    """No content in the response message."""


class MalformedResponseError(OpenRouterError):
    """Response JSON was missing expected fields."""


class OpenRouterClient:
    """Direct HTTP client to OpenRouter's OpenAI-compatible chat endpoint.

    Thread-safe for one-shot calls (each call creates its own request; the
    shared httpx.Client handles connection pooling).  Not multi-thread-safe
    for the config itself (reasoning_effort, max_tokens, etc.) — create one
    client per cfg and reuse it.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str = "max",
        max_tokens: int = 8192,
        timeout: int = 300,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=True)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Config) -> OpenRouterClient:
        """Build from a Config object, probing reasoning effort at startup."""
        client = cls(
            base_url=cfg.openrouter_base_url,
            api_key=cfg.openrouter_key,
            model=cfg.openrouter_model,
            reasoning_effort=cfg.openrouter_reasoning_effort,
            max_tokens=cfg.openrouter_max_tokens,
            timeout=cfg.openrouter_timeout,
            max_retries=cfg.openrouter_max_retries,
        )
        # Probe the highest supported reasoning effort — this is advisory
        # best-effort at startup, not a hard gate.  If every level fails
        # we keep the configured default (the model will still work, just
        # at whatever its default effort is).
        probed = client._probe_reasoning_effort()
        log.info("reasoning_effort probe: configured=%s probed=%s",
                 cfg.openrouter_reasoning_effort, probed)
        if probed != client.reasoning_effort:
            log.info("reasoning_effort updated from %s -> %s",
                     client.reasoning_effort, probed)
            client.reasoning_effort = probed
        return client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, brief: str) -> tuple[dict[str, int], str]:
        """Run the Neo brief through the model.

        Returns (usage, verdict) where usage is normalized to
        {input_tokens, output_tokens} and verdict is the extracted
        VERDICT: line or empty string (fail-closed).

        Raises OpenRouterError subclasses on failure — the caller should
        catch and return empty usage + empty verdict.
        """
        body = self._build_body(brief)
        last_error: OpenRouterError | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return self._chat_once(body)
            except OpenRouterError as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 30)
                    log.warning("attempt %d/%d: %s — retry in %ds",
                                attempt, self.max_retries, e, wait)
                    time.sleep(wait)
                else:
                    log.error("all %d attempts failed: %s", self.max_retries, e)

        raise last_error  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_body(self, brief: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "user", "content": brief},
            ],
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://neo.rhobear.ai",
            "X-Title": "rhobear-neo",
        }

    def _chat_once(self, body: dict[str, Any]) -> tuple[dict[str, int], str]:
        """Single attempt — no retry logic."""
        try:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=body,
            )
        except httpx.TimeoutException:
            raise TimeoutError("request timed out")
        except httpx.TransportError as e:
            raise ProviderError(f"transport error: {e}")

        if resp.status_code != 200:
            raise ProviderError(
                f"HTTP {resp.status_code}: {resp.text[:500]}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise MalformedResponseError(f"invalid JSON: {e}")

        usage = self._normalize_usage(data.get("usage") or {})
        verdict = self._extract_verdict(data)
        return usage, verdict

    def _normalize_usage(self, raw: dict[str, Any]) -> dict[str, int]:
        """Normalize OpenRouter/OpenAI usage to {input_tokens, output_tokens}."""
        return {
            "input_tokens": int(raw.get("prompt_tokens", 0) or 0),
            "output_tokens": int(raw.get("completion_tokens", 0) or 0),
        }

    def _extract_verdict(self, data: dict[str, Any]) -> str:
        """Extract verdict from the response, fail-closed on any anomaly."""
        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise MalformedResponseError("no choices in response")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise MalformedResponseError("choice is not an object")

        # --- truncation detection (fail-closed) ---
        finish_reason = choice.get("finish_reason") or ""
        if finish_reason == "length":
            raise TruncatedError("output truncated (finish_reason=length)")

        # --- content extraction ---
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise MalformedResponseError("message is not an object")

        content = (message.get("content") or "").strip()
        reasoning_content = (message.get("reasoning_content") or "").strip()

        log.debug("finish_reason=%s content_len=%d reasoning_len=%d",
                  finish_reason, len(content), len(reasoning_content))

        if not content:
            # Reasoning-only or empty — fail closed.  The model must
            # produce a substantive verdict.
            raise EmptyResponseError(
                f"no content in response (reasoning_only={bool(reasoning_content)})"
            )

        # --- verdict extraction ---
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped.upper().startswith("VERDICT:"):
                continue
            verdict = stripped.split(":", 1)[1]
            # Strip markdown formatting the model sometimes wraps around
            # the verdict value.
            verdict = verdict.replace("*", "").replace("`", "").strip()
            if verdict:
                return verdict
            # VERDICT: with nothing after the colon — fail closed.
            raise MalformedResponseError(
                f"VERDICT line with empty value: {line!r}"
            )

        # No VERDICT: line found at all — fail closed.
        return ""

    def _probe_reasoning_effort(self) -> str:
        """Probe the highest supported reasoning effort level.

        Sends a minimal request for each effort level in preference order
        and returns the first one that the provider accepts (HTTP 200).
        If none accept, returns 'high' as a safe fallback.
        """
        for effort in REASONING_EFFORTS:
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "ok"}],
                        "max_tokens": 5,
                        "reasoning_effort": effort,
                    },
                )
                if resp.status_code == 200:
                    log.info("reasoning_effort=%s accepted by %s", effort, self.model)
                    return effort
                log.info("reasoning_effort=%s rejected (%d): %.200s",
                         effort, resp.status_code, resp.text)
            except Exception as exc:
                log.info("reasoning_effort=%s probe error: %s", effort, exc)

        log.warning("all reasoning_effort levels rejected — falling back to 'high'")
        return "high"