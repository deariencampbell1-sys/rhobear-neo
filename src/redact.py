"""One shared secret redactor for untrusted text (advisor work orders).

The advisor reads attacker-controlled PR content and its frontier agent MAY
echo secrets read from its host filesystem into the work order (PEM blocks,
GitHub tokens, credential-bearing DSNs, 40-hex keys).  Those fields reach two
dangerous places:

  - `neo_jobs.payload` / `neo_actions.detail` — persistence at rest: the
    database, its backups, and anyone with DB access;
  - the fixer brief — egress to a `--yolo --accept-hooks` agent that holds a
    push token.

Both are redacted by THIS module.  There must be exactly one redactor: a
second implementation written from scratch is drift (W1-10-r1).

Used by:
  - ``src/escalation.py`` — BEFORE the work order is persisted.  Redacting at
    render time is too late: the row and its backups would already hold the
    secret.
  - ``src/neo_builder.py`` — before the work order is rendered into the
    fixer brief (defence in depth over the already-redacted payload).
"""
from __future__ import annotations

import re

__all__ = ["redact_secrets", "redact_data"]

# Anything matching a secret SHAPE is replaced.  Shape-based redaction is not
# a sandbox — novel secret shapes can still travel — but every known leak
# vector seen in this fabric (PEM blocks, ghs_/ghp_/sk- tokens, DSNs,
# key-context hex) is covered.
_SECRET_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    # PEM private-key blocks (whole block incl. the base64 body).
    (re.compile(
        r"-----BEGIN[^\n]*PRIVATE KEY-----"
        r".*?"
        r"-----END[^\n]*PRIVATE KEY-----",
        re.DOTALL), "[REDACTED:PRIVATE KEY]"),
    # A BEGIN with no END: swallow to end of text so no key material survives.
    (re.compile(r"-----BEGIN[^\n]*PRIVATE KEY-----.*\Z", re.DOTALL),
     "[REDACTED:PRIVATE KEY]"),
    # GitHub tokens: ghs_/ghp_/gho_/ghu_/ghr_ … and fine-grained PATs.
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"), "[REDACTED:GITHUB TOKEN]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}\b"), "[REDACTED:GITHUB TOKEN]"),
    # OpenAI / OpenRouter-style keys.
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "[REDACTED:API KEY]"),
    # DSNs with credentials: scheme://user[:pass]@host… (postgresql://…@ …).
    (re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/\"'@]*@[^\s\"']+", re.IGNORECASE),
     "[REDACTED:DSN]"),
    # 40-hex values in key context ("api_key: <40 hex>", "token=…").
    (re.compile(
        r"(?i)\b((?:api[_-]?)?key|token|secret|passwd|password|"
        r"auth[_-]?token|access[_-]?token)\b"
        r"(\s*[:=]\s*[\"']?|\s+(?:is|was)\s+)"
        r"[0-9a-f]{40}\b"),
     r"\1\2[REDACTED]"),
)


def redact_secrets(text: str) -> str:
    """Redact anything matching a secret shape (PEM private-key blocks,
    ghs_/ghp_/sk- tokens, credential-bearing DSNs, 40-hex values in key
    context) from untrusted advisor output.  Non-strings pass through
    unchanged."""
    if not isinstance(text, str):
        return text
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_data(value):
    """Recursively redact every string inside a JSON-ish structure
    (dict / list / tuple of scalars).  Keys and non-string scalars pass
    through unchanged — this is what escalation applies to the advisor work
    order BEFORE it is persisted."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {k: redact_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_data(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_data(v) for v in value)
    return value
