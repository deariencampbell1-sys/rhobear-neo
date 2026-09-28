"""Fast, bounded Jev decisions for Neo's review-of-the-reviewer step."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
MAX_DIFF_CHARS = 56_000
MAX_REVIEW_CHARS = 12_000


@dataclass(frozen=True)
class JevAssessment:
    available: bool
    risk: str = "unknown"
    security: float = 0.0
    correctness: float = 0.0
    review_quality: str = "unknown"
    review_confidence: float = 0.0
    escalate: float = 0.0
    reason: str = ""

    @property
    def needs_attention(self) -> bool:
        # Jev is a classifier, not proof that a patch is safe. Treat ambiguous
        # classifications as needing the ordinary Neo investigation.
        return (
            not self.available
            or self.risk != "low"
            or self.security >= 0.20
            or self.correctness >= 0.35
            or self.review_quality != "adequate"
            or self.review_confidence < 0.75
            or self.escalate >= 0.25
        )

    def brief(self) -> str:
        if not self.available:
            return f"Jev assessment unavailable ({self.reason}). Inspect diff and review yourself."
        return (
            f"Jev decisions: patch risk={self.risk}; security-surface probability={self.security:.2f}; "
            f"correctness-concern probability={self.correctness:.2f}; review quality={self.review_quality} "
            f"(confidence={self.review_confidence:.2f}); deeper-review probability={self.escalate:.2f}. "
            "These are triage signals, not findings. Inspect the diff and verify concrete evidence."
        )


def assess(diff: str, review: str, api_key: str, *, timeout: float = 8.0) -> JevAssessment:
    """Ask one Jev call to assess the patch and whether the review covers it.

    Oversize or missing evidence is intentionally uncertain; cutting off the
    end of a diff could silently omit the only dangerous hunk.
    """
    if not api_key:
        return JevAssessment(False, reason="no OpenRouter key")
    if not diff.strip() or not review.strip():
        return JevAssessment(False, reason="diff or review missing")
    if len(diff) > MAX_DIFF_CHARS or len(review) > MAX_REVIEW_CHARS:
        return JevAssessment(False, reason="diff or review exceeds Jev context budget")

    body = {
        "model": MODEL,
        "state": {"diff": diff, "review": review},
        "questions": {
            "risk": {
                "type": "choice",
                "instructions": "Classify the patch in `diff` by the consequence of a missed bug.",
                "criteria": {
                    "low": "Local or cosmetic change with no trust, data, money, or execution boundary",
                    "medium": "Behavioral change where a bug could affect users or reliability",
                    "high": "Auth, permissions, secrets, payments, persistence, shell, file, network, or other trust boundary",
                },
            },
            "security": {
                "type": "noul",
                "instructions": "Does `diff` change an auth, permission, secret, payment, SQL, shell, file, network, deserialization, or data-boundary surface?",
            },
            "correctness": {
                "type": "noul",
                "instructions": "Does `diff` contain a plausible correctness problem such as broken control flow, unchecked errors, deleted validation, or a race?",
            },
            "review_quality": {
                "type": "choice",
                "instructions": "How well does `review` cover the important changed behavior in `diff`? Judge coverage, not writing style.",
                "criteria": {
                    "adequate": "Important behavior and plausible risks are addressed with evidence, or the patch is genuinely trivial",
                    "unclear": "The review lacks enough specific evidence to establish coverage",
                    "poor": "The review appears to miss an important changed behavior or claims an unsupported issue",
                },
            },
            "escalate": {
                "type": "noul",
                "instructions": "Should a deeper code or security review inspect this `diff` given `review`, including any likely missed issue?",
            },
        },
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
        answers = data["answers"]
        risk = answers["risk"]["choice"]
        quality = answers["review_quality"]["choice"]
        confidence = float(answers["review_quality"]["confidence"])
        security = float(answers["security"]["noul"])
        correctness = float(answers["correctness"]["noul"])
        escalate = float(answers["escalate"]["noul"])
        if risk not in {"low", "medium", "high"} or quality not in {"adequate", "unclear", "poor"}:
            raise ValueError("unexpected Jev choice")
        if not all(0 <= x <= 1 for x in (confidence, security, correctness, escalate)):
            raise ValueError("Jev probability outside [0, 1]")
        return JevAssessment(True, risk, security, correctness, quality, confidence, escalate)
    except (KeyError, TypeError, ValueError, OSError, urllib.error.URLError) as exc:
        # Never log response bodies or credentials from provider errors.
        return JevAssessment(False, reason=type(exc).__name__)
