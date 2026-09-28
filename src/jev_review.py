"""Fast, bounded Jev decisions for Neo's review-of-the-reviewer step."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


ENDPOINT = "https://openrouter.ai/api/alpha/decisions"

# Findings that may remove automatic merge authority. Everything else Jev says
# is a triage signal for Neo's own inspection. Per the control-plane directive,
# a coarse adjective ("risk=high", "needs_attention") is NOT a blocking rule:
# blocking must be a concrete finding that maps to one of these rules.
BLOCKING_RULES: dict[str, str] = {
    "REVIEW_INCOMPLETE": "the change surface could not be fully inspected, so review coverage cannot be established",
    "REVIEW_NOT_ADEQUATE": "the reviewer output misses an important changed behaviour or claims an unsupported issue",
    "REVIEW_COVERAGE_UNCLEAR": "the reviewer output lacks the specific evidence needed to establish coverage",
}
MODEL = "typesafe/jev-1.13"
MAX_DIFF_CHARS = 56_000
MAX_REVIEW_CHARS = 12_000
MAX_CHUNKS = 12


def _bounded_diff(diff: str) -> tuple[str, bool]:
    """Put trust-boundary/source hunks first and retain an omission marker."""
    if len(diff) <= MAX_DIFF_CHARS:
        return diff, False
    sections = [part for part in re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE) if part]
    if not sections:
        return diff[:MAX_DIFF_CHARS], True

    def priority(section: str) -> tuple[int, str]:
        header = section.split("\n", 1)[0].lower()
        if re.search(r"auth|permission|secret|payment|sql|shell|file|network|admin|security", header):
            return (0, header)
        if re.search(r"\.(py|ts|tsx|js|jsx|go|rs|java|kt|swift|c|cpp|h|sh|sql)\b", header):
            return (1, header)
        return (2, header)

    selected = []
    remaining = MAX_DIFF_CHARS - 90
    omitted = 0
    for section in sorted(sections, key=priority):
        header = section.split("\n", 1)[0].lower()
        if any(name in header for name in ("package-lock.json", "pnpm-lock.yaml", "yarn.lock", "node_modules/")):
            section = section.split("\n", 1)[0] + "\n[generated dependency body omitted]\n"
            omitted += 1
        if len(section) > remaining:
            if remaining > 500:
                selected.append(section[:remaining])
            omitted += 1
            break
        selected.append(section)
        remaining -= len(section)
    return "".join(selected) + f"\n[diff budget reached; {omitted} section(s) omitted or shortened]", True


@dataclass(frozen=True)
class JevFinding:
    """One concrete Jev observation. Only `blocking` findings touch merge authority."""

    rule: str
    severity: str
    evidence: str = ""
    blocking: bool = False


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
    partial: bool = False
    findings: tuple[JevFinding, ...] = ()
    head_sha: str = ""

    @property
    def requires_full_review(self) -> bool:
        """Signals that Neo should inspect the diff itself. Never a veto.

        Jev is a classifier, not proof that a patch is safe: any trust-boundary
        surface (auth, money, SQL, shell) lands `risk=high` with zero defects,
        so this only asks for the ordinary Neo investigation.
        """
        return (
            not self.available
            or self.partial
            or self.risk != "low"
            or self.security >= 0.20
            or self.correctness >= 0.35
            or self.review_quality != "adequate"
            or self.review_confidence < 0.75
            or self.escalate >= 0.25
        )

    @property
    def needs_attention(self) -> bool:
        """Deprecated alias for `requires_full_review` (logging/back-compat)."""
        return self.requires_full_review

    def policy_findings(self) -> tuple[JevFinding, ...]:
        """Concrete findings. Provider findings win; otherwise policy-derived.

        Policy only converts *evidence about review coverage* into blocking
        findings. Risk adjectives alone never produce one.
        """
        if self.findings:
            return self.findings
        out: list[JevFinding] = []
        if self.partial:
            out.append(JevFinding(
                "REVIEW_INCOMPLETE", "high",
                "Jev's own input was shortened, so the change surface was not fully inspected.",
                True,
            ))
        if self.available and self.review_quality == "poor":
            out.append(JevFinding(
                "REVIEW_NOT_ADEQUATE", "high",
                "Jev judged the reviewer's output to miss an important changed behaviour "
                "or to claim an unsupported issue.",
                True,
            ))
        elif (self.available and self.review_quality == "unclear"
              and self.escalate >= 0.25):
            out.append(JevFinding(
                "REVIEW_COVERAGE_UNCLEAR", "medium",
                f"Jev could not establish coverage (review confidence {self.review_confidence:.2f}, "
                f"deeper-review probability {self.escalate:.2f}).",
                True,
            ))
        return tuple(out)

    @property
    def blockers(self) -> tuple[JevFinding, ...]:
        """Findings that may remove automatic merge authority."""
        return tuple(f for f in self.policy_findings() if f.blocking and f.rule in BLOCKING_RULES)

    def brief(self) -> str:
        if not self.available:
            return f"Jev assessment unavailable ({self.reason}). Inspect diff and review yourself."
        return (
            f"Jev decisions: patch risk={self.risk}; security-surface probability={self.security:.2f}; "
            f"correctness-concern probability={self.correctness:.2f}; review quality={self.review_quality} "
            f"(confidence={self.review_confidence:.2f}); deeper-review probability={self.escalate:.2f}. "
            f"{'The Jev input was shortened; inspect the full diff. ' if self.partial else ''}"
            "These are triage signals, not findings. Inspect the diff and verify concrete evidence."
        )


def _parse_findings(answers: dict) -> tuple[JevFinding, ...]:
    """Read provider findings when present. Unknown rules pass through as signals."""
    raw = answers.get("findings") or []
    out: list[JevFinding] = []
    if not isinstance(raw, list):
        return ()
    for item in raw:
        if not isinstance(item, dict):
            continue
        rule = str(item.get("rule") or "").strip()
        if not rule:
            continue
        out.append(JevFinding(
            rule=rule,
            severity=str(item.get("severity") or "medium"),
            evidence=str(item.get("evidence") or "")[:300],
            blocking=bool(item.get("blocking")) and rule in BLOCKING_RULES,
        ))
    return tuple(out)



def _questions() -> dict:
    """The fixed question set. Kept verbatim from the single-shot version."""
    return {
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
    }


def _payload(diff_text: str, review_text: str) -> dict:
    return {"model": MODEL,
            "state": {"diff": diff_text, "review": review_text},
            "questions": _questions()}


def _ask(payload: dict, api_key: str, timeout: float) -> dict | None:
    """One Jev call. None means the provider did not give a usable answer."""
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answers = json.load(response)["answers"]
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
    except (KeyError, TypeError, ValueError, OSError, urllib.error.URLError):
        # Never log response bodies or credentials from provider errors.
        return None
    return {"risk": risk, "security": security, "correctness": correctness,
            "quality": quality, "confidence": confidence, "escalate": escalate,
            "findings": _parse_findings(answers)}


_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
_QUALITY_ORDER = {"adequate": 0, "unclear": 1, "poor": 2}


def _merge(results: list[dict]) -> dict:
    """Worst case wins: the aggregate must never look better than any chunk."""
    findings: dict[tuple, JevFinding] = {}
    for item in results:
        for finding in item["findings"]:
            findings.setdefault((finding.rule, finding.evidence[:80]), finding)
    return {
        "risk": max((r["risk"] for r in results), key=lambda x: _RISK_ORDER[x]),
        "quality": max((r["quality"] for r in results), key=lambda x: _QUALITY_ORDER[x]),
        "confidence": min(r["confidence"] for r in results),
        "security": max(r["security"] for r in results),
        "correctness": max(r["correctness"] for r in results),
        "escalate": max(r["escalate"] for r in results),
        "findings": tuple(findings.values()),
    }


def _split_sections(diff: str) -> list[str]:
    return [p for p in re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE) if p]


def _chunk_diff(diff: str, budget: int = MAX_DIFF_CHARS) -> tuple[list[str], bool]:
    """Pack file sections into bounded chunks, each carrying a full file manifest.

    Returns (chunks, complete). `complete` is False only when a single file's
    own section cannot fit in one chunk — then that surface really was cut and
    the caller must treat the assessment as incomplete.
    """
    sections = _split_sections(diff)
    if not sections:
        return [diff[:budget]], len(diff) <= budget
    manifest = "\n".join(s.split("\n", 1)[0] for s in sections)
    header = f"[file manifest — {len(sections)} changed file(s), all of which are covered below]\n{manifest}\n"
    chunks: list[str] = []
    current = header
    complete = True
    for section in sections:
        if len(section) > budget:
            # One huge generated/lockfile-like body: keep its head, admit the cut.
            chunks.append(header + section[:budget])
            complete = False
            continue
        if len(current) + len(section) > budget and len(current) > len(header):
            chunks.append(current)
            current = header + section
        else:
            current += section
    if len(current) > len(header):
        chunks.append(current)
    return chunks, complete


def assess(diff: str, review: str, api_key: str, *, timeout: float = 8.0,
           head_sha: str = "") -> JevAssessment:
    """Assess the patch and the review's coverage of it.

    Small diffs are one call. Large diffs are split into bounded chunks with a
    file manifest so every changed file is inspected; incompleteness is then
    reported only when a chunk really could not be examined.
    """
    if not api_key:
        return JevAssessment(False, reason="no OpenRouter key")
    if not diff.strip() or not review.strip():
        return JevAssessment(False, reason="diff or review missing")

    review_partial = len(review) > MAX_REVIEW_CHARS
    bounded_review = review[-MAX_REVIEW_CHARS:] if review_partial else review

    if len(diff) <= MAX_DIFF_CHARS:
        chunks, complete = [diff], True
    else:
        chunks, complete = _chunk_diff(diff)
        if len(chunks) > MAX_CHUNKS:
            # More surface than the caller is willing to pay for: keep the
            # single-shot priority view and stay honest about the cut.
            bounded, _ = _bounded_diff(diff)
            chunks, complete = [bounded], False

    results = [r for r in (_ask(_payload(chunk, bounded_review), api_key, timeout)
                           for chunk in chunks) if r is not None]
    if not results:
        return JevAssessment(False, reason="jev call failed")

    merged = _merge(results)
    partial = (review_partial
               or not complete
               or len(results) < len(chunks))
    return JevAssessment(
        True, merged["risk"], merged["security"], merged["correctness"],
        merged["quality"], merged["confidence"], merged["escalate"],
        partial=partial, findings=merged["findings"], head_sha=head_sha,
    )
