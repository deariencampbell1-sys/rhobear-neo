"""The Neo protocol brief.

Neo runs as a headless Claude Code CLI agent routed through DeepSeek Direct
(Anthropic-compatible endpoint, exact `deepseek-v4-flash` model at max
reasoning effort). This module builds the brief that binds that agent to the
canon: review the reviewer's findings, drive to GREEN, fix-forward ONLY trivial
issues, dispatch a builder for substantial ones, ESCALATE as a last resort,
and return a verdict.  The agent NEVER holds merge credentials — the
deterministic merge executor (merge_executor.py) is the ONLY code path that
calls `gh pr merge`.

The agent's verdicts (ACCEPT, FIX-FORWARD, BOUNCE-BUILDER, ESCALATE) are
returned to the worker, which calls the merge executor or escalation logic.
The old ACCEPT-MERGED verdict is still parsed for backward compatibility.
"""
from __future__ import annotations

import re


def build_brief(*, repo: str, pr: int, head_sha: str, reviewer_context: str,
                reviewer_green: bool, auto_merge: bool, builder_round: int,
                max_builder_rounds: int, builder_model: str,
                jev_context: str = "") -> str:
    merge_clause = (
        "If ACCEPT and the reviewer is GREEN and all required checks pass and the PR is in order, "
        "do NOT merge — return ACCEPT and the deterministic merge executor will merge. "
        "Add the label `neo:accept`."
        if auto_merge else
        "Auto-merge is OFF for this install. If ACCEPT and GREEN, DO NOT merge — add the label "
        "`neo:ready`, post a one-line 'ready to merge' comment, and STOP. "
        "The deterministic merge executor respects the auto-merge policy."
    )

    builder_clause = (
        f"You have already spent {builder_round}/{max_builder_rounds} builder rounds on this PR. "
        + (f"You may dispatch ONE more Claude Code CLI agent (same {builder_model} model/tool loop) for a substantial fix."
           if builder_round < max_builder_rounds else
           "You are OUT of builder rounds — do NOT dispatch another builder. Report what is still "
           "outstanding as BOUNCE-BUILDER with the findings; the lane parks the PR (visible, "
           "owner-actionable) and never auto-dispatches. Running out of rounds is NOT an escalation.")
    )

    return f"""You are Neo, RHOBEAR's review-and-MERGE gate. Follow your canon EXACTLY — the
canon is inlined below.

TARGET: PR #{pr} in {repo} at head {head_sha[:12]}.
JEV TRIAGE: {jev_context or 'No Jev assessment available.'}
REVIEWER: `{reviewer_context}` reported {'GREEN' if reviewer_green else 'NOT green (changes requested / failing)'} on this head
({'commit status success' if reviewer_green else 'commit status failure/pending'}).
GREEN means the reviewer's COMMIT STATUS is `success` on this exact head. A reviewer verdict
line of `COMMENT` delivered with that success status is GREEN with non-blocking notes only —
triage the notes, fix or flag anything real, and you may ACCEPT when no CRITICAL/HIGH is
open; do not escalate on the verdict word `COMMENT` alone. `REQUEST_CHANGES` is never green,
even when an older status on the same head says success.

SETUP: You are in an empty working directory with `gh` authed. To read the PR you only need `gh`. If you
must edit files (a fix-forward, or to prep a builder), first: `gh repo clone {repo} repo && cd repo &&
git fetch origin && git checkout {head_sha}` (or the PR branch). Push fix-forward commits to the PR branch.

IMPORTANT: You do NOT hold merge credentials. You never run `gh pr merge`. On ACCEPT, return the
verdict and the deterministic merge executor will merge if the evidence permits.

CANON (do exactly this):
1. Pull the reviewer's findings: `gh pr view {pr} -R {repo} --json reviews,statusCheckRollup,comments`
   and `gh api repos/{repo}/commits/{head_sha}/status`. Triage each finding real vs false-positive.
2. Decide a VERDICT:
   - ACCEPT  → reviewer green (commit status success on this head) + no open CRITICAL/HIGH +
     required checks pass. COMMENT-with-success notes are advisory by the reviewer's own gate;
     a note that is CRITICAL or HIGH is an open finding and still blocks. {merge_clause}
   - FIX-FORWARD → TRIVIAL only (wrong import, typo, missing constant, a test expectation). Fix it
     yourself, commit to the PR branch, push. Your push re-triggers the reviewer → you'll be re-invoked.
   - BOUNCE (substantial bug) → {builder_clause}
     To dispatch a builder: write a precise brief (file · line · observed · expected · smallest fix) and
     run the same Claude Code CLI agent (same {builder_model} model/tool loop) in a checkout of {repo}@the PR
     branch; it fixes + pushes. Its push re-triggers reviewer → you'll be re-invoked to re-check.
   - ESCALATE → LAST RESORT, and it is a HANDOFF to the owner-proxy agent, not a stop. It is read by
     an agent holding the owner's standing authority who will adjudicate and keep the chain moving.
     Reserve it for what is genuinely irreversible or outside any agent's authority: spending money,
     rotating or issuing secrets/signing certs, a public launch, deleting production data.
     "Brand" alone is NOT sufficient. If the repo states a rule, APPLY it. If the repo contradicts
     itself, state which reading you take and why, then proceed — a contradiction is a judgement call,
     not an owner-gated fork. Running out of builder rounds is NOT an escalation (see above).
     When you do escalate: label `neo:escalate` and post the EXACT decision needed with the options
     enumerated (A1/A2 style) so it is answerable in one word. NEVER attempt to merge.
3. GUARDRAILS: never approve for merge with open CRITICAL/HIGH, out of order (a PR depending on an
   unmerged one), with expanded/undeclared scope, without test evidence when code changed, or
   bypassing the repo's OWN branch protection / required reviewers. Honor what the repo owner set.

OUTPUT exactly:
VERDICT: <ACCEPT|ACCEPT-READY|FIX-FORWARD|BOUNCE-BUILDER|ESCALATE>
REASONS:
- <one line>
- <one line>
ACTIONS: <what you did — commits/labels/builder dispatched>
"""


# ---------------------------------------------------------------------------
# Verdict parsing — Zeus's own answers
# ---------------------------------------------------------------------------
#
# Zeus's own verdicts are exactly ACCEPT, FIX-FORWARD and BOUNCE (owner, Q9:
# "Zeus answers: ACCEPT, FIX-FORWARD or BOUNCE").  Apollo's words (ACCEPT /
# REQUEST_CHANGES / COMMENT) are read from its review state on GitHub and are
# NEVER parsed from text.
ZEUS_VERDICTS = ("ACCEPT", "FIX-FORWARD", "BOUNCE")

# The internal protocol canon still speaks a few alias words (the OUTPUT
# contract above, the builder fixer contract, and pre-Q9 answers).  They map
# onto the three Q9 words for everything a person sees: the owner's vocabulary
# is exactly three words (see the `verdicts` / `merge-ask` flows — "BOUNCE
# means a person decides", which covers the ESCALATE owner-proxy handoff).
_ZEUS_ALIASES = {
    "ACCEPT": "ACCEPT",
    "ACCEPT-MERGED": "ACCEPT",
    "ACCEPT-READY": "ACCEPT",
    "FIXED": "ACCEPT",           # builder alias for ACCEPT-READY
    "FIX-FORWARD": "FIX-FORWARD",
    "BOUNCE": "BOUNCE",
    "BOUNCE-BUILDER": "BOUNCE",
    "ESCALATE": "BOUNCE",
}

# Words the builder fixer contract may answer (extracted raw for the builder
# lane); not Zeus's own verdicts — parse_verdict never returns these.
_BUILDER_WORDS = frozenset({"FIXED", "NO-CHANGE", "BLOCKED"})

_KNOWN_VERDICT_WORDS = frozenset(_ZEUS_ALIASES) | _BUILDER_WORDS

# The verdict is the closing line of an answer; scan only the tail so an
# example or a quoted earlier answer deeper in the text can never win.
_VERDICT_WINDOW = 40


def _strip_markup(line: str) -> str:
    """Remove markdown emphasis/decoration so `**VERDICT:** BOUNCE` and
    `> - **VERDICT: BOUNCE**` all reduce to `VERDICT: BOUNCE`."""
    cleaned = line
    for ch in ("*", "`", "_", "~"):
        cleaned = cleaned.replace(ch, "")
    cleaned = cleaned.strip()
    cleaned = re.sub(r"^(?:#{1,6}\s*|>\s*|[-+]\s+|\d+[.)]\s+)+", "", cleaned)
    return cleaned.strip()


_MARKER_WITH_VALUE_RE = re.compile(r"^VERDICT:\s*([A-Za-z][A-Za-z-]*)", re.IGNORECASE)
_MARKER_ONLY_RE = re.compile(r"^VERDICT:\s*$", re.IGNORECASE)
_BARE_WORD_RE = re.compile(r"^([A-Za-z][A-Za-z-]*)\s*[.!]?$", re.IGNORECASE)


def _line_verdict_value(line: str) -> str | None:
    """The verdict word on one line, or None.  Accepts `VERDICT: <word>`
    after any markdown bold/heading/bullet decoration, and the bolded-label
    shape where the colon sits inside the bold (`**VERDICT:** BOUNCE`).
    Trailing prose after the word (`**VERDICT: BOUNCE.** Both are small.`)
    keeps only the word.  Prose that merely mentions the marker
    ("Do not output VERDICT: ...") never matches: the marker must start the
    line after decoration is stripped."""
    cleaned = _strip_markup(line)
    if not cleaned:
        return None
    m = _MARKER_WITH_VALUE_RE.match(cleaned)
    if m:
        return m.group(1).upper()
    return None


def _is_bare_marker(line: str) -> bool:
    """True for a bolded/plain `VERDICT:` label with its word on the next
    line ("after markdown bold")."""
    return bool(_MARKER_ONLY_RE.match(_strip_markup(line)))


def _bare_verdict_word(line: str) -> str | None:
    cleaned = _strip_markup(line)
    m = _BARE_WORD_RE.match(cleaned)
    if m and m.group(1).upper() in _KNOWN_VERDICT_WORDS:
        return m.group(1).upper()
    return None


def _verdict_values(text: str, window: int = _VERDICT_WINDOW) -> list[str]:
    """Verdict words in the last `window` lines, most recent first."""
    lines = text.splitlines()[-window:]
    found: list[str] = []
    for idx, line in enumerate(lines):
        value = _line_verdict_value(line)
        if value is None and _is_bare_marker(line) and idx + 1 < len(lines):
            value = _bare_verdict_word(lines[idx + 1])
        if value:
            found.append(value)
    found.reverse()
    return found


def extract_verdict_line(text: str, *, window: int = _VERDICT_WINDOW) -> str | None:
    """The raw verdict word from the tail of an agent answer (any canon word:
    the protocol OUTPUT contract, the builder fixer contract, or a Q9 word).

    Scans the last `window` lines (default 40), most recent first.  Tolerates
    fenced blocks and markdown bold.  Returns the uppercased word or None.
    """
    values = _verdict_values(text, window)
    return values[0] if values else None


def parse_verdict(text: str, *, window: int = _VERDICT_WINDOW) -> str | None:
    """Zeus's own verdict from its answer text: exactly ACCEPT, FIX-FORWARD
    or BOUNCE (owner, Q9).

    Accepts `VERDICT: <word>` case-insensitively on a line anywhere in the
    last 40 lines, also inside a fenced block or after markdown bold.  The
    internal canon aliases (ACCEPT-MERGED, ACCEPT-READY, FIXED, BOUNCE-BUILDER,
    ESCALATE) map onto the three Q9 words; the most recent verdict in the
    window wins.  Returns one of ZEUS_VERDICTS (uppercased) or None — never a
    guess: Apollo's words ACCEPT / REQUEST_CHANGES / COMMENT are read from its
    review state, not parsed from text.
    """
    for value in _verdict_values(text, window):
        zeus_word = _ZEUS_ALIASES.get(value)
        if zeus_word:
            return zeus_word
    return None
