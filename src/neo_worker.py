"""Neo worker — turns a reviewer verdict into a Neo protocol run.

Flow per wake ({repo, sha, green, context[, pr_number]}):
  1. Resolve PR number from the head SHA (if not supplied).
  2. Loop/thrash guard: skip if we already acted on this SHA; read builder rounds.
  3. Entitlement/credits gate (per-install balance) — Neo actions only for a paid install.
  4. Build the Neo brief (neo_protocol) and run it headless via Claude Code CLI
     (deepseek-v4-flash at max reasoning effort via DeepSeek Direct, isolated
     config + temp workdir).  The brief itself dispatches a builder when a
     substantial fix is needed; the Claude Code agent handles gh, git, edit, and
     test runner tools.
  5. Record the action + credits (Gemini-baseline cost via the shared pricing) in state.

The heavy lifting (read findings, fix-forward, dispatch builder, merge) is done by the
LLM bound to the canon — this worker is the deterministic wrapper: guardrails,
credits, state. Merge authority + the auto-merge toggle live here.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import subprocess
from datetime import datetime, timezone

from .config import Config, DEFAULT_OWNER_INSTALL_ID
from .jobs import Jobs
from .neo_builder import LANE_BRANCH, lane_of
from .neo_state import NeoState, verdict_color
from .agent_claude import ClaudeAgent, AgentError
from .github_app import AppNotConfigured, GitHubAppError, NoInstallation, repo_token
from . import neo_protocol
from .jev_review import JevAssessment, assess as jev_assess
from .usage import UsageStore, UsageAttempt

log = logging.getLogger("rhobear_neo.worker")

# Gemini-baseline credit model (see plans/neo-pricing-model.md). Credits charged =
# gemini_cost * 1.26 * 100_000. We meter AFTER the run from the agent's reported usage.
MARGIN = 1.26
CREDITS_PER_USD = 100_000
GEMINI = {  # $/token
    "flash_in": 0.30 / 1e6, "flash_out": 2.50 / 1e6,
    "pro_in": 2.00 / 1e6, "pro_out": 12.00 / 1e6, "pro_cache": (2.00 * 0.25) / 1e6,
}


def _gh(token: str, *args: str, timeout: int = 30) -> tuple[int, str]:
    """Run `gh` as the repo's installation token (never a personal token)."""
    env = {**os.environ, "GH_TOKEN": token}
    p = subprocess.run(["gh", *args], capture_output=True, text=True, env=env, timeout=timeout)
    return p.returncode, (p.stdout or p.stderr).strip()


def resolve_pr(token: str, repo: str, sha: str, given: int | None) -> int | None:
    if given:
        return given
    rc, out = _gh(token, "api", f"repos/{repo}/commits/{sha}/pulls",
                  "-H", "Accept: application/vnd.github+json")
    if rc != 0:
        log.warning("resolve_pr gh failed repo=%s sha=%s: %s", repo, sha[:8], out[:120])
        return None
    try:
        prs = json.loads(out)
        opens = [p for p in prs if p.get("state") == "open"]
        return (opens or prs)[0]["number"] if prs else None
    except Exception:
        return None


def credits_for(usage: dict | None, builder: bool) -> int:
    """Gemini-baseline credits from the agent's reported usage.

    Accepts either the raw agent usage dict (with input_tokens/output_tokens/
    cache_read_input_tokens) or a summed usage dict (with uncached_input/
    cached_input/output_tokens/reasoning_tokens).  Falls back gracefully
    for both formats.

    `usage=None` means the agent reported no usage data: charge 0.  Token
    counts coalesce on None only — 0 is a real count, never missing
    (defect 9).
    """
    usage = usage or {}
    # Support both raw agent format and summed usage format.
    fin = usage.get("input_tokens")
    if fin is None:
        fin = usage.get("uncached_input")
    if fin is None:
        fin = 0
    fout = usage.get("output_tokens")
    if fout is None:
        fout = 0
    cread = usage.get("cache_read_input_tokens")
    if cread is None:
        cread = usage.get("cached_input")
    if cread is None:
        cread = 0
    if builder:
        cost = fin * GEMINI["pro_in"] + cread * GEMINI["pro_cache"] + fout * GEMINI["pro_out"]
    else:
        cost = fin * GEMINI["flash_in"] + fout * GEMINI["flash_out"]
    return int(cost * MARGIN * CREDITS_PER_USD)


# --- metering: the owner workspace never meters ----------------------------
# The owner workspace install id (cfg.owner_install_id — NEO_OWNER_INSTALL_ID
# in the environment, W1-13-r1 defect 7: configured, never hard-coded here)
# and any install whose `plan` is empty skip BOTH the credit gate and the
# debit. Customers with a plan are gated on balance <= 0 (and told so on the
# PR) and debited.  OWNER_INSTALL_ID stays as the legacy single-argument
# default (contract: tests/test_neo_protocol_metering.py imports it).
OWNER_INSTALL_ID = DEFAULT_OWNER_INSTALL_ID


def meters_credits(inst: dict, cfg: Config | None = None) -> bool:
    """True only for a paying customer install: a non-empty plan AND not the
    owner workspace. The owner workspace never meters (owner, W1-02).

    Which install IS the owner comes from ``cfg.owner_install_id``
    (NEO_OWNER_INSTALL_ID).  Without a Config the legacy OWNER_INSTALL_ID
    constant applies (the single-argument contract)."""
    owner = getattr(cfg, "owner_install_id", None) or OWNER_INSTALL_ID
    if str(inst.get("install_id") or "") == str(owner):
        return False
    return bool(inst.get("plan"))


def _compute_diff_stat(token: str, repo: str, pr: int) -> str:
    """Compute a diff stat for the PR from the pull request API.

    Returns a string like '3 files changed, +120/-40' or empty on failure.
    (gh has no `pr diff --stat`; the stat comes from GET /repos/{repo}/pulls/{pr}
    -> changed_files/additions/deletions.)
    """
    jq = '"\\(.changed_files) files changed, +\\(.additions)/-\\(.deletions)"'
    try:
        rc, out = _gh(token, "api", f"repos/{repo}/pulls/{pr}", "--jq", jq,
                      timeout=30)
        if rc == 0 and out:
            return out.strip()[:2000]
    except Exception:
        log.warning("diff_stat failed for %s#%s", repo, pr)
    return ""


def _trusted_contexts(cfg: Config) -> tuple[str, ...]:
    """Contexts whose status is the reviewer's own gate (config-driven)."""
    raw = getattr(cfg, "trusted_review_contexts", ()) or ()
    if isinstance(raw, str):
        raw = [c.strip() for c in raw.split(",") if c.strip()]
    return tuple(str(c) for c in raw)


def _is_trusted(context: str, trusted: tuple[str, ...]) -> bool:
    return any(context == c or context.startswith(c) for c in trusted)


def _parse_instant(value: str):
    """An aware datetime for a GitHub timestamp, or None when unusable.

    W1-07-r2 defect 7: created_at was compared as strings, so mixed `Z` and
    `+offset` timestamps sorted lexicographically wrong ('...T12:00:00+05:00'
    is OLDER than '...T10:00:00Z' but sorts above it as a string).  'Z' maps
    to +00:00; a value that parses to a NAIVE datetime counts as unparseable
    too — comparison must always be between aware instants."""
    s = str(value or "").strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else None


def _newer(candidate: str, incumbent: str) -> bool:
    """True when candidate's instant is strictly newer than incumbent's.
    A value that cannot be parsed sorts LAST: an unparseable candidate never
    wins; an unparseable incumbent is always beaten by a parseable one."""
    a, b = _parse_instant(candidate), _parse_instant(incumbent)
    if a is None:
        return False
    if b is None:
        return True
    return a > b


def _newest_status_per_context(statuses: list[dict]) -> dict[tuple[str, str], dict]:
    """Keep the newest status per lowercased context AND creator identity.

    Equal or unusable timestamps preserve the first item (GitHub normally sends
    newest first); any parseable newer timestamp replaces the incumbent."""
    newest: dict[tuple[str, str], dict] = {}
    for status in statuses or []:
        context = str(status.get("context") or "").lower()
        creator = status.get("creator") or {}
        identity = (str(creator.get("node_id") or "")
                    or str(creator.get("login") or "").lower())
        key = (context, identity)
        prior = newest.get(key)
        if prior is None or _newer(str(status.get("created_at") or ""),
                                   str(prior.get("created_at") or "")):
            newest[key] = status
    return newest


# --- W1-07-r1: the two greens on ONE head SHA (the merge rule (b)-(d)) -----
# Board `merge-authority` v7 / `trusted` / `verdicts`:
#   green 1: Apollo ACCEPT on the head commit — always required; a commit
#            status or check-run success is never enough and COMMENTED is not
#            approval;
#   green 2: the repo's own merge authority — the architect's approval bound
#            to the exact head, OR an enabled trusted reviewer's green.
#            Either one counts;
#   a trusted reviewer that is red (or requesting changes) always blocks.
# Every fact below is bound to the head SHA: any new push resets both greens.

ARCHITECT_ALWAYS_REPOS = (
    "deariencampbell1-sys/rhobear-reviews-app",
    "deariencampbell1-sys/rhobear-neo",
    "deariencampbell1-sys/rhobear-verifier",
)
"""The three lane repos where the architect source is always on (the board:
'on other repos only when that install has it on')."""


def _review_oid(r: dict) -> str:
    return str((r.get("commit") or {}).get("oid")
               or r.get("commit_id") or r.get("commitId") or "")


def _review_author(r: dict) -> str:
    return str((r.get("author") or {}).get("login") or r.get("author_login") or "")


_OLDEST_INSTANT = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _review_submitted_key(r: dict):
    return _parse_instant(str(r.get("submittedAt") or r.get("submitted_at") or "")) \
        or _OLDEST_INSTANT


def _reviewer_rows(token: str, repo: str, pr: int) -> list[dict]:
    """Read every REST review row; an unreadable/incomplete list raises."""
    from .merge_executor import read_evidence_pages
    return read_evidence_pages(
        _gh, token, f"repos/{repo}/pulls/{pr}/reviews", None)


def _reviews_with_bot_types(token: str, repo: str, pr: int, reviews: list) -> list:
    """Merge exact REST identities into GraphQL reviews and append REST-only rows."""
    rows = _reviewer_rows(token, repo, pr)
    types: dict[str, str] = {}
    by_node: dict[str, dict] = {}
    for row in rows:
        user = row.get("user") or {}
        login = str(user.get("login") or "").lower()
        if login:
            types[login] = "Bot" if str(user.get("type") or "") == "Bot" else "User"
        node_id = str(row.get("node_id") or "")
        if node_id:
            by_node[node_id] = row
    seen_nodes: set[str] = set()
    for r in reviews:
        if not isinstance(r, dict):
            continue
        login = str((r.get("author") or {}).get("login")
                    or r.get("author_login") or "").lower()
        author = dict(r.get("author") or {})
        review_node = str(r.get("id") or "")
        if review_node:
            seen_nodes.add(review_node)
        rest = by_node.get(review_node)
        exact_bot = types.get(f"{login}[bot]") if login and not login.endswith("[bot]") else None
        ambiguous = exact_bot == "Bot" and types.get(login) == "User"
        if ambiguous:
            log.warning("%s#%s ambiguous reviewer login %s: node id decides; login fallback fails closed",
                        repo, pr, login)
        if rest is not None:
            # Identity comes from the exact REST row, never a same-login account.
            user = rest.get("user") or {}
            author["login"] = str(user.get("login") or author.get("login") or "")
            author["type"] = "Bot" if str(user.get("type") or "") == "Bot" else "User"
            if user.get("node_id"):
                author["node_id"] = user["node_id"]
            if rest.get("html_url"):
                r["html_url"] = rest["html_url"]
            if rest.get("submitted_at"):
                r["submitted_at"] = rest["submitted_at"]
        elif ambiguous:
            author["type"] = "User"
        elif exact_bot == "Bot":
            author["type"] = "Bot"
        else:
            author["type"] = types.get(login, "User")
        r["author"] = author

    combined = list(reviews)
    for row in rows:
        node_id = str(row.get("node_id") or "")
        if node_id and node_id in seen_nodes:
            continue
        user = row.get("user") or {}
        combined.append({
            "id": node_id,
            "state": row.get("state"),
            "commit": {"oid": row.get("commit_id")},
            "commit_id": row.get("commit_id"),
            "author": {"login": user.get("login"), "type": user.get("type"),
                       "node_id": user.get("node_id")},
            "submittedAt": row.get("submitted_at"),
            "submitted_at": row.get("submitted_at"),
            "body": row.get("body"),
            "url": row.get("html_url"),
            "html_url": row.get("html_url"),
        })
    return combined


def _green_fields(cfg: Config, repo: str, sha: str, reviews: list, statuses: list,
                  rollup: list, enabled_sources: tuple = (), check_runs=()) -> dict:
    """The two greens + the trusted red for ONE head SHA. Fails closed.

    green 1 is Apollo's review state on `sha` (the newest one wins); green 2 is
    the architect's success status on `sha` from the owner's account, or the
    newest signal of any ENABLED trusted reviewer on `sha` (their status,
    check run, or review — counted once per reviewer).  A reviewer on no
    trusted list (including the PR author) is ignored entirely.
    """
    from . import trusted_reviewers as tr
    from .webhook_server import is_architect_context

    owner = repo.split("/", 1)[0]
    owner_logins = {owner.lower()}
    if repo in ARCHITECT_ALWAYS_REPOS:
        owner_logins |= ({str(getattr(cfg, "owner_install_id", "") or "").lower(),
                          DEFAULT_OWNER_INSTALL_ID.lower()} - {""})
    contexts = _trusted_contexts(cfg)
    apollo = contexts[0] if contexts else ""
    out: dict = {"trusted_red": [], "trusted_seen": []}

    # --- green 1: Apollo ACCEPT on this exact head (always required) -------
    ordered = sorted((r for r in (reviews or []) if isinstance(r, dict)),
                     key=_review_submitted_key)
    chosen = None
    for r in ordered:                             # oldest first — newest wins
        if _review_oid(r) != sha:                 # bound to this head only
            continue
        author = _review_author(r)
        # Defect 5: an empty/missing author never matches any source — the old
        # `if author and not ...` fell OPEN for a falsy author and chose a
        # nameless review as Apollo's ACCEPT.
        if not author or not tr.source_matches(apollo, author):
            continue                              # any other account is ignored
        if str((r.get("author") or {}).get("type") or "") != "Bot":
            continue
        chosen = r
    if chosen is None:
        out.update(green1_ok=False, green1_reason="no review by Apollo on this exact head",
                   green1_label="Apollo ACCEPT", green1_url="", green1_at="")
    else:
        state = str(chosen.get("state") or "").upper()
        accepted = state in ("ACCEPT", "APPROVED")
        out.update(green1_ok=accepted,
                   green1_reason="" if accepted else (state or "no verdict state"),
                   green1_label="Apollo ACCEPT",
                   green1_url=str(chosen.get("url") or chosen.get("html_url") or ""),
                   green1_at=str(chosen.get("submittedAt") or chosen.get("submitted_at") or ""))

    latest_statuses = _newest_status_per_context(statuses).values()

    # --- green 2a: the architect's approval on this exact head -------------
    arch = next((s for s in latest_statuses
                 if is_architect_context(str(s.get("context") or ""))), None)
    if (arch is not None
            and str(arch.get("state") or "").lower() == "success"
            and str((arch.get("creator") or {}).get("login") or "").lower() in owner_logins):
        out.update(green2_ok=True, green2_kind="architect",
                   green2_label="Zeus ACCEPT",
                   green2_url=str(arch.get("target_url") or ""),
                   green2_at=str(arch.get("created_at") or ""), green2_reason="")

    # --- green 2b: an enabled trusted reviewer's green (either counts) -----
    greens_found: list = []
    for src in enabled_sources:
        # A review establishes the trusted account's node identity.  Once known,
        # a same-login status from any other node is foreign, regardless of its
        # context.  Login+Bot is only the compatibility fallback with no node.
        trusted_nodes = {
            str((r.get("author") or {}).get("node_id") or "")
            for r in (reviews or [])
            if tr.source_matches(src, _review_author(r))
            and str((r.get("author") or {}).get("type") or "") == "Bot"
            and str((r.get("author") or {}).get("node_id") or "")
        }

        def trusted_status(status):
            creator = status.get("creator") or {}
            if str(creator.get("type") or "") != "Bot":
                return False
            node = str(creator.get("node_id") or "")
            if trusted_nodes:
                return bool(node and node in trusted_nodes)
            return tr.source_matches(src, str(creator.get("login") or ""))

        sts = [s for s in latest_statuses if trusted_status(s)]
        rv = next((r for r in reversed(reviews or [])
                   if _review_oid(r) == sha and _review_author(r)
                   and tr.source_matches(src, _review_author(r))
                   and str((r.get("author") or {}).get("type") or "") == "Bot"), None)
        cks = [c for c in (check_runs or [])
               if tr.source_matches(src, str((c.get("app") or {}).get("slug") or ""))]
        signals = []
        for st in sts:
            creator_login = str((st.get("creator") or {}).get("login") or "")
            signals.append((str(st.get("context") or ""),
                            str(st.get("state") or "").lower(), "status",
                            st.get("target_url") or "", st.get("created_at") or "",
                            creator_login or str(st.get("context") or "")))
        for ck in cks:
            label = str(ck.get("context") or ck.get("name") or "")
            signals.append((label,
                            str(ck.get("conclusion") or ck.get("state") or "").lower(),
                            "check", ck.get("html_url") or ck.get("details_url") or "",
                            ck.get("completed_at") or ck.get("started_at") or "", label))
        if rv is not None:
            label = _review_author(rv)
            signals.append((label, str(rv.get("state") or "").lower(),
                            "review", rv.get("url") or rv.get("html_url") or "",
                            rv.get("submittedAt") or rv.get("submitted_at") or "", label))
        if not signals:
            continue
        # Every latest non-success status from a trusted creator is red.  A
        # request-changes review or failing check also blocks until re-driven.
        red = [s for s in signals
               if (s[2] == "status" and s[1] != "success")
               or s[1] in ("failure", "error", "cancelled", "timed_out",
                           "changes_requested", "request_changes", "changes request")]
        green = [s for s in signals if s[1] in ("success", "accept", "accepted", "approved")]
        pick = red[0] if red else (green[0] if green else signals[0])
        out["trusted_seen"].append({"source": src, "match": pick[0],
                                    "state": pick[1], "login": pick[5]})
        if red:
            label, state = red[0][5], red[0][1]
            out["trusted_red"].append(
                f"{label} (" + ("requested changes" if "changes" in state or
                               "request" in state else state) + ")")
        elif green:
            greens_found.append(green[0])
    # Rule (d) is decided only after EVERY enabled source is collected
    # (W1-07-r2 defect 2): `break` on the first green hid a red reviewer
    # sorted after it, so "no enabled trusted reviewer is red on this
    # head" failed open on merge authority.  Any red refuses; a green
    # counts only when nothing is red.
    if not out.get("green2_ok") and greens_found and not out["trusted_red"]:
        match, _, kind, url, at, _ = greens_found[0]
        word = "accepted" if kind == "review" else "passed"
        out.update(green2_ok=True, green2_kind="trusted",
                   green2_label=f"{match} {word} (trusted)",
                   green2_url=str(url or ""), green2_at=str(at or ""),
                   green2_reason="")

    if not out.get("green2_ok"):
        out.update(green2_ok=False, green2_kind="", green2_label="", green2_url="",
                   green2_at="", green2_reason=(
                       "no architect approval (architect/* status success set from the "
                       "owner's account) on this exact head and no enabled trusted "
                       "reviewer green on this exact head"))

    # --- CI summary for the merge comment (one source: GitHub's rollup) ----
    out["ci_total"] = len(rollup or [])
    out["ci_green"] = sum(1 for c in (rollup or [])
                          if str(c.get("conclusion") or "").upper() in ("SUCCESS", "NEUTRAL")
                          or str(c.get("state") or "").upper() == "SUCCESS")
    return out


def _enabled_reviewers(cfg: Config, state, repo: str,
                       read_status: dict | None = None) -> tuple[str, ...]:
    """Read install-scoped reviewers and report whether the read was incomplete.

    An App/DB failure cannot be treated as an empty trusted-reviewer list: the
    two-green gate must know that it cannot prove the absence of a trusted red.
    A legacy state with no DSN keeps its established no-sources behavior.
    """
    dsn = getattr(state, "dsn", "")
    if not isinstance(dsn, str) or not dsn:
        return ()
    try:
        from . import github_app as _gh_app
        from . import trusted_reviewers
        app = _gh_app.github_app_for(cfg)
        if app is None:
            raise _gh_app.AppNotConfigured("no GitHub App identity configured")
        install_id = str(app.installation_for_repo(repo))
        owner_install_id = (
            install_id
            if repo.split("/", 1)[0].lower() == DEFAULT_OWNER_INSTALL_ID.lower()
            else None)
        trusted_reviewers.ensure_schema(dsn, owner_install_id=owner_install_id)
        return trusted_reviewers.enabled_sources(dsn, install_id, repo)
    except Exception as exc:  # noqa: BLE001 — fail closed, never strand the wake
        if read_status is not None:
            read_status["trusted_reviewers_unreadable"] = True
        log.warning("trusted_reviewers: installation id for %s unresolvable "
                    "(%s) — trusted reviewers unreadable (fail closed)",
                    repo, type(exc).__name__)
        return ()


def _review_evidence(token: str, cfg: Config, repo: str, pr: int, sha: str,
                     enabled_sources: tuple = (), reviewers_unreadable: bool = False) -> dict:
    """Machine-readable review evidence bound to ONE commit, not to the PR.

    A result produced for commit A may authorize only commit A.  The trusted
    producer is the Reviews App's CHECK RUN on exactly `sha` (GET
    /repos/{repo}/commits/{sha}/check-runs): only a check run whose `app.id`
    equals NEO_TRUSTED_REVIEWER_APP_ID and whose `head_sha` equals `sha`
    (full 40-char equality) counts.  A review or status from any other
    account (including the PR author) is ignored.  Unresolved CRITICAL/HIGH
    and coverage come from that run's output.summary JSON — never hard-coded.
    """
    try:
        rc, raw = _gh(token, "pr", "view", str(pr), "-R", repo,
                      "--json", "headRefOid,headRefName,reviews,statusCheckRollup", timeout=45)
    except Exception as exc:  # noqa: BLE001 — fail closed, never strand the wake
        return {"ok": False, "reason": f"evidence read failed ({type(exc).__name__})",
                "decided_sha": sha, "head_ref": ""}
    if rc != 0:
        return {"ok": False, "reason": "pr view unavailable", "decided_sha": sha, "head_ref": ""}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "pr view malformed", "decided_sha": sha, "head_ref": ""}
    head_ref = str(data.get("headRefName") or "")

    from .merge_executor import EvidenceReadError, read_evidence_pages

    # Every list that can hide a trusted red uses the same completeness rule.
    try:
        check_runs = read_evidence_pages(
            _gh, token, f"repos/{repo}/commits/{sha}/check-runs", "check_runs")
        data["check_runs"] = check_runs
    except EvidenceReadError as exc:
        return {"ok": False, "reason": str(exc), "error": type(exc).__name__,
                "decided_sha": sha, "head_ref": head_ref}

    trusted = _trusted_contexts(cfg)
    try:
        statuses = read_evidence_pages(
            _gh, token, f"repos/{repo}/commits/{sha}/status", "statuses")
    except EvidenceReadError as exc:
        return {"ok": False, "reason": str(exc), "error": type(exc).__name__,
                "decided_sha": sha, "head_ref": head_ref}
    trusted_states = [
        str(item.get("state") or "").lower()
        for item in _newest_status_per_context(statuses).values()
        if _is_trusted(str(item.get("context") or ""), trusted)
    ]
    if not trusted:
        return {"ok": False, "reason": "no trusted review contexts configured",
                "decided_sha": sha, "head_ref": head_ref}

    try:
        raw_reviews = data.get("reviews") or []
        if not isinstance(raw_reviews, list):
            raise TypeError("reviews must be a list")
        reviews = _reviews_with_bot_types(token, repo, pr, raw_reviews)
        review_sha = next((str(r.get("commit_id") or "") for r in reversed(reviews)
                           if r.get("body")), "")

        # Required checks come only from the complete REST lists.  GraphQL's
        # statusCheckRollup is unpaginated and cannot prove this head is green.
        rollup = [*check_runs, *statuses]
        data["statusCheckRollup"] = rollup
        failing = [c for c in rollup
                   if str(c.get("conclusion") or "").upper() in {"FAILURE", "TIMED_OUT", "CANCELLED"}
                   or str(c.get("state") or "").upper() in {"FAILURE", "ERROR"}]
        all_success = all(
            str(c.get("conclusion") or "").upper() == "SUCCESS"
            or str(c.get("state") or "").upper() == "SUCCESS"
            for c in rollup
        ) if rollup else None

        # Deterministic evidence plus the legacy keys the worker consumes.
        from . import review_evidence as rev_ev
        ev = rev_ev.build_evidence(cfg, data, sha)
        ev["review_sha"] = review_sha
        ev["review_bound_ok"] = ev.get("review_sha_match", True)
        ev["trusted_states"] = trusted_states
        ev["trusted_green"] = bool(trusted_states) and all(s == "success" for s in trusted_states)
        ev["failing_checks"] = ev.get("required_checks_failing", len(failing))
        ev["rollup_green"] = all_success
        ev["head_ref"] = head_ref
        ev["reviews"] = reviews
        ev["statuses"] = statuses
        ev["trusted_reviewers_unreadable"] = bool(reviewers_unreadable)
        ev.update(_green_fields(cfg, repo, sha, reviews, statuses, rollup,
                                enabled_sources, check_runs))
        return ev
    except EvidenceReadError as exc:
        return {"ok": False, "reason": str(exc), "error": type(exc).__name__,
                "decided_sha": sha, "head_ref": head_ref}
    except Exception as exc:  # noqa: BLE001 — malformed green evidence fails closed
        return {"ok": False, "reason": "green evaluation failed",
                "error": type(exc).__name__, "decided_sha": sha, "head_ref": head_ref}


#: Verdict → recorded phase for a completed (non-ACCEPT, non-builder-command)
#: run.  Production uses this exact mapping; tests must not rebuild it.
BUILDER_VERDICT_PHASES = {
    "FIXED": "ready",
    "NO-CHANGE": "ready",
    # Not "parked": credits/advisor parks share that phase and are
    # transient — the recovery gate must not treat them as acted.
    "BLOCKED": "builder_blocked",
    "FIX-FORWARD": "fix_forward",
    "ESCALATE": "escalated",
}


def resolve_auto_merge(install_auto_merge, cfg: Config) -> bool:
    """P0 hotfix: nullable inheritance for auto-merge.

    An explicit install value wins; only ``None`` inherits the global
    default.  (The old ``or`` made an explicit False inherit True.)
    """
    if install_auto_merge is None:
        return bool(cfg.auto_merge_default)
    return bool(install_auto_merge)


def merge_authority(install_auto_merge: bool, evidence: dict,
                    jev: JevAssessment, cfg: Config | None = None) -> tuple[bool, str]:
    """Exact-SHA merge authority. Fails closed; never a veto by adjective."""
    if not install_auto_merge:
        return False, "auto-merge disabled for this install"
    if not evidence.get("ok"):
        return False, f"review evidence unavailable ({evidence.get('reason')})"
    # An unreadable head branch cannot prove the PR is not a lane PR, so it may
    # never be merged under the non-lane rule (which skips architect and drift).
    if "head_ref" in evidence and not evidence.get("head_ref"):
        return False, "PR head branch unavailable — cannot rule out a lane PR"
    cfg = cfg if cfg is not None else Config()
    if lane_of(str(evidence.get("head_ref") or ""), cfg):
        return False, "lane PR head must use the lane merge gate"
    if evidence.get("head_sha") != evidence.get("decided_sha"):
        return False, "head moved since the wake — stale evidence cannot authorize this SHA"
    if not evidence.get("review_bound_ok", True):
        return False, "the reviewed commit is not this head SHA"
    # The merge rule (b)-(d): two greens on this exact head and no trusted
    # reviewer red (board `merge-authority` v7).  Presence-gated like the
    # W1-13 belt above: a bundle without the keys is older/patched evidence
    # and keeps the legacy behaviour; production evidence always carries them.
    from . import merge_executor as _merge_executor
    if _merge_executor.has_rule_fields(evidence):
        ok, why = _merge_executor.two_green_gate(evidence)
        if not ok:
            return False, why
    if not evidence.get("trusted_green"):
        return False, "the trusted review status is not success on this exact head SHA"
    # Trusted producer = the Reviews App's check run on this exact head
    # (item 3).  A dict without the key (older patched evidence) keeps the
    # legacy behavior; an explicit False means no trusted run -> no merge.
    if evidence.get("trusted_producer_ok") is False:
        return False, "no trusted check run from the Reviews App on this exact head"
    blockers = jev.blockers
    if blockers:
        return False, "Jev blocking findings: " + ",".join(f.rule for f in blockers)
    return True, "exact-head evidence complete"


# North Star lane branches ("l3-kit/...", "l8d-engine-mcps/...", "l8b/...") belong to their lane: one writer
# per branch, the lane's briefed worker. Neo never runs its agent on one (no fix-forward, no builder
# bounce) and merges only the exact head that Apollo passed AND the lane architect accepted with a
# commit status "architect/<lane>" set from the owner's account. L0 ruling 2026-10-25, north-star #1.
# LANE_BRANCH and lane_of are imported from neo_builder: one classifier for both paths.
ARCHITECT_CONTEXT = "architect/"
DRIFT_CONTEXT = "l0/drift"   # L0 sentinel, SPRINT.md section 9


def lane_merge_authority(lane_merge: bool, evidence: dict, lane: str,
                         statuses: list[dict], reviews: list[dict], owner: str,
                         require_drift: bool = False) -> tuple[bool, str]:
    """Merge authority for a lane PR. Fails closed. Follows NEO_LANE_MERGE, not the install switch."""
    if not lane_merge:
        return False, "lane merges disabled (NEO_LANE_MERGE is off)"
    if not evidence.get("ok"):
        return False, f"review evidence unavailable ({evidence.get('reason')})"
    head = evidence.get("head_sha") or ""
    if not head or head != evidence.get("decided_sha"):
        return False, "head moved since the wake — stale evidence cannot authorize this SHA"
    if not evidence.get("trusted_green"):
        return False, "Apollo's status is not success on this exact head SHA"
    if not evidence.get("required_checks_pass"):
        return False, "required checks have not passed on this exact head SHA"
    if evidence.get("green1_ok") is not True:
        return False, ("green 1 missing: Apollo's review is not ACCEPT on this exact head SHA "
                       f"({evidence.get('green1_reason') or 'no review by Apollo on this head'})")
    if evidence.get("trusted_red"):
        return False, ("a trusted reviewer is red on this exact head: "
                       + ", ".join(str(x) for x in evidence["trusted_red"]))
    latest_statuses = _newest_status_per_context(statuses).values()
    latest = next((s for s in latest_statuses
                   if str(s.get("context") or "").lower() == ARCHITECT_CONTEXT + lane), None)
    if not latest or str(latest.get("state") or "").lower() != "success":
        return False, f"waiting for the {lane} architect to accept this exact head"
    if str((latest.get("creator") or {}).get("login") or "").lower() != owner.lower():
        return False, f"the {ARCHITECT_CONTEXT}{lane} status was not set from the owner's account"
    if require_drift:
        drift = next((s for s in latest_statuses
                      if str(s.get("context") or "").lower() == DRIFT_CONTEXT), None)
        if not drift:
            return False, "waiting for L0's drift check (l0/drift) on this exact head"
        if str(drift.get("state") or "").lower() != "success":
            return False, f"L0's drift check fails on this exact head ({drift.get('description') or 'see the PR'})"
        drift_login = str((drift.get("creator") or {}).get("login") or "").lower()
        if not drift_login or drift_login != owner.lower():
            return False, "l0/drift status was not set from the owner's account"
    return True, "Apollo and the lane architect both passed this exact head"


def run_lane_gate(token: str, cfg: Config, state: NeoState, repo: str, pr: int, sha: str,
                  lane: str, evidence: dict | None = None,
                  enabled_sources: tuple[str, ...] | None = None) -> None:
    """Decide a lane PR without the agent: merge the exact head, or hold and say why."""
    if state.already_acted(repo, pr, sha, "merged"):
        log.info("%s#%s@%s already merged by Neo — skip", repo, pr, sha[:8])
        return
    org = repo.split("/", 1)[0]
    if evidence is None:
        try:
            reviewer_read: dict = {}
            if enabled_sources is None:
                enabled_sources = _enabled_reviewers(cfg, state, repo, reviewer_read)
            evidence = _review_evidence(
                token, cfg, repo, pr, sha, enabled_sources=enabled_sources,
                reviewers_unreadable=reviewer_read.get("trusted_reviewers_unreadable", False))
        except Exception as exc:  # noqa: BLE001 — edge 5: a broken read fails CLOSED
            reason = f"lane gate: evidence read broke ({type(exc).__name__}) — fail closed"
            state.record(repo, pr, sha, "lane_hold", verdict="HOLD",
                         detail={"lane": lane, "reason": reason})
            return
    statuses = evidence.get("statuses") or []
    fresh_reviews = evidence.get("reviews") or []
    # Defect 1: the gate's conditions dropped review_bound_ok — an ACCEPT on an
    # older commit passed. The reviewed commit must be this head SHA (belt
    # semantics as in merge_authority: a bundle without the key is legacy
    # evidence and keeps its behaviour).
    if not evidence.get("review_bound_ok", True):
        ok, reason = False, "lane gate: the reviewed commit is not this head SHA"
    else:
        ok, reason = lane_merge_authority(
            bool(getattr(cfg, "lane_merge", False)), evidence, lane,
            statuses if isinstance(statuses, list) else [],
            fresh_reviews if isinstance(fresh_reviews, list) else [], org,
            require_drift=getattr(cfg, "lane_require_drift", False) is True)
    log.info("%s#%s@%s lane=%s merge authority=%s (%s)", repo, pr, sha[:8], lane, ok, reason)
    if ok:
        evidence["green2_ok"] = True
        evidence["green2_kind"] = "architect"
        evidence["green2_label"] = f"{ARCHITECT_CONTEXT}{lane} ACCEPT"
        evidence["green2_url"] = ""
        evidence["green2_at"] = ""
        evidence["green2_reason"] = ""
        from . import merge_executor as _merge_executor
        try:
            result = _merge_executor.merge(repo, pr, sha, evidence, token)
            if result.get("merged"):
                state.record(repo, pr, sha, "merged", verdict="ACCEPT-MERGED",
                             detail={"lane": lane, "reason": reason})
                return
            reason = "merge executor returned no merge confirmation"
        except _merge_executor.MergeError as exc:
            reason = str(exc)
        except Exception as exc:  # noqa: BLE001 — unexpected failures must be visible holds
            log.warning("%s#%s@%s lane merge executor failed (%s)",
                        repo, pr, sha[:8], type(exc).__name__)
            reason = f"lane gate: merge executor failed ({type(exc).__name__})"
    state.record(repo, pr, sha, "lane_hold", verdict="HOLD", detail={"lane": lane, "reason": reason})


def _read_head_ref(token: str, repo: str, pr: int) -> str:
    """Independently read the branch name; an unavailable branch stays unknown."""
    try:
        rc, raw = _gh(token, "pr", "view", str(pr), "-R", repo,
                      "--json", "headRefName", timeout=30)
        if rc != 0:
            return ""
        data = json.loads(raw)
        branch = data.get("headRefName")
        return str(branch) if isinstance(branch, str) else ""
    except Exception:  # noqa: BLE001 — unknown must never fall through to the agent
        return ""


def _lane_pr(evidence: dict, cfg: Config | None = None) -> str | None:
    """Return the PR's North Star lane, or None for an ordinary branch."""
    branch = str(evidence.get("head_ref") or "")
    if not branch:
        return None
    return lane_of(branch, cfg) or None


def _jev_for_pr(token: str, cfg: Config, repo: str, pr: int, sha: str = "") -> JevAssessment:
    """Read this PR's patch and review evidence for one cheap Jev decision."""
    if not isinstance(cfg.jev_api_key, str) or not cfg.jev_api_key:
        return JevAssessment(False, reason="no OpenRouter key")
    rc, diff = _gh(token, "pr", "diff", str(pr), "-R", repo, timeout=45)
    if rc != 0:
        return JevAssessment(False, reason="PR diff unavailable")
    rc, raw = _gh(token, "pr", "view", str(pr), "-R", repo,
                  "--json", "reviews,comments", timeout=45)
    if rc != 0:
        return JevAssessment(False, reason="review evidence unavailable")
    try:
        data = json.loads(raw)
        # GitHub status arrives after the review is published. The most recent
        # formal review is more relevant than concatenating old bot output.
        reviews = data.get("reviews") or []
        comments = data.get("comments") or []
        review = next((str(item.get("body")) for item in reversed(reviews)
                       if item.get("body")), "")
        if not review:
            review = next((str(item.get("body")) for item in reversed(comments)
                           if item.get("body")), "")
    except (TypeError, ValueError, AttributeError):
        return JevAssessment(False, reason="review evidence malformed")
    return jev_assess(diff, review, cfg.jev_api_key, head_sha=sha)


def evidence_brief(evidence: dict, auto_merge: bool, reason: str) -> str:
    """Machine-readable evidence line handed to the Neo agent.

    The agent's prose must never outrank this: if it says merge authority is not
    granted, the agent may ACCEPT and label but must not merge.
    """
    if not evidence.get("ok"):
        return (f"MERGE_AUTHORITY: NOT_GRANTED — {reason}. Do not merge; continue non-merge work "
                "and record what blocked it.")
    head = (evidence.get("head_sha") or "?")[:12]
    review_sha = (evidence.get("review_sha") or "?")[:12]
    grant = "GRANTED" if auto_merge else "NOT_GRANTED"
    return (
        f"MERGE_AUTHORITY: {grant} ({reason}).\n"
        f"EVIDENCE head_sha={head} decided_sha={evidence.get('decided_sha', '?')[:12]} "
        f"review_sha={review_sha} review_bound_ok={evidence.get('review_bound_ok')} "
        f"trusted_review_status={evidence.get('trusted_states')} "
        f"(other checks in this repo: {evidence.get('failing_checks')} failing, informational).\n"
        "Merge only when MERGE_AUTHORITY is GRANTED for this exact head SHA."
    )


def _run_agent(
    agent: ClaudeAgent,
    brief: str,
    cfg: Config | None = None,
    gh_token: str = "",
    usage_store=None,
    job_id: int | None = None,
    attempt_number: int = 1,
) -> tuple[dict, str, int | None, object | None]:
    """Run the Neo brief headless via Claude Code CLI, walking the route chain.

    Uses run_with_chain when cfg and routes are available; falls back to the
    single configured route for backward compatibility.

    Returns (normalised usage, verdict line, winning_route_idx, winning_route).
    AgentError is caught here (the caller retries the job); any OTHER exception
    from the agent spawn is allowed to propagate so the caller's broad handler
    releases the job too (defect 3).
    """
    from .usage import available_routes

    try:
        if cfg is not None and cfg.parsed_routes:
            avail = available_routes(cfg.parsed_routes)
            if avail:
                usage, verdict, ridx, route = agent.run_with_chain(
                    brief, avail, cfg, gh_token,
                    usage_store=usage_store, job_id=job_id,
                    attempt_number=attempt_number,
                )
                return usage, verdict, ridx, route
        # Fallback: single configured route (backward compat).
        usage, verdict = agent.run(brief)
        return usage, verdict, None, None
    except AgentError:
        log.exception("neo claude agent run failed")
        return {}, "", None, None


def _wake_lease(cfg: Config) -> int:
    """Wake triage leases must cover the whole agent run (a 600 s default lease
    with a 30-minute agent would expire mid-run and let a second worker claim
    the same SHA)."""
    timeout = int(getattr(cfg, "deepseek_timeout", 1800) or 1800)
    return max(600, timeout + 120)


def run_neo(cfg: Config, state: NeoState, jobs: Jobs, wake: dict,
           usage_store: UsageStore | None = None) -> None:
    repo, sha = wake["repo"], wake["sha"]
    green = verdict_color(wake.get("green"))

    # --- durable wake job -------------------------------------------------
    # on_verdict() already persisted the job (keyed by repo + head sha +
    # verdict color, PR resolved later).  If we were called without a job row
    # (tests / manual replay), create it here — dedupe-on-outcome means a
    # terminal or expired job is re-armed, an active one is left alone.
    job_id = wake.get("job_id")
    if not job_id:
        job_id = jobs.create_wake(repo, sha, bool(green),
                                  pr_number=wake.get("pr_number"), payload=wake)
        if job_id is None:
            log.info("deduped wake %s@%s — an active job already owns it", repo, sha[:8])
            return

    # --- GitHub identity for THIS repo (the customer's App installation) ---
    # Neo acts as the GitHub App: every GitHub call below uses this repo's
    # installation token, never a personal token. A repo with no App
    # installation is refused with a clear log line and the job resolves
    # `no_installation` — terminal, not retried (owner re-arms by installing
    # the App on the repo).
    try:
        token = repo_token(cfg, repo, wake.get("installation_id"))
    except (NoInstallation, AppNotConfigured) as exc:
        log.warning("refusing wake %s@%s: %s — no GitHub App installation "
                    "(no_installation)", repo, sha[:8], exc)
        wtoken = secrets.token_hex(16)
        if jobs.claim(job_id, pr_number=wake.get("pr_number"),
                      lease_secs=_wake_lease(cfg), worker_token=wtoken):
            jobs.resolve(job_id, "no_installation", worker_token=wtoken,
                         payload={"reason": f"no GitHub App installation for {repo}"})
        return
    except GitHubAppError as exc:
        # Transient GitHub API failure (5xx, timeout, token-mint error).
        # Before the claim, nothing else would release this job — its
        # submitted_at stamp hides it from the sweep.  Release it retryably.
        log.warning("transient GitHub error for %s@%s: %s — job released retryably",
                    repo, sha[:8], exc)
        wtoken = secrets.token_hex(16)
        if jobs.claim(job_id, pr_number=wake.get("pr_number"),
                      lease_secs=_wake_lease(cfg), worker_token=wtoken):
            jobs.retry(job_id, worker_token=wtoken,
                       reason=f"transient GitHub identity error ({type(exc).__name__})")
        return

    try:
        pr = resolve_pr(token, repo, sha, wake.get("pr_number"))
    except Exception as exc:
        # gh hung (TimeoutExpired) or is missing: before the claim, nothing else would
        # release this job — its submitted_at stamp hides it from the sweep. Release it
        # retryably, exactly like the no-PR branch below.
        log.warning("resolve_pr crashed (%s) for %s@%s — job released retryably",
                    type(exc).__name__, repo, sha[:8])
        wtoken = secrets.token_hex(16)
        if jobs.claim(job_id, pr_number=None, lease_secs=_wake_lease(cfg),
                      worker_token=wtoken):
            jobs.retry(job_id, worker_token=wtoken,
                       reason=f"pre-claim crash ({type(exc).__name__})")
        return
    if not pr:
        # Transient gh failure, or a commit status that fired before any PR
        # exists.  Release the job retryably (claim-then-retry: claim() marks
        # the run so the attempt cap bounds it, retry() returns it to queued +
        # outcome='failed').  The recovery loop re-submits it; after the cap the
        # job parks and a future wake re-arms it — the SHA is never permanently
        # poisoned (defect 5 / GLM 5).
        log.info("no open PR for %s@%s — job released retryably", repo, sha[:8])
        wtoken = secrets.token_hex(16)
        if jobs.claim(job_id, pr_number=None, lease_secs=_wake_lease(cfg),
                      worker_token=wtoken):
            jobs.retry(job_id, worker_token=wtoken, reason="no open PR for head sha")
        else:
            log.info("job %s already claimed for %s@%s — another worker resolves PR",
                     job_id, repo, sha[:8])
        return

    # --- claim the durable job (dedupe + triage in one transaction) -------
    # Atomic UPDATE ... WHERE state='queued' — if another worker already
    # claimed it (concurrent re-submit), we skip.  The worker_token is our
    # claimant identity: every resolve/retry/lease below must present it, so a
    # stale claimant (lease lost under a DB outage) can never mutate a
    # replacement run.
    wtoken = secrets.token_hex(16)
    if not jobs.claim(job_id, pr_number=pr, lease_secs=_wake_lease(cfg),
                      worker_token=wtoken):
        log.info("job already claimed %s job=%s — skip (deduped)", repo, job_id)
        return

    # --- per-SHA idempotency gate (integrator finding 2 / defect 2) -------
    # A duplicate delivery of the same verdict (GitHub redelivery, or a
    # reviewer re-publishing success on the same SHA — both normal) is re-armed
    # by create_wake (dedupe on OUTCOME, so the verdict is never dropped).  But
    # re-running the whole triage would re-debit credits, re-merge an
    # already-merged PR, and mint an extra builder round.  The color-aware
    # already_acted gate makes the re-arm idempotent.  The triage row is
    # recorded only AFTER a completed agent run (see below), so a retried job
    # whose agent crashed mid-run is NOT gated — it re-runs — while a verdict
    # that actually completed is treated as already-acted.
    if state.already_acted(repo, pr, sha, "triage", green=green) or \
            state.already_acted(repo, pr, sha, "merged"):
        jobs.resolve(job_id, "no_change", worker_token=wtoken,
                     payload={"reason": "already acted on this SHA/verdict"})
        log.info("%s#%s@%s already acted (triage green=%s or merged) — "
                 "duplicate delivery skipped", repo, pr, sha[:8], green)
        return
    # A previous run of this verdict did its work (phase row, builder job) and then died
    # before writing the triage marker. Re-running the agent would mint a second builder
    # round; finish the bookkeeping instead.
    for done_phase in ("builder", "ready", "fix_forward", "escalated",
                       "escalating", "builder_blocked", "blocked"):
        if state.already_acted(repo, pr, sha, done_phase, green=green):
            state.record(repo, pr, sha, "triage",
                         detail={"green": bool(green), "recovered": done_phase})
            jobs.resolve(job_id, "done", worker_token=wtoken,
                         payload={"reason": f"{done_phase} already recorded; triage marker restored"})
            log.info("%s#%s@%s %s already recorded — triage marker restored, no re-run",
                     repo, pr, sha[:8], done_phase)
            return

    rounds = state.builder_rounds(repo, pr)

    # --- entitlement (per-install credit balance) ---------------------------
    org = repo.split("/", 1)[0]
    inst = state.install(org, org)          # dogfood: install keyed by org
    if meters_credits(inst, cfg) and inst.get("credits_balance", 0) <= 0:
        state.record(repo, pr, sha, "parked", verdict="PARKED",
                     detail={"reason": "out of credits — owner must top up"})
        _gh(token, "pr", "comment", str(pr), "-R", repo,
            "--body", "Neo: this install is out of credits — top up to resume auto-fix/merge.")
        jobs.park(job_id, worker_token=wtoken,
                  reason="out of credits — owner must top up")
        log.info("%s#%s: install out of credits — gated (parked)", repo, pr)
        return
    # P0 hotfix: nullable inheritance for auto-merge.
    install_auto_merge = resolve_auto_merge(inst.get("auto_merge"), cfg)

    # --- exact-SHA evidence (Phase A) --------------------------------------
    reviewer_read: dict = {}
    enabled_sources = _enabled_reviewers(cfg, state, repo, reviewer_read)
    evidence = _review_evidence(
        token, cfg, repo, pr, sha, enabled_sources=enabled_sources,
        reviewers_unreadable=reviewer_read.get("trusted_reviewers_unreadable", False))
    if evidence.get("ok") and evidence.get("head_sha") and evidence["head_sha"] != sha:
        state.record(repo, pr, sha, "stale_head", verdict="STALE",
                     detail={"context": wake.get("context"),
                             "head_sha": evidence["head_sha"], "decided_sha": sha})
        jobs.resolve(job_id, "no_change", worker_token=wtoken,
                     payload={"reason": "stale head"})
        log.info("%s#%s@%s head moved to %s — stale evidence, not mergeable",
                 repo, pr, sha[:8], evidence["head_sha"][:8])
        return

    # --- North Star lane branches: architect-gated merge, never the agent ---
    head_unknown = "head_ref" in evidence
    if head_unknown and not evidence.get("head_ref"):
        evidence["head_ref"] = (str(wake.get("head_ref") or "")
                                or _read_head_ref(token, repo, pr))
    if head_unknown and not evidence.get("head_ref"):
        state.record(repo, pr, sha, "lane_hold", verdict="HOLD",
                     detail={"lane": "unknown",
                             "reason": "PR head branch unavailable — cannot rule out a lane PR"})
        return
    try:
        lane_pr = _lane_pr(evidence, cfg)
    except (AgentError, TypeError, ValueError) as exc:
        log.warning("%s#%s@%s %s — skip (fail closed)", repo, pr, sha[:8], exc)
        return
    if lane_pr is not None:
        run_lane_gate(token, cfg, state, repo, pr, sha, lane_pr,
                      evidence=evidence, enabled_sources=enabled_sources)
        return
    # Same case-insensitive normaliser as the webhook dispatch (defect 4).
    from .webhook_server import is_architect_context, is_drift_context
    if is_architect_context(str(wake.get("context") or "")) \
            or is_drift_context(str(wake.get("context") or "")):
        return  # an architect or L0 drift status only ever drives a lane PR

    # Jev never supplies a green verdict.
    jev = _jev_for_pr(token, cfg, repo, pr, sha)
    auto_merge, authority_reason = merge_authority(install_auto_merge, evidence, jev, cfg)
    log.info("%s#%s@%s Jev available=%s risk=%s quality=%s partial=%s "
             "requires_full_review=%s blockers=%s",
             repo, pr, sha[:8], jev.available, jev.risk, jev.review_quality,
             jev.partial, jev.requires_full_review,
             ",".join(f.rule for f in jev.blockers) or "-")
    log.info("%s#%s@%s merge authority=%s (%s) head=%s review_sha=%s checks_green=%s",
             repo, pr, sha[:8], auto_merge, authority_reason,
             (evidence.get("head_sha") or "?")[:8], (evidence.get("review_sha") or "?")[:8],
             evidence.get("rollup_green"))

    # --- run the Neo protocol headless via Claude Code CLI --------------------
    # The triage row is recorded AFTER a completed run (below, with the phase
    # row), NOT here: a triage that crashes mid-agent must not leave a marker
    # that the per-SHA idempotency gate would then treat as already-acted on
    # the retry.  Recording it here is what stranded heads in the B0 baseline
    # (the old code had to clear() the marker on agent failure).  With the
    # durable job store the retry is the job's own lease cycle, so the marker
    # is only meaningful once a run actually completes.
    brief = neo_protocol.build_brief(
        repo=repo, pr=pr, head_sha=sha,
        reviewer_context=wake.get("context", "?"), reviewer_green=bool(green),
        auto_merge=auto_merge, builder_round=rounds,
        max_builder_rounds=cfg.max_builder_rounds, builder_model=cfg.deepseek_model,
        jev_context=jev.brief() + "\n" + evidence_brief(evidence, auto_merge, authority_reason),
    )
    # The agent gets the repo's installation token, never a personal one.
    # [Apollo r6] A run without GitHub identity must be loud AT THE CALLER:
    # ERROR-level here (the worker journal — which job ran unauthenticated),
    # not only from_config's once-per-process line and `had_gh_token=False`
    # on the metering rows.
    if not token:
        log.error("%s#%s@%s constructing triage agent with NO gh_token — "
                  "the run proceeds unauthenticated (its attempt rows read "
                  "had_gh_token=False)", repo, pr, sha[:8])
    agent = ClaudeAgent.from_config(cfg, gh_token=token)
    try:
        # Lease renewal: a static lease shorter than the agent run would let a
        # second worker claim the same SHA mid-run (GLM 6 applies to wakes too).
        with jobs.lease_renewal(job_id, lease_secs=_wake_lease(cfg),
                                worker_token=wtoken):
            # Read the job's real attempt counter for usage metering
            # (Defect 5: attempt_number comes from the claimed row, never a constant).
            job_row = jobs.get(job_id)
            attempt_number = job_row["attempt"] if job_row else 1
            usage, verdict, winning_route_idx, winning_route = _run_agent(
                agent, brief,
                cfg=cfg, gh_token=token,
                usage_store=usage_store, job_id=job_id,
                attempt_number=attempt_number,
            )
    except Exception as exc:
        # Any exception from the agent spawn (FileNotFoundError/OSError/
        # UnicodeDecodeError included) releases the job for retry instead of
        # stranding the SHA (defect 3 / GLM 4).
        log.exception("%s#%s@%s agent crashed (%s) — job retried",
                      repo, pr, sha[:8], type(exc).__name__)
        jobs.retry(job_id, worker_token=wtoken,
                   reason=f"agent crashed ({type(exc).__name__})")
        return

    if not verdict:
        log.warning("%s#%s@%s agent produced no verdict — job retried", repo, pr, sha[:8])
        jobs.retry(job_id, worker_token=wtoken, reason="agent produced no verdict")
        return

    # --- B2: Merge executor (deterministic, not agent) --------------------
    # This is the only merge path in Neo: on ACCEPT/ACCEPT-MERGED we call
    # merge_executor.merge(), which re-reads the head, re-checks evidence, and
    # merges with the expected SHA.  The agent is instructed never to merge and
    # never receives a personal token, but its repo-scoped installation token
    # (Contents: write, needed for fix-forward pushes) is technically capable
    # of merging — branch protection (required review + required checks) is
    # the enforced boundary, not the credential scope.  See DEPLOY.md.
    verdict_base = verdict.split()[0] if verdict else ""
    is_accept = verdict_base in ("ACCEPT", "ACCEPT-MERGED", "ACCEPT-READY")
    is_builder = verdict_base == "BOUNCE-BUILDER"
    advice_id = ""  # set when the escalation path submits to the advisor

    if is_accept:
        # Use the evidence already fetched by _review_evidence above.
        # Convert the old evidence dict to the new format for merge_authority.
        # Item 4: unresolved_critical_high and coverage_complete come from the
        # trusted check run's output.summary — nothing is hard-coded here;
        # missing -> merge denied.
        from . import review_evidence as rev_ev
        evidence_b2 = {
            "ok": evidence.get("ok", False),
            "reason": evidence.get("reason", ""),
            "head_sha": evidence.get("head_sha", ""),
            "decided_sha": evidence.get("decided_sha", sha),
            "trusted_producer_ok": evidence.get(
                "trusted_producer_ok", evidence.get("trusted_green", False)),
            "review_sha_match": evidence.get(
                "review_sha_match", evidence.get("review_bound_ok", False)),
            # Fail closed: unknown rollup (None — no rollup data) is NOT a
            # pass.  Only an explicit green rollup authorizes a merge.
            "required_checks_pass": evidence.get("rollup_green") is True,
            "required_checks_failing": evidence.get("failing_checks", 0),
            # From the trusted check run's output — not hard-coded.  Missing
            # (None) is denied by merge_authority.
            "unresolved_critical_high": evidence.get("unresolved_critical_high"),
            "coverage_complete": evidence.get("coverage_complete", False),
            "auto_merge": install_auto_merge,
        }
        # The two greens travel with the bundle (kept present-only, so older
        # patched evidence keeps its legacy shape) — merge() re-runs the same
        # gate on it immediately before the merge API call (W1-07 step 5).
        for _k in ("green1_ok", "green1_reason", "green1_label", "green1_url",
                   "green1_at", "green2_ok", "green2_kind", "green2_label",
                   "green2_url", "green2_at", "green2_reason", "trusted_red",
                   "trusted_seen", "ci_total", "ci_green", "trusted_reviewers_unreadable"):
            if _k in evidence:
                evidence_b2[_k] = evidence[_k]
        granted, reason = rev_ev.merge_authority(evidence_b2)
        if granted and ("green1_ok" in evidence_b2 or "green2_ok" in evidence_b2):
            from . import merge_executor as _merge_executor
            granted, reason = _merge_executor.two_green_gate(evidence_b2)

        if granted and auto_merge and verdict_base in ("ACCEPT", "ACCEPT-MERGED"):
            # Call the deterministic merge executor.
            from . import merge_executor
            try:
                merge_result = merge_executor.merge(
                    repo, pr, sha, evidence_b2, token,
                )
                phase = "merged"
                if merge_result.get("merged"):
                    log.info("%s#%s@%s merged by executor (sha=%s)",
                             repo, pr, sha[:8], merge_result.get("sha", "?")[:12])
            except merge_executor.HeadMoved as exc:
                log.warning("%s#%s@%s head moved before merge: %s",
                            repo, pr, sha[:8], exc)
                phase = "stale_head"
            except merge_executor.MergeDenied as exc:
                log.warning("%s#%s@%s merge denied: %s",
                            repo, pr, sha[:8], exc)
                phase = "ready"  # mark as ready but not merged
            except merge_executor.MergeError as exc:
                log.warning("%s#%s@%s merge failed: %s",
                            repo, pr, sha[:8], exc)
                phase = "ready"
        else:
            phase = "ready"
            if not auto_merge:
                log.info("%s#%s@%s ACCEPT but auto-merge disabled — ready",
                         repo, pr, sha[:8])
            else:
                log.info("%s#%s@%s ACCEPT but evidence blocks merge: %s",
                         repo, pr, sha[:8], reason)

    elif is_builder:
        # --- B2: Escalation check before dispatching builder ---------------
        rounds = state.builder_rounds(repo, pr)
        bounces_before = getattr(cfg, "bounces_before_escalation", 5)
        if state.already_escalated(repo, pr):
            # This PR already had its ONE escalated round; a further bounce
            # parks it for the owner — no second escalation, no more rounds.
            log.info("%s#%s@%s already escalated — bounce parks the PR",
                     repo, pr, sha[:8])
            phase = "parked"
            state.record(repo, pr, sha, "parked", verdict="PARKED",
                         detail={"reason": "escalated builder round bounced again"})
        elif rounds >= bounces_before:
            log.info("%s#%s@%s %d builder rounds — escalating to advisor",
                     repo, pr, sha[:8], rounds)
            from . import escalation
            try:
                round_records = [
                    escalation.RoundRecord(
                        round_no=r, sha=sha, outcome="bounced",
                    )
                    for r in range(1, rounds + 1)
                ]
                # Async: submit returns immediately (advisor runs on the build
                # box); the recovery loop polls and creates the ONE escalated
                # round when the work order is done.
                outcome = escalation.submit_escalation(
                    cfg, state, jobs, repo, pr, sha,
                    rounds=round_records,
                    diff_stat=_compute_diff_stat(token, repo, pr),
                    installation_id=wake.get("installation_id"),
                )
                if outcome["action"] == "escalating":
                    phase = "escalating"
                    advice_id = outcome.get("advice_id", "")
                else:
                    phase = "parked"
                    advice_id = ""
            except Exception as exc:
                log.warning("%s#%s@%s escalation failed: %s — parking",
                            repo, pr, sha[:8], exc)
                phase = "parked"
                state.record(repo, pr, sha, "parked", verdict="PARKED",
                             detail={"reason": f"escalation failed: {exc}"})
        else:
            # Normal builder dispatch.
            phase = "builder"
            jobs.create_builder(repo, pr, sha, rounds + 1, payload={
                "verdict": verdict, "wake_job": job_id,
                "head_sha": sha, "phase": "builder", "round": rounds + 1,
                "context": wake.get("context"),
                "installation_id": wake.get("installation_id"),
            })
    else:
        # FIXED/NO-CHANGE/BLOCKED from a builder-style agent, plus
        # FIX-FORWARD, ESCALATE, and unknown verdicts.
        # Builder contract: FIXED → wait for new head's review;
        # NO-CHANGE → counts as a bounce (ready); BLOCKED → park with reason.
        # Unknown verdicts → BLOCKED: unparseable, never silently dropped.
        phase = BUILDER_VERDICT_PHASES.get(verdict_base, "blocked")

    # --- B4: record which route actually produced the verdict -------------
    # The route chain already recorded per-attempt usage inside run_with_chain.
    # The phase row records the winning route (the route that produced the
    # verdict), not a guess.
    if winning_route is not None:
        route_provider = winning_route.provider
        route_model = winning_route.model
    elif cfg.parsed_routes:
        from .usage import available_routes
        avail = available_routes(cfg.parsed_routes)
        if avail:
            first_idx, first_route = avail[0]
            route_provider = first_route.provider
            route_model = first_route.model
        else:
            route_provider = "unknown"
            route_model = ""
    else:
        # Fallback: no parsed routes — hardcode deepseek for the configured direct route.
        route_provider = "deepseek"
        route_model = cfg.deepseek_model

    # Record the successful attempt in the usage store if not already recorded
    # by the route chain (idempotent via ON CONFLICT DO NOTHING).
    if usage_store is not None and job_id is not None:
        from .usage import UsageAttempt
        try:
            success_attempt = UsageAttempt(
                job_id=job_id,
                attempt_number=attempt_number,
                route_idx=winning_route_idx if winning_route_idx is not None else 0,
                provider=route_provider,
                model=route_model,
                # Defect 9: use None-coalescing get, never x or y (0 is real, None is unknown).
                uncached_input=usage.get("input_tokens") if usage.get("input_tokens") is not None else usage.get("uncached_input"),
                cached_input=usage.get("cache_read_input_tokens") if usage.get("cache_read_input_tokens") is not None else usage.get("cached_input"),
                output_tokens=usage.get("output_tokens"),
                outcome="success" if verdict else "error",
            )
            usage_store.record_attempt(success_attempt)
        except Exception:  # noqa: BLE001
            log.exception("failed to record successful usage for job %s", job_id)

    credits = credits_for(usage, builder=is_builder)
    new_round = (rounds + 1) if is_builder else rounds

    # The phase row carries the verdict colour so a retry after a crash can tell this
    # verdict's work is already done (see the gate above).  An `escalating` row
    # carries the advisor advice_id (submit_escalation already recorded one with
    # the full detail; this row keeps the advice_id too so the recovery loop's
    # newest-first escalating_prs() always finds it) and the webhook
    # installation_id so the escalated round uses the id Neo already had instead
    # of a repo->installation lookup (defect 4).  It is recorded under the
    # per-PR advisory lock (record_escalating) so a concurrent take_escalation
    # can never clobber it (defect 1).
    # The route that actually ran (B4).
    route_detail = {"route_provider": route_provider, "route_model": route_model}

    if phase == "escalating":
        state.record_escalating(repo, pr, sha, verdict=verdict,
                                builder_round=new_round, credits=credits,
                                advice_id=advice_id,
                                installation_id=wake.get("installation_id"),
                                extra={"green": bool(green),
                                       "wake_job": job_id,
                                       **route_detail})
    else:
        state.record(repo, pr, sha, phase, verdict=verdict,
                     builder_round=new_round, credits=credits,
                     detail={"green": bool(green), "wake_job": job_id,
                             **route_detail})
    # The triage marker goes LAST: it vouches that the phase row and any builder job above
    # exist. A crash before this line leaves the phase row, which the gate above detects.
    state.record(repo, pr, sha, "triage", detail={"context": wake.get("context"),
                                                   "green": bool(green),
                                                   "jev": jev.brief(),
                                                   "jev_attention": jev.requires_full_review,
                                                   "head_sha": evidence.get("head_sha"),
                                                   "review_sha": evidence.get("review_sha"),
                                                   "rollup_green": evidence.get("rollup_green"),
                                                   "jev_blockers": [f.rule for f in jev.blockers],
                                                   "auto_merge": auto_merge,
                                                   "reason": authority_reason,
                                                   "route_provider": route_provider,
                                                   "route_model": route_model})

    # Resolve the wake job with a terminal outcome.
    jobs.resolve(job_id, "done", worker_token=wtoken,
                 payload={"verdict": verdict, "phase": phase,
                          "credits": credits})
    if meters_credits(inst, cfg):
        state.debit(org, credits)
    log.info("%s#%s@%s verdict=%s phase=%s credits=%d (auto_merge=%s round=%d)",
             repo, pr, sha[:8], verdict or "?", phase, credits, auto_merge, rounds)



