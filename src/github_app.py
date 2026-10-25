"""GitHub App identity for Neo — App JWT + per-installation tokens.

Neo acts as the customer's GitHub App, not as the owner's personal OAuth
token. A customer grants Neo access to a private repo by installing the App on
it; every GitHub action for a repo is then made with THAT repo's installation
token. There is no fallback to a personal token: a repo with no installation is
refused (job outcome `no_installation`, terminal).

  * App JWT — RS256, 9 minutes, signed with the App's private key
    (NEO_GITHUB_APP_ID + NEO_GITHUB_APP_KEY_PATH, or the PEM text in
    NEO_GITHUB_APP_KEY). The key is never logged.
  * Installation token — POST /app/installations/{id}/access_tokens, cached
    per installation and refreshed 5 minutes before its 1 hour expiry.
  * installation_for_repo — GET /repos/{owner}/{repo}/installation, cached
    with a short TTL so a newly installed repo is picked up promptly.

Uses PyJWT + cryptography (already in requirements) and stdlib `urllib` over
plain HTTPS — no new dependency. Caches are read and written under a lock; the
HTTP call itself runs outside it, so one slow GitHub call never stalls the other
repos (concurrent cold misses may each mint a token — every one is valid).
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import jwt  # PyJWT

log = logging.getLogger("rhobear_neo.github_app")

# Token validity and refresh policy (GitHub installation tokens live 1 hour).
TOKEN_TTL_SECS = 3600
TOKEN_REFRESH_LEAD_SECS = 300      # refresh 5 min before expiry
JWT_TTL_SECS = 540                 # 9 minutes
JWT_CLOCK_SKEW_SECS = 60           # backdate iat so a skewed clock never mints a bad JWT
REPO_CACHE_TTL_SECS = 300          # short TTL for the repo -> installation map

API_BASE = "https://api.github.com"


class GitHubAppError(Exception):
    """Base for all GitHub App identity failures — fail closed."""


class NoInstallation(GitHubAppError):
    """The repo has no GitHub App installation — Neo must refuse to act on it.

    Terminal (not retried): the customer has not approved the App on this
    repo, and Neo never falls back to a personal token.
    """


class InstallationGone(GitHubAppError):
    """A stored installation id no longer exists (POST /app/installations/{id}/access_tokens
    returned 404 — the App was uninstalled, or the id is stale).

    NOT transient: the caller re-resolves the repo's installation (defect 6)
    instead of retrying the dead id forever.
    """


class AppNotConfigured(GitHubAppError):
    """No App identity is configured (no app id / key). Fail closed."""


def _parse_iso(iso: str) -> float:
    """ISO-8601 (GitHub returns e.g. '2026-10-19T12:34:56Z') -> epoch float."""
    s = iso.strip()
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def _next_link(link_header: str) -> str:
    """Extract the rel='next' URL from a GitHub Link header, or '' if absent.

    Handles '<https://...>; rel="next"' and '<...>; rel="next", <...>; rel="last"'
    multi-part headers.
    """
    for part in (link_header or "").split(","):
        segs = [s.strip() for s in part.split(";")]
        if len(segs) < 2:
            continue
        if not any('rel="next"' in s for s in segs[1:]):
            continue
        url = segs[0].strip()
        if url.startswith("<") and url.endswith(">"):
            return url[1:-1]
    return ""


def load_app_key(cfg) -> str:
    """The App private key: the file named by NEO_GITHUB_APP_KEY_PATH, else the
    PEM text in NEO_GITHUB_APP_KEY. Returns '' when neither is configured."""
    path = str(getattr(cfg, "github_app_key_path", "") or "").strip()
    if path:
        try:
            return Path(path).expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            # Never include key content; the path is safe to log.
            raise AppNotConfigured(
                f"cannot read GitHub App private key at {path}: {exc}"
            ) from exc
    return str(getattr(cfg, "github_app_key", "") or "").strip()


class GitHubApp:
    """App identity: JWT signing + per-installation token minting + caching.

    Thread-safe: caches are read and written under `_lock`; HTTP runs outside
    it (a slow call never blocks other repos). Concurrent cold misses on one
    installation may each mint a token; all are valid and the last one cached.
    """

    def __init__(
        self,
        app_id,
        key_pem,
        *,
        http=None,
        time_fn=time.time,
        repo_ttl: int = REPO_CACHE_TTL_SECS,
        token_refresh_lead: int = TOKEN_REFRESH_LEAD_SECS,
        http_timeout: int = 30,
    ) -> None:
        self.app_id = str(app_id).strip()
        self._key_pem = key_pem
        self._http = http or self._default_http
        self._time = time_fn
        self._repo_ttl = repo_ttl
        self._token_refresh_lead = token_refresh_lead
        self._http_timeout = http_timeout
        self._lock = threading.Lock()
        # repo -> (installation_id, fetched_at)
        self._repo_cache: dict[str, tuple[int, float]] = {}
        # installation_id -> (token, expires_at_epoch)
        self._token_cache: dict[int, tuple[str, float]] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "GitHubApp":
        app_id = str(getattr(cfg, "github_app_id", "") or "").strip()
        key_pem = load_app_key(cfg)
        if not app_id or not key_pem:
            raise AppNotConfigured(
                "GitHub App identity missing: NEO_GITHUB_APP_ID and "
                "NEO_GITHUB_APP_KEY_PATH (or NEO_GITHUB_APP_KEY) are required"
            )
        return cls(app_id, key_pem)

    # ------------------------------------------------------------------
    # App JWT
    # ------------------------------------------------------------------

    def jwt(self, *, now: float | None = None) -> str:
        """RS256 App JWT valid for 9 minutes. `iat` is backdated for clock skew
        so a slightly-ahead peer never rejects a just-minted token."""
        now = self._time() if now is None else now
        iat = int(now) - JWT_CLOCK_SKEW_SECS
        exp = iat + JWT_TTL_SECS
        return jwt.encode(
            {"iat": iat, "exp": exp, "iss": self.app_id},
            self._key_pem,
            algorithm="RS256",
        )

    # ------------------------------------------------------------------
    # Installation resolution
    # ------------------------------------------------------------------

    def installation_for_repo(self, repo: str, *, now: float | None = None) -> int:
        """The App installation id for `repo` (GET /repos/{owner}/{repo}/installation).

        Raises NoInstallation when the repo has no installation (GitHub returns
        404 for that). Cached with a short TTL.  The HTTP fetch happens OUTSIDE
        the lock so a slow call does not stall all workers.
        """
        now = self._time() if now is None else now
        with self._lock:
            cached = self._repo_cache.get(repo)
            if cached and now - cached[1] < self._repo_ttl:
                return cached[0]
        # Cache miss — fetch outside the lock (concurrent misses for the same
        # repo may both call the API; that is a minor inefficiency, not a stall).
        status, raw = self._api(
            "GET", f"/repos/{repo}/installation"
        )
        if status == 404:
            raise NoInstallation(
                f"repo {repo} has no GitHub App installation"
            )
        if status != 200:
            raise GitHubAppError(
                f"installation lookup for {repo} failed: HTTP {status}"
            )
        data = json.loads(raw)
        installation_id = int(data["id"])
        with self._lock:
            self._repo_cache[repo] = (installation_id, now)
        return installation_id

    def installation_token(
        self, installation_id: int, *, now: float | None = None,
        permissions: dict | None = None,
    ) -> str:
        """A 1-hour installation token for `installation_id`.

        `permissions`, when given, requests a token with ONLY those permissions
        (e.g. {"contents": "read", "pull_requests": "read"} for a read-only
        token used by the escalation advisor).  The cache key includes the
        permissions dict so a read-only and a write token for the same
        installation are distinct entries.

        Cached and refreshed 5 minutes before expiry. The HTTP fetch happens
        OUTSIDE the lock so a slow token-mint does not stall all workers.
        """
        now = self._time() if now is None else now
        inst = int(installation_id)
        cache_key = (inst, json.dumps(permissions or {}, sort_keys=True))
        with self._lock:
            cached = self._token_cache.get(cache_key)
            if cached and cached[1] - now > self._token_refresh_lead:
                return cached[0]
        # Cache miss — fetch outside the lock.
        body: dict = {}
        if permissions:
            body["permissions"] = permissions
        status, raw = self._api(
            "POST",
            f"/app/installations/{inst}/access_tokens",
            body=body,
        )
        if status == 404:
            # The installation no longer exists (the App was uninstalled or the
            # id is stale).  Drop the cached entry so a later call re-mints,
            # and raise a TYPED error so token_for_repo can re-resolve from the
            # repo instead of treating this as a transient error forever
            # (defect 6).
            with self._lock:
                self._token_cache.pop(cache_key, None)
            raise InstallationGone(
                f"installation {inst} no longer exists (HTTP 404)"
            )
        if status != 201:
            raise GitHubAppError(
                f"installation token for {inst} failed: HTTP {status}"
            )
        data = json.loads(raw)
        token = str(data["token"])
        expires = _parse_iso(str(data["expires_at"]))
        with self._lock:
            self._token_cache[cache_key] = (token, expires)
        return token

    def token_for_repo(
        self,
        repo: str,
        installation_id: int | None = None,
        *,
        now: float | None = None,
        permissions: dict | None = None,
    ) -> str:
        """The installation token to use when acting on `repo`.

        `installation_id`, when known (the webhook payload carries
        installation.id), is used directly; otherwise it is resolved from the
        repo. `permissions` scopes the token (default: full installation
        permissions). Raises NoInstallation when the repo has no installation.

        Defect 6: when the stored installation id 404s on token mint (the App
        was uninstalled), the cached mapping is dropped, the repo is
        re-resolved with installation_for_repo, and if that also 404s the call
        fails closed with NoInstallation (terminal) — never an infinite
        transient retry.
        """
        if installation_id is None:
            installation_id = self.installation_for_repo(repo, now=now)
        try:
            return self.installation_token(installation_id, now=now,
                                           permissions=permissions)
        except InstallationGone:
            with self._lock:
                self._repo_cache.pop(repo, None)
            try:
                fresh = self.installation_for_repo(repo, now=now)
            except NoInstallation:
                raise NoInstallation(
                    f"repo {repo} has no GitHub App installation (stale "
                    f"installation {installation_id} also 404s)"
                ) from None
            return self.installation_token(fresh, now=now, permissions=permissions)

    def list_installations(self, *, per_page: int = 100) -> list[int]:
        """Every GitHub App installation id (for the repo allowlist / scope).

        Paged: GET /app/installations?per_page=100, following the Link
        header's rel='next' until exhausted (defect 6: any installation/repo
        listing must page).  Raises GitHubAppError on failure.
        """
        out: list[int] = []
        path = f"/app/installations?per_page={per_page}"
        for _ in range(100):  # hard bound — never loop on a malformed header
            status, raw, resp_headers = self._api_hdrs("GET", path)
            if status != 200:
                raise GitHubAppError(
                    f"list installations failed: HTTP {status}"
                )
            try:
                data = json.loads(raw)
                out.extend(int(d["id"]) for d in (data or []))
            except (TypeError, ValueError, KeyError) as exc:
                raise GitHubAppError(
                    f"list installations malformed: {exc}"
                ) from exc
            # Header names are case-insensitive (HTTP/2 sends `link`).
            link = next((v for k, v in (resp_headers or {}).items()
                         if k.lower() == "link"), "")
            nxt = _next_link(link)
            if not nxt:
                break
            # Link headers carry ABSOLUTE URLs; keep only the path for _api
            # (which prefixes API_BASE itself).
            parts = urllib.parse.urlsplit(nxt)
            if parts.netloc:
                nxt = urllib.parse.urlunsplit(("", "", parts.path, parts.query, ""))
            path = nxt
        return out

    # ------------------------------------------------------------------
    # Internal: HTTPS
    # ------------------------------------------------------------------

    def _api_hdrs(
        self, method: str, path: str, *, headers: dict | None = None, body=None
    ) -> tuple[int, bytes, dict]:
        """One authenticated call to api.github.com, also returning the response
        headers (for Link-header pagination).

        `body` is JSON-encoded when not already bytes. Returns (status, raw
        body, headers). 4xx/5xx are returned, not raised, so callers can
        distinguish e.g. a 404 (no installation) from a transport failure.
        """
        hdrs = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.jwt()}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if headers:
            hdrs.update(headers)
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        try:
            result = self._http(method, path, hdrs, body)
        except GitHubAppError:
            raise
        except Exception as exc:  # noqa: BLE001 — network layer, fail closed
            raise GitHubAppError(
                f"github api {method} {path} failed: {type(exc).__name__}"
            ) from exc
        if len(result) == 3:
            status, raw, resp_headers = result
        else:
            status, raw = result
            resp_headers = {}
        return status, raw, resp_headers

    def _api(
        self, method: str, path: str, *, headers: dict | None = None, body=None
    ) -> tuple[int, bytes]:
        """One authenticated call to api.github.com (headers discarded)."""
        status, raw, _ = self._api_hdrs(method, path, headers=headers, body=body)
        return status, raw

    def _default_http(
        self, method: str, path: str, headers: dict, body: bytes | None
    ) -> tuple[int, bytes, dict]:
        url = API_BASE + path
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self._http_timeout) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers)


# ---------------------------------------------------------------------------
# Config glue — one per-repo token resolver shared by the worker and builder.
# ---------------------------------------------------------------------------

_GITHUB_ATTR = "_rhobear_github_app"


def github_app_for(cfg) -> GitHubApp | None:
    """The GitHubApp for `cfg` (built once and cached on the cfg instance).

    Returns None when the App identity is not configured. Production refuses
    to serve without it (see __main__ / neo_builder.main), so callers treat a
    None here as "no identity — cannot act on any repo".
    """
    existing = getattr(cfg, _GITHUB_ATTR, None)
    if existing is not None:
        return existing
    try:
        app = GitHubApp.from_config(cfg)
    except AppNotConfigured:
        return None
    setattr(cfg, _GITHUB_ATTR, app)
    return app


def repo_token(cfg, repo: str, installation_id: int | None = None,
               permissions: dict | None = None) -> str:
    """The installation token to use when acting on `repo`.

    Every GitHub action for a repo uses THAT repo's installation token
    (minted by the GitHub App).  There is NO fallback to a personal token:
    a repo with no installation raises NoInstallation (terminal).  A cfg
    with no App identity raises AppNotConfigured.

    `permissions` scopes the token (e.g. {"contents": "read",
    "pull_requests": "read"} for a read-only token).  The escalation advisor
    uses a read-only token; the merge executor uses the full installation
    permissions.

    Callers that need a token for tests inject a fake GitHubApp (set it on
    the cfg via the `_rhobear_github_app` attribute or a fixture) instead
    of relying on cfg.gh_token — that field is ignored as an identity.
    """
    app = github_app_for(cfg)
    if app is None:
        raise AppNotConfigured(
            f"no GitHub identity for {repo} (GitHub App not configured; "
            "personal GH_TOKEN is never used as an identity)"
        )
    return app.token_for_repo(repo, installation_id=installation_id,
                              permissions=permissions)
