"""GitHub App identity for Neo — App JWT + per-installation tokens.

Neo acts as the customer's GitHub App, not as the owner's personal `gho_`
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
plain HTTPS — no new dependency. Caches are guarded by a lock (double-checked
read under the same lock as the write, so a cold-cache fetch happens exactly
once even under concurrency).
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
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


class AppNotConfigured(GitHubAppError):
    """No App identity is configured (no app id / key). Fail closed."""


def _parse_iso(iso: str) -> float:
    """ISO-8601 (GitHub returns e.g. '2026-10-19T12:34:56Z') -> epoch float."""
    s = iso.strip()
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


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

    Thread-safe: caches are read and written under `_lock`, and the cold-cache
    fetch happens inside the lock so exactly one HTTP call mints a token even
    when several workers hit the same installation at once.
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
        404 for that). Cached with a short TTL.
        """
        now = self._time() if now is None else now
        with self._lock:
            cached = self._repo_cache.get(repo)
            if cached and now - cached[1] < self._repo_ttl:
                return cached[0]
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
            self._repo_cache[repo] = (installation_id, now)
            return installation_id

    def installation_token(
        self, installation_id: int, *, now: float | None = None
    ) -> str:
        """A 1-hour installation token for `installation_id`.

        Cached and refreshed 5 minutes before expiry. Thread-safe (the fetch
        happens under the lock so a cold cache mints exactly one token).
        """
        now = self._time() if now is None else now
        inst = int(installation_id)
        with self._lock:
            cached = self._token_cache.get(inst)
            if cached and cached[1] - now > self._token_refresh_lead:
                return cached[0]
            status, raw = self._api(
                "POST",
                f"/app/installations/{inst}/access_tokens",
                body={},
            )
            if status != 201:
                raise GitHubAppError(
                    f"installation token for {inst} failed: HTTP {status}"
                )
            data = json.loads(raw)
            token = str(data["token"])
            expires = _parse_iso(str(data["expires_at"]))
            self._token_cache[inst] = (token, expires)
            return token

    def token_for_repo(
        self,
        repo: str,
        installation_id: int | None = None,
        *,
        now: float | None = None,
    ) -> str:
        """The installation token to use when acting on `repo`.

        `installation_id`, when known (the webhook payload carries
        installation.id), is used directly; otherwise it is resolved from the
        repo. Raises NoInstallation when the repo has no installation.
        """
        if installation_id is None:
            installation_id = self.installation_for_repo(repo, now=now)
        return self.installation_token(installation_id, now=now)

    # ------------------------------------------------------------------
    # Internal: HTTPS
    # ------------------------------------------------------------------

    def _api(
        self, method: str, path: str, *, headers: dict | None = None, body=None
    ) -> tuple[int, bytes]:
        """One authenticated call to api.github.com.

        `body` is JSON-encoded when not already bytes. Returns (status, raw
        body). 4xx/5xx are returned, not raised, so callers can distinguish
        e.g. a 404 (no installation) from a transport failure.
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
        try:
            return self._http(method, path, hdrs, body)
        except GitHubAppError:
            raise
        except Exception as exc:  # noqa: BLE001 — network layer, fail closed
            raise GitHubAppError(
                f"github api {method} {path} failed: {type(exc).__name__}"
            ) from exc

    def _default_http(
        self, method: str, path: str, headers: dict, body: bytes | None
    ) -> tuple[int, bytes]:
        url = API_BASE + path
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self._http_timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()


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


def repo_token(cfg, repo: str, installation_id: int | None = None) -> str:
    """The installation token to use when acting on `repo`.

    Uses the GitHub App when configured; otherwise falls back to
    `cfg.gh_token`. The fallback exists ONLY for the pre-App unit tests that
    pass a placeholder token via cfg.gh_token and assert the token-propagation
    mechanism — production always configures the App (startup refuses to serve
    without it), so a personal token can never be used as an identity here.

    Raises NoInstallation (repo not installed) or AppNotConfigured (no
    identity at all) — both fail closed, never a personal-token fallback in
    production.
    """
    app = github_app_for(cfg)
    if app is not None:
        return app.token_for_repo(repo, installation_id=installation_id)
    token = str(getattr(cfg, "gh_token", "") or "").strip()
    if not token:
        raise AppNotConfigured(
            f"no GitHub identity for {repo} (GitHub App not configured)"
        )
    return token
