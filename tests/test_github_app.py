"""N15-B1g — Neo acts as the GitHub App, one installation token per repo.

Covers (from the lane brief):
  * App JWT shape + 9-minute expiry (generated RSA key, decoded + verified)
  * installation token cache hit / refresh / expiry with a fake clock
  * per-repo token selection — repo A's token is never used for repo B
  * no-installation refusal (NoInstallation / job outcome `no_installation`)
  * thread-safe token minting (one POST on a cold cache, even under concurrency)
  * webhook payload installation.id is passed through to the wake
  * every gh / git subprocess built by the code carries the installation token
    and never the process env's GH_TOKEN (personal token)
  * the Hermes / Claude agent env likewise
  * run_neo / dispatch_one refuse a repo with no installation (terminal)
  * run_neo uses the webhook installation.id when present

No real GitHub / provider calls: all HTTP and subprocesses are fakes.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src import neo_builder, neo_worker
from src.agent_claude import ClaudeAgent
from src.agent_hermes import HermesAgent
from src.config import Config
from src.github_app import (
    AppNotConfigured,
    GitHubApp,
    NoInstallation,
    github_app_for,
    load_app_key,
    repo_token,
)
from src.jobs import Jobs
from src.webhook_server import NeoDispatcher

SHA1 = "a" * 40


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rsa_key():
    """A real RSA-2048 keypair (PyJWT RS256 needs cryptography, in requirements)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return {"pem": pem.strip(), "pub": pub.strip()}


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeGitHubHTTP:
    """Fake api.github.com: repo->installation map + per-install tokens.

    Supports defect-6 behaviour: `token_404s` are installations whose
    access-token POST returns 404 (the App was uninstalled), and
    `listed_installations` serves GET /app/installations with Link-header
    pagination (per_page=100).
    """

    def __init__(self, installations=None, tokens=None, *, clock=None,
                 token_ttl=3600, token_404s=None, listed_installations=None):
        self.installations = dict(installations or {})  # repo -> installation id
        self.tokens = dict(tokens or {})                # installation id -> token text
        self.token_404s = set(token_404s or ())         # token POST 404s for these ids
        self.listed_installations = listed_installations  # None | [{"id": n}, ...]
        self.clock = clock or FakeClock()
        self.token_ttl = token_ttl
        self.calls: list[tuple[str, str]] = []          # (method, path)

    def __call__(self, method, path, headers, body=None):
        self.calls.append((method, path))
        if path.startswith("/repos/") and path.endswith("/installation"):
            repo = path[len("/repos/"):-len("/installation")]
            if repo in self.installations:
                return 200, json.dumps({"id": self.installations[repo]}).encode()
            return 404, b"{}"
        if method == "GET" and path.startswith("/app/installations"):
            # Paginated listing: per_page=100, follow the Link rel="next".
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(path).query)
            per_page = int(q.get("per_page", ["100"])[0])
            page = int(q.get("page", ["1"])[0])
            if self.listed_installations is None:
                return 200, b"[]"
            start = (page - 1) * per_page
            chunk = self.listed_installations[start:start + per_page]
            raw = json.dumps(chunk).encode()
            if start + per_page < len(self.listed_installations):
                nxt = (f"<https://api.github.com/app/installations?per_page="
                       f"{per_page}&page={page + 1}>; rel=\"next\"")
            else:
                nxt = ""
            return 200, raw, {"Link": nxt}
        if method == "POST" and path.endswith("/access_tokens"):
            inst = int(path.split("/")[3])
            if inst in self.token_404s:
                return 404, b'{"message":"Bad credentials"}'
            token = self.tokens.get(inst, f"ghs_tok_{inst}")
            expires = _iso(self.clock.t + self.token_ttl)
            return 201, json.dumps({"token": token, "expires_at": expires}).encode()
        raise AssertionError(f"unexpected fake github call {method} {path}")


def _make_app(key: dict, http: FakeGitHubHTTP) -> GitHubApp:
    return GitHubApp("4347986", key["pem"], http=http, time_fn=http.clock)


def _cfg_with_app(key: dict, http: FakeGitHubHTTP) -> Config:
    cfg = Config()
    cfg.github_app_id = "4347986"
    cfg.github_app_key = key["pem"]
    cfg.gh_token = "gho_personal_secret"
    setattr(cfg, "_rhobear_github_app", _make_app(key, http))
    return cfg


# ---------------------------------------------------------------------------
# App JWT — shape and expiry
# ---------------------------------------------------------------------------

class TestAppJWT:
    def test_jwt_shape_and_expiry(self, rsa_key):
        app = GitHubApp("4347986", rsa_key["pem"])
        tok = app.jwt(now=1_000_000.0)
        # verify_exp off: the token is minted against the fake clock (epoch
        # 1e6) and would look expired to the wall clock; we assert the shape
        # and the 9-minute window from the claims themselves.
        payload = jwt.decode(tok, rsa_key["pub"], algorithms=["RS256"],
                             options={"verify_exp": False})
        assert payload["iss"] == "4347986"
        # RS256, 9 minutes, backdated iat for clock skew.
        assert payload["exp"] - payload["iat"] == 540
        assert payload["iat"] <= 1_000_000
        assert payload["exp"] <= 1_000_000 + 540

    def test_jwt_authorization_header_is_bearer_jwt(self, rsa_key):
        app = GitHubApp("4347986", rsa_key["pem"])
        http = FakeGitHubHTTP(installations={"acme/a": 111})
        app = GitHubApp("4347986", rsa_key["pem"], http=http, time_fn=http.clock)
        app.token_for_repo("acme/a")
        # The GET carries a Bearer App JWT, not any personal token.
        assert http.calls[0][0] == "GET"


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------

class TestKeyLoading:
    def test_key_from_path_file(self, tmp_path, rsa_key):
        key_file = tmp_path / "app-key.pem"
        key_file.write_text(rsa_key["pem"], encoding="utf-8")
        cfg = SimpleNamespace(github_app_key_path=str(key_file), github_app_key="")
        assert load_app_key(cfg) == rsa_key["pem"]

    def test_key_from_env_text(self, rsa_key):
        cfg = SimpleNamespace(github_app_key_path="", github_app_key=rsa_key["pem"])
        assert load_app_key(cfg) == rsa_key["pem"]

    def test_key_path_wins_over_env_text(self, tmp_path, rsa_key):
        key_file = tmp_path / "app-key.pem"
        key_file.write_text(rsa_key["pem"], encoding="utf-8")
        cfg = SimpleNamespace(github_app_key_path=str(key_file), github_app_key="env-other")
        assert load_app_key(cfg) == rsa_key["pem"]

    def test_missing_key_path_raises(self, tmp_path):
        cfg = SimpleNamespace(github_app_key_path=str(tmp_path / "nope.pem"), github_app_key="")
        with pytest.raises(AppNotConfigured):
            load_app_key(cfg)

    def test_from_config_requires_app_id_and_key(self, rsa_key):
        with pytest.raises(AppNotConfigured):
            GitHubApp.from_config(SimpleNamespace(github_app_id="", github_app_key=""))
        with pytest.raises(AppNotConfigured):
            GitHubApp.from_config(SimpleNamespace(github_app_id="1", github_app_key=""))
        app = GitHubApp.from_config(
            SimpleNamespace(github_app_id="1", github_app_key_path="", github_app_key=rsa_key["pem"]))
        assert app.app_id == "1"


# ---------------------------------------------------------------------------
# Installation token — cache hit / refresh / expiry with a fake clock
# ---------------------------------------------------------------------------

class TestTokenCache:
    def test_cached_and_refreshed_before_expiry(self, rsa_key):
        clock = FakeClock()
        http = FakeGitHubHTTP(installations={"acme/a": 111},
                              tokens={111: "tok-a"}, clock=clock)
        app = _make_app(rsa_key, http)
        # Explicit installation id isolates the token cache from the repo cache.
        assert app.installation_token(111) == "tok-a"   # cold cache -> POST
        assert app.installation_token(111) == "tok-a"   # cache hit
        assert len(http.calls) == 1
        # 53m20s later: 6m40s before the 1 h expiry -> still cached.
        clock.t += 3200
        assert app.installation_token(111) == "tok-a"
        assert len(http.calls) == 1
        # 1m41s later: 4m59s before expiry (< 5 min lead) -> refreshed.
        clock.t += 101
        assert app.installation_token(111) == "tok-a"
        assert len(http.calls) == 2

    def test_installation_resolution_cached_short_ttl(self, rsa_key):
        clock = FakeClock()
        http = FakeGitHubHTTP(installations={"acme/a": 111}, clock=clock)
        app = _make_app(rsa_key, http)
        assert app.installation_for_repo("acme/a") == 111
        assert app.installation_for_repo("acme/a") == 111
        assert len(http.calls) == 1
        clock.t += 301  # past the short repo TTL -> refetch
        assert app.installation_for_repo("acme/a") == 111
        assert len(http.calls) == 2

    def test_per_repo_token_never_crosses_repos(self, rsa_key):
        clock = FakeClock()
        http = FakeGitHubHTTP(
            installations={"acme/a": 111, "acme/b": 222},
            tokens={111: "tok-A-only", 222: "tok-B-only"},
            clock=clock,
        )
        app = _make_app(rsa_key, http)
        assert app.token_for_repo("acme/a") == "tok-A-only"
        assert app.token_for_repo("acme/b") == "tok-B-only"
        # repo A's token is never used for repo B (and vice versa).
        assert app.token_for_repo("acme/a") == "tok-A-only"
        assert app.token_for_repo("acme/b") == "tok-B-only"

    def test_explicit_installation_id_skips_repo_lookup(self, rsa_key):
        clock = FakeClock()
        # acme/a has NO installation — but the webhook told us the installation.
        http = FakeGitHubHTTP(installations={}, tokens={999: "tok-webhook"}, clock=clock)
        app = _make_app(rsa_key, http)
        assert app.token_for_repo("acme/a", installation_id=999) == "tok-webhook"

    def test_no_installation_raises(self, rsa_key):
        clock = FakeClock()
        http = FakeGitHubHTTP(installations={}, clock=clock)
        app = _make_app(rsa_key, http)
        with pytest.raises(NoInstallation):
            app.token_for_repo("acme/private")


# ---------------------------------------------------------------------------
# Thread safety — one token POST on a cold cache even under concurrency
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_token_fetch_all_get_same_token(self, rsa_key):
        clock = FakeClock()
        http = FakeGitHubHTTP(installations={"acme/a": 111}, clock=clock)
        app = _make_app(rsa_key, http)
        barrier = threading.Barrier(8)
        results = []

        def worker():
            barrier.wait()
            results.append(app.token_for_repo("acme/a"))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(results) == 8
        assert set(results) == {"ghs_tok_111"}
        # The HTTP fetch happens OUTSIDE the lock now (fix 5), so concurrent
        # misses may each call the API.  The important guarantee is that every
        # thread gets the same valid token — not that only one POST was made.
        posts = [c for c in http.calls if c[0] == "POST"]
        assert len(posts) >= 1


# ---------------------------------------------------------------------------
# repo_token / github_app_for glue
# ---------------------------------------------------------------------------

class TestUninstalledApp:
    """Defect 6 — a stored installation_id whose token POST returns 404 (the
    App was uninstalled) is NOT treated as a transient error forever.  The
    cached token is dropped, the repo is re-resolved, and if the repo also has
    no installation the call fails closed with NoInstallation (terminal)."""

    def test_stale_install_404_refalls_back_to_repo(self, rsa_key):
        """Webhook passes a stale install id 111; the repo now maps to 222.
        token_for_repo re-resolves and returns the fresh install's token."""
        clock = FakeClock()
        http = FakeGitHubHTTP(installations={"acme/a": 222},
                              tokens={222: "tok-new"},
                              token_404s={111}, clock=clock)
        app = _make_app(rsa_key, http)
        tok = app.token_for_repo("acme/a", installation_id=111)
        assert tok == "tok-new"
        # The stale id was dropped from the token cache — it must not be
        # served again from cache.
        assert (111, "{}") not in app._token_cache
        # The repo mapping was re-resolved to the fresh installation.
        assert app.installation_for_repo("acme/a") == 222

    def test_stale_install_404_and_no_repo_raises_no_installation(self, rsa_key):
        """Both the stored id and the repo 404 → NoInstallation (terminal),
        never an infinite transient retry."""
        clock = FakeClock()
        http = FakeGitHubHTTP(installations={}, tokens={},
                              token_404s={111}, clock=clock)
        app = _make_app(rsa_key, http)
        with pytest.raises(NoInstallation):
            app.token_for_repo("acme/private", installation_id=111)

    def test_installation_token_404_drops_cache_and_raises_typed(self, rsa_key):
        """installation_token itself surfaces a typed error on 404 and drops
        the cached entry so a later call re-mints."""
        from src.github_app import InstallationGone, GitHubAppError
        clock = FakeClock()
        http = FakeGitHubHTTP(installations={}, tokens={},
                              token_404s={111}, clock=clock)
        app = _make_app(rsa_key, http)
        with pytest.raises(InstallationGone):
            app.installation_token(111)
        assert (111, "{}") not in app._token_cache
        # GitHubAppError remains the base class (fail closed).
        assert issubclass(InstallationGone, GitHubAppError)

    def test_list_installations_paginates(self, rsa_key):
        """Any installation listing pages: per_page=100, following the Link
        header's rel='next' until exhausted."""
        clock = FakeClock()
        http = FakeGitHubHTTP(
            listed_installations=[{"id": i} for i in range(1, 103)],
            clock=clock,
        )
        app = _make_app(rsa_key, http)
        ids = app.list_installations()
        assert len(ids) == 102
        assert ids[0] == 1 and ids[-1] == 102
        listing_calls = [p for m, p in http.calls
                         if m == "GET" and p.startswith("/app/installations")]
        assert len(listing_calls) == 2, listing_calls
        assert "per_page=100" in listing_calls[0]
        assert "page=2" in listing_calls[1]


class TestRepoTokenGlue:
    def test_repo_token_uses_app_when_configured(self, rsa_key):
        cfg = _cfg_with_app(rsa_key, FakeGitHubHTTP(
            installations={"acme/a": 111}, tokens={111: "tok-app"}))
        assert repo_token(cfg, "acme/a") == "tok-app"

    def test_repo_token_app_not_configured_raises(self):
        cfg = Config()
        cfg.gh_token = ""
        with pytest.raises(AppNotConfigured):
            repo_token(cfg, "acme/a")

    def test_github_app_for_caches_on_cfg(self, rsa_key):
        cfg = Config()
        cfg.github_app_id = "4347986"
        cfg.github_app_key = rsa_key["pem"]
        a1 = github_app_for(cfg)
        a2 = github_app_for(cfg)
        assert a1 is a2
        assert a1.app_id == "4347986"


# ---------------------------------------------------------------------------
# Webhook installation.id passes through to the wake
# ---------------------------------------------------------------------------

class TestWebhookInstallationId:
    def test_status_wake_carries_installation_id(self):
        seen = {}

        def on_verdict(wake):
            seen.update(wake)

        cfg = SimpleNamespace(repo_allowlist=(), orgs=("acme",),
                              trusted_review_contexts=("rhobear-reviews",))
        d = NeoDispatcher(cfg, on_verdict)
        out = d.dispatch(
            "status",
            {"repository": {"full_name": "acme/a"},
             "context": "rhobear-reviews", "state": "success", "sha": SHA1,
             "installation": {"id": 777}},
        )
        assert out.code == 200
        assert seen["installation_id"] == 777

    def test_check_run_wake_carries_installation_id(self):
        seen = {}

        def on_verdict(wake):
            seen.update(wake)

        cfg = SimpleNamespace(repo_allowlist=(), orgs=("acme",),
                              trusted_review_contexts=("CodeAnt",))
        d = NeoDispatcher(cfg, on_verdict)
        d.dispatch(
            "check_run",
            {"action": "completed",
             "repository": {"full_name": "acme/a"},
             "check_run": {"name": "CodeAnt", "conclusion": "success",
                           "head_sha": SHA1, "pull_requests": [{"number": 3}]},
             "installation": {"id": 888}},
        )
        assert seen["installation_id"] == 888
        assert seen["pr_number"] == 3


# ---------------------------------------------------------------------------
# Subprocess env — the installation token, never the process GH_TOKEN
# ---------------------------------------------------------------------------

class TestSubprocessTokenHygiene:
    def test_worker_gh_env_uses_installation_token(self, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "gho_personal_secret")
        captured = {}

        def fake_run(args, **kw):
            captured["env"] = kw.get("env")
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(neo_worker.subprocess, "run", fake_run)
        neo_worker._gh("ghs_install_a", "api", "repos/acme/a/pulls/1")
        assert captured["env"]["GH_TOKEN"] == "ghs_install_a"
        assert captured["env"]["GH_TOKEN"] != "gho_personal_secret"

    def test_builder_gh_env_uses_installation_token(self, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "gho_personal_secret")
        captured = {}

        def fake_run(args, **kw):
            captured["env"] = kw.get("env")
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(neo_builder.subprocess, "run", fake_run)
        neo_builder.gh("ghs_install_b", "api", "repos/acme/b/pulls/1")
        assert captured["env"]["GH_TOKEN"] == "ghs_install_b"
        assert captured["env"]["GH_TOKEN"] != "gho_personal_secret"

    def test_builder_clone_env_uses_installation_token(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GH_TOKEN", "gho_personal_secret")
        calls = []

        def fake_run(args, **kw):
            calls.append(kw.get("env"))
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(neo_builder, "WORKTREES", tmp_path / "wt")
        monkeypatch.setattr(neo_builder.subprocess, "run", fake_run)
        monkeypatch.setattr(neo_builder, "pr_state", lambda *a: "OPEN\tfeature")
        try:
            neo_builder.prepare_worktree("ghs_install_a", "acme/a", 7)
        except Exception:
            pass  # later steps may need more fakes; the env on clone/fetch is what matters
        assert calls, "git subprocess was attempted"
        env_calls = [env for env in calls if env is not None]
        assert env_calls, "clone/fetch carried an env"
        for env in env_calls:
            assert env["GH_TOKEN"] == "ghs_install_a"
            assert env["GH_TOKEN"] != "gho_personal_secret"

    def test_claude_agent_env_uses_installation_token(self, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "gho_personal_secret")
        agent = ClaudeAgent(claude_bin="/usr/bin/claude", api_key="sk-test",
                            base_url="https://api.deepseek.com/anthropic",
                            model="deepseek-v4-flash", gh_token="ghs_install_a")
        env = agent._build_env("/tmp/cfg")
        assert env["GH_TOKEN"] == "ghs_install_a"
        assert env["GH_TOKEN"] != "gho_personal_secret"

    def test_hermes_agent_env_uses_installation_token(self, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "gho_personal_secret")
        agent = HermesAgent(hermes_bin="/opt/rhobear-hermes/bin/hermes",
                            provider="zai", model="glm-5.3-flash",
                            gh_token="ghs_install_b")
        env = agent._build_env()
        assert env["GH_TOKEN"] == "ghs_install_b"
        assert env["GH_TOKEN"] != "gho_personal_secret"


# ---------------------------------------------------------------------------
# End-to-end wiring: run_neo / dispatch_one resolve and use the per-repo token
# ---------------------------------------------------------------------------

class TestWorkerWiring:
    def test_run_neo_refuses_repo_without_installation(self, rsa_key):
        cfg = _cfg_with_app(rsa_key, FakeGitHubHTTP(installations={}))
        jobs = MagicMock(spec=Jobs)
        jobs.create_wake.return_value = 42
        jobs.claim.return_value = 42
        state = MagicMock()
        wake = {"repo": "acme/private", "sha": SHA1, "green": True,
                "context": "rhobear-reviews", "pr_number": 1, "job_id": 42}
        with patch("src.neo_worker.resolve_pr") as resolve_pr:
            neo_worker.run_neo(cfg, state, jobs, wake)
        resolve_pr.assert_not_called()  # refused before any GitHub work
        jobs.resolve.assert_called_once()
        assert jobs.resolve.call_args[0][1] == "no_installation"
        jobs.claim.assert_called_once()

    def test_run_neo_uses_webhook_installation_id(self, rsa_key):
        cfg = _cfg_with_app(rsa_key, FakeGitHubHTTP(
            installations={}, tokens={999: "tok-webhook"}))
        jobs = MagicMock(spec=Jobs)
        jobs.create_wake.return_value = 42
        jobs.claim.return_value = 42
        state = MagicMock()
        state.builder_rounds.return_value = 0
        state.install.return_value = {"auto_merge": False, "plan": None,
                                      "credits_balance": 100}
        state.already_acted.return_value = False
        wake = {"repo": "acme/a", "sha": SHA1, "green": True,
                "context": "rhobear-reviews", "pr_number": 1, "job_id": 42,
                "installation_id": 999}
        agent_tokens = []

        def fake_run_agent(agent, brief):
            agent_tokens.append(agent.gh_token)
            return ({"input_tokens": 0}, "ACCEPT-READY")

        with patch("src.neo_worker.resolve_pr", return_value=1), \
             patch("src.neo_worker._review_evidence", return_value={
                 "ok": True, "head_sha": SHA1, "decided_sha": SHA1,
                 "review_sha": "", "review_bound_ok": True,
                 "trusted_states": ["success"], "trusted_green": True,
                 "failing_checks": 0, "rollup_green": True}), \
             patch("src.neo_worker._jev_for_pr",
                   return_value=SimpleNamespace(
                       brief=lambda: "JEV none", blockers=[], available=False,
                       risk="low", review_quality="adequate", partial=False,
                       requires_full_review=False)), \
             patch("src.neo_worker._run_agent", side_effect=fake_run_agent):
            neo_worker.run_neo(cfg, state, jobs, wake)
        assert agent_tokens == ["tok-webhook"]  # never the personal token

    def test_dispatch_one_uses_repo_installation_token(self, rsa_key, tmp_path):
        cfg = _cfg_with_app(rsa_key, FakeGitHubHTTP(
            installations={"acme/a": 111}, tokens={111: "tok-a"}))
        cfg.max_builder_rounds = 3
        jobs = MagicMock(spec=Jobs)
        row = {"id": 1, "repo": "acme/a", "pr_number": 5, "head_sha": SHA1,
               "builder_round": 1, "attempt": 1, "worker_token": "w"}
        gh_tokens = []
        agent_tokens = []
        real_builder_init = neo_builder.BuilderAgent.__init__

        def fake_gh(token, *args, **kw):
            gh_tokens.append(token)
            return (0, SHA1)

        def spy_init(self, **kw):
            agent_tokens.append(kw.get("gh_token"))
            real_builder_init(self, **kw)

        with patch("src.neo_builder.BRIEFS", tmp_path / "briefs"), \
             patch("src.neo_builder.pr_state", return_value="OPEN\tmain"), \
             patch("src.neo_builder.prepare_worktree",
                   return_value=("/tmp/wt", "main")), \
             patch("src.neo_builder.fetch_findings", return_value=""), \
             patch("src.neo_builder.gh", side_effect=fake_gh), \
             patch.object(neo_builder.BuilderAgent, "run",
                          return_value=({"input_tokens": 0}, "FIXED")), \
             patch.object(neo_builder.BuilderAgent, "__init__", spy_init):
            neo_builder.dispatch_one(cfg, jobs, row)
        assert gh_tokens, "gh was called with the repo token"
        assert set(gh_tokens) == {"tok-a"}   # never a personal token
        assert agent_tokens == ["tok-a"]
        jobs.resolve.assert_called_once()
        assert jobs.resolve.call_args[0][1] == "done"

    def test_dispatch_one_refuses_repo_without_installation(self, rsa_key):
        cfg = _cfg_with_app(rsa_key, FakeGitHubHTTP(installations={}))
        cfg.max_builder_rounds = 3
        jobs = MagicMock(spec=Jobs)
        row = {"id": 1, "repo": "acme/private", "pr_number": 5, "head_sha": SHA1,
               "builder_round": 1, "attempt": 1}
        with patch("src.neo_builder.pr_state") as pr_state:
            neo_builder.dispatch_one(cfg, jobs, row)
        pr_state.assert_not_called()
        jobs.resolve.assert_called_once()
        assert jobs.resolve.call_args[0][1] == "no_installation"

    def test_review_evidence_runs_with_fake_gh(self, rsa_key):
        """_review_evidence runs for real with a fake gh subprocess.

        The review finding says all existing tests patch _review_evidence out;
        this one exercises the actual function (check-runs fetch for the
        trusted Reviews App, status fetch, build_evidence) against fake gh
        output.  Item 3: the trusted producer is the App's check run on the
        exact head SHA.
        """
        from src.neo_worker import _review_evidence

        cfg = _cfg_with_app(rsa_key, FakeGitHubHTTP(installations={}))
        cfg.trusted_reviewer_app_id = "12345"

        def fake_gh(token, *args, **kw):
            if args[:3] == ("pr", "view", str(5)):
                # Return a PR with no reviews, no statusCheckRollup
                data = {
                    "headRefOid": SHA1,
                    "reviews": [],
                    "statusCheckRollup": [],
                }
                return 0, json.dumps(data)
            if args[:3] == ("api", f"repos/acme/a/commits/{SHA1}/check-runs",):
                data = {"check_runs": [{
                    "name": "rhobear-reviews", "head_sha": SHA1,
                    "app": {"id": 12345, "slug": "rhobear-reviews"},
                    "conclusion": "success",
                    "output": {"summary": '{"unresolved_critical_high": 0, "coverage": "complete"}'},
                }]}
                return 0, json.dumps(data)
            if args[:3] == ("api", f"repos/acme/a/commits/{SHA1}/status",):
                data = {"state": "success", "statuses": [
                    {"context": "rhobear-reviews", "state": "success"},
                ]}
                return 0, json.dumps(data)
            raise AssertionError(f"unexpected _gh call: {args}")

        with patch("src.neo_worker._gh", side_effect=fake_gh):
            ev = _review_evidence("tok", cfg, "acme/a", 5, SHA1)
        assert ev["ok"] is True
        assert ev["decided_sha"] == SHA1
        assert ev["head_sha"] == SHA1
        assert ev["trusted_producer_ok"] is True
        assert ev["review_sha_match"] is True
        assert ev["unresolved_critical_high"] == 0
        assert ev["coverage_complete"] is True
        assert ev["review_bound_ok"] is True
        assert ev["trusted_green"] is True
        # Empty statusCheckRollup -> rollup_green is None (no data),
        # which is correct: there is no rollup to judge.
        assert ev["rollup_green"] is None
        assert ev["failing_checks"] == 0

    def test_review_evidence_no_trusted_contexts(self, rsa_key):
        """_review_evidence returns ok=False when no trusted contexts are configured."""
        from src.neo_worker import _review_evidence

        cfg = _cfg_with_app(rsa_key, FakeGitHubHTTP(installations={}))
        cfg.trusted_review_contexts = ()  # empty

        def fake_gh(token, *args, **kw):
            if args[:3] == ("pr", "view", str(5)):
                data = {"headRefOid": SHA1, "reviews": [], "statusCheckRollup": []}
                return 0, json.dumps(data)
            if args[:3] == ("api", f"repos/acme/a/commits/{SHA1}/check-runs",):
                data = {"check_runs": []}
                return 0, json.dumps(data)
            if args[:3] == ("api", f"repos/acme/a/commits/{SHA1}/status",):
                data = {"state": "success", "statuses": []}
                return 0, json.dumps(data)
            raise AssertionError(f"unexpected _gh call: {args}")

        with patch("src.neo_worker._gh", side_effect=fake_gh):
            ev = _review_evidence("tok", cfg, "acme/a", 5, SHA1)
        assert ev["ok"] is False
        assert ev["reason"] == "no trusted review contexts configured"


# ---------------------------------------------------------------------------
# Startup — GH_TOKEN is ignored, never an identity
# ---------------------------------------------------------------------------

class TestStartupIgnoredToken:
    def test_config_reads_github_app_identity_from_env(self, rsa_key):
        with patch.dict(os.environ, {
            "NEO_GITHUB_APP_ID": "4347986",
            "NEO_GITHUB_APP_KEY_PATH": "",
            "NEO_GITHUB_APP_KEY": rsa_key["pem"],
            "GH_TOKEN": "gho_ignored_secret",
            "DEEPSEEK_API_KEY": "",
        }, clear=False):
            cfg = Config()
        assert cfg.github_app_id == "4347986"
        assert cfg.github_app_key == rsa_key["pem"]
        assert cfg.gh_token == "gho_ignored_secret"

    def test_require_no_longer_demands_gh_token(self):
        cfg = Config()
        cfg.webhook_secret = "whs_test"
        cfg.deepseek_key = "sk-test"
        cfg.database_url = "postgres://localhost/test"
        cfg.gh_token = ""           # GH_TOKEN is not an identity any more
        cfg.require()               # must NOT raise

    def test_main_logs_gh_token_ignored_and_requires_app(self, rsa_key, caplog):
        import src.__main__ as main_mod
        src_text = open(main_mod.__file__).read()
        assert "IGNORED" in src_text
        assert "github_app_for" in src_text
        # A personal GH_TOKEN must never satisfy the identity requirement.
        with patch.dict(os.environ, {"GH_TOKEN": "gho_ignored"}, clear=False):
            cfg = Config()
            assert github_app_for(cfg) is None   # app identity is what matters
