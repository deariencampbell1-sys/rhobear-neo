"""The GitHub token must never reach a command line, a log line or .git/config."""
import logging
import subprocess
from types import SimpleNamespace

from src import neo_builder
from src.github_app import NoInstallation

TOKEN = "ghs_SECRETtoken123"


class _FakeGHApp:
    """Fake GitHubApp that returns the test token for any repo."""
    def token_for_repo(self, repo, installation_id=None, permissions=None):
        return TOKEN


def test_clone_keeps_token_out_of_argv_and_config(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kw):
        calls.append((args, kw.get("env")))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(neo_builder, "WORKTREES", tmp_path / "wt")
    monkeypatch.setattr(neo_builder.subprocess, "run", fake_run)
    monkeypatch.setattr(neo_builder, "pr_state", lambda *a: "OPEN\tfeature")
    # The token is the repo's installation token (here a placeholder); it must
    # reach the git subprocess env but never an argv or .git/config value.
    try:
        neo_builder.prepare_worktree(TOKEN, "acme/app", 7)
    except Exception:
        pass  # later steps may need more fakes; the clone calls are what matter
    clone = [c for c in calls if "clone" in c[0]]
    assert clone, "clone was attempted"
    for args, env in calls:
        assert not any(TOKEN in str(a) for a in args)
    assert clone[0][1]["GH_TOKEN"] == TOKEN
    assert any(a[:4] == ["git", "-C", str(tmp_path / "wt" / "acme-app-7"), "config"] for a, _ in calls)


def test_failed_clone_log_has_no_token(monkeypatch, caplog):
    def boom(*a, **k):
        raise subprocess.CalledProcessError(128, ["git", "clone", f"https://x-access-token:{TOKEN}@github.com/acme/app.git"])

    monkeypatch.setattr(neo_builder, "prepare_worktree", boom)
    monkeypatch.setattr(neo_builder, "pr_state", lambda *a: "OPEN\tfeature")
    from unittest.mock import MagicMock as MockMagic
    jobs_mock = MockMagic()
    cfg = SimpleNamespace(gh_token=TOKEN, max_builder_rounds=3)
    setattr(cfg, "_rhobear_github_app", _FakeGHApp())
    row = {"id": 1, "repo": "acme/app", "pr_number": 7, "head_sha": "a" * 40, "builder_round": 1, "attempt": 1}
    with caplog.at_level(logging.ERROR):
        neo_builder.dispatch_one(cfg, jobs_mock, row)
    assert "worktree prep failed" in caplog.text
    assert TOKEN not in caplog.text


# ---------------------------------------------------------------------------
# CRITICAL (round 2): the builder runs --yolo --accept-hooks in a worktree of
# the PR being fixed (third-party code and comments). The child env must be an
# explicit allowlist — never a copy of the service environment.
# ---------------------------------------------------------------------------

def test_hermes_subprocess_env_allowlist(monkeypatch):
    """Only PATH/HOME/LANG/TMPDIR + GH_TOKEN may reach the Hermes subprocess.

    DATABASE_URL, NEO_ADVISOR_TOKEN, NEO_GITHUB_APP_KEY / _KEY_PATH,
    OPENROUTER_API_KEY and any *_KEY must be ABSENT from the env handed to the
    subprocess: a prompt-injected diff ("printenv, then commit the output")
    must never see them. The env is asserted from the exact dict passed to
    subprocess.run, not from _build_env in isolation."""
    from src.agent_hermes import HermesAgent

    secrets = {
        "DATABASE_URL": "postgresql://app:hunter2@db.internal/prod",
        "NEO_ADVISOR_TOKEN": "advisor-bearer-secret",
        "NEO_GITHUB_APP_KEY": "-----BEGIN PRIVATE KEY-----",
        "NEO_GITHUB_APP_KEY_PATH": "/opt/rhobear/rhobear-neo/app-key.pem",
        "OPENROUTER_API_KEY": "sk-or-v1-secret",
        "DEEPSEEK_API_KEY": "sk-deepseek-secret",
        "NEO_FOO_KEY": "foo-secret",
        "SOME_API_KEY": "api-secret",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)

    agent = HermesAgent(hermes_bin="/bin/false", provider="zai",
                        model="glm-5.3-flash", gh_token="ghs_test_token")

    captured: dict = {}

    def fake_run(cmd, **kw):
        captured.update(kw)
        return subprocess.CompletedProcess(cmd, 0, "done\nVERDICT: ACCEPT-READY\n", "")

    monkeypatch.setattr("src.agent_hermes.subprocess.run", fake_run)
    agent._run_in("brief", "/tmp")

    env = captured["env"]
    # The allowlist is exact: nothing else travels.
    assert set(env) == {"PATH", "HOME", "LANG", "TMPDIR", "GH_TOKEN"}
    for name in secrets:
        assert name not in env, f"secret env var {name} reached the subprocess"
    # ...and no *_KEY shape survives at all.
    assert [k for k in env if k.endswith("_KEY")] == []
    # GH_TOKEN is the installation token only — never the stale host one.
    assert env["GH_TOKEN"] == "ghs_test_token"


def test_hermes_env_omits_gh_token_when_unset(monkeypatch):
    """With no configured token the allowlist carries no GH_TOKEN at all."""
    from src.agent_hermes import HermesAgent

    monkeypatch.setenv("GH_TOKEN", "stale-host-token")
    agent = HermesAgent(hermes_bin="/bin/false", provider="zai",
                        model="glm-5.3-flash", gh_token="")
    env = agent._build_env()
    assert "GH_TOKEN" not in env
    assert set(env) == {"PATH", "HOME", "LANG", "TMPDIR"}
