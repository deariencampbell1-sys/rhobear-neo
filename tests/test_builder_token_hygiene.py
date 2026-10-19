"""The GitHub token must never reach a command line, a log line or .git/config."""
import logging
import subprocess
from types import SimpleNamespace

from src import neo_builder

TOKEN = "ghs_SECRETtoken123"


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
    monkeypatch.setattr(neo_builder, "_mark", lambda *a, **k: None)
    monkeypatch.setattr(neo_builder, "_db", lambda: None)
    from unittest.mock import MagicMock as MockMagic
    jobs_mock = MockMagic()
    row = {"id": 1, "repo": "acme/app", "pr_number": 7, "head_sha": "a" * 40, "builder_round": 1, "attempt": 1}
    with caplog.at_level(logging.ERROR):
        neo_builder.dispatch_one(SimpleNamespace(gh_token=TOKEN, max_builder_rounds=3), jobs_mock, row)
    assert "worktree prep failed" in caplog.text
    assert TOKEN not in caplog.text
