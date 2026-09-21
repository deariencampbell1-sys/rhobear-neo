"""A failed triage must not strand a PR at its head forever."""
from unittest import mock
from src import neo_worker


class FakeState:
    def __init__(self):
        self.rows = []
    def already_acted(self, repo, pr, sha, phase):
        return any(r[:4] == (repo, pr, sha, phase) for r in self.rows)
    def builder_rounds(self, repo, pr):
        return 0
    def install(self, *a):
        return {}
    def record(self, repo, pr, sha, phase, **kw):
        self.rows.append((repo, pr, sha, phase, kw.get("verdict", "")))
    def clear(self, repo, pr, sha, phase):
        before = len(self.rows)
        self.rows = [r for r in self.rows if r[:4] != (repo, pr, sha, phase)]
        return before - len(self.rows)
    def debit(self, *a):
        pass


def _cfg():
    return mock.Mock(auto_merge_default=True, max_builder_rounds=4, deepseek_model="m")


def _wake():
    return {"repo": "o/r", "sha": "a" * 40, "pr_number": 7, "context": "rhobear-reviews", "green": True}


def test_agent_failure_clears_triage_marker_so_next_wake_retries():
    st = FakeState()
    with mock.patch.object(neo_worker, "resolve_pr", return_value=7), \
         mock.patch.object(neo_worker.ClaudeAgent, "from_config"), \
         mock.patch.object(neo_worker, "_run_agent", return_value=({}, "")):
        neo_worker.run_neo(_cfg(), st, _wake())
    assert not st.already_acted("o/r", 7, "a" * 40, "triage"), \
        "a failed agent must not leave the PR marked as triaged"


def test_successful_triage_still_idempotent():
    st = FakeState()
    with mock.patch.object(neo_worker, "resolve_pr", return_value=7), \
         mock.patch.object(neo_worker.ClaudeAgent, "from_config"), \
         mock.patch.object(neo_worker, "_run_agent", return_value=({}, "FIX-FORWARD")):
        neo_worker.run_neo(_cfg(), st, _wake())
    assert st.already_acted("o/r", 7, "a" * 40, "triage")
