"""A failed triage must not strand a PR at its head forever."""
from unittest import mock
from src import neo_worker


class FakeState:
    """In-memory NeoState stand-in with the same color-aware dedup contract.

    rows: (repo, pr, sha, phase, green, verdict); green None == legacy row. The
    real SQL treats a missing detail->>'green' as false (red), so legacy rows
    only block red wakes -- a first green wake still re-triggers.
    """
    def __init__(self):
        self.rows = []
    def claim_action(self, repo, pr, sha, phase, *, green=None, detail=None):
        """In-memory simulation of NeoState.claim_action(). Returns True if not already acted."""
        if green is None:
            for r in self.rows:
                if r[:4] == (repo, pr, sha, phase):
                    return False
        else:
            for r in self.rows:
                if r[:4] != (repo, pr, sha, phase):
                    continue
                if r[4] is None:
                    if green is False:
                        return False
                    continue
                if bool(r[4]) == bool(green):
                    return False
        # Not found: claim the slot.
        self.rows.append((repo, pr, sha, phase, green, ""))
        return True
    def already_acted(self, repo, pr, sha, phase, green=None):
        for r in self.rows:
            if r[:4] != (repo, pr, sha, phase):
                continue
            if green is None:
                return True
            if r[4] is None:
                if green is False:
                    return True
                continue
            if bool(r[4]) == bool(green):
                return True
        return False
    def builder_rounds(self, repo, pr):
        return 0
    def install(self, *a):
        return {}
    def record(self, repo, pr, sha, phase, **kw):
        detail = kw.get("detail") or {}
        self.rows.append((repo, pr, sha, phase, detail.get("green"),
                          kw.get("verdict", "")))
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


# ===================================================================
# Verdict-color redelivery -- a green flip on a red-triaged head must re-run
# ===================================================================

def _run(st, wake, verdict="ACCEPT-READY"):
    with mock.patch.object(neo_worker, "resolve_pr", return_value=7), \
         mock.patch.object(neo_worker.ClaudeAgent, "from_config"), \
         mock.patch.object(neo_worker, "_run_agent", return_value=({}, verdict)) as m:
        neo_worker.run_neo(_cfg(), st, wake)
    return m


def test_green_flip_after_red_triage_is_not_skipped():
    """The green-stall bug: reviewer passes the same head Neo already triaged
    red. The old guard skipped forever; it must triage again."""
    st = FakeState()
    st.record("o/r", 7, "a" * 40, "triage", detail={"green": False})
    m = _run(st, _wake())  # _wake() green=True
    assert m.called, "green verdict after a red triage must re-run the agent"
    assert any(r[:4] == ("o/r", 7, "a" * 40, "triage") and r[4] is True
               for r in st.rows), "green triage must be recorded"


def test_same_color_red_wake_is_still_idempotent():
    """A red wake re-delivered for a red-triaged head stays skipped."""
    st = FakeState()
    st.record("o/r", 7, "a" * 40, "triage", detail={"green": False})
    m = _run(st, {**_wake(), "green": False})
    assert not m.called, "same-color red redelivery must stay idempotent"


def test_legacy_row_without_color_counts_as_red():
    """Rows written before the color field must not block a first green wake."""
    st = FakeState()
    st.record("o/r", 7, "a" * 40, "triage", detail={})
    m = _run(st, _wake())
    assert m.called, "legacy colorless row must not swallow a green verdict"


def test_red_flip_after_green_triage_re_runs():
    """The inverse: reviewer turns red after a green triage (regression)."""
    st = FakeState()
    st.record("o/r", 7, "a" * 40, "triage", detail={"green": True})
    m = _run(st, {**_wake(), "green": False}, verdict="BOUNCE-BUILDER")
    assert m.called, "red verdict after a green triage must re-run the agent"


def test_already_merged_head_never_re_runs():
    """Even a color flip must not re-open a head Neo already merged."""
    st = FakeState()
    st.record("o/r", 7, "a" * 40, "merged", detail={"green": True})
    m = _run(st, _wake())
    assert not m.called, "a merged head must never be triaged again"
