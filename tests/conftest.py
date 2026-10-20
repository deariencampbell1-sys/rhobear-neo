"""Shared fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _worktree_at_round_sha(request, monkeypatch):
    """Tests patch prepare_worktree to a fake path; treat that worktree as being
    at the round's SHA. Tests of the stale-head path patch head_moved themselves."""
    if "real_head_moved" in request.keywords:
        return
    try:
        import src.neo_builder as nb
    except Exception:  # noqa: BLE001
        return
    monkeypatch.setattr(nb, "head_moved", lambda worktree, sha: None)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_head_moved: use the real neo_builder.head_moved")
