"""Shared fixtures.

There is deliberately no autouse stub for ``neo_builder.head_moved``: tests
that patch ``prepare_worktree`` to a fake path must opt in explicitly (by
patching ``head_moved`` themselves) so a stale-head regression cannot hide
behind a global monkeypatch.
"""
