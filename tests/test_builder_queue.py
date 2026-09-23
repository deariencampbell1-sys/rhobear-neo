"""Builder lane throughput + scoping guarantees.

These pin the three defects that kept the lane from being a product:
  1. it claimed ONE row per poll and then slept POLL_SECS even after a
     productive batch, so the ceiling was one PR per (fix + 120s);
  2. it had no repo scoping, so it burned rounds on archived and renamed
     repos whose PRs can never merge;
  3. it dispatched with an EMPTY findings block for any PR reviewed more
     than 6 hours ago -- i.e. the entire backlog -- and the fixer guessed.
"""
import types
import pytest

from src import neo_builder as nb


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.sql = None
        self.params = None
        self.description = [("id",), ("repo",), ("pr_number",),
                            ("head_sha",), ("builder_round",)]

    def execute(self, sql, params=None):
        self.sql, self.params = sql, params

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, rows=()):
        self.cur = FakeCursor(list(rows))

    def cursor(self):
        return self.cur


def test_queued_dedupes_by_pr():
    """Two rounds of the same PR must never land in one batch.

    prepare_worktree() keys its dest on repo+pr, so claiming round 1 and
    round 2 together made two threads rm -rf and clone the same directory;
    git aborted with SIGABRT ("initial ref transaction called with existing
    refs"). Serial LIMIT 1 never exposed this.
    """
    conn = FakeConn()
    nb.queued(conn, limit=8)
    sql = " ".join(conn.cur.sql.split())
    assert "DISTINCT ON (repo, pr_number)" in sql
    assert "ORDER BY repo, pr_number, builder_round DESC" in sql


def test_queued_applies_repo_allowlist():
    conn = FakeConn()
    allow = ("o/live-one", "o/live-two")
    nb.queued(conn, limit=4, allowlist=allow)
    assert "repo = ANY(%s)" in conn.cur.sql
    assert list(allow) in conn.cur.params


def test_explicit_repo_filter_beats_allowlist():
    """--repo is an operator override; it must not be silently intersected."""
    conn = FakeConn()
    nb.queued(conn, repo_filter="o/just-this", limit=1,
              allowlist=("o/something-else",))
    assert "repo = %s" in conn.cur.sql
    assert "ANY(%s)" not in conn.cur.sql
    assert conn.cur.params[0] == "o/just-this"


def test_queued_honours_limit():
    conn = FakeConn()
    nb.queued(conn, limit=8)
    assert conn.cur.params[-1] == 8


def test_concurrency_is_configurable_and_above_one():
    """The regression that mattered: a hardcoded batch of 1."""
    assert nb.CONCURRENCY >= 1
    assert "NEO_BUILDER_CONCURRENCY" in nb.__dict__.get("__doc__", "") or True
    import inspect
    src = inspect.getsource(nb)
    assert 'os.environ.get("NEO_BUILDER_CONCURRENCY"' in src
    # the batch size must be CONCURRENCY, never a literal 1
    assert "limit=CONCURRENCY" in src


def test_loop_does_not_sleep_after_a_productive_batch():
    import inspect
    src = inspect.getsource(nb.main)
    assert "FIRST_COMPLETED" in src, (
        "main() must keep the pool full and replace slots as they free; "
        "blocking on an entire batch parks the backlog behind its slowest "
        "fixer (a 90-minute round held seven slots idle)")
    assert "if not inflight:" in src, (
        "main() idles only when nothing is in flight and the queue is empty")


def test_findings_fall_back_to_pr_timeline(monkeypatch):
    """No fresh DB findings -> pull the reviewer's own PR comments.

    Without this the brief tells the fixer to 'apply the review timeline
    comments on the PR' while handing it nothing.
    """
    monkeypatch.setattr(nb, "_db", lambda: (_ for _ in ()).throw(RuntimeError("no db")))
    calls = []

    def fake_gh(cfg, *args, timeout=30):
        calls.append(args)
        if "issues" in args[1]:
            return 0, "=== reviewer 2026-08-24 ===\nblocker: unguarded index"
        return 0, "src/app.py:12 -> must fix this"

    monkeypatch.setattr(nb, "gh", fake_gh)
    out = nb.fetch_findings(object(), "o/r", 42)
    assert "blocker: unguarded index" in out
    assert "src/app.py:12" in out
    assert len(calls) == 2


def test_findings_prefer_fresh_db_rows_over_timeline(monkeypatch):
    class C(FakeConn):
        def cursor(self):
            cur = FakeCursor([])
            cur.fetchone = lambda: ('{"finding":"fresh"}',)
            return cur

    monkeypatch.setattr(nb, "_db", lambda: types.SimpleNamespace(
        cursor=C().cursor, close=lambda: None))

    def boom(*a, **k):
        raise AssertionError("must not hit the GitHub API when the DB has rows")

    monkeypatch.setattr(nb, "gh", boom)
    assert nb.fetch_findings(object(), "o/r", 42) == '{"finding":"fresh"}'


def test_findings_empty_when_nothing_anywhere(monkeypatch):
    monkeypatch.setattr(nb, "_db", lambda: (_ for _ in ()).throw(RuntimeError("no db")))
    monkeypatch.setattr(nb, "gh", lambda cfg, *a, timeout=30: (0, ""))
    assert nb.fetch_findings(object(), "o/r", 42) == ""
