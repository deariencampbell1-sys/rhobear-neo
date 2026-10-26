"""trusted_reviewers — which outside reviewers count as green 2 (Wave 3 API).

Table trusted_reviewers(install_id, repo, source, enabled, updated_at) with
repo='*' as the install-wide wildcard. Seeded once (idempotent) for the owner
install with CodeAnt on; a customer's explicit disabled row is never flipped
back. list/set/delete are install-scoped for Wave 3.

No test touches a real database: psycopg.connect is replaced with a recording
fake. The NEO_TEST_DSN-gated integration tests below run against the
integrator's throwaway Postgres only.
"""
from __future__ import annotations

from unittest import mock

import pytest

OWNER = "deariencampbell1-sys"


class FakeCursor:
    def __init__(self, rows=()):
        self.calls: list[tuple] = []
        self.rows = list(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchall(self):
        return list(self.rows)


class FakeConn:
    def __init__(self, rows=()):
        self.cur = FakeCursor(rows)
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1


def _install(module, rows=()):
    """A recording fake DB under module._connect; returns the FakeConn."""
    conn = FakeConn(rows)
    module._connect = lambda dsn: conn
    return conn


def test_ensure_schema_creates_the_table_and_seeds_the_owner_install_once():
    from src import trusted_reviewers as tr
    conn = _install(tr)
    tr.ensure_schema("dsn")
    sqls = [sql for sql, _ in conn.cur.calls]
    assert any("CREATE TABLE IF NOT EXISTS trusted_reviewers" in sql for sql in sqls), sqls
    assert any("updated_at" in sql and "enabled" in sql for sql in sqls), sqls
    seed = [p for sql, p in conn.cur.calls if "INSERT" in sql.upper() and "ON CONFLICT" in sql.upper()]
    assert seed and seed[0][:2] == (OWNER, "CodeAnt"), seed
    assert "DO NOTHING" in [sql for sql, _ in conn.cur.calls if "INSERT" in sql.upper()][0].upper(), \
        "a customer's disabled row must never be flipped back"


def test_set_list_delete_are_install_scoped():
    from src import trusted_reviewers as tr
    conn = _install(tr)
    tr.set_source("dsn", OWNER, "*", "Greptile", enabled=True)
    upsert = [c for c in conn.cur.calls if "INSERT" in c[0].upper()]
    assert upsert, conn.cur.calls
    sql, params = upsert[0]
    assert "ON CONFLICT (install_id, repo, source) DO UPDATE" in sql, sql
    assert params == (OWNER, "*", "Greptile", True), params
    assert conn.commits >= 1

    conn = _install(tr, rows=[(OWNER, "*", "Greptile", True, "2026-10-26T00:00:00Z")])
    rows = tr.list_sources("dsn", OWNER)
    assert rows == [{"install_id": OWNER, "repo": "*", "source": "Greptile",
                     "enabled": True, "updated_at": "2026-10-26T00:00:00Z"}]
    sql, params = conn.cur.calls[-1]
    assert "install_id = %s" in sql and params == (OWNER,), (sql, params)

    conn = _install(tr)
    tr.delete_source("dsn", OWNER, "*", "Greptile")
    sql, params = conn.cur.calls[-1]
    assert sql.strip().upper().startswith("DELETE"), sql
    assert params == (OWNER, "*", "Greptile"), params


def test_enabled_sources_wildcard_is_overridden_by_a_repo_row():
    from src import trusted_reviewers as tr
    rows = [
        (OWNER, "*", "CodeAnt", True, ""),
        (OWNER, "deariencampbell1-sys/rhobear-neo", "CodeAnt", False, ""),
        (OWNER, "*", "Greptile", False, ""),
        (OWNER, "*", "CodeRabbit", True, ""),
    ]
    _install(tr, rows)
    # The repo row overrides the wildcard per source.
    assert tr.enabled_sources("dsn", OWNER, "deariencampbell1-sys/rhobear-neo") == ("CodeRabbit",)
    assert set(tr.enabled_sources("dsn", OWNER, "deariencampbell1-sys/rhobear-reviews-app")) == {
        "CodeAnt", "CodeRabbit"}
    # Another install never sees this install's rows.
    _install(tr, rows=[])
    assert tr.enabled_sources("dsn", "acme", "acme/checkout") == ()


def test_source_matches_is_normalised_and_never_matches_an_unknown_reviewer():
    from src import trusted_reviewers as tr
    assert tr.source_matches("CodeAnt", "code-ant")
    assert tr.source_matches("Greptile", "Greptile[bot]")
    assert tr.source_matches("CodeRabbit", "coderabbit")
    assert not tr.source_matches("CodeAnt", "codecov")
    assert not tr.source_matches("CodeAnt", "mallory")
    assert not tr.source_matches("", "anything")


def test_disabled_source_is_not_in_the_enabled_set():
    """Disabled CodeAnt does not count — the toggle is the customer's choice."""
    from src import trusted_reviewers as tr
    _install(tr, rows=[(OWNER, "*", "CodeAnt", False, "")])
    assert tr.enabled_sources("dsn", OWNER, "deariencampbell1-sys/rhobear-neo") == ()


# ---------------------------------------------------------------------------
# Integration (the integrator's throwaway PG: NEO_TEST_DSN) — real SQL.
# ---------------------------------------------------------------------------

def _dsn():
    import os
    return os.environ.get("NEO_TEST_DSN") or ""


@pytest.mark.skipif(not _dsn(), reason="NEO_TEST_DSN not set (throwaway Postgres)")
def test_integration_real_sql_roundtrip():
    from src import trusted_reviewers as tr
    dsn = _dsn()
    tr.ensure_schema(dsn)
    tr.ensure_schema(dsn)                      # idempotent
    tr.set_source(dsn, "w1-07-test", "w1-07-test/repo", "Greptile", enabled=True)
    assert "Greptile" in tr.enabled_sources(dsn, "w1-07-test", "w1-07-test/repo")
    tr.set_source(dsn, "w1-07-test", "w1-07-test/repo", "Greptile", enabled=False)
    assert tr.enabled_sources(dsn, "w1-07-test", "w1-07-test/repo") == ()
    tr.delete_source(dsn, "w1-07-test", "w1-07-test/repo", "Greptile")
    assert tr.list_sources(dsn, "w1-07-test", "w1-07-test/repo") == []
    # The seed is present exactly once and never re-enabled a disabled row.
    tr.set_source(dsn, tr.DEFAULT_OWNER_INSTALL_ID, "*", tr.SEED_SOURCE, enabled=False)
    tr.ensure_schema(dsn)
    assert tr.enabled_sources(dsn, tr.DEFAULT_OWNER_INSTALL_ID, "any/repo") == ()
    tr.set_source(dsn, tr.DEFAULT_OWNER_INSTALL_ID, "*", tr.SEED_SOURCE, enabled=True)