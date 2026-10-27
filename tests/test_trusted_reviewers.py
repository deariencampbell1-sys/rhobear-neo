"""trusted_reviewers — which outside reviewers count as green 2 (Wave 3 API).

Table trusted_reviewers(install_id, repo, source, enabled, updated_at) with
repo='*' as the install-wide wildcard. Seeded once (idempotent) for the owner
install with CodeAnt on; a customer's explicit disabled row is never flipped
back. list/set/delete are install-scoped for Wave 3.

No test touches a real database: psycopg.connect is replaced with a recording
fake that is RESTORED on exit (defect 3: the old _install leaked the fake, so
the NEO_TEST_DSN-gated integration test below ran against stale fake SQL and
never touched real SQL). The integration test runs against the integrator's
throwaway Postgres only, and skips with a printed reason otherwise.
"""
from __future__ import annotations

import contextlib
import os

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
        # The real SQL filters by install_id; a faithful fake does too, so a
        # query under the wrong id finds no rows (defect 6's test depends on
        # this being discriminating).
        if self.calls:
            sql, params = self.calls[-1]
            if params and len(params) == 1 and sql.strip().upper().startswith("SELECT"):
                return [r for r in self.rows if r and r[0] == params[0]]
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


@contextlib.contextmanager
def recording_db(rows=()):
    """A recording fake DB installed as trusted_reviewers._connect — and
    RESTORED on exit (defect 3, HIGH): the fixture installs and restores, so
    after it exits module._connect is the original object again and no later
    test (the NEO_TEST_DSN integration one in particular) can run against a
    stale fake."""
    from src import trusted_reviewers as tr
    conn = FakeConn(rows)
    original = tr._connect
    tr._connect = lambda dsn: conn
    try:
        yield conn
    finally:
        tr._connect = original


def test_ensure_schema_creates_the_table_and_seeds_the_owner_install_once():
    from src import trusted_reviewers as tr
    with recording_db() as conn:
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
    with recording_db() as conn:
        tr.set_source("dsn", OWNER, "*", "Greptile", enabled=True)
        upsert = [c for c in conn.cur.calls if "DO UPDATE" in c[0].upper()]
        assert upsert, conn.cur.calls
        sql, params = upsert[0]
        assert "ON CONFLICT (install_id, repo, source) DO UPDATE" in sql, sql
        assert params == (OWNER, "*", "Greptile", True), params
        assert conn.commits >= 1

    with recording_db(rows=[(OWNER, "*", "Greptile", True, "2026-10-26T00:00:00Z")]) as conn:
        rows = tr.list_sources("dsn", OWNER)
        assert rows == [{"install_id": OWNER, "repo": "*", "source": "Greptile",
                         "enabled": True, "updated_at": "2026-10-26T00:00:00Z"}]
        sql, params = conn.cur.calls[-1]
        assert "install_id = %s" in sql and params == (OWNER,), (sql, params)

    with recording_db() as conn:
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
    with recording_db(rows):
        # The repo row overrides the wildcard per source.
        assert tr.enabled_sources("dsn", OWNER, "deariencampbell1-sys/rhobear-neo") == ("CodeRabbit",)
        assert set(tr.enabled_sources("dsn", OWNER, "deariencampbell1-sys/rhobear-reviews-app")) == {
            "CodeAnt", "CodeRabbit"}
    # Another install never sees this install's rows.
    with recording_db(rows=[]):
        assert tr.enabled_sources("dsn", "acme", "acme/checkout") == ()


def test_source_matches_is_normalised_and_never_matches_an_unknown_reviewer():
    from src import trusted_reviewers as tr
    assert tr.source_matches("CodeAnt", "code-ant")
    assert tr.source_matches("Greptile", "Greptile[bot]")
    assert tr.source_matches("CodeRabbit", "coderabbit")
    assert not tr.source_matches("CodeAnt", "codecov")
    assert not tr.source_matches("CodeAnt", "mallory")
    assert not tr.source_matches("", "anything")


# ---------------------------------------------------------------------------
# Defect 1 (CRITICAL): source_matches is normalised EQUALITY — a prefix in
# either direction must never match, or any GitHub account named `ap` or
# `apollox` could spoof the Apollo ACCEPT that authorises a merge.
# One row per case (the W1-07-r2 table).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source,name,expected", [
    ("apollo", "ap", False),
    ("apollo", "apol", False),
    ("apollo", "apollox", False),
    ("apollo", "apollobot2", False),
    ("apollo", "Apollo", True),
    ("apollo", "apollo[bot]", True),
    ("CodeAnt", "code-ant", True),
    ("CodeAnt", "CodeAnt[bot]", True),
    ("CodeAnt", "codecov", False),
    ("Greptile", "greptile[bot]", True),
])
def test_source_matches_table(source, name, expected):
    """Normalised equality only; [bot]/-bot/_bot suffixes and separators stay
    tolerated (the reason the function exists) and nothing else does."""
    from src import trusted_reviewers as tr
    assert tr.source_matches(source, name) is expected, (source, name)


def test_an_empty_review_author_matches_no_source():
    """Defect 5 (MEDIUM): an empty/missing/whitespace author never matches any
    source — an anonymous review must never fall open to any trusted
    reviewer (including Apollo)."""
    from src import trusted_reviewers as tr
    for empty in ("", None, "   ", "\t\n"):
        assert tr.source_matches("apollo", empty) is False, repr(empty)
        assert tr.source_matches("CodeAnt", empty) is False, repr(empty)
        assert tr.source_matches(empty, "apollo") is False, repr(empty)


def test_disabled_source_is_not_in_the_enabled_set():
    """Disabled CodeAnt does not count — the toggle is the customer's choice."""
    from src import trusted_reviewers as tr
    with recording_db(rows=[(OWNER, "*", "CodeAnt", False, "")]):
        assert tr.enabled_sources("dsn", OWNER, "deariencampbell1-sys/rhobear-neo") == ()


# ---------------------------------------------------------------------------
# Defect 6 (MEDIUM): _enabled_reviewers must query under the repo's REAL
# GitHub App installation id — never the org login, which silently disabled
# the feature for every customer whose install_id differs from the login.
# ---------------------------------------------------------------------------

class _FakeApp:
    def __init__(self, install_id=None, error=None):
        self._install_id = install_id
        self._error = error
        self.repos: list[str] = []

    def installation_for_repo(self, repo):
        self.repos.append(repo)
        if self._error is not None:
            raise self._error
        return self._install_id


def _state():
    from types import SimpleNamespace
    return SimpleNamespace(dsn="dsn")


def test_enabled_reviewers_resolves_the_real_installation_id(monkeypatch):
    """A row written under install id 12345 is found when the resolver returns
    12345 — the query is keyed on the installation id, not the org login."""
    from src import neo_worker
    from src.config import Config
    from src import github_app
    app = _FakeApp(install_id=12345)
    monkeypatch.setattr(github_app, "github_app_for", lambda cfg: app)
    with recording_db(rows=[("12345", "*", "Greptile", True, "")]) as conn:
        got = neo_worker._enabled_reviewers(Config(), _state(), "acme/checkout")
        selects = [p for sql, p in conn.cur.calls
                   if sql.strip().upper().startswith("SELECT") and p]
    assert got == ("Greptile",), got
    assert app.repos == ["acme/checkout"], app.repos
    assert ("12345",) in selects, selects


def test_owner_seed_and_lookup_share_the_resolved_installation_id(monkeypatch):
    from src import neo_worker
    from src.config import Config
    from src import github_app

    app = _FakeApp(install_id=12345)
    monkeypatch.setattr(github_app, "github_app_for", lambda cfg: app)
    with recording_db(rows=[("12345", "*", "CodeAnt", True, "")]) as conn:
        got = neo_worker._enabled_reviewers(Config(), _state(), f"{OWNER}/rhobear-neo")
        selects = [p for sql, p in conn.cur.calls
                   if sql.strip().upper().startswith("SELECT") and p]
        seeds = [p for sql, p in conn.cur.calls
                 if "INSERT" in sql.upper() and "ON CONFLICT" in sql.upper()]
    assert got == ("CodeAnt",), got
    assert selects == [("12345",)], selects
    assert seeds == [("12345", "CodeAnt")], seeds


def test_enabled_reviewers_fail_closed_when_the_installation_cannot_resolve(
        monkeypatch, caplog):
    """A resolver failure logs a WARNING naming the repo and returns no
    sources (fail closed) — a customer's trusted reviewers must never
    silently count, and must never silently NOT count either."""
    import logging
    from src import neo_worker
    from src.config import Config
    from src import github_app
    errors = (github_app.NoInstallation("repo acme/checkout has no installation"),
              github_app.GitHubAppError("lookup failed"))
    for error in errors:
        app = _FakeApp(error=error)
        monkeypatch.setattr(github_app, "github_app_for", lambda cfg, _app=app: _app)
        with recording_db(rows=[("12345", "*", "Greptile", True, "")]):
            with caplog.at_level(logging.WARNING, logger="rhobear_neo.worker"):
                got = neo_worker._enabled_reviewers(Config(), _state(), "acme/checkout")
        assert got == (), got
        warned = [r for r in caplog.records if r.levelno >= logging.WARNING
                  and "acme/checkout" in r.getMessage()]
        assert warned, [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# Integration (the integrator's throwaway PG: NEO_TEST_DSN) — real SQL.
# ---------------------------------------------------------------------------

def _require_dsn() -> str:
    """The integration test's own guard: without NEO_TEST_DSN it SKIPS with a
    printed reason — it never silently passes without touching real SQL."""
    dsn = os.environ.get("NEO_TEST_DSN") or ""
    if not dsn:
        reason = ("NEO_TEST_DSN is unset — real-SQL integration needs the "
                  "integrator's throwaway Postgres; skipping with this "
                  "printed reason (never silently passing)")
        print(reason)
        pytest.skip(reason)
    return dsn


def test_the_connect_fake_is_restored_after_the_fixture_exits():
    """After the fixture exits, module._connect is the original object (the
    old _install leaked the fake, so the NEO_TEST_DSN integration test ran
    against stale fake SQL and proved nothing)."""
    from src import trusted_reviewers as tr
    original = tr._connect
    with recording_db(rows=[(OWNER, "*", "Greptile", True, "2026-10-26T00:00:00Z")]) as conn:
        assert tr._connect is not original
        assert tr.enabled_sources("dsn", OWNER, "deariencampbell1-sys/rhobear-neo") == ("Greptile",)
        assert conn.cur.calls, "the fake must actually serve the read"
    assert tr._connect is original


def test_the_integration_guard_skips_with_a_printed_reason(monkeypatch, capsys):
    """NEO_TEST_DSN unset: the integration guard SKIPS with a printed reason —
    it never silently passes without touching real SQL."""
    monkeypatch.delenv("NEO_TEST_DSN", raising=False)
    with pytest.raises(BaseException) as exc:   # pytest.skip.Exception
        _require_dsn()
    out = capsys.readouterr().out
    assert "NEO_TEST_DSN" in out, out
    assert "NEO_TEST_DSN" in str(exc.value), str(exc.value)


def test_integration_real_sql_roundtrip():
    from src import trusted_reviewers as tr
    dsn = _require_dsn()
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
