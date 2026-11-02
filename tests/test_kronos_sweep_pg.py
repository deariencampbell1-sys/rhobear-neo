"""W1-08-r3 — the judge sweep against REAL Postgres (HIGH 3 + 4 + 6).

Runs only when NEO_TEST_DSN is set (the same throwaway PG the jobs tests
use). Fakes alone cannot prove the claims these tests make:

- HIGH 3: the sweep's SQL itself excludes terminal (merged / ready) and
  already-judged (escalated) rows and in-flight (recent) PRs, and pages
  fairly through a persisted keyset cursor — >20 rows, the first 20
  terminal: the 21st still gets judged; 25 live rows get judged across
  passes, none twice;
- HIGH 4: Zeus's consume-then-act gap — take_escalation() commits the
  marker's deletion BEFORE escalate_pr() records the outcome. A Kronos
  decision attempted IN that gap (the two-worker pause) must defer; Zeus's
  own escalation then completes as the only one;
- HIGH 6: a real Zeus-shaped builder action row (detail carries only
  green/wake_job/route metadata) plus its job payload (which carries the
  installation id) drives consumer `reviews:kronos/<id>`; with no provable
  identity the judge defers loudly instead of degrading to the bare lane.

Run:  NEO_TEST_DSN='<dsn>' .venv/bin/python -m pytest tests/test_kronos_sweep_pg.py -q
"""
from __future__ import annotations

import json
import os

import psycopg
import pytest
from psycopg.rows import dict_row

from src.jobs import Jobs
from src.neo_state import NeoState

_sweep = pytest.importorskip(
    "src.kronos.__main__",
    reason="src/kronos/__main__.py not present yet (red-first commit)")
_judge_sweep = _sweep._judge_sweep
SqlWrites = pytest.importorskip("src.kronos.store").SqlWrites

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO_TEST_DSN"),
    reason="NEO_TEST_DSN not set (no throwaway Postgres for this run)",
)

HEAD = "a" * 40


@pytest.fixture()
def dsn():
    return os.environ["NEO_TEST_DSN"]


@pytest.fixture()
def fresh(dsn):
    NeoState(dsn).ensure_schema()
    Jobs(dsn).ensure_schema()
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("TRUNCATE neo_actions")
        c.execute("TRUNCATE neo_jobs")
    yield dsn
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("TRUNCATE neo_actions")
        c.execute("TRUNCATE neo_jobs")
        c.execute("SET statement_timeout = 0")


def _quiet(dsn, repo: str, pr: int, *, hours: float = 3.0) -> None:
    """Make a PR look quiet — at judge time a real loop's rows are old."""
    h = int(hours)
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "UPDATE neo_actions SET created_at = now() - make_interval("
            "hours => %s) WHERE repo=%s AND pr_number=%s",
            (h, repo, pr))
        c.execute(
            "UPDATE neo_jobs SET created_at = now() - make_interval("
            "hours => %s), updated_at = now() - make_interval(hours => %s) "
            "WHERE kind='builder' AND repo=%s AND pr_number=%s",
            (h, h, repo, pr))


def _builder_rounds(dsn, repo: str, pr: int, n: int, *, shape: str = "bare",
                    head: str = HEAD) -> None:
    """Record n builder rounds. `shape`:
      - 'bare':   the test-suite shape (tree_sha detail);
      - 'zeus':   EXACTLY what Zeus's worker writes — detail carries only
                  green/wake_job/route metadata, NO installation id.
    """
    state = NeoState(dsn)
    for i in range(1, n + 1):
        if shape == "zeus":
            detail = {"green": False, "wake_job": 1000 + i,
                      "route_provider": "routegate",
                      "route_model": "summit"}
        else:
            detail = {"tree_sha": f"tree{i}"}
        state.record(repo, pr, f"{head[:39]}{i}", "builder", verdict="BOUNCE",
                     builder_round=i, detail=detail)


class _RecordingWrites:
    """The sweep's CAS writer stand-in: every decision 'succeeds' (the write
    SQL itself is proven in test_kronos_store_pg.py)."""

    def __init__(self):
        self.escalates: list[tuple] = []
        self.stops: list[tuple] = []

    def escalate(self, repo, pr, head_sha, *, advisor_round, detail,
                 builder_payload, authority=None, **kw):
        self.escalates.append((repo, pr, advisor_round))
        return True, "escalated (test write)"

    def stop(self, repo, pr, head_sha, *, advisor_round, detail,
             authority=None, **kw):
        self.stops.append((repo, pr, advisor_round))
        return True, "stop marker (test write)"


class _SweepServices:
    """Real db() and writes() shapes; GitHub + Summit + comments faked."""

    def __init__(self, dsn, *, pulls_head: str = HEAD):
        self.dsn = dsn
        self.pulls_head = pulls_head
        self.summit_consumers: list[str] = []
        self.comment_urls: list[str] = []

    def db(self, dsn, **kw):
        outer = self

        class _Db:
            def rows(_self, sql, params=()):
                with psycopg.connect(dsn, autocommit=True,
                                     row_factory=dict_row) as c:
                    if kw.get("statement_timeout_ms"):
                        c.execute("SET statement_timeout = "
                                  f"{max(1, int(kw['statement_timeout_ms']))}")
                    cur = c.execute(sql, params)
                    try:
                        return list(cur.fetchall())
                    except psycopg.ProgrammingError:
                        return []

        return _Db()

    def writes(self, dsn, **kw):
        return _RecordingWrites()

    def http(self, method, url, *, headers=None, body=None, timeout=30):
        if "chat/completions" in url:
            self.summit_consumers.append(
                str((headers or {}).get("X-RHOBEAR-Consumer", "")))
            answer = {"progress": "no", "plan": "", "reason": "r3 sweep stop"}
            return 200, json.dumps({"choices": [{"message": {
                "content": json.dumps(answer)}}]})
        if "api.github.com" in url and "/reviews" in url:
            return 200, json.dumps([])
        if "api.github.com" in url and "/pulls/" in url:
            return 200, json.dumps({"state": "open", "merged": False,
                                    "draft": False,
                                    "head": {"sha": self.pulls_head}})
        if "/issues/" in url and method == "POST":
            self.comment_urls.append(url)
            return 201, "{}"
        return 200, "[]"

    def runner(self, argv, *, timeout=30):
        import subprocess
        return subprocess.CompletedProcess(argv, 0, stdout="0\n", stderr="")

    def alert(self, *a, **k):  # never wired: the sweep builds its own
        raise AssertionError("alerter built through _alerter")


def _zeus_pr(dsn, repo: str, pr: int, *, install=4242, head: str = HEAD) -> None:
    """One Zeus-shaped PR: 5 builder rounds (actions) + the round-5 job whose
    payload carries the installation id (the authoritative place)."""
    _builder_rounds(dsn, repo, pr, 5, head=head)
    Jobs(dsn).create_builder(repo, pr, f"{head[:39]}5", 5, payload={
        "verdict": "BOUNCE", "wake_job": 1000 + pr, "head_sha": f"{head[:39]}5",
        "phase": "builder", "round": 5, "context": "green",
        "installation_id": install,
    })


def _run_judge_sweep(cfg, dsn, services, state, out, *, monkeypatch):
    monkeypatch.setattr("src.kronos.__main__._readonly_token",
                        lambda cfg2, repo, install_id: "ro-token")
    _judge_sweep(cfg, state, services.db(dsn), services, out,
                 now_fn=lambda: 1000.0, left_fn=lambda: 100.0,
                 clamp=lambda t: t)


def _cfg(tmp_path, dsn):
    from src.kronos.config import KronosConfig
    return KronosConfig(state_dir=str(tmp_path), routegate_url="http://127.0.0.1:8645",
                        judge_model="summit", judge_timeout_secs=20,
                        reviews_database_url=dsn)


# ---------------------------------------------------------------------------
# HIGH 3 — no LIMIT 20 starvation
# ---------------------------------------------------------------------------

def test_twenty_one_rows_first_twenty_terminal_the_21st_gets_judged(
        fresh, tmp_path, monkeypatch):
    """The review's exact scenario: >20 qualifying rows, the first 20 already
    terminal (merged) — the sweep must still reach the 21st on this pass."""
    dsn = fresh
    for pr in range(1, 21):
        _zeus_pr(dsn, "o/r", pr, install=4242)
        NeoState(dsn).record("o/r", pr, HEAD, "merged", verdict="MERGED",
                             detail={})
        _quiet(dsn, "o/r", pr)
    _zeus_pr(dsn, "o/r", 21, install=4242)
    _quiet(dsn, "o/r", 21)

    services = _SweepServices(dsn)
    from src.kronos import KronosState
    state = KronosState(str(tmp_path / "state.json"))
    lines: list[str] = []
    _run_judge_sweep(_cfg(tmp_path, dsn), dsn, services, state,
                     lines.append, monkeypatch=monkeypatch)
    assert len(services.summit_consumers) == 1, \
        (f"expected exactly the 21st PR judged, got "
         f"{len(services.summit_consumers)} summit calls")
    assert "o/r#21" in state.slot("judged"), lines
    assert "o/r#21" not in ["o/r#%d" % p for p in range(1, 21)]


def test_twenty_five_live_rows_are_judged_across_passes_none_twice(
        fresh, tmp_path, monkeypatch):
    """25 live (quiet, undecided) PRs: pass 1 judges its page and persists
    the keyset cursor; pass 2 continues AFTER it — all 25 judged, none
    judged twice."""
    dsn = fresh
    for pr in range(1, 26):
        _zeus_pr(dsn, "o/r", pr, install=4242)
        _quiet(dsn, "o/r", pr)

    services = _SweepServices(dsn)
    from src.kronos import KronosState
    state = KronosState(str(tmp_path / "state.json"))
    _run_judge_sweep(_cfg(tmp_path, dsn), dsn, services, state,
                     lambda l: None, monkeypatch=monkeypatch)
    judged_after_one = set(state.slot("judged"))
    assert len(judged_after_one) == 20, \
        f"pass 1 judged {len(judged_after_one)} PRs, expected the 20-row page"
    _run_judge_sweep(_cfg(tmp_path, dsn), dsn, services, state,
                     lambda l: None, monkeypatch=monkeypatch)
    judged_after_two = set(state.slot("judged"))
    assert len(judged_after_two) == 25, \
        f"the 21st-25th PRs were starved: {sorted(judged_after_two)}"
    assert len(services.summit_consumers) == 25, \
        "a PR was sent to Summit twice"


# ---------------------------------------------------------------------------
# HIGH 4 — Zeus's consume-then-act gap (two workers, one DB)
# ---------------------------------------------------------------------------

def test_kronos_defers_inside_zeus_take_then_escalate_gap(fresh):
    """THE two-worker race, paused exactly where it bites: Zeus committed
    take_escalation() (marker deleted) but escalate_pr() has not run yet.
    Kronos's CAS must DEFER here — then Zeus's act() completes as the only
    escalation."""
    repo, pr = "o/r", 601
    _builder_rounds(fresh, repo, pr, 5)  # FRESH timestamps: Zeus is active
    state = NeoState(fresh)
    state.record_escalating(repo, pr, HEAD, advice_id="zeus-adv-1")
    assert state.take_escalation(repo, pr, "zeus-adv-1") is True
    # ---- paused: the escalating row is gone; escalate_pr() not yet run ----
    writes = SqlWrites(fresh)
    ok, note = writes.escalate(repo, pr, HEAD, advisor_round=6,
                               detail={}, builder_payload={"escalated": True})
    assert not ok, "Kronos escalated inside Zeus's consume-then-act gap"
    assert ("mid-flight" in note.lower() or "recent" in note.lower()
            or "quiet" in note.lower()), note
    # ---- Zeus's act() completes AFTER the gap ----
    state.record(repo, pr, HEAD, "escalated", verdict="ESCALATE",
                 builder_round=6, detail={"advisor_round": 6})
    with psycopg.connect(fresh, row_factory=dict_row) as c:
        rows = c.execute(
            "SELECT builder_round, detail FROM neo_actions WHERE repo=%s "
            "AND pr_number=%s AND phase='escalated'", (repo, pr)).fetchall()
        jobs = c.execute(
            "SELECT COUNT(*) AS n FROM neo_jobs WHERE kind='builder' "
            "AND repo=%s AND pr_number=%s", (repo, pr)).fetchone()
    assert len(rows) == 1, f"two escalations survived the gap: {rows}"
    assert rows[0]["builder_round"] == 6
    detail = rows[0]["detail"]
    if isinstance(detail, str):
        detail = json.loads(detail)
    assert "kronos" not in detail, "the surviving escalation is Kronos's"
    assert jobs["n"] == 0, "Kronos's round-6 builder job was written"


def test_kronos_defers_the_stop_inside_zeus_gap_too(fresh):
    """The same gap, `no` verdict: a Kronos stop in the gap would park a PR
    Zeus is about to escalate — it must defer."""
    repo, pr = "o/r", 602
    _builder_rounds(fresh, repo, pr, 5)
    state = NeoState(fresh)
    state.record_escalating(repo, pr, HEAD, advice_id="zeus-adv-2")
    assert state.take_escalation(repo, pr, "zeus-adv-2") is True
    writes = SqlWrites(fresh)
    ok, note = writes.stop(repo, pr, HEAD, advisor_round=6, detail={})
    assert not ok, "Kronos stopped a PR Zeus was escalating"
    assert ("mid-flight" in note.lower() or "recent" in note.lower()
            or "quiet" in note.lower()), note
    state.record(repo, pr, HEAD, "escalated", verdict="ESCALATE",
                 builder_round=6, detail={"advisor_round": 6})
    with psycopg.connect(fresh, row_factory=dict_row) as c:
        rows = c.execute(
            "SELECT detail FROM neo_actions WHERE repo=%s AND pr_number=%s "
            "AND phase='escalated'", (repo, pr)).fetchall()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# HIGH 6 — installation identity from the authoritative job payload
# ---------------------------------------------------------------------------

def test_consumer_scope_comes_from_the_job_payload(fresh, tmp_path,
                                                   monkeypatch):
    """A real Zeus-shaped PR: builder ACTION rows carry NO installation id;
    the job payload does. The sweep must read the payload and call Summit as
    reviews:kronos/<id>."""
    dsn = fresh
    repo, pr = "o/r", 700
    _builder_rounds(dsn, repo, pr, 5, shape="zeus")
    Jobs(dsn).create_builder(repo, pr, f"{HEAD[:39]}5", 5, payload={
        "verdict": "BOUNCE", "wake_job": 1005, "head_sha": f"{HEAD[:39]}5",
        "phase": "builder", "round": 5, "context": "green",
        "installation_id": 4242,
    })
    _quiet(dsn, repo, pr)

    services = _SweepServices(dsn)
    from src.kronos import KronosState
    state = KronosState(str(tmp_path / "state.json"))
    _run_judge_sweep(_cfg(tmp_path, dsn), dsn, services, state,
                     lambda l: None, monkeypatch=monkeypatch)
    assert len(services.summit_consumers) == 1, "the PR was never judged"
    assert services.summit_consumers[0] == "reviews:kronos/4242", \
        services.summit_consumers


def test_unproven_identity_defers_loudly(fresh, tmp_path, monkeypatch):
    """No installation id in any job payload or recorded action: no bare-lane
    fallback, no repo-lookup token — the judge defers loudly."""
    dsn = fresh
    repo, pr = "o/r", 701
    _builder_rounds(dsn, repo, pr, 5, shape="zeus")
    Jobs(dsn).create_builder(repo, pr, f"{HEAD[:39]}5", 5, payload={
        "verdict": "BOUNCE", "wake_job": 2005, "head_sha": f"{HEAD[:39]}5",
        "phase": "builder", "round": 5, "context": "green",
    })
    _quiet(dsn, repo, pr)

    services = _SweepServices(dsn)
    from src.kronos import KronosState
    state = KronosState(str(tmp_path / "state.json"))
    lines: list[str] = []
    _run_judge_sweep(_cfg(tmp_path, dsn), dsn, services, state,
                     lines.append, monkeypatch=monkeypatch)
    assert services.summit_consumers == [], \
        "Summit was asked without a proven installation identity"
    assert any("deferred" in l.lower() or "unproven" in l.lower()
               for l in lines), lines
    assert "o/r#701" not in state.slot("judged")
