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
import time

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
            # r4: an optional override lets a test drive the `yes` path
            override = getattr(self, "summit_answer", None)
            if override:
                return 200, json.dumps({"choices": [{"message": {
                    "content": override}}]})
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

    services = _SweepServices(dsn, pulls_head=f"{HEAD[:39]}5")
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

    services = _SweepServices(dsn, pulls_head=f"{HEAD[:39]}5")
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

    services = _SweepServices(dsn, pulls_head=f"{HEAD[:39]}5")
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
# ---------------------------------------------------------------------------
# r4 HIGH B — a cross-repository PR-number collision must not suppress loops
# ---------------------------------------------------------------------------

def test_two_repos_same_pr_number_both_served(fresh, tmp_path, monkeypatch):
    """Repo A's PR #800 has a ready row (terminal); repo B's UNRELATED PR
    #800 has five unproductive rounds. The r3 NOT EXISTS correlated only the
    PR number (`other.repo = repo` bound the unqualified column to the INNER
    neo_actions row), so repo B's PR vanished from every judge page forever.
    Repo B's PR must still be judged; repo A's must stay excluded."""
    dsn = fresh
    _zeus_pr(dsn, "o/ready", 800, install=11)
    NeoState(dsn).record("o/ready", 800, HEAD, "ready", verdict="ACCEPT",
                         detail={})
    _quiet(dsn, "o/ready", 800)
    _zeus_pr(dsn, "o/live", 800, install=22)
    _quiet(dsn, "o/live", 800)

    services = _SweepServices(dsn, pulls_head=f"{HEAD[:39]}5")
    from src.kronos import KronosState
    state = KronosState(str(tmp_path / "state.json"))
    lines: list[str] = []
    _run_judge_sweep(_cfg(tmp_path, dsn), dsn, services, state,
                     lines.append, monkeypatch=monkeypatch)
    assert services.summit_consumers == ["reviews:kronos/22"], \
        (f"repo B's PR #800 was suppressed by repo A's terminal row: "
         f"{services.summit_consumers}")
    assert "o/live#800" in state.slot("judged"), state.slot("judged")
    assert "o/ready#800" not in state.slot("judged"), \
        "repo A's terminal PR was judged after all"
# ---------------------------------------------------------------------------
# r4 HIGH C — Zeus's consume-then-act gap is sealed by a durable in-flight
# marker (AUTHORIZED Zeus interop change, Morpheus ruling 2026-11-02):
# take_escalation() re-phases the consumed marker to 'escalation-inflight'
# instead of deleting it; the marker is cleared only when the outcome
# records, and is reclaimable after a stated stale timeout (3600 s).
# Kronos defers whenever the marker exists.
# ---------------------------------------------------------------------------

def test_kronos_defers_on_the_just_consumed_marker_with_stale_history(fresh):
    """THE review's counterexample: five builder rounds and their jobs more
    than 3600 s old; Zeus creates a fresh escalating marker, consumes it and
    pauses before escalate_pr(). The r3 quiet-window defer sees only stale
    rows — on the r3 head Kronos's CAS WROTE round six here. With the r4
    Zeus change a durable in-flight marker remains: both Kronos decisions
    must defer until Zeus's outcome records."""

    from src.neo_state import ESCALATION_INFLIGHT_PHASE, NeoState as _NS
    repo, pr = "o/r", 603
    _builder_rounds(fresh, repo, pr, 5)
    _quiet(fresh, repo, pr, hours=3.0)   # stale (>3600 s) history
    state = _NS(fresh)
    state.record_escalating(repo, pr, HEAD, advice_id="zeus-adv-3")
    assert state.take_escalation(repo, pr, "zeus-adv-3") is True
    # ---- paused: marker consumed; escalate_pr() has not run yet ----
    writes = SqlWrites(fresh)
    ok, note = writes.escalate(repo, pr, HEAD, advisor_round=6,
                               detail={}, builder_payload={"escalated": True})
    assert not ok, "Kronos escalated onto a just-consumed Zeus escalation"
    assert ("in-flight" in note.lower() or "mid-flight" in note.lower()), note
    ok, note = writes.stop(repo, pr, HEAD, advisor_round=6, detail={})
    assert not ok, "Kronos stopped a PR Zeus was escalating"
    assert ("in-flight" in note.lower() or "mid-flight" in note.lower()), note
    # ---- Zeus's act() completes: outcome recorded, marker cleared ----
    state.record(repo, pr, HEAD, "escalated", verdict="ESCALATE",
                 builder_round=6, detail={"advisor_round": 6})
    state.clear_inflight_markers(repo, pr)
    with psycopg.connect(fresh, row_factory=dict_row) as c:
        n = c.execute(
            "SELECT COUNT(*) AS n FROM neo_actions WHERE repo=%s AND "
            "pr_number=%s AND phase=%s",
            (repo, pr, ESCALATION_INFLIGHT_PHASE)).fetchone()["n"]
        jobs = c.execute(
            "SELECT COUNT(*) AS n FROM neo_jobs WHERE kind='builder' "
            "AND repo=%s AND pr_number=%s", (repo, pr)).fetchone()["n"]
    assert n == 0, "the in-flight marker outlived the outcome"
    ok2, note2 = writes.escalate(repo, pr, HEAD, advisor_round=6,
                                 detail={}, builder_payload={})
    assert not ok2 and "already" in note2.lower(), (ok2, note2)
    assert jobs == 0


def test_kronos_defers_whenever_the_marker_exists_even_when_stale(fresh):
    """The marker is reclaimable by ZEUS after the stale timeout — but until
    Zeus reclaims it, it exists and Kronos defers. After the reclaim the
    advice re-enters Zeus's pipeline; when the outcome records (here: the
    advisor lost the advice and Zeus parked it) every marker is gone and
    Kronos can stop the loop — a crashed Zeus cannot block forever."""
    from src.escalation import _consume_and_act
    from src.neo_state import ESCALATION_INFLIGHT_PHASE, NeoState as _NS
    repo, pr = "o/r", 605
    _builder_rounds(fresh, repo, pr, 5)
    _quiet(fresh, repo, pr, hours=3.0)
    state = _NS(fresh)
    state.record_escalating(repo, pr, HEAD, advice_id="zeus-adv-4")
    assert state.take_escalation(repo, pr, "zeus-adv-4") is True
    _age_marker(fresh, repo, pr, seconds=72000)
    writes = SqlWrites(fresh)
    ok, note = writes.escalate(repo, pr, HEAD, advisor_round=6,
                               detail={}, builder_payload={"escalated": True})
    assert not ok, "Kronos wrote round six over Zeus's in-flight marker"
    assert ("in-flight" in note.lower() or "mid-flight" in note.lower()), note
    # Zeus recovers: the stale marker is reclaimed into its pipeline...
    reclaimed = state.reclaim_stale_inflight(stale_secs=3600)
    assert reclaimed == [{"repo": repo, "pr": pr, "action": "re-recorded"}]
    # ...the advice is consumed again and the outcome records (a park)...
    out = _consume_and_act(
        state, repo, pr, HEAD, "zeus-adv-4", None,
        lambda: state.record(repo, pr, HEAD, "parked", verdict="PARKED",
                             detail={"reason": "advisor lost the advice"})
        or {"action": "parked"},
        detail={"advice_id": "zeus-adv-4"})
    assert out["action"] == "parked", out
    # ...the parked rows age (the recovery happened hours ago in real life)...
    _quiet(fresh, repo, pr)
    # ...and the PR is judgeable again: no marker of any kind is left
    ok, note = writes.stop(repo, pr, HEAD, advisor_round=6,
                           detail={"kronos_stop": True})
    assert ok, f"Kronos could not stop a settled PR: {note}"


def _age_marker(dsn, repo: str, pr: int, *, seconds: float) -> None:
    from src.neo_state import ESCALATION_INFLIGHT_PHASE
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "UPDATE neo_actions SET detail = jsonb_set(detail, "
            "'{consumed_at}', to_jsonb(%s::double precision)) "
            "WHERE repo=%s AND pr_number=%s AND phase=%s",
            (time.time() - seconds, repo, pr, ESCALATION_INFLIGHT_PHASE))
class _RealWritesSweepServices(_SweepServices):
    """The sweep with the REAL CAS writer, so the fresh-only insert really
    hits the pre-existing job row and mark_notified really lands in the
    committed marker's detail."""

    def writes(self, dsn, **kw):
        return SqlWrites(dsn)


# ---------------------------------------------------------------------------
# r4 HIGH D — a committed decision survives a process death before the flush
# ---------------------------------------------------------------------------

class _FakeTokenApp:
    """The PR commenter's Zeus-App stand-in (no credentials on the box)."""

    def __init__(self, *a, **k):
        pass

    def token_for_repo(self, *a, **k):
        return "tok"


class _CapturingAlerter:
    """What _alerter() returns in these tests: every alert and close is
    recorded for the assertions."""

    def __init__(self):
        self.alerts: list[tuple[str, str, str]] = []
        self.closes: list[str] = []

    def alert(self, fp, subject, body):
        self.alerts.append((fp, subject, body))
        return "created"

    def resolved(self, fp, note):
        self.closes.append(fp)


def _committed_stop_row(dsn, repo: str, pr: int, *,
                        reason="kronos judge stopped the loop: dead end"):
    """Exactly the row writes.stop() commits — inserted directly, as if the
    process died between the DB COMMIT and the state flush."""
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "INSERT INTO neo_actions(repo,pr_number,head_sha,phase,"
            "builder_round,verdict,credits,detail) "
            "VALUES (%s,%s,%s,'escalated',0,'PARKED',0,%s)",
            (repo, pr, HEAD, json.dumps({
                "reason": reason, "advisor_round": 6,
                "advice_id": f"kronos-stop:{repo}#{pr}",
                "kronos_stop": True, "kronos": True})))


def test_stop_committed_without_outbox_still_alerts_after_a_crash(
        fresh, tmp_path, monkeypatch):
    """Process death right after the stop's COMMIT: the fresh pass has an
    EMPTY outbox and must still alert the owner — reconstructed from the
    committed marker — and mark the row notified so the next pass stays
    quiet."""
    dsn = fresh
    repo, pr = "o/r", 620
    _builder_rounds(dsn, repo, pr, 5)
    _quiet(dsn, repo, pr)
    _committed_stop_row(dsn, repo, pr)

    from src.kronos import KronosState
    state = KronosState(str(tmp_path / "state.json"))   # FRESH: outbox empty
    services = _RealWritesSweepServices(dsn, pulls_head=f"{HEAD[:39]}5")
    alerter = _CapturingAlerter()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg2, state2, now_fn, **kw: alerter)
    monkeypatch.setattr("src.github_app.GitHubApp", _FakeTokenApp)
    lines: list[str] = []
    _run_judge_sweep(_cfg(tmp_path, dsn), dsn, services, state,
                     lines.append, monkeypatch=monkeypatch)
    fps = [fp for fp, _s, _b in alerter.alerts]
    assert "loop:o/r#620" in fps, \
        f"the owner was never told about the committed stop: {fps}"
    assert services.summit_consumers == [], \
        "a committed stop was sent to Summit"
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        row = c.execute(
            "SELECT detail FROM neo_actions WHERE repo=%s AND pr_number=%s "
            "AND phase='escalated'", (repo, pr)).fetchone()
    detail = row["detail"]
    if isinstance(detail, str):
        detail = json.loads(detail)
    assert detail.get("kronos_notified") is True, \
        "the durable delivered-flag was not written after delivery"
    alerts_after_one = list(alerter.alerts)
    _run_judge_sweep(_cfg(tmp_path, dsn), dsn, services, state,
                     lambda l: None, monkeypatch=monkeypatch)
    assert alerter.alerts == alerts_after_one, \
        "the second pass re-alerted a notified decision"


def test_committed_escalation_comment_reconstructed_after_a_crash(
        fresh, tmp_path, monkeypatch):
    """The escalated decision's explanatory comment survives the same crash
    window: rebuilt from the marker's plan, delivered once, never twice."""
    dsn = fresh
    repo, pr = "o/r", 621
    _builder_rounds(dsn, repo, pr, 5)
    _quiet(dsn, repo, pr)
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "INSERT INTO neo_actions(repo,pr_number,head_sha,phase,"
            "builder_round,verdict,credits,detail) "
            "VALUES (%s,%s,%s,'escalated',6,'ESCALATE',0,%s)",
            (repo, pr, HEAD, json.dumps({
                "reason": "kronos judge: one escalated round with Summit's plan",
                "advisor_round": 6, "advice_id": f"kronos:{repo}#{pr}",
                "installation_id": 4242, "kronos": True,
                "work_order": {"route": "summit",
                               "summary": "the plan",
                               "steps_count": 1}})))

    from src.kronos import KronosState
    state = KronosState(str(tmp_path / "state.json"))
    services = _RealWritesSweepServices(dsn, pulls_head=f"{HEAD[:39]}5")
    alerter = _CapturingAlerter()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg2, state2, now_fn, **kw: alerter)
    monkeypatch.setattr("src.github_app.GitHubApp", _FakeTokenApp)
    _run_judge_sweep(_cfg(tmp_path, dsn), dsn, services, state,
                     lambda l: None, monkeypatch=monkeypatch)
    assert len(services.comment_urls) == 1, \
        f"the reconstructed comment was not delivered: {services.comment_urls}"
    assert alerter.alerts == [], \
        "an escalation raised an owner alert"
    assert services.summit_consumers == []
    _run_judge_sweep(_cfg(tmp_path, dsn), dsn, services, state,
                     lambda l: None, monkeypatch=monkeypatch)
    assert len(services.comment_urls) == 1, \
        "the second pass re-posted the comment"


# ---------------------------------------------------------------------------
# r4 MEDIUM F — the job-conflict alert stays open while the conflicting
# round job exists in ANY state (PARKED, DONE — the unique key still blocks)
# ---------------------------------------------------------------------------

def test_job_conflict_alert_stays_open_through_parked_and_done_rows(
        fresh, tmp_path, monkeypatch):
    """The r3 resolution closed the alert whenever no queued/running job
    matched — a PARKED or DONE round-six job kept blocking every future
    write while the alert claimed the problem was gone."""
    dsn = fresh
    repo, pr = "o/r", 630
    # the parked round-six job exists BEFORE the loop's own round-5 job (so
    # the identity lookup still reads the round-5 payload, newest first)
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "INSERT INTO neo_jobs (kind, repo, pr_number, head_sha, "
            "wake_color, builder_round, state, outcome, payload) "
            "VALUES ('builder', %s, %s, %s, false, 6, 'parked', "
            "'parked', '{}')",
            (repo, pr, f"{HEAD[:39]}5"))
    _zeus_pr(dsn, repo, pr, install=4242)
    _quiet(dsn, repo, pr)

    services = _RealWritesSweepServices(dsn, pulls_head=f"{HEAD[:39]}5")
    services.summit_answer = json.dumps(
        {"progress": "yes", "plan": "one more round",
         "reason": "worth it"})
    from src.kronos import KronosState
    from src.kronos.__main__ import _resolve_fixed
    state = KronosState(str(tmp_path / "state.json"))
    alerter = _CapturingAlerter()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg2, state2, now_fn, **kw: alerter)
    lines: list[str] = []
    _run_judge_sweep(_cfg(tmp_path, dsn), dsn, services, state,
                     lines.append, monkeypatch=monkeypatch)
    assert "job-conflict:o/r#630" in state.slot("open_alerts"), \
        (f"the conflict was never alerted: {lines}")
    assert state.slot("job_conflicts").get("o/r#630") == 6, \
        "the conflicting round was not recorded for the resolution pass"
    # PARKED blocking job: the resolution pass must keep it open
    _resolve_fixed(_cfg(tmp_path, dsn), state, [], services.db(dsn),
                   lambda l: None, now_fn=lambda: 1.0)
    assert "job-conflict:o/r#630" in state.slot("open_alerts"), \
        "a PARKED blocking job resolved the alert"
    # DONE blocking job: STILL open
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "UPDATE neo_jobs SET state='done', outcome='done' "
            "WHERE kind='builder' AND repo=%s AND pr_number=%s "
            "AND builder_round=6", (repo, pr))
    _resolve_fixed(_cfg(tmp_path, dsn), state, [], services.db(dsn),
                   lambda l: None, now_fn=lambda: 1.0)
    assert "job-conflict:o/r#630" in state.slot("open_alerts"), \
        "a DONE blocking job resolved the alert"
    # the row is gone -> closes
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "DELETE FROM neo_jobs WHERE kind='builder' AND repo=%s "
            "AND pr_number=%s AND builder_round=6", (repo, pr))
    _resolve_fixed(_cfg(tmp_path, dsn), state, [], services.db(dsn),
                   lambda l: None, now_fn=lambda: 1.0)
    assert "job-conflict:o/r#630" not in state.slot("open_alerts"), \
        "the alert never closed after the blocking job was deleted"
