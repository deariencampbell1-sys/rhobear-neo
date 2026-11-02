"""W1-08-r3 — fix-round tests for the ASTRA re-review (issuecomment-6578963903).

Every test here is written RED against the r2 head (6a17ca8): the assertions
name defects the re-review confirmed at that SHA.

- HIGH 2: PRs that are merged / ready / two-greens-on-the-same-head are
  skipped BEFORE the model call and AGAIN under the decision CAS;
- HIGH 3: no fixed LIMIT 20 starvation — terminal/judged rows are filtered in
  SQL and the sweep pages fairly through a persisted keyset cursor;
- HIGH 5: an App JWT failure becomes a BROKEN deliveries:<app> check that
  alerts — it never aborts the pass;
- HIGH 6: the installation identity comes from the authoritative job payload,
  never from the builder action detail Zeus does not write; unproven defers;
- HIGH 7: a committed stop survives a failed PR comment — the owner alert is
  independent (durable pending-notification outbox, retried next pass);
- MEDIUM 1: alert and delivery HTTP timeouts clamp to the pass budget;
- MEDIUM 2: loop/flap/flap-detect alerts register and CLOSE on Zeus's real
  terminal shapes (merged / ready-on-newest-head), not distinct-ACCEPT counts;
- MEDIUM 3: a failed restart-count probe raises into the flap-detect alert
  instead of silently reading as zero;
- LOW 1: the retired remedy window configuration is gone.

The real-Postgres halves of HIGH 1/3/4/6 live in tests/test_kronos_store_pg.py
and tests/test_kronos_sweep_pg.py (NEO_TEST_DSN).
"""
from __future__ import annotations

import json
import subprocess

import pytest

from src.kronos import checks
from src.kronos.config import KronosConfig
from src.kronos.judge import judge_pr

HEAD = "a" * 40
OTHER_HEAD = "b" * 40
APOLLO = "rhobear-reviews[bot]"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

def _rounds(n: int = 5, head: str = HEAD) -> list[dict]:
    return [{"round": i, "head_sha": head, "verdict": "BOUNCE",
             "tree_sha": f"t{i}", "diff_stat": "", "findings": ""}
            for i in range(1, n + 1)]


class FakeDbR3:
    """rows(sql, params) over canned tables. Branches are keyed on the r3 SQL
    shapes so the r2 sweep SQL (no NOT EXISTS / no cursor params) reads as an
    empty page — a red run must show the defect, not a fake leak."""

    def __init__(self, *, rounds=None, escalating=False, escalated=False,
                 merged=False, ready=False, assessments=None, jobs=None,
                 actions_with_install=None, sweep_page=None, greens=0,
                 ready_on_newest_head=False):
        self.rounds = rounds if rounds is not None else _rounds()
        self.escalating = escalating
        self.escalated = escalated
        self.merged = merged
        self.ready = ready
        self.assessments = assessments or []
        self.jobs = jobs or []
        self.actions_with_install = actions_with_install or []
        self.sweep_page = sweep_page
        self.greens = greens
        self.ready_on_newest_head = ready_on_newest_head
        self.sweep_calls: list[tuple[str, tuple]] = []

    def rows(self, sql: str, params=()) -> list[dict]:
        s = " ".join(str(sql).split())
        if "NOT EXISTS" in s:  # the r3 sweep (filtered + fair)
            self.sweep_calls.append((s, tuple(params)))
            return list(self.sweep_page or [])
        if "COUNT(DISTINCT" in s:
            return [{"greens": self.greens}]
        if "phase = 'ready'" in s and "ORDER BY id DESC LIMIT 1)" in s:
            return [{"n": 1}] if self.ready_on_newest_head else []
        if "phase = 'ready'" in s or "phase='ready'" in s:
            return [{"n": 1}] if self.ready else []
        if "phase = 'merged'" in s or "phase='merged'" in s:
            return [{"n": 1}] if self.merged else []
        if "phase='escalating'" in s or "phase = 'escalating'" in s:
            return [{"repo": "o/r", "pr": 1}] if self.escalating else []
        if "phase='escalated'" in s or "phase = 'escalated'" in s:
            return [{"n": 1}] if self.escalated else []
        if "ORDER BY id ASC" in s:
            return self.rounds
        if "FROM assessments" in s:
            return self.assessments
        if "FROM neo_jobs" in s:
            return list(self.jobs)
        if "installation_id" in s:
            return list(self.actions_with_install)
        return []


class FakeGithubR3:
    """api.github.com as the judge sees it: the PR's CURRENT state and its
    reviews (read-only). Records every call."""

    def __init__(self, *, merged=False, state="open", head: str = HEAD,
                 reviews=None):
        self.merged = merged
        self.state = state
        self.head = head
        self.reviews = reviews or []
        self.calls: list[dict] = []

    def __call__(self, method, url, *, headers=None, body=None, timeout=30):
        self.calls.append({"method": method, "url": url, "timeout": timeout})
        if url.endswith("/reviews") or "/reviews?" in url:
            return 200, json.dumps(self.reviews)
        if "/pulls/" in url:
            return 200, json.dumps({"state": self.state,
                                    "merged": self.merged, "draft": False,
                                    "head": {"sha": self.head}})
        return 200, "{}"


class FakeHttpR3:
    """The RouteGate POST (Summit). Records headers so the consumer scope can
    be asserted."""

    def __init__(self, content: str = "", *, on_call=None):
        self.content = content
        self.calls: list[dict] = []
        self.on_call = on_call

    def __call__(self, method, url, *, headers=None, body=None, timeout=30):
        self.calls.append({"method": method, "url": url,
                           "headers": dict(headers or {}), "timeout": timeout})
        if self.on_call:
            self.on_call()
        msg = {"choices": [{"message": {"content": self.content}}]}
        return 200, json.dumps(msg)


class FakeWritesR3:
    """The CAS writer: honours an injected authority callable exactly like
    src.kronos.store.SqlWrites does (the re-check runs inside the lock)."""

    def __init__(self, defer_reason: str | None = None):
        self.escalates: list[dict] = []
        self.stops: list[dict] = []
        self.authority_results: list[str] = []
        self.defer_reason = defer_reason

    def _gate(self, authority):
        if authority is None:
            return True, ""
        ok, why = authority()
        self.authority_results.append(why or "ok")
        return ok, why

    def escalate(self, repo, pr, head_sha, *, advisor_round, detail,
                 builder_payload, authority=None, **kw):
        ok, why = self._gate(authority)
        if not ok:
            return False, f"deferred: {why}"
        self.escalates.append({"repo": repo, "pr": pr, "head_sha": head_sha,
                               "advisor_round": advisor_round,
                               "detail": detail,
                               "builder_payload": builder_payload})
        if self.defer_reason:
            return False, self.defer_reason
        return True, f"escalated round {advisor_round} recorded"

    def stop(self, repo, pr, head_sha, *, advisor_round, detail,
             authority=None, **kw):
        ok, why = self._gate(authority)
        if not ok:
            return False, f"deferred: {why}"
        self.stops.append({"repo": repo, "pr": pr, "head_sha": head_sha,
                           "advisor_round": advisor_round, "detail": detail})
        if self.defer_reason:
            return False, self.defer_reason
        return True, "stop marker recorded"


class FakeStateR3:
    def __init__(self):
        self.data: dict = {}

    def slot(self, name):
        return self.data.setdefault(name, {})

    def flush(self):
        pass


class FakeCommentsR3:
    def __init__(self, fail: bool = False):
        self.posted: list[tuple[str, int, str]] = []
        self.fail = fail

    def __call__(self, repo: str, pr: int, body: str):
        if self.fail:
            raise RuntimeError("PR comment -> HTTP 403")
        self.posted.append((repo, pr, body))


class FakeAlerterR3:
    def __init__(self):
        self.alerts: list[tuple[str, str, str]] = []
        self.closes: list[str] = []

    def alert(self, fingerprint, subject, body):
        self.alerts.append((fingerprint, subject, body))
        return "created"

    def resolved(self, fingerprint, note):
        self.closes.append(fingerprint)


GOOD_NO = json.dumps({"progress": "no", "plan": "",
                      "reason": "the same tests fail in the same way"})
GOOD_YES = json.dumps({"progress": "yes",
                       "plan": "split the module and fix the imports",
                       "reason": "converging, one more round is worth it"})


def _cfg(**kw) -> KronosConfig:
    return KronosConfig(routegate_url="http://127.0.0.1:8645",
                        judge_model="summit", judge_timeout_secs=20, **kw)


def _judge_pr(db, *, github=None, http=None, comments=None, alerter=None,
              state=None, writes=None, install_id="999", head: str = HEAD,
              **kw):
    """judge_pr with every r3 dependency wired. On the r2 head judge_pr has
    no `github` parameter: the shim drops it so the DB-level gates fail on
    their assertions (the defect), not on the signature."""
    kwargs = dict(comments=comments or FakeCommentsR3(),
                  alerter=alerter or FakeAlerterR3(),
                  state=state or FakeStateR3(),
                  repo="o/r", pr=52, head_sha=head,
                  install_id=install_id, read_token="ro-token",
                  http=http or FakeHttpR3(GOOD_NO),
                  github=github if github is not None else FakeGithubR3(),
                  now_fn=lambda: 1000.0,
                  writes=writes or FakeWritesR3(),
                  timeout=15)
    kwargs.update(kw)
    try:
        return judge_pr(_cfg(), db, **kwargs)
    except TypeError as exc:
        if "github" in str(exc) and "argument" in str(exc):
            kwargs.pop("github")
            return judge_pr(_cfg(), db, **kwargs)
        raise


class _SweepServices:
    """Minimal services for _judge_sweep: the CAS writer shape, faked GitHub
    and Summit, faked token minting (identity is tested in the PG suite)."""

    def __init__(self, db, *, pulls_head: str = HEAD, reviews=None):
        self._db = db
        self.pulls_head = pulls_head
        self.reviews = reviews or []
        self.summit_calls: list[dict] = []
        self.comments: list[str] = []

    def db(self, dsn, **kw):
        return self._db

    def writes(self, dsn, **kw):
        return FakeWritesR3()

    def http(self, method, url, *, headers=None, body=None, timeout=30):
        if "chat/completions" in url:
            self.summit_calls.append(dict(headers or {}))
            return 200, json.dumps(
                {"choices": [{"message": {"content": GOOD_NO}}]})
        if "api.github.com" in url and "/reviews" in url:
            return 200, json.dumps(self.reviews)
        if "api.github.com" in url and "/pulls/" in url:
            return 200, json.dumps({"state": "open", "merged": False,
                                    "draft": False,
                                    "head": {"sha": self.pulls_head}})
        if "/issues/" in url and method == "POST":
            self.comments.append(url)
            return 201, "{}"
        return 200, "[]"


def _run_sweep(db, services, state, *, page):
    """One _judge_sweep pass over a canned page."""
    from src.kronos.__main__ import _judge_sweep

    def rows(sql, params=()):
        s = " ".join(str(sql).split())
        if "NOT EXISTS" in s:
            db.sweep_calls.append((s, tuple(params)))
            return list(page)
        return FakeDbR3.rows(db, sql, params)

    db.rows = rows
    _judge_sweep(_cfg(), state, db, services, out=lambda l: None,
                 now_fn=lambda: 1000.0, left_fn=lambda: 100.0,
                 clamp=lambda t: t)


# ---------------------------------------------------------------------------
# HIGH 2 — merged / ready / two-greens are skipped, before the model AND
# under the decision CAS
# ---------------------------------------------------------------------------

def test_merged_pr_is_skipped_before_the_model():
    """merged-after-five: Zeus recorded the merge — no Summit call, no stop,
    no escalation."""
    db = FakeDbR3(merged=True)
    http = FakeHttpR3(GOOD_NO)
    writes = FakeWritesR3()
    outcome = _judge_pr(db, http=http, writes=writes)
    assert outcome.action == "skip", outcome
    assert http.calls == [], "the model was called for a merged PR"
    assert writes.stops == [] and writes.escalates == []


def test_ready_pr_is_skipped_before_the_model():
    """Zeus recorded phase='ready' (the worker ACCEPTed) — Kronos never
    judges it, even at 5 rounds."""
    db = FakeDbR3(ready=True)
    http = FakeHttpR3(GOOD_NO)
    writes = FakeWritesR3()
    outcome = _judge_pr(db, http=http, writes=writes)
    assert outcome.action == "skip", outcome
    assert http.calls == []
    assert writes.stops == [] and writes.escalates == []


def test_github_merged_is_skipped():
    """GitHub says merged — the DB has no row yet, but the authoritative
    state wins: no Summit call."""
    db = FakeDbR3()
    github = FakeGithubR3(merged=True)
    http = FakeHttpR3(GOOD_NO)
    writes = FakeWritesR3()
    outcome = _judge_pr(db, github=github, http=http, writes=writes)
    assert outcome.action == "skip", outcome
    assert http.calls == []
    assert writes.stops == [] and writes.escalates == []


def test_two_greens_on_the_same_head_are_skipped():
    """green-after-five: the loop recovered — Apollo ACCEPT plus a second
    approval, BOTH bound to the current head — no Summit call."""
    db = FakeDbR3()
    github = FakeGithubR3(head=HEAD, reviews=[
        {"user": {"login": APOLLO}, "state": "APPROVED", "commit_id": HEAD},
        {"user": {"login": "the-architect"}, "state": "APPROVED",
         "commit_id": HEAD},
    ])
    http = FakeHttpR3(GOOD_NO)
    writes = FakeWritesR3()
    outcome = _judge_pr(db, github=github, http=http, writes=writes)
    assert outcome.action == "skip", outcome
    assert http.calls == []
    assert writes.stops == [] and writes.escalates == []


def test_two_greens_on_an_older_head_do_not_skip():
    """Approvals bound to an OLD head are not the proof: a new push resets
    them (the merge rule is same-head)."""
    db = FakeDbR3()
    github = FakeGithubR3(head=HEAD, reviews=[
        {"user": {"login": APOLLO}, "state": "APPROVED",
         "commit_id": OTHER_HEAD},
        {"user": {"login": "the-architect"}, "state": "APPROVED",
         "commit_id": OTHER_HEAD},
    ])
    outcome = _judge_pr(db, github=github)
    assert outcome.action != "skip", outcome


def test_head_moved_defers_loudly():
    """GitHub's current head differs from the newest recorded round — Zeus is
    mid-flight on a newer head: defer, never judge stale history."""
    db = FakeDbR3()
    github = FakeGithubR3(head=OTHER_HEAD)
    http = FakeHttpR3(GOOD_NO)
    outcome = _judge_pr(db, github=github, http=http)
    assert outcome.action == "deferred", outcome
    assert http.calls == []


def test_authoritative_state_is_rechecked_under_the_decision_cas():
    """The PR merges WHILE the model is answering: the write's authority
    re-check (inside the CAS) must refuse — no stop marker, no comment, no
    alert."""
    db = FakeDbR3()
    github = FakeGithubR3()
    http = FakeHttpR3(GOOD_NO, on_call=lambda: setattr(github, "merged", True))
    writes = FakeWritesR3()
    comments = FakeCommentsR3()
    alerter = FakeAlerterR3()
    state = FakeStateR3()
    outcome = _judge_pr(db, github=github, http=http, writes=writes,
                        comments=comments, alerter=alerter, state=state)
    assert outcome.action == "deferred", outcome
    assert writes.authority_results, "the CAS never re-checked the PR state"
    assert any("merged" in r.lower() for r in writes.authority_results), \
        writes.authority_results
    assert writes.stops == [] and writes.escalates == []
    assert comments.posted == [] and alerter.alerts == []
    assert "o/r#52" not in state.slot("judged")


# ---------------------------------------------------------------------------
# HIGH 3 — no LIMIT 20 starvation: SQL filters + persisted fair cursor
# ---------------------------------------------------------------------------

def test_sweep_sql_filters_terminal_and_judged_rows():
    """The sweep's query must itself exclude terminal/judged PRs (merged /
    ready / escalated-with-advisor_round) and the quiet ones — the r2 query
    returned the same first 20 historical rows on every pass."""
    db = FakeDbR3()
    _run_sweep(db, _SweepServices(db), FakeStateR3(), page=[])
    assert db.sweep_calls, "the sweep never queried"
    sql, params = db.sweep_calls[0]
    assert "NOT EXISTS" in sql, "terminal rows are not filtered in SQL"
    assert "phase IN ('merged', 'ready')" in sql, sql
    assert "advisor_round" in sql, "judged (escalated) rows are not filtered"
    assert "make_interval" in sql, "in-flight (recent) rows are not filtered"
    assert "LIMIT" in sql, "the page is unbounded"


def test_judge_cursor_pages_fairly_across_passes():
    """>20 qualifying rows: pass 1 judges the first page and PERSISTS the
    cursor; pass 2 starts AFTER it — the 21st+ PRs get their turn."""
    page1 = [{"repo": "o/r", "pr_number": i, "rounds": 5, "head_sha": HEAD}
             for i in range(1, 21)]
    page2 = [{"repo": "o/r", "pr_number": i, "rounds": 5, "head_sha": HEAD}
             for i in range(21, 26)]
    pages = [page1, page2]
    db = FakeDbR3()

    def rows(sql, params=()):
        s = " ".join(str(sql).split())
        if "NOT EXISTS" in s:
            db.sweep_calls.append((s, tuple(params)))
            return list(pages[min(len(db.sweep_calls) - 1, len(pages) - 1)])
        return FakeDbR3.rows(db, sql, params)

    db.rows = rows
    from src.kronos.__main__ import _judge_sweep
    import os
    import tempfile
    from src.kronos import KronosState
    with tempfile.TemporaryDirectory() as td:
        state = KronosState(os.path.join(td, "state.json"))
        services = _SweepServices(db)
        _judge_sweep(_cfg(), state, db, services, out=lambda l: None,
                     now_fn=lambda: 1000.0, left_fn=lambda: 100.0,
                     clamp=lambda t: t)
        assert len(db.sweep_calls) == 1
        first_params = db.sweep_calls[0][1]
        assert len(first_params) >= 6, \
            f"the sweep query carries no cursor/page params: {first_params}"
        _judge_sweep(_cfg(), state, db, services, out=lambda l: None,
                     now_fn=lambda: 1000.0, left_fn=lambda: 100.0,
                     clamp=lambda t: t)
        assert len(db.sweep_calls) == 2
        second_params = db.sweep_calls[1][1]
        assert second_params[-3] > first_params[-3] or \
            second_params[-2] > first_params[-2], \
            (f"pass 2 re-fetched the same page: "
             f"{first_params} -> {second_params}")


# ---------------------------------------------------------------------------
# HIGH 5 — an App JWT failure is a BROKEN check that alerts, never a crash
# ---------------------------------------------------------------------------

class _EmptyDb:
    def rows(self, sql, params=()):
        return []


def test_app_jwt_failure_marks_deliveries_broken_and_alerts(tmp_path,
                                                            monkeypatch):
    """_app_jwt raising (unreadable key / signing failure) must not abort the
    pass: deliveries:<app> goes BROKEN with a non-secret reason and alerts."""
    from src.kronos.__main__ import run_pass
    from src.kronos import KronosState

    class _Services:
        def db(self, dsn, **kw):
            return _EmptyDb()

        def runner(self, argv, *, timeout=30):
            return subprocess.CompletedProcess(argv, 0, stdout="0\n",
                                               stderr="")

        def http(self, method, url, *, headers=None, body=None, timeout=30):
            raise AssertionError("no network in this test")

        def writes(self, dsn, **kw):
            return FakeWritesR3()

    cfg = KronosConfig(state_dir=str(tmp_path),
                       reviews_database_url="postgresql://fake/fake",
                       reviews_app_id="1", reviews_app_key="k",
                       zeus_app_id="2", zeus_app_key="k")
    alerter = FakeAlerterR3()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg2, state, now_fn, **kw: alerter)

    def _boom(cfg2, app):
        raise ValueError("signing failed")

    monkeypatch.setattr("src.kronos.__main__._app_jwt", _boom)
    lines: list[str] = []
    rc = run_pass(cfg, dry_run=False, state=KronosState(":memory:"),
                  services=_Services(), out=lines.append,
                  now_fn=lambda: 1000.0, mono_fn=lambda: 0.0)
    assert rc == 0, "a JWT failure crashed the pass"
    assert any("deliveries:reviews" in l and "broken" in l for l in lines), \
        lines
    assert any("deliveries:zeus" in l and "broken" in l for l in lines), lines
    assert alerter.alerts, "the broken delivery check never alerted"
    assert any(fp.startswith("check:deliveries:") for fp, _s, _b
               in alerter.alerts), alerter.alerts


# ---------------------------------------------------------------------------
# HIGH 7 — a committed stop never loses the owner notification
# ---------------------------------------------------------------------------

def test_stop_alerts_even_when_the_comment_403s():
    """The stop marker is committed, the PR comment 403s — the owner alert
    must STILL go out (alert independent of comment)."""
    db = FakeDbR3()
    writes = FakeWritesR3()
    comments = FakeCommentsR3(fail=True)
    alerter = FakeAlerterR3()
    state = FakeStateR3()
    outcome = _judge_pr(db, http=FakeHttpR3(GOOD_NO), writes=writes,
                        comments=comments, alerter=alerter, state=state)
    assert outcome.action == "stopped", outcome
    assert writes.stops, "the stop was not committed"
    fps = [fp for fp, _s, _b in alerter.alerts]
    assert "loop:o/r#52" in fps, f"the owner was never told: {fps}"
    box = state.slot("notify_outbox")
    assert "o/r#52" in box, "the undelivered comment was dropped"
    assert box["o/r#52"]["needs_comment"] is True
    assert box["o/r#52"]["needs_alert"] is False


def test_outbox_is_retried_on_a_later_pass():
    """The pending comment survives the pass and is delivered on a later one;
    the outbox entry clears only when everything is delivered."""
    db = FakeDbR3()
    writes = FakeWritesR3()
    comments = FakeCommentsR3(fail=True)
    alerter = FakeAlerterR3()
    state = FakeStateR3()
    _judge_pr(db, http=FakeHttpR3(GOOD_NO), writes=writes, comments=comments,
              alerter=alerter, state=state)
    assert state.slot("notify_outbox")
    from src.kronos.judge import drain_notify_outbox
    good_comments = FakeCommentsR3()
    drain_notify_outbox(state, good_comments, alerter)
    assert good_comments.posted, "the pending comment was never delivered"
    assert state.slot("notify_outbox") == {}, \
        "a fully delivered entry stayed in the outbox"


def test_yes_persists_the_comment_in_the_outbox_too():
    """A `yes` writes its builder job, then the comment fails — the comment
    is pending, not lost."""
    db = FakeDbR3()
    writes = FakeWritesR3()
    comments = FakeCommentsR3(fail=True)
    state = FakeStateR3()
    outcome = _judge_pr(db, http=FakeHttpR3(GOOD_YES), writes=writes,
                        comments=comments, state=state)
    assert outcome.action == "escalated", outcome
    assert len(writes.escalates) == 1, "the job was not written"
    box = state.slot("notify_outbox")
    assert "o/r#52" in box and box["o/r#52"]["needs_comment"] is True


def test_job_conflict_on_the_write_alerts_the_owner():
    """HIGH 1's alert half: the CAS refuses because a builder job for the
    round already exists (fresh-only insert) — the owner is told."""
    db = FakeDbR3()
    writes = FakeWritesR3(
        defer_reason="deferred: builder job for this round already exists "
                     "(fresh-only insert — never re-armed)")
    alerter = FakeAlerterR3()
    state = FakeStateR3()
    outcome = _judge_pr(db, http=FakeHttpR3(GOOD_YES), writes=writes,
                        alerter=alerter, state=state)
    assert outcome.action == "deferred", outcome
    fps = [fp for fp, _s, _b in alerter.alerts]
    assert "job-conflict:o/r#52" in fps, f"no conflict alert: {fps}"
    assert "job-conflict:o/r#52" in state.slot("open_alerts"), \
        "the conflict alert is not tracked for resolution"


# ---------------------------------------------------------------------------
# HIGH 6 — installation identity from the authoritative job payload
# ---------------------------------------------------------------------------

def test_install_id_comes_from_the_job_payload_not_the_action_detail():
    """Zeus's builder ACTION rows carry only green/wake_job/route metadata;
    the installation id lives in neo_jobs.payload. The sweep must read it
    there and the consumer must be reviews:kronos/<id> — never the bare
    lane."""
    db = FakeDbR3(jobs=[{"payload": {"installation_id": 4242,
                                     "verdict": "BOUNCE"}}])
    services = _SweepServices(db)
    _run_sweep(db, services, FakeStateR3(),
               page=[{"repo": "o/r", "pr_number": 52, "rounds": 5,
                      "head_sha": HEAD}])
    assert services.summit_calls, "the model was never asked"
    consumer = services.summit_calls[0].get("X-RHOBEAR-Consumer", "")
    assert consumer == "reviews:kronos/4242", consumer


def test_unproven_install_defers_loudly():
    """No job payload, no recorded identity — no bare-lane fallback, no
    repo-lookup token: the judge defers loudly."""
    db = FakeDbR3(jobs=[], actions_with_install=[])
    services = _SweepServices(db)
    lines: list[str] = []
    from src.kronos.__main__ import _judge_sweep

    def rows(sql, params=()):
        s = " ".join(str(sql).split())
        if "NOT EXISTS" in s:
            db.sweep_calls.append((s, tuple(params)))
            return [{"repo": "o/r", "pr_number": 52, "rounds": 5,
                     "head_sha": HEAD}]
        return FakeDbR3.rows(db, sql, params)

    db.rows = rows
    _judge_sweep(_cfg(), FakeStateR3(), db, services, out=lines.append,
                 now_fn=lambda: 1000.0, left_fn=lambda: 100.0,
                 clamp=lambda t: t)
    assert services.summit_calls == [], \
        "the model was asked without a proven installation identity"
    assert any("deferred" in l.lower() or "unproven" in l.lower()
               for l in lines), lines


# ---------------------------------------------------------------------------
# MEDIUM 1 — the 240 s budget bounds the alert and delivery I/O too
# ---------------------------------------------------------------------------

def test_alert_http_timeout_clamps_to_the_remaining_budget(monkeypatch):
    """_alert_once builds its alerter WITH the budget: with <1s left the
    issue calls time out at 1s, not the default 30s."""
    from src.kronos.__main__ import _alert_once
    from src.kronos import KronosState

    cfg = _cfg()
    state = KronosState(":memory:")
    seen: list[int] = []

    class _Svc:
        def http(self, method, url, *, headers=None, body=None, timeout=30):
            seen.append(timeout)
            return 200, "[]"

    class _FakeApp:
        def __init__(self, *a, **k):
            pass

        def token_for_repo(self, *a, **k):
            return "tok"

        def jwt(self):
            return "jwt"

    # the alerter imports GitHubApp lazily; patch it at the source module
    monkeypatch.setattr("src.github_app.GitHubApp", _FakeApp)
    _alert_once(cfg, state, lambda l: None, fingerprint="check:x",
                subject="s", body="b", now_fn=lambda: 0.0,
                services=_Svc(), left_fn=lambda: 0.4)
    assert seen, "the alerter never made an HTTP call"
    assert max(seen) <= 1, f"alert HTTP not clamped to the budget: {seen}"


def test_delivery_list_timeout_clamps_to_the_remaining_budget(tmp_path,
                                                              monkeypatch):
    """The delivery probe's 30 s GET is clamped to what is LEFT of the pass
    budget (the r2 code ran it independently of the clock)."""
    from src.kronos.__main__ import run_pass
    from src.kronos import KronosState

    class _Clock:
        def __init__(self):
            self.t = 0.0

    clock = _Clock()

    class _BumpDb:
        def rows(self, sql, params=()):
            clock.t = 239.5  # 0.5s of the 240s budget left
            return []

    class _Services:
        def db(self, dsn, **kw):
            return _BumpDb()

        def runner(self, argv, *, timeout=30):
            return subprocess.CompletedProcess(argv, 0, stdout="0\n",
                                               stderr="")

        def http(self, method, url, *, headers=None, body=None, timeout=30):
            raise AssertionError("no network in this test")

        def writes(self, dsn):
            raise AssertionError("no writes in this test")

    recorded: list[float] = []

    def _factory(http_opener=None):
        def _list(jwt, *, per_page=100, timeout=30):
            recorded.append(timeout)
            return []
        return _list

    monkeypatch.setattr("src.kronos.__main__.list_deliveries_via_app",
                        _factory)

    class _NoAlert:
        def alert(self, *a, **k):
            return "suppressed"

        def resolved(self, *a, **k):
            pass

    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg2, state, now_fn, **kw: _NoAlert())
    monkeypatch.setattr("src.kronos.__main__._app_jwt",
                        lambda cfg2, app: "fake-jwt")

    cfg = KronosConfig(state_dir=str(tmp_path), pass_budget_secs=240,
                       reviews_database_url="postgresql://fake/fake",
                       reviews_app_id="1", reviews_app_key="k",
                       zeus_app_id="2", zeus_app_key="k")
    run_pass(cfg, dry_run=False, state=KronosState(":memory:"),
             services=_Services(), out=lambda l: None,
             now_fn=lambda: clock.t, mono_fn=lambda: clock.t)
    assert recorded, "the delivery probe never ran"
    assert max(recorded) <= 1, \
        f"delivery GET not clamped to the budget: {recorded}"


# ---------------------------------------------------------------------------
# MEDIUM 2 — every alert kind registers and closes
# ---------------------------------------------------------------------------

def test_loop_alert_closes_when_zeus_records_the_merge(tmp_path,
                                                       monkeypatch):
    """The loop alert closes on Zeus's REAL terminal shape (a merged row) —
    not on distinct-ACCEPT counts Zeus never writes."""
    from src.kronos.__main__ import _resolve_fixed
    from src.kronos import KronosState
    cfg = _cfg()
    state = KronosState(":memory:")
    state.slot("open_alerts")["loop:o/r#52"] = "2026-11-01T00:00:00Z"
    alerter = FakeAlerterR3()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg2, state2, now_fn, **kw: alerter)
    db = FakeDbR3(merged=True)
    _resolve_fixed(cfg, state, [], db, lambda l: None, now_fn=lambda: 1.0)
    assert "loop:o/r#52" in alerter.closes, \
        "the loop alert stayed open after Zeus recorded the merge"
    assert state.slot("open_alerts") == {}


def test_loop_resolved_uses_zeus_terminal_shapes():
    """checks.loop_resolved: merged -> True; a ready row on the NEWEST builder
    head -> True; distinct ACCEPTs on builder rows (the r2 signal Zeus never
    writes) -> False."""
    assert checks.loop_resolved(FakeDbR3(merged=True), "o/r", 52) is True
    assert checks.loop_resolved(
        FakeDbR3(ready_on_newest_head=True), "o/r", 52) is True
    assert checks.loop_resolved(
        FakeDbR3(greens=2), "o/r", 52) is False, \
        "distinct ACCEPTs on builder rows are not two greens"
    assert checks.loop_resolved(FakeDbR3(), "o/r", 52) is False


def test_flap_alert_resolves_after_a_quiet_window(tmp_path, monkeypatch):
    """A healthy unit's flap alert closes once the quiet window has passed —
    flap_status's window reset must not keep it open forever."""
    from src.kronos.__main__ import _resolve_fixed
    from src.kronos import KronosState
    from src.kronos.checks import flap_status
    cfg = _cfg()
    state = KronosState(":memory:")
    alerter = FakeAlerterR3()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg2, state2, now_fn, **kw: alerter)
    now = 1000.0
    flap_slot = state.slot("flaps")
    should, flap_slot = flap_status(flap_slot, "rhobear-neo", nrestarts=5,
                                    now=now)
    assert should is True
    state.slot("open_alerts")["flap:rhobear-neo"] = "x"
    results = [checks.CheckResult("unit:rhobear-neo", checks.OK, "active")]
    # healthy passes go by (the detection window ages, no new restarts)...
    for t in (now + 300, now + 900):
        flap_slot = state.slot("flaps")
        should, flap_slot = flap_status(flap_slot, "rhobear-neo",
                                        nrestarts=5, now=t)
        assert should is False
        _resolve_fixed(cfg, state, results, None, lambda l: None,
                       now_fn=lambda: t)
        if t - now <= 600:
            assert "flap:rhobear-neo" in state.slot("open_alerts"), \
                "closed inside the quiet window"
    assert "flap:rhobear-neo" not in state.slot("open_alerts"), \
        "a settled unit's flap alert never closed"


def test_flap_detect_alert_resolves_when_the_probe_recovers(tmp_path,
                                                            monkeypatch):
    """check:flap-detect:<unit> has no CheckResult of its own — it closes when
    the unit is ok AND the restart probe succeeded again."""
    from src.kronos.__main__ import _resolve_fixed
    from src.kronos import KronosState
    cfg = _cfg()
    state = KronosState(":memory:")
    alerter = FakeAlerterR3()
    monkeypatch.setattr("src.kronos.__main__._alerter",
                        lambda cfg2, state2, now_fn, **kw: alerter)
    state.slot("open_alerts")["check:flap-detect:rhobear-neo"] = "x"
    results = [checks.CheckResult("unit:rhobear-neo", checks.OK, "active")]
    _resolve_fixed(cfg, state, results, None, lambda l: None,
                   now_fn=lambda: 1.0, flap_blind=set())
    assert "check:flap-detect:rhobear-neo" not in state.slot("open_alerts"), \
        "the flap-detect alert never closed"


# ---------------------------------------------------------------------------
# MEDIUM 3 — a failed restart probe raises, never reads as zero
# ---------------------------------------------------------------------------

def test_nrestarts_raises_on_a_failed_probe():
    def runner(argv, *, timeout=30):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="denied")

    with pytest.raises(RuntimeError):
        checks.nrestarts("rhobear-neo", runner=runner)


def test_nrestarts_raises_on_malformed_output():
    def runner(argv, *, timeout=30):
        return subprocess.CompletedProcess(argv, 0, stdout="active\n",
                                           stderr="")

    with pytest.raises(RuntimeError):
        checks.nrestarts("rhobear-neo", runner=runner)


# ---------------------------------------------------------------------------
# LOW 1 — the retired remedy configuration is gone
# ---------------------------------------------------------------------------

def test_remedy_window_config_is_gone():
    cfg = KronosConfig()
    assert getattr(cfg, "remedy_window_secs", None) is None, \
        "KRONOS_REMEDY_WINDOW_SECS is dead configuration (remedies are " \
        "retired) — it implies an auto-repair feature that does not exist"


def test_state_is_not_documented_as_a_remedy_ledger():
    from src.kronos import KronosState
    doc = KronosState.__doc__ or ""
    assert "remedy ledger" not in doc, doc
