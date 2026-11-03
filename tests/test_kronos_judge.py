"""W1-08-r2 — the judge (escalator) with fakes.

r2 changes (fix round on the r1 review):

- HIGH 6: the round-six ceiling and the Zeus/Kronos deconfliction are
  ATOMIC — the judge writes through a compare-and-set `writes` adapter
  (one advisory-locked transaction); six stored rounds are refused before
  Summit is ever asked, by the STORED round number, not the list length;
  a lost race (writes defer) never sets the judged marker and never
  comments, so Zeus's path owns the PR;
- HIGH 7: the installation token NEVER rides in a model message, and every
  model output string is redacted (fail closed) before it reaches the DB,
  a PR comment, an alert or a log;
- HIGH 2: the judge no longer writes through the read-only rows adapter —
  writes go through the `writes` adapter (SqlWrites against the real
  schema in tests/test_kronos_store_pg.py);
- MEDIUM 3: the Summit timeout is clamped by the caller to the remaining
  pass budget (the plumbing is pinned here, the clamp math in the CLI
  tests).

The DB, RouteGate HTTP, GitHub comments and the alerter are fakes.
"""
from __future__ import annotations

import json

from src.kronos import judge
from src.kronos.judge import (
    JUDGE_ROUNDS_TRIGGER,
    JudgeTimeout,
    ask_summit,
    parse_judge_answer,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeDb:
    """rows(sql, params) from a dict of canned tables. READ-ONLY: the judge
    may never write through this object again (that was HIGH 2)."""

    def __init__(self, rounds: list[dict] | None = None,
                 escalated: bool = False,
                 escalating: bool = False,
                 assessments: list[dict] | None = None,
                 merged: bool = False,
                 ready: bool = False):
        self.rounds = rounds or []
        self.escalated = escalated
        self.escalating = escalating
        self.assessments = assessments or []
        self.merged = merged
        self.ready = ready

    # read side -------------------------------------------------------------
    def rows(self, sql: str, params=()) -> list[dict]:
        s = " ".join(str(sql).split())
        if "phase='escalation-inflight'" in s:
            # r4 HIGH C — no Zeus in-flight marker in these fixtures; this
            # branch must come before the generic FROM neo_actions catch-all.
            return []
        if "phase='escalated'" in s or "phase = 'escalated'" in s:
            return [{"n": 1}] if self.escalated else []
        if "phase='escalating'" in s or "phase = 'escalating'" in s:
            return [{"repo": "o/r", "pr": 1}] if self.escalating else []
        if "phase = 'merged'" in s or "phase='merged'" in s:
            return [{"n": 1}] if self.merged else []
        if "phase = 'ready'" in s or "phase='ready'" in s:
            return [{"n": 1}] if self.ready else []
        if "FROM neo_actions" in s:
            return self.rounds
        if "FROM assessments" in s:
            return self.assessments
        return []


class FakeGithub:
    """api.github.com as the judge sees it (r3 HIGH 2): the PR's CURRENT
    state and its reviews, read-only. Defaults to an open PR on the same
    head the judge was given, with no reviews — the judging path."""

    def __init__(self, *, merged=False, state="open", head="f" * 40,
                 reviews=None):
        self.merged = merged
        self.state = state
        self.head = head
        self.reviews = reviews or []
        self.calls: list[dict] = []

    def __call__(self, method, url, *, headers=None, body=None, timeout=30):
        self.calls.append({"method": method, "url": url})
        if "/reviews" in url:
            return 200, json.dumps(self.reviews)
        if "/pulls/" in url:
            return 200, json.dumps({"state": self.state,
                                    "merged": self.merged, "draft": False,
                                    "head": {"sha": self.head}})
        return 200, "{}"


class FakeWrites:
    """The judge's ONLY writer: escalate/stop, each an atomic CAS that can
    defer (Zeus won the race). Records every call for assertions. Honours an
    injected authority callable exactly like src.kronos.store.SqlWrites."""

    def __init__(self, defer_reason: str | None = None):
        self.escalates: list[dict] = []
        self.stops: list[dict] = []
        self.defer_reason = defer_reason
        self.authority_results: list[str] = []

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


class FakeState:
    """The minimal KronosState shape the judge uses (no file)."""

    def __init__(self):
        self.data: dict = {}

    def slot(self, name):
        return self.data.setdefault(name, {})

    def flush(self):
        pass


class FakeComments:
    def __init__(self):
        self.posted: list[tuple[str, int, str]] = []

    def __call__(self, repo: str, pr: int, body: str):
        self.posted.append((repo, pr, body))


class FakeAlerter:
    def __init__(self):
        self.alerts: list[tuple[str, str, str]] = []

    def alert(self, fingerprint, subject, body):
        self.alerts.append((fingerprint, subject, body))
        return "created"

    def resolved(self, fingerprint, note):
        pass


def _cfg():
    from src.kronos.config import KronosConfig
    return KronosConfig(
        routegate_url="http://127.0.0.1:8645",
        judge_model="summit",
        judge_timeout_secs=20,
    )


def _rounds(n: int, tree: str = "") -> list[dict]:
    return [{"round": i, "head_sha": f"{i:04d}" * 10,
             "verdict": "BOUNCE", "tree_sha": tree or f"t{i}",
             "findings": f"round {i} findings"}
            for i in range(1, n + 1)]


class FakeHttp:
    def __init__(self, status: int = 200, content: str = "",
                 err: Exception | None = None):
        self.status = status
        self.content = content
        self.err = err
        self.calls: list[dict] = []

    def __call__(self, method, url, *, headers=None, body=None, timeout=30):
        self.calls.append({"method": method, "url": url, "headers": headers,
                           "body": body, "timeout": timeout})
        if self.err:
            raise self.err
        msg = {"choices": [{"message": {"content": self.content}}]}
        return self.status, json.dumps(msg)


GOOD_YES = json.dumps({"progress": "yes", "plan": "split the module and fix imports",
                       "reason": "converging structure, one more round is worth it"})
GOOD_NO = json.dumps({"progress": "no", "plan": "",
                      "reason": "the same tests fail in the same way each round"})


# ---------------------------------------------------------------------------
# the Summit call itself
# ---------------------------------------------------------------------------

def test_ask_summit_never_puts_a_token_in_the_messages():
    """HIGH 7 — the read-only installation token never travels in a model
    message (r1 put it in the user content; a model echo would leak it)."""
    http = FakeHttp(content=GOOD_YES)
    ask_summit(
        "http://127.0.0.1:8645", "summit", "REPORT", install_id="123456",
        read_token="ghs_AAAA0123456789abcdefgh", http=http, timeout=5)
    call = http.calls[0]
    assert call["url"] == "http://127.0.0.1:8645/v1/chat/completions"
    assert call["headers"]["X-RHOBEAR-Consumer"] == "reviews:kronos/123456"
    body = json.loads(call["body"])
    assert body["model"] == "summit"
    assert "REPORT" in body["messages"][-1]["content"]
    for msg in body["messages"]:
        assert "ghs_" not in str(msg.get("content")), (
            "an installation token reached a model message")


def test_parse_rejects_missing_fields_and_wrong_shape():
    assert parse_judge_answer('{"progress": "maybe"}') is None
    assert parse_judge_answer('not json at all') is None
    assert parse_judge_answer('{"progress": "no"}') is None, "reason required"
    assert parse_judge_answer('{"progress": "yes", "plan": "", "reason": "r"}') is None, \
        "a yes without a plan is a bad answer"
    ok = parse_judge_answer('{"progress": "no", "plan": "", "reason": "dead end"}')
    assert ok == {"progress": "no", "plan": "", "reason": "dead end"}


def test_timeout_raises_judge_timeout():
    http = FakeHttp(err=TimeoutError("slow"))
    try:
        ask_summit("http://127.0.0.1:8645", "summit", "R", install_id="1",
                   read_token="t", http=http, timeout=1)
        raise AssertionError("expected JudgeTimeout")
    except JudgeTimeout:
        pass


# ---------------------------------------------------------------------------
# the judge flow
# ---------------------------------------------------------------------------

def _run_judge(db: FakeDb, http, *, install_id="9999",
               state=None, writes=None, timeout=None, github=None):
    comments = FakeComments()
    alerter = FakeAlerter()
    state = state or FakeState()
    writes = writes if writes is not None else FakeWrites()
    out = judge.judge_pr(
        _cfg(), db, comments=comments, alerter=alerter, state=state,
        repo="o/r", pr=52, head_sha="f" * 40, install_id=install_id,
        http=http, now_fn=lambda: 1.0, writes=writes, timeout=timeout,
        github=github if github is not None else FakeGithub())
    return out, comments, alerter, state, writes


def test_five_rounds_trigger_the_judge():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_YES)
    out, *_rest = _run_judge(db, http)
    assert http.calls, "the judge never asked Summit"
    assert out.action == "escalated"


def test_fewer_than_five_rounds_never_judges():
    db = FakeDb(rounds=_rounds(4))
    http = FakeHttp(content=GOOD_YES)
    out, *_rest = _run_judge(db, http)
    assert http.calls == []
    assert out.action == "skip"


def test_yes_creates_exactly_one_escalated_round():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_YES)
    out, comments, alerter, state, writes = _run_judge(db, http)
    assert out.action == "escalated"
    assert len(writes.escalates) == 1, "ONE escalated Zeus round"
    call = writes.escalates[0]
    assert call["advisor_round"] == 6, "5 normal rounds + 1 escalated = 6"
    payload = call["builder_payload"]
    assert payload["escalated"] is True
    assert "split the module and fix imports" in json.dumps(payload)
    assert call["detail"]["advisor_round"] == 6
    assert any("judged this loop" in b for _, _, b in comments.posted)
    assert state.slot("judged"), "the judged marker is set only after the CAS won"


def test_no_stops_zeus_with_comment_and_alert():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_NO)
    out, comments, alerter, _state, writes = _run_judge(db, http)
    assert out.action == "stopped"
    assert writes.escalates == [], "a 'no' never creates a round"
    assert len(writes.stops) == 1
    assert writes.stops[0]["detail"].get("kronos_stop") is True, \
        "Zeus is stopped via the marker it already consumes"
    assert any("stopped" in b.lower() for _, _, b in comments.posted)
    assert alerter.alerts, "the owner is alerted"


def test_timeout_stops():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(err=TimeoutError("slow"))
    out, _comments, alerter, _state, writes = _run_judge(db, http)
    assert out.action == "stopped"
    assert writes.escalates == []
    assert alerter.alerts, "a timeout stops and tells the owner"


def test_bad_answer_stops():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content="I think it looks fine?!")
    out, *_rest = _run_judge(db, http)
    assert out.action == "stopped"


def test_judge_never_fires_twice_for_the_same_pr():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_YES)
    shared_state = FakeState()
    out1, *_ = _run_judge(db, http, state=shared_state)
    # a second pass on the same PR (state remembers the judgment)
    out2, _c2, _a2, _s2, writes2 = _run_judge(db, http, state=shared_state)
    assert out1.action == "escalated"
    assert out2.action == "skip"
    assert writes2.escalates == [], "never a second escalated round"


def test_judge_defers_while_zeus_advisor_path_is_escalating():
    """Zeus's own advisor escalation (src/escalation.py) owns the PR while an
    escalating marker exists — Kronos must not double-escalate."""
    db = FakeDb(rounds=_rounds(5), escalating=True)
    http = FakeHttp(content=GOOD_YES)
    out, *_ = _run_judge(db, http)
    assert out.action == "skip"
    assert http.calls == []


def test_judge_defers_when_already_escalated():
    db = FakeDb(rounds=_rounds(6), escalated=True)
    http = FakeHttp(content=GOOD_YES)
    out, *_ = _run_judge(db, http)
    assert out.action == "skip"


# ---------------------------------------------------------------------------
# HIGH 6 — the round-six ceiling and the CAS deconfliction
# ---------------------------------------------------------------------------

def test_six_stored_rounds_are_never_judged():
    """HIGH 6 — at six stored rounds every decision is refused BEFORE Summit
    is asked: 5 normal + 1 escalated = 6 is the ceiling, never a 7th."""
    db = FakeDb(rounds=_rounds(6))
    http = FakeHttp(content=GOOD_YES)
    out, _c, _a, _s, writes = _run_judge(db, http)
    assert http.calls == [], "Summit was asked at the ceiling"
    assert out.action == "skip"
    assert "6" in out.reason or "ceiling" in out.reason, out.reason
    assert writes.escalates == [] and writes.stops == []


def test_sparse_round_numbers_hit_the_ceiling_too():
    """HIGH 6 — the ceiling reads the STORED round number, not the list
    length: five rows whose rounds are 1,2,3,4,6 (a lost marker row) are at
    the ceiling and must be refused."""
    rounds = _rounds(5)
    rounds[-1]["round"] = 6
    db = FakeDb(rounds=rounds)
    http = FakeHttp(content=GOOD_YES)
    out, _c, _a, _s, writes = _run_judge(db, http)
    assert http.calls == [], "list length hid a stored round 6"
    assert out.action == "skip"
    assert writes.escalates == [] and writes.stops == []


def test_a_yes_at_five_rounds_targets_round_six_not_seven():
    """The escalated round is the stored MAX + 1 — with 5 rows that is 6,
    never len()+1 against a sparse history."""
    rounds = _rounds(5)
    rounds[-1]["round"] = 5
    rounds[0]["round"] = 1
    db = FakeDb(rounds=rounds)
    http = FakeHttp(content=GOOD_YES)
    _out, _c, _a, _s, writes = _run_judge(db, http)
    assert writes.escalates[0]["advisor_round"] == 6


def test_cas_deferral_writes_nothing_and_comments_nothing():
    """HIGH 6 — when the CAS defers (Zeus won the race between the read and
    the write), the judge must not set the judged marker, must not comment,
    must not alert as if IT had decided — and must not raise."""
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_YES)
    state = FakeState()
    writes = FakeWrites(defer_reason="deferred: Zeus advisor escalation is in flight")
    out, comments, alerter, state, _w = _run_judge(
        db, http, state=state, writes=writes)
    assert out.action == "deferred", out
    assert len(writes.escalates) == 1, "exactly ONE write attempt"
    assert state.slot("judged") == {}, "a lost race must not park the PR"
    assert comments.posted == [], "a lost race must not comment"
    assert alerter.alerts == [], "a lost race must not alert"


def test_stop_cas_deferral_writes_nothing():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_NO)
    state = FakeState()
    writes = FakeWrites(defer_reason="deferred: already escalated or stopped")
    out, comments, alerter, state, _w = _run_judge(
        db, http, state=state, writes=writes)
    assert out.action == "deferred"
    assert state.slot("judged") == {}
    assert comments.posted == []
    assert alerter.alerts == []


# ---------------------------------------------------------------------------
# HIGH 7 — every model output string is redacted before it lands anywhere
# ---------------------------------------------------------------------------

def test_model_output_is_redacted_before_db_comment_alert_and_log(caplog):
    """HIGH 7 — Summit echoes a token in the reason: no model text carrying a
    token shape may reach the DB detail, the PR comment, the owner alert,
    the returned outcome or the log."""
    import logging
    echoed = json.dumps({
        "progress": "no",
        "plan": "",
        "reason": "leaked ghs_AAAA0123456789abcdefgh while reading the diff",
    })
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=echoed)
    with caplog.at_level(logging.WARNING, logger="rhobear_neo.kronos.judge"):
        out, comments, alerter, _state, writes = _run_judge(db, http)
    assert out.action == "stopped"
    surfaces = json.dumps(writes.stops) + json.dumps(alerter.alerts) + \
        json.dumps(comments.posted) + out.reason + caplog.text
    assert "ghs_AAAA0123456789abcdefgh" not in surfaces, surfaces


def test_escalated_plan_is_redacted_too():
    echoed = json.dumps({
        "progress": "yes",
        "plan": "first paste ghs_BBBB0123456789abcdefgh then split the module",
        "reason": "converging",
    })
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=echoed)
    out, comments, _a, _s, writes = _run_judge(db, http)
    assert out.action == "escalated"
    surfaces = json.dumps(writes.escalates) + json.dumps(comments.posted)
    assert "ghs_BBBB0123456789abcdefgh" not in surfaces, surfaces
    assert "split the module" in json.dumps(writes.escalates), \
        "the harmless plan text survives the redaction"


def test_redaction_failure_fails_closed():
    """If the redactor itself fails, NO model text may reach any surface:
    the judge stops with a fixed literal reason instead. This exercises the
    REAL chain: src.redact.redact_data fails -> _redact converts ->
    judge_pr fails closed."""
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_NO)
    state = FakeState()
    writes = FakeWrites()

    import src.redact as redact_mod

    def boom(_value):
        raise RuntimeError("redactor exploded")

    original = redact_mod.redact_data
    redact_mod.redact_data = boom
    try:
        out, comments, alerter, state, writes2 = _run_judge(
            db, http, state=state, writes=writes)
    finally:
        redact_mod.redact_data = original
    assert out.action == "stopped"
    assert writes2.stops, "the fail-closed stop still parks the PR"
    surfaces = json.dumps(writes2.stops) + json.dumps(alerter.alerts) + \
        json.dumps(comments.posted) + out.reason
    assert "converging structure" not in surfaces, \
        "unredactable model text reached a surface"
    assert "judge answer withheld" in out.reason, out.reason


# ---------------------------------------------------------------------------
# MEDIUM 3 — the Summit timeout the caller clamps
# ---------------------------------------------------------------------------

def test_judge_passes_the_clamped_timeout_to_summit():
    """The sweep clamps the judge's timeout to the remaining pass budget;
    judge_pr must forward it to the RouteGate call verbatim."""
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_YES)
    _run_judge(db, http, timeout=37)
    assert http.calls[0]["timeout"] == 37, http.calls[0]["timeout"]


def test_trigger_threshold_is_five():
    assert JUDGE_ROUNDS_TRIGGER == 5


def test_judge_report_carries_verdicts_findings_and_diff_stats():
    rounds = _rounds(5)
    rounds[1]["diff_stat"] = "3 files changed, +120/-40"
    db = FakeDb(rounds=rounds,
                assessments=[{"repo": "o/r", "pr_number": 52,
                              "head_sha": rounds[0]["head_sha"],
                              "data": {"findings": [
                                  {"severity": "HIGH",
                                   "title": "sql injection"}]}}])
    http = FakeHttp(content=GOOD_YES)
    _run_judge(db, http)
    sent = json.loads(http.calls[0]["body"])["messages"][-1]["content"]
    assert "BOUNCE" in sent, "verdicts travel in the report"
    assert "sql injection" in sent, "Apollo's findings travel"
    assert "round 5" in sent, "the round history travels"
    assert "3 files changed, +120/-40" in sent, "diff stats travel"
