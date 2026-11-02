"""W1-08-r1 — the judge (escalator) with fakes (acceptance).

- 5 Zeus fix rounds without two greens triggers the judge: one Summit call
  through RouteGate with consumer `reviews:kronos/<install_id>`, a READ-ONLY
  installation token in the payload, and the bounded round history;
- `progress: yes` -> exactly ONE escalated Zeus round whose brief is the plan
  (never a 7th round — 5 normal + 1 escalated = 6 max);
- `progress: no`, a bad answer, or a timeout -> Zeus stops on that PR (the
  marker Zeus already consumes), one PR comment with the reason, owner alert;
- the judge never fires twice for the same PR.

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
    """rows(sql, params) from a dict of canned tables."""

    def __init__(self, rounds: list[dict] | None = None,
                 escalated: bool = False,
                 escalating: bool = False,
                 assessments: list[dict] | None = None):
        self.rounds = rounds or []
        self.escalated = escalated
        self.escalating = escalating
        self.assessments = assessments or []
        self.recorded: list[dict] = []
        self.builder_jobs: list[dict] = []

    # read side -------------------------------------------------------------
    def rows(self, sql: str, params=()) -> list[dict]:
        if "FROM neo_actions" in sql and "phase='escalated'" in sql:
            return [{"n": 1}] if self.escalated else []
        if "phase='escalating'" in sql:
            return [{"repo": "o/r", "pr": 1}] if self.escalating else []
        if "FROM neo_actions" in sql:
            return self.rounds
        if "FROM assessments" in sql:
            return self.assessments
        return []

    # write side (the two things the judge may write) ------------------------
    def record(self, repo, pr, head_sha, phase, **kw):
        self.recorded.append({"repo": repo, "pr": pr, "head_sha": head_sha,
                              "phase": phase, **kw})

    def create_builder(self, repo, pr, head_sha, builder_round, *, payload=None,
                       **kw):
        self.builder_jobs.append({"repo": repo, "pr": pr, "round": builder_round,
                                  "payload": payload})
        return 55


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

def test_ask_summit_uses_routegate_consumer_and_readonly_token():
    http = FakeHttp(content=GOOD_YES)
    answer = ask_summit(
        "http://127.0.0.1:8645", "summit", "REPORT", install_id="123456",
        read_token="ght_ro_token", http=http, timeout=5)
    call = http.calls[0]
    assert call["url"] == "http://127.0.0.1:8645/v1/chat/completions"
    assert call["headers"]["X-RHOBEAR-Consumer"] == "reviews:kronos/123456"
    body = json.loads(call["body"])
    assert body["model"] == "summit"
    assert "REPORT" in body["messages"][-1]["content"]
    assert "ght_ro_token" in body["messages"][-1]["content"], \
        "the read-only token rides the payload so Summit can read the PR"
    assert answer == {"progress": "yes",
                      "plan": "split the module and fix imports",
                      "reason": "converging structure, one more round is worth it"}


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

def _run_judge(db: FakeDb, http, *, install_id="9999", read_token="ro-token"):
    comments = FakeComments()
    alerter = FakeAlerter()
    state = FakeState()
    out = judge.judge_pr(
        _cfg(), db, comments=comments, alerter=alerter, state=state,
        repo="o/r", pr=52, head_sha="f" * 40, install_id=install_id,
        read_token=read_token, http=http, now_fn=lambda: 1.0)
    return out, comments, alerter, state


def test_five_rounds_trigger_the_judge():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_YES)
    out, comments, alerter, state = _run_judge(db, http)
    assert http.calls, "the judge never asked Summit"
    assert out.action == "escalated"


def test_fewer_than_five_rounds_never_judges():
    db = FakeDb(rounds=_rounds(4))
    http = FakeHttp(content=GOOD_YES)
    out, comments, alerter, _ = _run_judge(db, http)
    assert http.calls == []
    assert out.action == "skip"


def test_yes_creates_exactly_one_escalated_round():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_YES)
    out, comments, alerter, _ = _run_judge(db, http)
    assert out.action == "escalated"
    assert len(db.builder_jobs) == 1, "ONE escalated Zeus round"
    job = db.builder_jobs[0]
    assert job["round"] == 6, "5 normal rounds + 1 escalated = 6, never a 7th"
    assert job["payload"]["escalated"] is True
    assert "split the module and fix imports" in json.dumps(job["payload"])
    esc = [r for r in db.recorded if r["phase"] == "escalated"]
    assert len(esc) == 1
    assert esc[0]["detail"]["advisor_round"] == 6
    assert any("judged this loop" in b for _, _, b in comments.posted)


def test_no_stops_zeus_with_comment_and_alert():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_NO)
    out, comments, alerter, _ = _run_judge(db, http)
    assert out.action == "stopped"
    assert db.builder_jobs == [], "a 'no' never creates a round"
    stops = [r for r in db.recorded if r["phase"] == "escalated"
             and r["detail"].get("kronos_stop")]
    assert stops, "Zeus is stopped via the marker it already consumes"
    assert any("stopped" in b.lower() for _, _, b in comments.posted)
    assert alerter.alerts, "the owner is alerted"


def test_timeout_stops():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(err=TimeoutError("slow"))
    out, comments, alerter, _ = _run_judge(db, http)
    assert out.action == "stopped"
    assert db.builder_jobs == []
    assert alerter.alerts, "a timeout stops and tells the owner"


def test_bad_answer_stops():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content="I think it looks fine?!")
    out, comments, alerter, _ = _run_judge(db, http)
    assert out.action == "stopped"
    assert db.builder_jobs == []


def test_judge_never_fires_twice_for_the_same_pr():
    db = FakeDb(rounds=_rounds(5))
    http = FakeHttp(content=GOOD_YES)
    out1, *_ = _run_judge(db, http)
    # a second pass on the same PR (state remembers the judgment)
    out2, comments2, alerter2, _ = _run_judge(db, http)
    assert out2.action == "skip"
    assert len(db.builder_jobs) == 1, "never a second escalated round"


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


def test_trigger_threshold_is_five():
    assert JUDGE_ROUNDS_TRIGGER == 5


def test_judge_report_carries_verdicts_findings_and_diff_stats():
    db = FakeDb(rounds=_rounds(3),
                assessments=[{"repo": "o/r", "pr_number": 52,
                              "head_sha": _rounds(3)[0]["head_sha"],
                              "findings": "3 HIGH findings: sql injection"}])
    http = FakeHttp(content=GOOD_YES)
    _run_judge(db, http)
    sent = json.loads(http.calls[0]["body"])["messages"][-1]["content"]
    assert "BOUNCE" in sent, "verdicts travel in the report"
    assert "3 HIGH findings: sql injection" in sent, "Apollo's findings travel"
    assert "round 2 findings" in sent, "the round history travels"
