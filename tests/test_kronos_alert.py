"""W1-08-r1 — GitHub issue alerts with fakes (acceptance).

- never the same fingerprint twice inside 6 hours; a fresh alert after the
  window is a comment on the SAME issue (one issue per problem);
- commented and closed when fixed;
- re-opened as a comment when the problem returns after a close.
"""
from __future__ import annotations

from src.kronos.alert import Alerter


class FakeIssues:
    def __init__(self):
        self.created: list[dict] = []
        self.comments: list[tuple[int, str]] = []
        self.closed: list[int] = []
        self._next = 100

    def search_open(self, fingerprint: str):
        for issue in self.created:
            if issue["fingerprint"] == fingerprint and issue["open"]:
                return issue
        return None

    def create(self, title: str, body: str, labels: list[str]):
        self._next += 1
        import re
        m = re.search(r"<!-- kronos-fingerprint: (.+?) -->", body)
        fp = m.group(1) if m else ""
        issue = {"number": self._next, "title": title, "body": body,
                 "labels": labels, "open": True, "fingerprint": fp}
        self.created.append(issue)
        return issue

    def comment(self, number: int, body: str):
        self.comments.append((number, body))

    def close(self, number: int, note: str):
        if note:
            self.comments.append((number, note))
        for issue in self.created:
            if issue["number"] == number:
                issue["open"] = False
        self.closed.append(number)


def _alerter(now: list[float]):
    from src.kronos import KronosState
    issues = FakeIssues()
    state = KronosState(":memory:")
    alerter = Alerter(issues, state, now_fn=lambda: now[0],
                      window_secs=6 * 3600)
    return alerter, issues


def test_same_fingerprint_not_alerted_twice_within_six_hours():
    now = [1000.0]
    alerter, issues = _alerter(now)
    fp = "unit-down:rhobear-neo"
    assert alerter.alert(fp, "down", "failed (exit 1)") == "created"
    now[0] += 3600  # one hour later
    assert alerter.alert(fp, "down", "failed (exit 1)") == "suppressed"
    assert len(issues.created) == 1, "never the same alert twice in 6 hours"


def test_alert_after_window_comments_on_the_same_issue():
    now = [1000.0]
    alerter, issues = _alerter(now)
    fp = "unit-down:rhobear-neo"
    alerter.alert(fp, "down", "failed (exit 1)")
    now[0] += 6 * 3600 + 1
    assert alerter.alert(fp, "down", "failed (exit 1) again") == "commented"
    assert len(issues.created) == 1, "one issue per problem, ever"
    assert issues.comments, "the repeat arrived as a comment"


def test_resolved_comments_and_closes():
    now = [1000.0]
    alerter, issues = _alerter(now)
    fp = "unit-down:rhobear-neo"
    alerter.alert(fp, "down", "failed (exit 1)")
    alerter.resolved(fp, "active again")
    assert issues.closed == [issues.created[0]["number"]]
    assert any("active again" in b for _, b in issues.comments)


def test_resolved_clears_the_dedupe_window():
    now = [1000.0]
    alerter, issues = _alerter(now)
    fp = "unit-down:rhobear-neo"
    alerter.alert(fp, "down", "failed (exit 1)")
    alerter.resolved(fp, "active again")
    now[0] += 60  # well inside the 6h window
    assert alerter.alert(fp, "down", "failed (exit 1)") == "created", (
        "a fix resets the problem: the next occurrence alerts immediately"
    )
    assert len(issues.created) == 2


def test_different_fingerprints_are_different_issues():
    now = [1000.0]
    alerter, issues = _alerter(now)
    alerter.alert("unit-down:rhobear-neo", "down", "x")
    alerter.alert("unit-down:rhobear-reviews", "down", "y")
    assert len(issues.created) == 2


def test_issue_body_carries_the_fingerprint_marker():
    now = [1000.0]
    alerter, issues = _alerter(now)
    alerter.alert("credits:install-42", "credits", "gated")
    body = issues.created[0]["body"]
    assert body.startswith("<!-- kronos-fingerprint: credits:install-42 -->")


def test_alerts_go_to_the_reviews_app_repo_with_the_kronos_label():
    from src.kronos.alert import ALERT_LABEL, ALERT_REPO
    assert ALERT_REPO == "deariencampbell1-sys/rhobear-reviews-app"
    assert ALERT_LABEL == "kronos"
