"""Kronos alerts — one GitHub issue per problem, told once, told when fixed.

Telegram is retired (owner 2026-06-17): alerts are GitHub issues in the
reviews-app repo (`deariencampbell1-sys/rhobear-reviews-app`), label
`kronos`, opened AS THE ZEUS APP so the owner gets GitHub's phone push.

One issue PER PROBLEM: the fingerprint (a stable problem key like
`unit-down:rhobear-neo` or `loop:o/r#52`) rides in the issue body as a
hidden marker, so a repeat of the same problem comments on the same issue
instead of opening another. Fixed -> comment + close, and the dedupe window
clears (a fix resets the problem). Never the same fingerprint twice inside
6 hours.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import quote

ALERT_REPO = "deariencampbell1-sys/rhobear-reviews-app"
ALERT_LABEL = "kronos"
API_BASE = "https://api.github.com"


def fingerprint_marker(fingerprint: str) -> str:
    return f"<!-- kronos-fingerprint: {fingerprint} -->"


class GithubIssueClient:
    """Issues API as an installation token (the Zeus App on the alert repo).

    `token` is a callable -> the current installation token (Kronos mints
    and refreshes it through src.github_app). No credential is ever logged;
    failures surface as exceptions with the status code only.
    """

    def __init__(self, repo: str, token, *, label: str = ALERT_LABEL,
                 http=None):
        self.repo = repo
        self._token = token
        self.label = label
        self._http = http or self._default_http

    @staticmethod
    def _default_http(method: str, url: str, *, headers: dict, body=None,
                      timeout: int = 30):
        req = urllib.request.Request(url, data=body, method=method,
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    def _call(self, method: str, path: str, body: dict | None = None):
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token()}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if payload:
            headers["Content-Type"] = "application/json"
        status, raw = self._http(method, f"{API_BASE}{path}",
                                 headers=headers, body=payload)
        if status >= 300:
            raise RuntimeError(f"github {method} {path} -> HTTP {status}")
        return json.loads(raw) if raw.strip() else {}

    def search_open(self, fingerprint: str):
        """The open issue carrying this fingerprint, or None (label-filtered
        list, marker scanned in-body — no search endpoint, no new scope)."""
        data = self._call(
            "GET",
            f"/repos/{self.repo}/issues?labels={quote(self.label)}"
            f"&state=open&per_page=100&sort=updated&direction=desc")
        marker = fingerprint_marker(fingerprint)
        for issue in data or []:
            if marker in (issue.get("body") or ""):
                return issue
        return None

    def create(self, title: str, body: str, labels: list[str]):
        return self._call("POST", f"/repos/{self.repo}/issues",
                          {"title": title, "body": body, "labels": labels})

    def comment(self, number: int, body: str):
        return self._call("POST",
                          f"/repos/{self.repo}/issues/{number}/comments",
                          {"body": body})

    def close(self, number: int, note: str):
        if note:
            self.comment(number, note)
        return self._call("PATCH", f"/repos/{self.repo}/issues/{number}",
                          {"state": "closed"})


class Alerter:
    """Dedupe + issue lifecycle over any client with the FakeIssues shape
    used by the tests (search_open / create / comment / close)."""

    def __init__(self, client, state, *, now_fn, window_secs: int = 6 * 3600,
                 label: str = ALERT_LABEL):
        self._client = client
        self._window = state.slot("alert_windows")
        self._now = now_fn
        self._window_secs = window_secs
        self._label = label

    def alert(self, fingerprint: str, subject: str, body: str) -> str:
        """'created' | 'commented' | 'suppressed' — never two issues per
        problem, never two alerts inside the window."""
        last = self._window.get(fingerprint)
        if last is not None:
            try:
                if self._now() - float(last) < self._window_secs:
                    return "suppressed"
            except (TypeError, ValueError):
                pass
        issue = self._client.search_open(fingerprint)
        full_body = f"{fingerprint_marker(fingerprint)}\n\n{body}"
        if issue is None:
            self._client.create(subject, full_body, [self._label])
            action = "created"
        else:
            self._client.comment(issue["number"], body)
            action = "commented"
        self._window[fingerprint] = self._now()
        return action

    def resolved(self, fingerprint: str, note: str) -> None:
        """Commented and closed when fixed; the window clears so a fresh
        occurrence alerts immediately."""
        issue = self._client.search_open(fingerprint)
        if issue is not None:
            self._client.close(issue["number"], f"Fixed: {note}")
        self._window.pop(fingerprint, None)
