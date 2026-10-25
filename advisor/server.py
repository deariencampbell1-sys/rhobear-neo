"""Advisor HTTP server — bound to the tailnet address.

Receives escalation requests from Neo at POST /v1/advise:

  Body: {repo, pr, head_sha, rounds: [...], diff_stat, read_token}
  Auth: Bearer <NEO_ADVISOR_TOKEN>

  Returns 202 {"advice_id": "<uuid>"} immediately.  The agent runs in a
  background thread (one agent at a time; at most 3 queued, then 429).

  GET /v1/advise/<advice_id> (same bearer) returns
    {status: queued|running|done|failed, work_order?, error?}
  Results are kept 24 h.

Flow behind a submission:
  1. Check bearer token.
  2. Enqueue the request (one at a time; max 3 queued, then 429).
  3. Clone the repo read-only at head_sha using read_token via credential
     helper pattern (never in URL or .git/config).
  4. Run a read-only frontier agent with a 20-minute timeout.
     First available of ADVISOR_ROUTES:
       - claude: claude -p --model opus --permission-mode plan
       - codex: codex exec -m <model> -c model_reasoning_effort=<effort>
  5. Produce a work order (strict JSON; one repair retry, then fail).
  6. Delete the temp dir.

Deployed on the build box as a capped systemd service (not on prod).  The
advisor NEVER pushes, comments, or merges.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("rhobear_neo.advisor.server")

# Advice records are kept for 24 h, then pruned.
RECORD_TTL_SECS = 24 * 60 * 60


class AdvisorError(Exception):
    """Base for advisor failures."""


class AuthFailed(AdvisorError):
    """Bearer token mismatch."""


class QueueFull(AdvisorError):
    """Too many queued requests."""


class AgentFailed(AdvisorError):
    """The frontier agent failed or returned malformed output."""


def _parse_route(route_spec: str) -> tuple[str, dict]:
    """Parse a route spec like 'claude:opus' or 'codex:gpt-6.1-sol:high'.

    Returns (engine, params) where engine is 'claude' or 'codex'.
    """
    parts = route_spec.split(":")
    engine = parts[0].strip().lower()
    params = {}
    if engine == "claude":
        params["model"] = parts[1].strip() if len(parts) > 1 else "opus"
    elif engine == "codex":
        params["model"] = parts[1].strip() if len(parts) > 1 else "gpt-6.1-sol"
        params["effort"] = parts[2].strip() if len(parts) > 2 else "high"
    return engine, params


def _check_cli(engine: str) -> bool:
    """Check if a CLI tool is available on PATH."""
    try:
        result = subprocess.run(
            ["which", engine],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


_AGENT_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR")


def _agent_env() -> dict[str, str]:
    """A minimal environment for the frontier agent subprocess.

    The brief embeds PR-derived text (diff stat, prior-round findings), so a
    prompt-injected PR could ask the agent to read /proc/self/environ.  Pass
    only non-secret process basics — never NEO_*, GH_TOKEN, GITHUB_*,
    DEEPSEEK_*, ANTHROPIC_*, OPENROUTER_*, or any *_KEY/*_TOKEN variable.
    The CLI resolves its own provider credentials from HOME/config.
    """
    defaults = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": os.path.expanduser("~") or "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TERM": "dumb",
        "TMPDIR": tempfile.gettempdir(),
    }
    return {key: (os.environ.get(key) or defaults[key])
            for key in _AGENT_ENV_KEYS}


def _run_agent(engine: str, params: dict, brief: str, timeout: int = 1200,
               cwd: str | None = None) -> str:
    """Run a read-only frontier agent and return its raw output.

    `cwd` is the read-only checkout at head_sha (the agent's working
    directory), so the agent's HEAD equals the PR head it was asked about.

    Engine-specific commands:
      - claude: claude -p <brief> --model <model> --permission-mode plan
      - codex: codex exec -m <model> -c model_reasoning_effort=<effort>
               --sandbox read-only <brief>
    """
    if engine == "claude":
        cmd = [
            "claude", "-p", brief,
            "--model", params.get("model", "opus"),
            "--permission-mode", "plan",
        ]
    elif engine == "codex":
        model = params.get("model", "gpt-6.1-sol")
        effort = params.get("effort", "high")
        cmd = [
            "codex", "exec", "-m", model,
            "-c", f"model_reasoning_effort={effort}",
            "--sandbox", "read-only",
            brief,
        ]
    else:
        raise AgentFailed(f"unknown engine: {engine}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout,
            cwd=cwd,
            env=_agent_env(),
        )
    except subprocess.TimeoutExpired:
        raise AgentFailed(f"agent timed out after {timeout}s")
    except FileNotFoundError:
        raise AgentFailed(f"CLI not found: {cmd[0]}")

    if proc.returncode != 0:
        raise AgentFailed(
            f"{engine} exited {proc.returncode}: "
            f"{proc.stderr[:500] or proc.stdout[:500]}"
        )

    return (proc.stdout or "").strip()


def _parse_work_order(raw: str) -> dict:
    """Parse the agent's output as a JSON work order.

    Attempts one repair retry (strip markdown fences / extract embedded JSON)
    before failing.
    """
    # First attempt: direct JSON parse.
    for attempt in range(2):
        text = raw.strip()
        # Strip markdown code fences if present.
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
            text = text.strip("`").strip()
        try:
            data = json.loads(text)
            # Validate required fields.
            assert isinstance(data, dict), "not a dict"
            assert "summary" in data, "missing summary"
            assert "root_cause" in data, "missing root_cause"
            steps = data.get("steps")
            if not isinstance(steps, list) or not all(
                    isinstance(s, dict) for s in steps):
                raise AgentFailed("work order steps must be a list of objects")
            return data
        except (json.JSONDecodeError, AssertionError, TypeError) as exc:
            if attempt == 0:
                # Repair: try to extract JSON from the output.
                # Find the first { and last }.
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    raw = text[start:end + 1]
                    continue
            raise AgentFailed(f"cannot parse work order: {exc}")

    raise AgentFailed("cannot parse work order after repair attempt")


def _valid_advise_request(repo, pr, head_sha, read_token) -> bool:
    """Validate the request fields that become git argv / URL components.

    Fail closed: a 40-char lowercase hex head SHA (git would otherwise parse
    a `--`-prefixed value as an option after the remote), an ``owner/name``
    repo, a positive JSON int PR (bools are ints in Python — reject them),
    and a non-empty string read token.
    """
    return (
        isinstance(repo, str) and _REPO_RE.fullmatch(repo) is not None
        and isinstance(pr, int) and not isinstance(pr, bool) and pr > 0
        and isinstance(head_sha, str)
        and _HEAD_SHA_RE.fullmatch(head_sha) is not None
        and isinstance(read_token, str) and bool(read_token.strip())
    )


def _clone_head(tmpdir: str, repo: str, head_sha: str, read_token: str,
                origin_url: str | None = None) -> str:
    """Clone `repo` read-only and check out `head_sha` (detached).

    `git init <dir>`; `git -C <dir> fetch --depth 1 origin <head_sha>` with the
    read token via the env credential helper (never in a URL or .git/config);
    `git -C <dir> checkout --detach FETCH_HEAD`.  Every returncode is checked.

    A failure raises AgentFailed with the git stderr (token redacted).  The
    agent's working directory is the checkout, so its HEAD equals `head_sha`
    regardless of the remote's default branch.

    `origin_url` overrides the origin URL (tests point it at a local bare
    repo); default is the GitHub URL with NO token in it.
    """
    repo_url = origin_url or f"https://github.com/{repo}.git"
    clone_dir = os.path.join(tmpdir, "repo")

    # Credential helper: GIT_ASKPASS prints the read_token from the child
    # environment.  The token never appears in the URL, .git/config, or the
    # generated script.
    askpass_path = os.path.join(tmpdir, "git-askpass.sh")
    with open(askpass_path, "w") as f:
        f.write("#!/bin/sh\n")
        f.write("printf '%s\\n' \"$NEO_ADVISOR_GIT_TOKEN\"\n")
    os.chmod(askpass_path, 0o700)
    env = {**os.environ, "GIT_ASKPASS": askpass_path,
           "GIT_TERMINAL_PROMPT": "0",
           "NEO_ADVISOR_GIT_TOKEN": read_token}

    steps = [
        ["git", "init", clone_dir],
        ["git", "-C", clone_dir, "remote", "add", "origin", repo_url],
        ["git", "-C", clone_dir, "fetch", "--depth", "1", "origin", head_sha],
        ["git", "-C", clone_dir, "checkout", "--detach", "FETCH_HEAD"],
    ]
    for step in steps:
        try:
            proc = subprocess.run(
                step, capture_output=True, text=True, timeout=120, env=env,
            )
        except subprocess.TimeoutExpired:
            raise AgentFailed(f"git {' '.join(step)} timed out")
        if proc.returncode != 0:
            stderr = (proc.stderr or "").replace(read_token, "[REDACTED]")
            raise AgentFailed(
                f"git {' '.join(step)} failed: {stderr[:500]}"
            )
    return clone_dir


def run_advisor(
    repo: str,
    pr: int,
    head_sha: str,
    rounds: list[dict],
    diff_stat: str,
    read_token: str,
    routes: tuple[str, ...],
    timeout: int = 1200,
    origin_url: str | None = None,
) -> dict:
    """Run the advisor for one escalation request.

    Clones the repo read-only at head_sha, runs the first available frontier
    agent with cwd = the checkout, and returns a work order.

    The read_token is used via git credential helper pattern (never in URL
    or .git/config).
    """
    with tempfile.TemporaryDirectory(prefix="advisor-") as tmpdir:
        clone_dir = _clone_head(tmpdir, repo, head_sha, read_token,
                                origin_url=origin_url)

        # --- Build the brief for the frontier agent ---------------------
        brief = _build_advisor_brief(repo, pr, head_sha, rounds, diff_stat, clone_dir)

        # --- Run the first available route ------------------------------
        last_error = None
        for route_spec in routes:
            engine, params = _parse_route(route_spec)
            if not _check_cli(engine):
                log.info("route %s unavailable (CLI not found), trying next", route_spec)
                continue
            try:
                raw = _run_agent(engine, params, brief, timeout=timeout,
                                 cwd=clone_dir)
                work_order = _parse_work_order(raw)
                work_order["route"] = route_spec
                return work_order
            except AgentFailed as exc:
                last_error = exc
                log.warning("route %s failed: %s, trying next", route_spec, exc)
                continue

        raise AgentFailed(
            f"all routes exhausted: {last_error or 'no routes available'}"
        )


def _build_advisor_brief(
    repo: str,
    pr: int,
    head_sha: str,
    rounds: list[dict],
    diff_stat: str,
    clone_dir: str,
) -> str:
    """Build the brief for the frontier advisor agent.

    The advisor is a READ-ONLY agent that analyzes the code and produces
    a work order.  It never edits, pushes, comments, or merges.
    """
    rounds_text = "\n".join(
        f"  Round {r.get('round', '?')}: SHA={str(r.get('sha', ''))[:12]} "
        f"outcome={r.get('outcome', '?')} "
        f"findings={r.get('findings', '')[:200]}"
        for r in rounds
    )

    return f"""You are a READ-ONLY frontier advisor for RHOBEAR's Neo merge gate.

CONTEXT:
  Repo: {repo}
  PR: #{pr}
  Head SHA: {head_sha[:12]}
  Diff stat: {diff_stat or '(not available)'}

PRIOR BUILDER ROUNDS:
{rounds_text or '  (none)'}

You have a read-only checkout of the repo at {clone_dir}.
You may read files, search the codebase, and analyze the diff.
You MUST NOT edit, push, comment, or merge anything.

TASK:
Analyze the PR and the prior builder rounds.  Determine why the builder
has been unable to fix the issues, and produce a work order that tells
the Builder EXACTLY what to do.

Your output must be STRICT JSON (no markdown, no prose wrapper):
{{
  "summary": "one-line summary of the root problem",
  "root_cause": "detailed root cause analysis",
  "steps": [
    {{"file": "path/to/file.py", "change": "what to change", "why": "why this change"}},
    ...
  ],
  "tests": ["test_path::test_name", ...],
  "risks": "potential risks or side effects"
}}

Be specific.  Include file paths, line numbers, and exact changes.
This will be given directly to a Builder agent as its task brief.
"""


class AdvisorService:
    """Holds advice records, the wait queue, and the active flag.

    One agent runs at a time; at most `queue_max` wait, then QueueFull (429).
    Records are kept 24 h.  Thread-safe: submit/status/_next all use `_lock`.
    """

    def __init__(self, routes: tuple[str, ...] = ("claude:opus",),
                 timeout: int = 1200, queue_max: int = 3,
                 origin_url: str | None = None) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, dict] = {}
        self._queue: list[str] = []
        self._active = False
        self.routes = routes
        self.timeout = timeout
        self.queue_max = queue_max
        self.origin_url = origin_url

    # ------------------------------------------------------------------
    def submit(self, payload: dict) -> str:
        """Queue or start the advisor for `payload`.

        Returns the advice_id.  Raises QueueFull when already active and the
        wait queue is full.  Never blocks on the agent.
        """
        advice_id = uuid.uuid4().hex
        with self._lock:
            self._prune_locked()
            if self._active:
                if len(self._queue) >= self.queue_max:
                    raise QueueFull("queue full")
                self._records[advice_id] = {
                    "status": "queued",
                    "payload": payload,
                    "work_order": None,
                    "error": "",
                    "created_at": time.time(),
                }
                self._queue.append(advice_id)
                return advice_id
            self._active = True
            self._records[advice_id] = {
                "status": "running",
                "payload": payload,
                "work_order": None,
                "error": "",
                "created_at": time.time(),
            }
        threading.Thread(target=self._worker, args=(advice_id, payload),
                         daemon=True).start()
        return advice_id

    def status(self, advice_id: str) -> dict | None:
        """The record for `advice_id`, or None when unknown/expired."""
        with self._lock:
            self._prune_locked()
            rec = self._records.get(advice_id)
            if rec is None:
                return None
            return {
                "status": rec["status"],
                "work_order": rec.get("work_order"),
                "error": rec.get("error", ""),
            }

    # ------------------------------------------------------------------
    def _worker(self, advice_id: str, payload: dict) -> None:
        """Run the agent for one advice and record the outcome."""
        try:
            work_order = run_advisor(
                str(payload.get("repo", "")),
                int(payload.get("pr", 0) or 0),
                str(payload.get("head_sha", "")),
                payload.get("rounds") or [],
                str(payload.get("diff_stat", "") or ""),
                str(payload.get("read_token", "") or ""),
                routes=self.routes,
                timeout=self.timeout,
                origin_url=self.origin_url,
            )
            with self._lock:
                rec = self._records.get(advice_id)
                if rec is not None:
                    rec["status"] = "done"
                    rec["work_order"] = work_order
                    rec["error"] = ""
        except AgentFailed as exc:
            log.warning("advisor failed %s: %s", advice_id, exc)
            with self._lock:
                rec = self._records.get(advice_id)
                if rec is not None:
                    rec["status"] = "failed"
                    rec["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            log.exception("advisor worker crashed %s", advice_id)
            with self._lock:
                rec = self._records.get(advice_id)
                if rec is not None:
                    rec["status"] = "failed"
                    rec["error"] = f"internal error: {type(exc).__name__}"
        finally:
            self._next()

    def _next(self) -> None:
        """Start the next queued advice, or free the active slot."""
        payload = None
        advice_id = None
        with self._lock:
            self._prune_locked()
            while self._queue:
                candidate = self._queue.pop(0)
                rec = self._records.get(candidate)
                if rec is not None:
                    advice_id = candidate
                    rec["status"] = "running"
                    payload = rec["payload"]
                    break
            if advice_id is None:
                self._active = False
                return
        threading.Thread(target=self._worker, args=(advice_id, payload),
                         daemon=True).start()

    def _prune_locked(self) -> None:
        """Drop records older than 24 h (caller holds `_lock`)."""
        cutoff = time.time() - RECORD_TTL_SECS
        stale = [aid for aid, rec in self._records.items()
                 if rec["created_at"] < cutoff]
        for aid in stale:
            del self._records[aid]
        # Queue entries whose record was pruned cannot be served.
        if stale:
            self._queue = [aid for aid in self._queue if aid in self._records]


_HEAD_SHA_RE = re.compile(r"[0-9a-f]{40}")
_REPO_RE = re.compile(r"[\w.-]+/[\w.-]+")
MAX_BODY_BYTES = 1 << 20  # 1 MiB


class AdvisorRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the advisor service (ThreadingHTTPServer)."""

    def log_message(self, *a):
        pass  # silence default stderr spam

    def _reply(self, code: int, body: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def _authorized(self) -> bool:
        auth = self.headers.get("Authorization", "")
        expected_token = getattr(self.server, "advisor_token", "")
        if not expected_token or not auth.startswith("Bearer "):
            return False
        # Constant-time compare: a byte-by-byte context leak on the bearer
        # would let an attacker recover the advisor token.
        return hmac.compare_digest(auth[7:], expected_token)

    def do_POST(self):
        if self.path != "/v1/advise":
            return self._reply(404, {"error": "not found"})
        if not self._authorized():
            return self._reply(401, {"error": "unauthorized"})

        # --- Parse body -------------------------------------------------
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            return self._reply(400, {"error": "bad content-length"})
        if length < 0 or length > MAX_BODY_BYTES:
            return self._reply(400, {"error": "request body too large"})
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return self._reply(400, {"error": "bad json"})
        if not isinstance(payload, dict):
            return self._reply(400, {"error": "bad json"})

        repo = payload.get("repo")
        pr = payload.get("pr")
        head_sha = payload.get("head_sha")
        read_token = payload.get("read_token")
        # These values reach a `git fetch` argv and a GitHub URL downstream:
        # validate shape here, before the service ever sees them.
        if not _valid_advise_request(repo, pr, head_sha, read_token):
            return self._reply(400, {"error": "invalid required fields"})

        # --- Queue / start (never waits on the agent) -------------------
        service = getattr(self.server, "advisor_service", None)
        if service is None:
            return self._reply(500, {"error": "advisor service not configured"})
        try:
            advice_id = service.submit(payload)
        except QueueFull:
            return self._reply(429, {"error": "queue full"})
        return self._reply(202, {"advice_id": advice_id})

    def do_GET(self):
        prefix = "/v1/advise/"
        if not self.path.startswith(prefix):
            return self._reply(404, {"error": "not found"})
        if not self._authorized():
            return self._reply(401, {"error": "unauthorized"})
        advice_id = self.path[len(prefix):].strip("/")
        service = getattr(self.server, "advisor_service", None)
        if service is None:
            return self._reply(500, {"error": "advisor service not configured"})
        record = service.status(advice_id)
        if record is None:
            return self._reply(404, {"error": "unknown advice_id"})
        return self._reply(200, record)


def serve(
    bind_host: str = "100.111.118.24",
    bind_port: int = 8787,
    advisor_token: str = "",
    advisor_routes: tuple[str, ...] = ("claude:opus",),
    advisor_queue_max: int = 3,
    advisor_timeout: int = 1200,
    origin_url: str | None = None,
) -> None:
    """Start the advisor HTTP server (ThreadingHTTPServer + AdvisorService)."""
    server = ThreadingHTTPServer((bind_host, bind_port), AdvisorRequestHandler)
    server.advisor_token = advisor_token
    server.advisor_service = AdvisorService(
        routes=advisor_routes,
        timeout=advisor_timeout,
        queue_max=advisor_queue_max,
        origin_url=origin_url,
    )
    log.info("advisor listening on %s:%d (routes=%s queue_max=%d)",
             bind_host, bind_port, advisor_routes, advisor_queue_max)
    server.serve_forever()


def main() -> None:
    """Entry point for the advisor service."""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)-5s %(name)s :: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    bind_host = os.environ.get("NEO_ADVISOR_BIND_HOST", "100.111.118.24")
    bind_port = int(os.environ.get("NEO_ADVISOR_BIND_PORT", "8787"))
    advisor_token = os.environ.get("NEO_ADVISOR_TOKEN", "")
    routes_raw = os.environ.get(
        "ADVISOR_ROUTES",
        "claude:opus,codex:gpt-6.1-sol:high,codex:gpt-6-sol:high",
    )
    advisor_routes = tuple(
        r.strip() for r in routes_raw.split(",") if r.strip()
    )
    advisor_queue_max = int(os.environ.get("ADVISOR_QUEUE_MAX", "3"))
    advisor_timeout = int(os.environ.get("ADVISOR_TIMEOUT", "1200"))

    if not advisor_token:
        log.error("ADVISOR_TOKEN not set — refusing to start")
        sys.exit(1)

    serve(
        bind_host=bind_host,
        bind_port=bind_port,
        advisor_token=advisor_token,
        advisor_routes=advisor_routes,
        advisor_queue_max=advisor_queue_max,
        advisor_timeout=advisor_timeout,
    )


if __name__ == "__main__":
    main()
