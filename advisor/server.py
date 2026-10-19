"""Advisor HTTP server — bound to the tailnet address.

Receives escalation requests from Neo at POST /v1/advise:

  Body: {repo, pr, head_sha, rounds: [...], diff_stat, read_token}
  Auth: Bearer <NEO_ADVISOR_TOKEN>

  Returns: {route, summary, root_cause, steps: [{file, change, why}],
            tests: [...], risks}

Flow:
  1. Check bearer token.
  2. Enqueue the request (one at a time; max 3 queued, then 429).
  3. Clone the repo read-only at head_sha using read_token via credential
     helper pattern (never in URL or .git/config).
  4. Run a read-only frontier agent with a 20-minute timeout.
     First available of ADVISOR_ROUTES:
       - claude: claude -p --model opus --permission-mode plan
       - codex: codex exec -m <model> -c model_reasoning_effort=<effort>
  5. Return a work order (strict JSON; one repair retry, then fail).
  6. Delete the temp dir.

Deployed on the build box as a capped systemd service (not on prod).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

log = logging.getLogger("rhobear_neo.advisor.server")


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


def _run_agent(engine: str, params: dict, brief: str, timeout: int = 1200) -> str:
    """Run a read-only frontier agent and return its raw output.

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

    Attempts one repair retry (strip markdown fences) before failing.
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
            assert "steps" in data, "missing steps"
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


def run_advisor(
    repo: str,
    pr: int,
    head_sha: str,
    rounds: list[dict],
    diff_stat: str,
    read_token: str,
    routes: tuple[str, ...],
    timeout: int = 1200,
) -> dict:
    """Run the advisor for one escalation request.

    Clones the repo read-only, runs the first available frontier agent,
    and returns a work order.

    The read_token is used via git credential helper pattern (never in URL
    or .git/config).
    """
    with tempfile.TemporaryDirectory(prefix="advisor-") as tmpdir:
        # --- Clone the repo read-only using credential helper ------------
        # We set GIT_ASKPASS to a script that returns the read_token.
        # This avoids putting the token in the URL or .git/config.
        askpass_path = os.path.join(tmpdir, "git-askpass.sh")
        with open(askpass_path, "w") as f:
            f.write("#!/bin/sh\n")
            f.write(f'echo "{read_token}"\n')
        os.chmod(askpass_path, 0o700)

        repo_url = f"https://x-access-token@github.com/{repo}.git"
        clone_dir = os.path.join(tmpdir, "repo")

        try:
            subprocess.run(
                ["git", "clone", "--depth=1", repo_url, clone_dir],
                capture_output=True, text=True,
                timeout=120,
                env={**os.environ, "GIT_ASKPASS": askpass_path,
                     "GIT_TERMINAL_PROMPT": "0"},
            )
        except subprocess.TimeoutExpired:
            raise AgentFailed("git clone timed out")
        except subprocess.CalledProcessError as exc:
            raise AgentFailed(f"git clone failed: {exc.stderr[:500]}")

        # Checkout the specific head SHA.
        try:
            subprocess.run(
                ["git", "-C", clone_dir, "checkout", head_sha],
                capture_output=True, text=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as exc:
            raise AgentFailed(f"git checkout {head_sha[:12]} failed: {exc.stderr[:500]}")

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
                raw = _run_agent(engine, params, brief, timeout=timeout)
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


class AdvisorRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the advisor service."""

    # Class-level state for queue management.
    queue: list[dict] = []
    queue_lock = threading.Lock()
    active = False
    active_lock = threading.Lock()

    def log_message(self, *a):
        pass  # silence default stderr spam

    def _reply(self, code: int, body: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def do_POST(self):
        if self.path != "/v1/advise":
            return self._reply(404, {"error": "not found"})

        # --- Auth -----------------------------------------------------------
        auth = self.headers.get("Authorization", "")
        expected_token = getattr(self.server, "advisor_token", "")
        if not auth.startswith("Bearer ") or auth[7:] != expected_token:
            return self._reply(401, {"error": "unauthorized"})

        # --- Parse body -----------------------------------------------------
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return self._reply(400, {"error": "bad json"})

        repo = payload.get("repo", "")
        pr = payload.get("pr", 0)
        head_sha = payload.get("head_sha", "")
        rounds = payload.get("rounds", [])
        diff_stat = payload.get("diff_stat", "")
        read_token = payload.get("read_token", "")

        if not repo or not pr or not head_sha or not read_token:
            return self._reply(400, {"error": "missing required fields"})

        # --- Queue management -----------------------------------------------
        with self.queue_lock:
            if self.active:
                if len(self.queue) >= getattr(self.server, "advisor_queue_max", 3):
                    return self._reply(429, {"error": "queue full"})
                self.queue.append(payload)
                return self._reply(202, {"status": "queued"})

            self.active = True

        # --- Process (outside lock) -----------------------------------------
        try:
            routes = getattr(self.server, "advisor_routes", ("claude:opus",))
            timeout = getattr(self.server, "advisor_timeout", 1200)
            work_order = run_advisor(
                repo, pr, head_sha, rounds, diff_stat, read_token,
                routes=routes, timeout=timeout,
            )
            self._reply(200, work_order)
        except AgentFailed as exc:
            self._reply(500, {"error": str(exc)})
        except Exception as exc:
            log.exception("advisor failed on %s#%s", repo, pr)
            self._reply(500, {"error": f"internal error: {type(exc).__name__}"})
        finally:
            # Process next in queue.
            with self.queue_lock:
                if self.queue:
                    next_req = self.queue.pop(0)
                    # Spawn a thread to process the next request.
                    t = threading.Thread(
                        target=self._process,
                        args=(next_req,),
                        daemon=True,
                    )
                    t.start()
                else:
                    self.active = False

    def _process(self, payload: dict):
        """Process a queued request in a background thread."""
        repo = payload.get("repo", "")
        pr = payload.get("pr", 0)
        head_sha = payload.get("head_sha", "")
        rounds = payload.get("rounds", [])
        diff_stat = payload.get("diff_stat", "")
        read_token = payload.get("read_token", "")
        try:
            routes = getattr(self.server, "advisor_routes", ("claude:opus",))
            timeout = getattr(self.server, "advisor_timeout", 1200)
            work_order = run_advisor(
                repo, pr, head_sha, rounds, diff_stat, read_token,
                routes=routes, timeout=timeout,
            )
            log.info("advisor completed %s#%s route=%s", repo, pr,
                     work_order.get("route", "?"))
        except Exception as exc:
            log.error("advisor failed on queued %s#%s: %s", repo, pr, exc)
        finally:
            with self.queue_lock:
                if self.queue:
                    next_req = self.queue.pop(0)
                    t = threading.Thread(
                        target=self._process,
                        args=(next_req,),
                        daemon=True,
                    )
                    t.start()
                else:
                    self.active = False


def serve(
    bind_host: str = "100.111.118.24",
    bind_port: int = 8787,
    advisor_token: str = "",
    advisor_routes: tuple[str, ...] = ("claude:opus",),
    advisor_queue_max: int = 3,
    advisor_timeout: int = 1200,
) -> None:
    """Start the advisor HTTP server."""
    server = HTTPServer((bind_host, bind_port), AdvisorRequestHandler)
    server.advisor_token = advisor_token
    server.advisor_routes = advisor_routes
    server.advisor_queue_max = advisor_queue_max
    server.advisor_timeout = advisor_timeout
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
