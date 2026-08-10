# rhobear-neo — the review-and-merge gate

Live path: `/opt/rhobear/rhobear-neo` on rhobear-vps (root user).
Serves: the Neo review gate on `127.0.0.1:8767`, sibling to rhobear-reviews.
Unit: `rhobear-neo.service` (installed from `deploy/rhobear-neo.service`).
Runs: `.venv/bin/python -m src`, reads `.env` via `EnvironmentFile`.

## Secrets (never commit)
`.env`, `app-key.pem` (GitHub App private key), `.local/` — all on the box only.
`config.py` reads everything from the environment; there is no secret in source.

## Claude Code CLI migration (wave0-neo bounce 1, 2026-08-09) — history only

The old Pi/direct-DeepSeek path and direct-HTTP OpenRouterClient are both gone.
Neo runs as a **Claude Code CLI agent** (`/usr/bin/claude`) with full tool
access (gh, git, edit, test runner) in an isolated per-run temp work directory.
That first CLI migration routed through OpenRouter; the cutover below replaces
it with DeepSeek Direct. The env vars in this old section are OBSOLETE — see
the cutover section for what `.env` must contain today.

## DeepSeek Direct cutover (2026-08-10)

### What changed
The OpenRouter route is disabled. Neo's Claude Code agent now talks to
**DeepSeek Direct** at the Anthropic-compatible endpoint
`https://api.deepseek.com/anthropic`, model exactly `deepseek-v4-flash`
(unprefixed direct ID — no `/deepseek` prefix, no `[1m]` suffix) at `max`
reasoning effort, `--effort max` / `CLAUDE_CODE_EFFORT_LEVEL=max` explicit,
`CLAUDE_CODE_MAX_OUTPUT_TOKENS=32000` output headroom.

### New env vars (REQUIRED — add to `.env` on the VPS before restart)
```bash
DEEPSEEK_API_KEY=sk-...            # rotated key, owned by Neo (never commit)
NEO_DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic   # default, can be omitted
NEO_DEEPSEEK_MODEL=deepseek-v4-flash                       # default, can be omitted
NEO_REASONING_EFFORT=max            # default; no runtime probe/fallback
NEO_MAX_TOKENS=32000                # default; validation floor is 32000
NEO_TIMEOUT=300                     # default (seconds), can be omitted
NEO_CLAUDE_BIN=/usr/bin/claude     # default, can be omitted
```

### Removed/deprecated env vars
- `OPENROUTER_API_KEY` — removed; the OpenRouter route is disabled. Remove it from
  `.env` so a stale route cannot silently select itself.
- `NEO_OPENROUTER_BASE_URL` / `NEO_OPENROUTER_MODEL` — removed; config no longer reads
  them and startup validation rejects the old `https://openrouter.ai/api` value and
  the old `deepseek/...`-prefixed model outright.
- `NEO_OPENROUTER_API_KEY` — renamed to `DEEPSEEK_API_KEY` (Neo-owned secret now).
- `NEO_DEEPSEEK_API_KEY` — no longer used. The old Pi/direct-DeepSeek path is deleted.
- `NEO_ANTHROPIC_BASE_URL` — no longer used.
- `NEO_MODEL_TRIAGE` / `NEO_MODEL_BUILDER` — no longer used; one model now.
- `NEO_MAX_RETRIES` — no longer needed; the Claude Code CLI handles its own retries.

### Rollout steps
1. Add `DEEPSEEK_API_KEY` (and optional overrides) to `/opt/rhobear/rhobear-neo/.env`.
2. Remove stale `OPENROUTER_API_KEY`, `NEO_OPENROUTER_BASE_URL`, `NEO_OPENROUTER_MODEL`,
   `NEO_OPENROUTER_API_KEY`, `NEO_DEEPSEEK_API_KEY`, `NEO_ANTHROPIC_BASE_URL`,
   `NEO_MODEL_TRIAGE`, `NEO_MODEL_BUILDER`, `NEO_MAX_RETRIES` from `.env`.
3. Copy the source tree to the VPS — this is NOT a git checkout:
   ```bash
   rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
     C:/Users/slang/.rhobear/silo-h/imports/rhobear-neo/ root@rhobear-vps:/opt/rhobear/rhobear-neo/
   ```
   Or zip and scp:
   ```bash
   cd /opt/rhobear/rhobear-neo && rm -rf src tests deploy requirements.txt DEPLOY.md README.md
   # then copy the new files from the build machine
   ```
   **Do NOT `git pull`** — `/opt/rhobear/rhobear-neo` is not a Git checkout.
4. Run the installer:
   ```bash
   ssh rhobear-vps 'cd /opt/rhobear/rhobear-neo && bash deploy/install.sh'
   ```
5. Verify the service starts:
   ```bash
   ssh rhobear-vps 'systemctl status rhobear-neo --no-pager -l'
   ```
6. Verify the log:
   ```bash
   ssh rhobear-vps 'journalctl -u rhobear-neo -n 20 --no-pager'
   ```
   Look for: `rhobear-neo up` with no errors.
7. Send a test verdict webhook (or wait for a real PR) and confirm Neo processes it.

## Deploy
```bash
# Copy source from the build machine to VPS (rsync or scp).
# Then on the VPS:
cd /opt/rhobear/rhobear-neo && bash deploy/install.sh
```
`install.sh` rebuilds the venv, reinstalls the unit and restarts it.
Do NOT hand-edit the live tree — commit here and copy.