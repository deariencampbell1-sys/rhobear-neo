# rhobear-neo — the review-and-merge gate

Live path: `/opt/rhobear/rhobear-neo` on rhobear-vps (root user).
Serves: the Neo review gate on `127.0.0.1:8767`, sibling to rhobear-reviews.
Unit: `rhobear-neo.service` (installed from `deploy/rhobear-neo.service`).
Runs: `.venv/bin/python -m src`, reads `.env` via `EnvironmentFile`.

## Secrets (never commit)
`.env`, `app-key.pem` (GitHub App private key), `.local/` — all on the box only.
`config.py` reads everything from the environment; there is no secret in source.

## Claude Code CLI migration (wave0-neo bounce 1, 2026-08-09)

### What changed
The old Pi/direct-DeepSeek path and direct-HTTP OpenRouterClient are both gone.
Neo now runs as a **Claude Code CLI agent** (`/usr/bin/claude`) with full tool
access (gh, git, edit, test runner) in an isolated per-run temp work directory.
OpenRouter provides the Anthropic-compatible backend; the model is exact
`deepseek/deepseek-v4-flash` at `max` reasoning effort.

### New env vars (REQUIRED — add to `.env` on the VPS before restart)
```bash
OPENROUTER_API_KEY=sk-or-v1-...   # single shared secret (same as rhobear-reviews)
NEO_OPENROUTER_BASE_URL=https://openrouter.ai/api   # default, can be omitted
NEO_OPENROUTER_MODEL=deepseek/deepseek-v4-flash     # default, can be omitted
NEO_REASONING_EFFORT=max            # default; no runtime probe/fallback
NEO_MAX_TOKENS=32000                # default; was 8192 in the old client
NEO_TIMEOUT=300                     # default (seconds), can be omitted
NEO_CLAUDE_BIN=/usr/bin/claude     # default, can be omitted
```

### Removed/deprecated env vars
- `NEO_OPENROUTER_API_KEY` — renamed to `OPENROUTER_API_KEY` (shared with Reviews).
- `NEO_DEEPSEEK_API_KEY` — no longer used. The old Pi/direct-DeepSeek path is deleted.
- `NEO_ANTHROPIC_BASE_URL` — no longer used.
- `NEO_MODEL_TRIAGE` / `NEO_MODEL_BUILDER` — no longer used; one model now.
- `DEEPSEEK_API_KEY` — no longer needed in the service environment.
- `NEO_MAX_RETRIES` — no longer needed; the Claude Code CLI handles its own retries.
- `NEO_OPENROUTER_BASE_URL` — still used but default changed to `https://openrouter.ai/api`
  (no `/v1` suffix — the CLI appends the Anthropic-compatible path).

### Rollout steps
1. Add `OPENROUTER_API_KEY` (and optional overrides) to `/opt/rhobear/rhobear-neo/.env`.
2. Remove old `NEO_OPENROUTER_API_KEY`, `NEO_DEEPSEEK_API_KEY`, `NEO_ANTHROPIC_BASE_URL`,
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