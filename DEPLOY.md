# rhobear-neo — the review-and-merge gate

Live path: `/opt/rhobear/rhobear-neo` on rhobear-vps (root user).
Serves: the Neo review gate on `127.0.0.1:8767`, sibling to rhobear-reviews.
Unit: `rhobear-neo.service` (installed from `deploy/rhobear-neo.service`).
Runs: `.venv/bin/python -m src`, reads `.env` via `EnvironmentFile`.

## Secrets (never commit)
`.env`, `app-key.pem` (GitHub App private key), `.local/` — all on the box only.
`config.py` reads everything from the environment; there is no secret in source.

## OpenRouter migration (wave0-neo, 2026-08-09)

### New env vars (REQUIRED — add to `.env` on the VPS before restart)
```bash
NEO_OPENROUTER_API_KEY=sk-or-v1-...   # OpenRouter API key (replaces NEO_DEEPSEEK_API_KEY)
NEO_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1   # default, can be omitted
NEO_OPENROUTER_MODEL=deepseek/deepseek-v4-flash         # default, can be omitted
NEO_REASONING_EFFORT=max                # default; probed at startup, falls back to high
NEO_MAX_TOKENS=8192                     # default, can be omitted
NEO_TIMEOUT=300                         # default (seconds), can be omitted
NEO_MAX_RETRIES=3                       # default, can be omitted
```

### Removed/deprecated env vars
- `NEO_DEEPSEEK_API_KEY` — no longer used. The old Pi/direct-DeepSeek path is deleted.
- `NEO_ANTHROPIC_BASE_URL` — no longer used.
- `NEO_MODEL_TRIAGE` — no longer used; model is `NEO_OPENROUTER_MODEL` now.
- `DEEPSEEK_API_KEY` — no longer needed in the service environment.

### Rollout steps
1. Add `NEO_OPENROUTER_API_KEY` (and optional overrides) to `/opt/rhobear/rhobear-neo/.env`.
2. Remove old `NEO_DEEPSEEK_API_KEY`, `NEO_ANTHROPIC_BASE_URL`, `NEO_MODEL_TRIAGE` from `.env`.
3. `ssh rhobear-vps 'cd /opt/rhobear/rhobear-neo && git pull && ./deploy/install.sh'`
4. Verify the service starts: `ssh rhobear-vps 'systemctl status rhobear-neo --no-pager -l'`
5. Verify the probe log: `ssh rhobear-vps 'journalctl -u rhobear-neo -n 20 --no-pager'`
   Look for: `reasoning_effort probe: configured=max probed=max`
6. Send a test verdict webhook (or wait for a real PR) and confirm Neo processes it.

## Deploy
```bash
ssh rhobear-vps 'cd /opt/rhobear/rhobear-neo && git pull && ./deploy/install.sh'
```
`install.sh` rebuilds the venv, reinstalls the unit and restarts it.
Do NOT hand-edit the live tree — commit here and pull.
