# rhobear-neo

The **review-and-merge gate**, sold as its own product (SKU) and shipped as its own GitHub App
(`rhobear-neo`, App ID 4347986) — separate from `rhobear-reviews` because the permission grant *is*
the product boundary: Reviews only reads + posts a verdict; Neo **writes and merges your code**.

## What it does
1. Wakes on a **trusted reviewer's verdict** landing on a PR head SHA — GitHub `status` (Reviews uses
   the Statuses API) or `check_run: completed` (Checks-API reviewers like CodeAnt). It does NOT
   subscribe to `pull_request`, so it never races the reviewer.
2. Runs the **Neo protocol** (`neo.md` canon) in one agentic Claude Code CLI loop pinned to
   DeepSeek Direct (`deepseek-v4-flash` via `https://api.deepseek.com/anthropic`) at maximum
   effort with at least 32K output headroom:
   triage findings → **fix-forward** trivial issues → repair substantial bugs with tools →
   **ESCALATE** owner-gated forks → **merge on green** behind the per-install auto-merge toggle.
3. The loop **closes itself**: a fix-forward commit or a builder push is a new `pull_request` event →
   Reviews re-fires → Neo re-fires. Bounded retry (≤2 builder rounds) → escalate, never thrash.

## Architecture
- Stdlib HTTP webhook server on `127.0.0.1:8767`; Caddy fronts `https://neo.rhobear.ai/webhook`.
- State in the shared VPS Postgres (`reviews_state`) — Neo-owned tables (`neo_actions`, `neo_installs`):
  loop guard (keyed PR+SHA), builder-round counter, per-install auto-merge toggle + credit balance.
- Engine: Claude Code CLI with tool access, routed only through DeepSeek Direct
  (`deepseek-v4-flash`); config validation rejects model, endpoint, effort, or output-budget
  drift before the worker starts. **Priced at the Gemini baseline + margin** (RHOBEAR credits model)
  — DeepSeek's lower cost is our margin. See `~/.claude/plans/neo-pricing-model.md`.

## Files
- `src/config.py` — env-driven config (no secrets baked in).
- `src/webhook_server.py` — HMAC-verified receiver; maps status/check_run → a verdict wake.
- `src/neo_worker.py` — the deterministic wrapper: loop guard, entitlement/credits, runs the protocol.
- `src/neo_protocol.py` — the Neo brief (port of `neo.md`).
- `src/neo_state.py` — Postgres state (loop guard, installs, credits).
- `src/__main__.py` — webhook server + background worker queue.
- `deploy/` — systemd unit, installer, Caddy snippet.

## Deploy (on rhobear-vps)
`.env` + `app-key.pem` must be present (chmod 600). Then: `bash deploy/install.sh`.
