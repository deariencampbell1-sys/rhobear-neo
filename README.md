# rhobear-neo

The **review-and-merge gate**, sold as its own product (SKU) and shipped as its own GitHub App
(`rhobear-neo`, App ID 4347986) — separate from `rhobear-reviews` because the permission grant *is*
the product boundary: Reviews only reads + posts a verdict; Neo **writes and merges your code**.

## What it does
1. Wakes on a **trusted reviewer's verdict** landing on a PR head SHA — GitHub `status` (Reviews uses
   the Statuses API) or `check_run: completed` (Checks-API reviewers like CodeAnt). It does NOT
   subscribe to `pull_request`, so it never races the reviewer.
2. Runs the **Neo protocol** (`neo.md` canon): triage findings → **fix-forward** trivial issues (DeepSeek
   Flash) → dispatch a **DeepSeek-Pro builder** for substantial bugs → **ESCALATE** owner-gated forks →
   **merge on green** behind the per-install auto-merge toggle.
3. The loop **closes itself**: a fix-forward commit or a builder push is a new `pull_request` event →
   Reviews re-fires → Neo re-fires. Bounded retry (≤2 builder rounds) → escalate, never thrash.

## Architecture
- Stdlib HTTP webhook server on `127.0.0.1:8767`; Caddy fronts `https://neo.rhobear.ai/webhook`.
- State in the shared VPS Postgres (`reviews_state`) — Neo-owned tables (`neo_actions`, `neo_installs`):
  loop guard (keyed PR+SHA), builder-round counter, per-install auto-merge toggle + credit balance.
- Engine: DeepSeek via the Anthropic-compatible endpoint (Flash triage, Pro builder). **Priced at the
  Gemini baseline + margin** (RHOBEAR credits model) — DeepSeek's lower cost is our margin. See
  `~/.claude/plans/neo-pricing-model.md`.

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
