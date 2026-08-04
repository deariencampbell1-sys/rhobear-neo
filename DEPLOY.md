# rhobear-neo — the review-and-merge gate

Live path: `/opt/rhobear/rhobear-neo` on rhobear-vps (root user).
Serves: the Neo review gate on `127.0.0.1:8767`, sibling to rhobear-reviews.
Unit: `rhobear-neo.service` (installed from `deploy/rhobear-neo.service`).
Runs: `.venv/bin/python -m src`, reads `.env` via `EnvironmentFile`.

## Secrets (never commit)
`.env`, `app-key.pem` (GitHub App private key), `.local/` — all on the box only.
`config.py` reads everything from the environment; there is no secret in source.

## Deploy
```bash
ssh rhobear-vps 'cd /opt/rhobear/rhobear-neo && git pull && ./deploy/install.sh'
```
`install.sh` rebuilds the venv, reinstalls the unit and restarts it.
Do NOT hand-edit the live tree — commit here and pull.
