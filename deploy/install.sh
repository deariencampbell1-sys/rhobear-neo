#!/usr/bin/env bash
# rhobear-neo installer — run ON rhobear-vps as root, from /opt/rhobear/rhobear-neo.
# Assumes .env + app-key.pem are already present (chmod 600).
set -euo pipefail
cd /opt/rhobear/rhobear-neo

python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt

install -m 644 deploy/rhobear-neo.service /etc/systemd/system/rhobear-neo.service
systemctl daemon-reload
systemctl enable rhobear-neo
systemctl restart rhobear-neo
sleep 2
systemctl --no-pager --lines=15 status rhobear-neo || true
echo "--- listening? ---"
ss -ltnp 2>/dev/null | grep 8767 || echo "NOT yet on :8767"
