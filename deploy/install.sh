#!/usr/bin/env bash
# rhobear-neo installer — run ON rhobear-vps as root, from /opt/rhobear/rhobear-neo.
# The only secret a deploy needs is the vault (/etc/rhobear/llm-keys.env): the
# generated /opt/rhobear/bin/vault-env helper builds /run/<service>/env at start.
set -euo pipefail
cd /opt/rhobear/rhobear-neo

python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt

# --- vault-backed service environments --------------------------------------
# /opt/rhobear/bin/vault-env (mode 0755) + /etc/rhobear/vault-env.d/*.names
# (mode 0644). The units run the helper as ExecStartPre; it parses vault
# NAME=value records and builds /run/<service>/env, failing closed and
# naming the missing vault name if anything is absent.
install_vault_env_helper() {
  local dest="$1" body
  body=$(cat <<'VAULT_ENV_HELPER'
#!/bin/sh
PATH=/usr/bin:/bin; export PATH
# vault-env <service> - materialise <RUN_DIR>/<service>/env from the vault,
# fail closed. NAME=value records are parsed, never sourced or printed; *_B64
# names decode to 0600 pems whose absolute path is exported. Overrides:
# VAULT_FILE, RUN_DIR, NAMES_DIR. Diagnostics never carry values.
set -eu; set +x; umask 077
ok=0; tmpd=""; pem_finals=""
cleanup() { [ "$ok" -eq 1 ] || /usr/bin/rm -f ${pem_finals:-}; [ -z "$tmpd" ] || /usr/bin/rm -rf "$tmpd"; return 0; }
trap cleanup EXIT
fail() { echo "vault-env: $1" >&2; exit 1; }
esc() { printf '%s' "$1" | /usr/bin/sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
valid_name() { case "$1" in ""|[0-9]*|*[!A-Za-z0-9_]*) return 1 ;; esac; return 0; }
service="${1:-}"
[ -n "$service" ] || fail "usage: vault-env <service>"
case "$service" in ''|.|..|*/*) fail "invalid service name $service" ;; esac
VAULT_FILE="${VAULT_FILE:-/etc/rhobear/llm-keys.env}"
RUN_DIR="${RUN_DIR:-/run}"
NAMES_DIR="${NAMES_DIR:-/etc/rhobear/vault-env.d}"
case "$RUN_DIR" in /*) ;; *) RUN_DIR="$(pwd)/$RUN_DIR" ;; esac
names_file="$NAMES_DIR/$service.names"
[ -f "$names_file" ] || fail "missing names file $names_file"
[ -f "$VAULT_FILE" ] || fail "missing vault file $VAULT_FILE"
vault_value() {
  while IFS= read -r ln || [ -n "$ln" ]; do
    case "$ln" in ""|"#"*|*=*) ;; *) continue ;; esac
    [ "${ln%%=*}" = "$1" ] || continue
    val="${ln#*=}"; [ -n "$val" ] || return 1
    printf '%s' "$val"; return 0
  done < "$VAULT_FILE"
  return 1
}
pairs=""; outs=" "; n=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ""|"#"*) continue ;; esac
  case "$line" in *=*) out="${line%%=*}"; vn="${line#*=}" ;; *) out="$line"; vn="$line" ;; esac
  n=$((n + 1))
  valid_name "$out" || fail "malformed names entry in $names_file line $n"
  valid_name "$vn" || fail "malformed names entry in $names_file line $n"
  case "$outs" in *" $out "*) fail "duplicate output name $out" ;; esac
  outs="$outs$out "; pairs="$pairs $out=$vn"
done < "$names_file"
for p in $pairs; do
  vault_value "${p#*=}" >/dev/null || fail "missing vault name ${p#*=}"
done
svc_dir="$RUN_DIR/$service"
/usr/bin/mkdir -p "$svc_dir" && /usr/bin/chmod 0700 "$svc_dir" || fail "cannot prepare $svc_dir"
owner_ref="$svc_dir"
tmpd=$(/usr/bin/mktemp -d "$(/usr/bin/dirname "$svc_dir")/.vault-env.XXXXXX") || fail "cannot create temp dir"
env_tmp="$tmpd/env"; : > "$env_tmp" && /usr/bin/chmod 600 "$env_tmp" || fail "cannot write $env_tmp"
/usr/bin/chown --reference="$owner_ref" "$env_tmp" || fail "cannot set owner of $env_tmp"
for p in $pairs; do
  out="${p%%=*}"; vn="${p#*=}"; val="$(vault_value "$vn")"
  case "$vn" in
  *_B64)
    low="$(printf '%s' "$out" | /usr/bin/tr 'A-Z' 'a-z')"
    pem="$svc_dir/$low.pem"; pt="$tmpd/$low.pem"
    printf '%s' "$val" | /usr/bin/base64 -d > "$pt" 2>/dev/null || { /usr/bin/rm -f "$pt"; fail "invalid base64 for vault name $vn"; }
    /usr/bin/chmod 600 "$pt"; /usr/bin/chown --reference="$owner_ref" "$pt" || fail "cannot set owner of $pt"
    /usr/bin/mv -T -f "$pt" "$pem"; pem_finals="$pem_finals $pem"
    printf '%s="%s"\n' "$out" "$(esc "$pem")" >> "$env_tmp"
    ;;
  *) printf '%s="%s"\n' "$out" "$(esc "$val")" >> "$env_tmp" ;;
  esac
done
/usr/bin/mv -T -f "$env_tmp" "$svc_dir/env" || fail "cannot install $svc_dir/env"
ok=1
VAULT_ENV_HELPER
)
  case "$body" in
    '#!/bin/sh
PATH=/usr/bin:/bin; export PATH
'*) ;;
    *) echo "install.sh: refusing to write $dest: helper PATH is not pinned" >&2; return 1 ;;
  esac
  printf '%s\n' "$body" > "$dest"
}

install_vault_env() {
  install -d -m 0755 /opt/rhobear/bin /etc/rhobear/vault-env.d

  install_vault_env_helper /opt/rhobear/bin/vault-env || return 1
  chmod 0755 /opt/rhobear/bin/vault-env

  local svc
  for svc in rhobear-neo rhobear-neo-builder; do
    cat > "/etc/rhobear/vault-env.d/${svc}.names" <<'VAULT_ENV_NAMES'
NEO_GITHUB_APP_KEY_PATH=RHOBEAR_ZEUS_GH_APP_KEY_B64
RHOBEAR_NEO_WEBHOOK_SECRET=RHOBEAR_ZEUS_WEBHOOK_SECRET
DATABASE_URL=REVIEWS_DATABASE_URL
DEEPSEEK_API_KEY
DEEPINFRA_API_KEY
VAULT_ENV_NAMES
  done
  chmod 0644 /etc/rhobear/vault-env.d/*.names
}
install_vault_env

install -m 644 deploy/rhobear-neo.service /etc/systemd/system/rhobear-neo.service
systemctl daemon-reload
systemctl enable rhobear-neo
systemctl restart rhobear-neo
sleep 2
systemctl --no-pager --lines=15 status rhobear-neo || true
echo "--- listening? ---"
ss -ltnp 2>/dev/null | grep 8767 || echo "NOT yet on :8767"
