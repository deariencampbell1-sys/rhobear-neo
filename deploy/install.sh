#!/bin/bash
# rhobear-neo installer — run ON rhobear-vps as root, from /opt/rhobear/rhobear-neo.
# The only secret a deploy needs is the vault (/etc/rhobear/llm-keys.env): the
# generated /opt/rhobear/bin/vault-env helper builds /run/<service>/env at start.
set -euo pipefail

# --- command resolution and pre-flight (F22-r1-r2) ---------------------------
# The shebang is absolute (/bin/bash): the kernel runs exactly that
# interpreter, never a `bash` looked up in the caller's PATH.  /usr/bin/env is
# *defined* as a PATH lookup, so `#!/usr/bin/env bash` was itself the one
# command a caller could hijack — before this file's first line ran.  With the
# interpreter fixed, pin PATH before anything else so every later command
# comes from the known system set; the absolute paths below are the belt for a
# future edit that forgets the pin.
PATH=/usr/bin:/bin; export PATH

fail() { /usr/bin/printf '%s\n' "install.sh: $1" >&2; exit 1; }

# Interrupt safety: a write mktemps next to its destination, so a root-owned
# .install-rhobear.XXXXXX can exist between mktemp and mv; remove every
# registered temp path on exit or signal.
TMP_FILES=""
cleanup_tmp() {
  [ -z "$TMP_FILES" ] || /usr/bin/rm -f $TMP_FILES
  return 0
}
trap cleanup_tmp EXIT INT TERM HUP

# atomic_write <dest> <mode> <refuse-symlink> <content>
# Write content to a mktemp file in dest's directory, set its mode, then
# mv -T -f it into place — never a bare check-then-write through $dest.  With
# refuse-symlink=1 a pre-existing symlink is refused and left in place;
# otherwise the symlink itself is replaced, never followed.  The temp path is
# registered with cleanup_tmp before it exists on disk.
atomic_write() {
  local dest="$1" mode="$2" refuse="$3" content="$4" dir tmp
  dir=$(/usr/bin/dirname -- "$dest") || return 1
  tmp=$(/usr/bin/mktemp "$dir/.install-rhobear.XXXXXX") || {
    echo "install.sh: cannot create a temp file next to $dest" >&2
    return 1
  }
  TMP_FILES="${TMP_FILES:-} $tmp"
  /usr/bin/printf '%s\n' "$content" > "$tmp" || {
    echo "install.sh: cannot write $tmp" >&2
    /usr/bin/rm -f "$tmp"
    return 1
  }
  /usr/bin/chmod "$mode" "$tmp" || {
    echo "install.sh: cannot set mode $mode on $tmp" >&2
    /usr/bin/rm -f "$tmp"
    return 1
  }
  if [ "$refuse" = "1" ] && [ -L "$dest" ]; then
    echo "install.sh: refusing to write $dest: destination is a symlink" >&2
    /usr/bin/rm -f "$tmp"
    return 1
  fi
  /usr/bin/mv -T -f "$tmp" "$dest" || {
    echo "install.sh: cannot install $dest" >&2
    /usr/bin/rm -f "$tmp"
    return 1
  }
}

# Resolve the interpreter once, deliberately, never through PATH.
PY=/usr/bin/python3
[ -x "$PY" ] || fail "python3 not found at $PY"
VENV_DIR=/opt/rhobear/rhobear-neo/.venv

# Test seam: stop before the root check and before the first write, so the
# resolution/pre-flight path is exercisable unprivileged in a temp sandbox.
# An argument, never an environment variable: an inherited variable can no
# longer turn a real install into a green no-op.
if [ "${1:-}" = "--dry-run" ]; then
  /usr/bin/printf '%s\n' "install.sh: dry run: no changes made" >&2
  exit 0
fi

[ "$(/usr/bin/id -u)" = "0" ] || fail "must run as root"
cd /opt/rhobear/rhobear-neo

"$PY" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q -r requirements.txt

# --- vault-backed service environments --------------------------------------
# /opt/rhobear/bin/vault-env (mode 0755) + /etc/rhobear/vault-env.d/*.names
# (mode 0644). The units run the helper as ExecStartPre; it parses vault
# NAME=value records and builds /run/<service>/env, failing closed and
# naming the missing vault name if anything is absent.
install_vault_env_helper() {
  local dest="$1" body
  body=$(/usr/bin/cat <<'VAULT_ENV_HELPER'
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
fail() { printf '%s\n' "vault-env: $1" >&2; exit 1; }
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
    case "$ln" in
      *=*) if [ "${ln%%=*}" = "$1" ]; then val="${ln#*=}"; [ -n "$val" ] || return 1; printf '%s' "$val"; return 0; fi ;;
    esac
  done < "$VAULT_FILE"
  return 1
}
pairs=""; outs=" "; n=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    ""|"#"*) ;;
    *)
      case "$line" in *=*) out="${line%%=*}"; vn="${line#*=}" ;; *) out="$line"; vn="$line" ;; esac
      n=$((n + 1))
      valid_name "$out" || fail "malformed names entry in $names_file line $n"
      valid_name "$vn" || fail "malformed names entry in $names_file line $n"
      case "$outs" in *" $out "*) fail "duplicate output name $out" ;; esac
      outs="$outs$out "; pairs="$pairs $out=$vn"
      ;;
  esac
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
  case "$body" in
    *"PATH="*"PATH="*)
      echo "install.sh: refusing to write $dest: second PATH= assignment in the helper body" >&2
      return 1 ;;
  esac
  atomic_write "$dest" 0755 1 "$body"
}

install_vault_env() {
  /usr/bin/install -d -m 0755 /opt/rhobear/bin /etc/rhobear/vault-env.d

  install_vault_env_helper /opt/rhobear/bin/vault-env || return 1

  local svc names_body
  names_body=$(/usr/bin/cat <<'VAULT_ENV_NAMES'
NEO_GITHUB_APP_KEY_PATH=RHOBEAR_ZEUS_GH_APP_KEY_B64
RHOBEAR_NEO_WEBHOOK_SECRET=RHOBEAR_ZEUS_WEBHOOK_SECRET
DATABASE_URL=REVIEWS_DATABASE_URL
DEEPSEEK_API_KEY
DEEPINFRA_API_KEY
VAULT_ENV_NAMES
)
  for svc in rhobear-neo rhobear-neo-builder; do
    atomic_write "/etc/rhobear/vault-env.d/${svc}.names" 0644 0 "$names_body" || return 1
  done
}
install_vault_env

/usr/bin/install -m 644 deploy/rhobear-neo.service /etc/systemd/system/rhobear-neo.service
/usr/bin/systemctl daemon-reload
/usr/bin/systemctl enable rhobear-neo
/usr/bin/systemctl restart rhobear-neo
/usr/bin/sleep 2
/usr/bin/systemctl --no-pager --lines=15 status rhobear-neo || true
echo "--- listening? ---"
/usr/bin/ss -ltnp 2>/dev/null | /usr/bin/grep 8767 || echo "NOT yet on :8767"
