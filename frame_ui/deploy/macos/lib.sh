#!/bin/sh

# Shared paths and deliberately small helpers for the macOS deployment scripts.
# This file is sourced; callers enable `set -eu` themselves.

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd -P)
REPO_ROOT=$(CDPATH= cd "$SCRIPT_DIR/../.." && pwd -P)
APP_SUPPORT="$HOME/Library/Application Support/Second Brain UI"
RELEASES_DIR="$APP_SUPPORT/releases"
CURRENT_LINK="$APP_SUPPORT/current"
PREVIOUS_LINK="$APP_SUPPORT/previous"
RUNTIME_ENV="$APP_SUPPORT/runtime.env"
LOG_DIR="$HOME/Library/Logs/Second Brain UI"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
PLIST="$LAUNCH_AGENTS/com.secondbrain.ui.plist"
LABEL="com.secondbrain.ui"
CADDYFILE="$REPO_ROOT/deploy/macos/Caddyfile"

die() {
  printf '%s\n' "second-brain-ui: $*" >&2
  exit 1
}

require_macos() {
  [ "$(uname -s)" = "Darwin" ] || die "this command must run on macOS"
}

runtime_token() {
  [ -f "$RUNTIME_ENV" ] || die "missing $RUNTIME_ENV; run install.sh first"
  token=$(sed -n 's/^SB_HTTP_TOKEN=//p' "$RUNTIME_ENV" | sed -n '1p')
  validate_token "$token"
  printf '%s' "$token"
}

validate_token() {
  candidate=$1
  [ -n "$candidate" ] || die "the token cannot be empty"
  # The token is substituted into a quoted Caddyfile value and stripped by
  # frontend_http. Restrict it to a conventional opaque-token alphabet rather
  # than trying to escape executable configuration syntax.
  case "$candidate" in
    *[!A-Za-z0-9._~+/=-]*)
      die "use a token containing only letters, digits, and . _ ~ + / = -" ;;
  esac
}

prompt_token() {
  printf 'Second Brain secret_http_token: ' >&2
  if [ -t 0 ]; then
    saved_tty=$(stty -g)
    trap 'stty "$saved_tty"; exit 130' HUP INT TERM
    stty -echo
    if ! IFS= read -r entered; then
      stty "$saved_tty"
      trap - HUP INT TERM
      die "could not read the token"
    fi
    stty "$saved_tty"
    trap - HUP INT TERM
    printf '\n' >&2
  else
    IFS= read -r entered || die "could not read the token"
  fi
  printf '%s' "$entered"
}

validate_gateway() {
  # The Caddyfile is the security perimeter — the Origin check that stands in
  # front of a route Caddy itself credentials — so a config that will not load
  # must be caught while the old one is still serving.
  #
  # Both variables have to be present or the parse fails on the placeholders
  # rather than on anything worth knowing about.
  command -v caddy >/dev/null 2>&1 || die "Caddy is not installed"
  SB_HTTP_TOKEN=$(runtime_token)
  SB_UI_CURRENT=$CURRENT_LINK
  export SB_HTTP_TOKEN SB_UI_CURRENT
  caddy validate --config "$CADDYFILE" --adapter caddyfile
}

backend_status() {
  # An unauthenticated frontend_http request must finish immediately with 401.
  # 503 means the kernel owns the socket but no frontend owns its request queue;
  # 000 means nothing accepted the connection at all.
  curl --max-time 3 --silent --output /dev/null --write-out '%{http_code}' \
    http://127.0.0.1:8787/events 2>/dev/null || true
}

require_backend_listener() {
  status=$(backend_status)
  case "$status" in
    401) return 0 ;;
    503) die "127.0.0.1:8787 is open, but frontend_http does not own it; check for a duplicate enabled frontend and restart Second Brain" ;;
    000|"") die "nothing is listening on 127.0.0.1:8787; start the Second Brain LaunchAgent and check its logs" ;;
    *) die "127.0.0.1:8787 answered HTTP $status instead of frontend_http's 401; inspect the process with: lsof -nP -iTCP:8787 -sTCP:LISTEN" ;;
  esac
}

atomic_link() {
  target=$1
  destination=$2
  temporary="${destination}.new.$$"
  rm -f "$temporary"
  ln -s "$target" "$temporary"
  # `mv` implementations disagree about whether a destination symlink to a
  # directory should be followed. Node is already a deployment prerequisite,
  # and renameSync maps to an atomic POSIX rename that replaces the link itself.
  node -e 'require("node:fs").renameSync(process.argv[1], process.argv[2])' \
    "$temporary" "$destination"
}
