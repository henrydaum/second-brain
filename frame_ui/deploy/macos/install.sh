#!/bin/sh
set -eu

. "$(CDPATH= cd "$(dirname "$0")" && pwd -P)/lib.sh"
require_macos

command -v brew >/dev/null 2>&1 || die "Homebrew is required; install it from https://brew.sh"
command -v node >/dev/null 2>&1 || die "Node.js is required to build the frontend"

if ! command -v caddy >/dev/null 2>&1; then
  brew install caddy
fi

mkdir -p "$APP_SUPPORT" "$LOG_DIR" "$LAUNCH_AGENTS"
chmod 700 "$APP_SUPPORT"

if [ ! -f "$RUNTIME_ENV" ]; then
  if [ -n "${SB_HTTP_TOKEN:-}" ]; then
    token=$SB_HTTP_TOKEN
  else
    token=$(prompt_token)
  fi
  validate_token "$token"
  umask 077
  printf 'SB_HTTP_TOKEN=%s\n' "$token" > "$RUNTIME_ENV"
fi
chmod 600 "$RUNTIME_ENV"

require_backend_listener
sh "$SCRIPT_DIR/build-release.sh"

xml_escape() {
  printf '%s' "$1" | sed \
    -e 's/&/\&amp;/g' \
    -e 's/</\&lt;/g' \
    -e 's/>/\&gt;/g' \
    -e 's/"/\&quot;/g' \
    -e "s/'/\&apos;/g"
}

run_script=$(xml_escape "$SCRIPT_DIR/run-caddy.sh")
working_dir=$(xml_escape "$REPO_ROOT")
stdout_log=$(xml_escape "$LOG_DIR/caddy.stdout.log")
stderr_log=$(xml_escape "$LOG_DIR/caddy.stderr.log")

umask 077
{
  printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>'
  printf '%s\n' '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
  printf '%s\n' '<plist version="1.0">'
  printf '%s\n' '<dict>'
  printf '%s\n' '  <key>Label</key>'
  printf '%s\n' "  <string>$LABEL</string>"
  printf '%s\n' '  <key>ProgramArguments</key>'
  printf '%s\n' '  <array>'
  printf '%s\n' '    <string>/bin/sh</string>'
  printf '%s\n' "    <string>$run_script</string>"
  printf '%s\n' '  </array>'
  printf '%s\n' '  <key>WorkingDirectory</key>'
  printf '%s\n' "  <string>$working_dir</string>"
  printf '%s\n' '  <key>EnvironmentVariables</key>'
  printf '%s\n' '  <dict>'
  printf '%s\n' '    <key>PATH</key>'
  printf '%s\n' '    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>'
  printf '%s\n' '  </dict>'
  printf '%s\n' '  <key>RunAtLoad</key><true/>'
  printf '%s\n' '  <key>KeepAlive</key><true/>'
  printf '%s\n' '  <key>ThrottleInterval</key><integer>5</integer>'
  printf '%s\n' '  <key>StandardOutPath</key>'
  printf '%s\n' "  <string>$stdout_log</string>"
  printf '%s\n' '  <key>StandardErrorPath</key>'
  printf '%s\n' "  <string>$stderr_log</string>"
  printf '%s\n' '</dict>'
  printf '%s\n' '</plist>'
} > "$PLIST"
chmod 600 "$PLIST"
plutil -lint "$PLIST"

validate_gateway

domain="gui/$(id -u)"
launchctl bootout "$domain/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap "$domain" "$PLIST"
launchctl kickstart -k "$domain/$LABEL"

printf '\nSecond Brain UI is installed.\n'
printf 'Open: http://127.0.0.1:4173\n'
printf 'Status: sh %s/manage.sh status\n' "$SCRIPT_DIR"
