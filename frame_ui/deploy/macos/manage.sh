#!/bin/sh
set -eu

. "$(CDPATH= cd "$(dirname "$0")" && pwd -P)/lib.sh"
require_macos
domain="gui/$(id -u)"

usage() {
  printf 'Usage: %s {update|status|restart|rollback|set-token|uninstall}\n' "$0" >&2
  exit 2
}

case "${1:-}" in
  update)
    sh "$SCRIPT_DIR/build-release.sh"
    ;;
  status)
    launchctl print "$domain/$LABEL" 2>/dev/null || true
    printf '\nfrontend_http: HTTP %s (expected 401)\n' "$(backend_status)"
    printf '\nGateway: '
    if curl --fail --silent --show-error http://127.0.0.1:4173/healthz; then
      printf '\n'
    else
      printf 'unavailable\n'
      exit 1
    fi
    ;;
  restart)
    [ -f "$PLIST" ] || die "not installed; run install.sh"
    # Checked before the restart rather than discovered after it. `KeepAlive` is
    # true, so a Caddyfile that will not load does not fail once — it fails
    # every five seconds with the UI down, and the reason is only in a log.
    validate_gateway
    launchctl kickstart -k "$domain/$LABEL"
    ;;
  rollback)
    [ -L "$CURRENT_LINK" ] || die "no current release"
    [ -L "$PREVIOUS_LINK" ] || die "no previous release"
    current=$(readlink "$CURRENT_LINK")
    previous=$(readlink "$PREVIOUS_LINK")
    [ -d "$current" ] || die "current release is missing: $current"
    [ -d "$previous" ] || die "previous release is missing: $previous"
    atomic_link "$previous" "$CURRENT_LINK"
    atomic_link "$current" "$PREVIOUS_LINK"
    printf 'Rolled back to %s\n' "$previous"
    ;;
  set-token)
    token=$(prompt_token)
    validate_token "$token"
    mkdir -p "$APP_SUPPORT"
    umask 077
    temporary="$RUNTIME_ENV.new.$$"
    printf 'SB_HTTP_TOKEN=%s\n' "$token" > "$temporary"
    chmod 600 "$temporary"
    mv -f "$temporary" "$RUNTIME_ENV"
    launchctl kickstart -k "$domain/$LABEL"
    ;;
  uninstall)
    launchctl bootout "$domain/$LABEL" >/dev/null 2>&1 || true
    rm -f "$PLIST"
    printf 'LaunchAgent removed. Releases, logs, and runtime.env were preserved in:\n%s\n%s\n' "$APP_SUPPORT" "$LOG_DIR"
    ;;
  *) usage ;;
esac
