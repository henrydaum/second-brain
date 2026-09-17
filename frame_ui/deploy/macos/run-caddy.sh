#!/bin/sh
set -eu

. "$(CDPATH= cd "$(dirname "$0")" && pwd -P)/lib.sh"
require_macos

SB_HTTP_TOKEN=$(runtime_token)
SB_UI_CURRENT=$CURRENT_LINK
export SB_HTTP_TOKEN SB_UI_CURRENT

[ -f "$CURRENT_LINK/index.html" ] || die "no active frontend release; run build-release.sh"
command -v caddy >/dev/null 2>&1 || die "Caddy is not installed"

exec caddy run --config "$CADDYFILE" --adapter caddyfile
