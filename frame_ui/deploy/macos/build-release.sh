#!/bin/sh
set -eu

. "$(CDPATH= cd "$(dirname "$0")" && pwd -P)/lib.sh"
require_macos

command -v node >/dev/null 2>&1 || die "Node.js is not installed"
command -v npm >/dev/null 2>&1 || die "npm is not installed"

# `engines` in package.json says what the toolchain needs and npm only warns
# about it. A release built on a Node that Vite does not support fails in ways
# that read as application bugs, so it is checked here — while this build can
# still be abandoned without touching the one currently being served.
#
# Kept in step with the `engines` field by hand; there are two numbers and one
# other place they live.
node -e 'const [a,b]=process.versions.node.split(".").map(Number);
  if (!((a===20&&b>=19)||(a===22&&b>=12)||a>=23)) {
    console.error("Node "+process.versions.node+" is too old; package.json wants ^20.19.0 || >=22.12.0");
    process.exit(1);
  }' || die "upgrade Node.js, then run this again"

mkdir -p "$RELEASES_DIR"
release_name="$(date -u '+%Y%m%dT%H%M%SZ')-$$"
release_dir="$RELEASES_DIR/$release_name"

cleanup_failed_release() {
  if [ ! -f "$release_dir/index.html" ]; then
    case "$release_dir" in
      "$RELEASES_DIR"/*) rm -rf "$release_dir" ;;
    esac
  fi
}
trap cleanup_failed_release EXIT HUP INT TERM

cd "$REPO_ROOT"
npm ci
npm test
npm run lint
npm run build -- --outDir "$release_dir"

[ -f "$release_dir/index.html" ] || die "build completed without index.html"

old_current=""
if [ -L "$CURRENT_LINK" ]; then
  old_current=$(readlink "$CURRENT_LINK")
  [ -d "$old_current" ] || die "current release link is broken: $old_current"
fi

if [ -n "$old_current" ]; then
  atomic_link "$old_current" "$PREVIOUS_LINK"
fi
atomic_link "$release_dir" "$CURRENT_LINK"

# Keep exactly the releases needed for the active build and one-step rollback.
previous=""
if [ -L "$PREVIOUS_LINK" ]; then
  previous=$(readlink "$PREVIOUS_LINK")
fi
for held in "$RELEASES_DIR"/*; do
  [ -e "$held" ] || continue
  [ "$held" = "$release_dir" ] && continue
  [ -n "$previous" ] && [ "$held" = "$previous" ] && continue
  case "$held" in
    "$RELEASES_DIR"/*) rm -rf "$held" ;;
  esac
done

trap - EXIT HUP INT TERM
printf 'Activated %s\n' "$release_dir"
