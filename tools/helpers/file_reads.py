"""Shared read-before-edit tracking for the file tools.

Records which files the model has actually seen this conversation, keyed in
``session.plugin_state["file_reads"]`` (persisted with the conversation
marker). read_file records; edit_file checks before mutating. Timestamps are
``st_mtime_ns`` ints compared with ``!=`` — JSON-safe, and inequality also
catches a file restored to an *older* version.
"""

dependencies_files = []
dependencies_pip = []

import os

PLUGIN = "file_reads"
MAX_ENTRIES = 200

FRESH = "fresh"      # read, unchanged since
STALE = "stale"      # read, but the file changed on disk afterwards
UNREAD = "unread"    # never read this conversation
UNKNOWN = "unknown"  # no session state reachable — enforcement is skipped


def _key(path) -> str:
    """Canonical map key: symlinks resolved, case folded for Windows."""
    return os.path.normcase(os.path.realpath(str(path)))


def _bag(context) -> dict | None:
    """The session's read map, created on demand; None when unreachable."""
    runtime, key = getattr(context, "runtime", None), getattr(context, "session_key", None)
    session = getattr(runtime, "sessions", {}).get(key) if runtime and key else None
    state = getattr(session, "plugin_state", None)
    if state is None:
        return None
    return state.setdefault(PLUGIN, {})


def _mtime_ns(path) -> int | None:
    """Current st_mtime_ns, or None when unreadable."""
    try:
        return os.stat(str(path)).st_mtime_ns
    except OSError:
        return None


def record_read(context, path) -> None:
    """Mark ``path`` as seen at its current mtime."""
    bag = _bag(context)
    mtime = _mtime_ns(path)
    if bag is None or mtime is None:
        return
    key = _key(path)
    bag.pop(key, None)  # re-insert at the end so eviction is oldest-read-first
    bag[key] = mtime
    while len(bag) > MAX_ENTRIES:
        bag.pop(next(iter(bag)))


def forget(context, path) -> None:
    """Drop ``path`` from the read map (after a delete)."""
    bag = _bag(context)
    if bag is not None:
        bag.pop(_key(path), None)


def check(context, path) -> str:
    """Classify ``path`` as fresh / stale / unread / unknown for enforcement."""
    bag = _bag(context)
    if bag is None:
        return UNKNOWN
    recorded = bag.get(_key(path))
    if recorded is None:
        return UNREAD
    return FRESH if _mtime_ns(path) == recorded else STALE
