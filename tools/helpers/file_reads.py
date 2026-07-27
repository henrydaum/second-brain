"""Shared read-before-edit tracking for the file tools.

Records which files the model has actually seen this conversation, kept in the
session's ``file_reads`` plugin-state namespace (persisted with the
conversation marker). read_file records; edit_file checks before mutating.

Timestamps are ``st_mtime_ns`` ints compared with ``!=`` — JSON-safe, and
inequality also catches a file restored to an *older* version.

Sandboxed: every effect is a Request, so this loads in a subprocess as
happily as in-process. Two consequences worth knowing:

- The bag is read and written whole, because session state is one value per
  namespace rather than a live dict. At ``MAX_ENTRIES`` keys that is cheap,
  and it is the honest shape — nothing here holds a reference to kernel state
  between calls.
- Keys are normalized without resolving symlinks (``sdk.path.normalize`` never
  touches disk). Two names for one file through a link therefore read as
  different files, which fails toward "not read yet" — the strict direction.
"""

dependencies_files = []
dependencies_pip = []

PLUGIN = "file_reads"
MAX_ENTRIES = 200

FRESH = "fresh"      # read, unchanged since
STALE = "stale"      # read, but the file changed on disk afterwards
UNREAD = "unread"    # never read this conversation
UNKNOWN = "unknown"  # no session state reachable — enforcement is skipped


def _bag(sdk) -> dict | None:
    """The session's read map, or None when session state is unreachable.

    None is not an error: a tool called outside a session has nowhere to keep
    this, and the callers treat that as "skip enforcement" rather than
    "refuse the edit".
    """
    try:
        stored = sdk.session.state_get(namespace=PLUGIN)
    except sdk.Failed:
        return None
    return dict(stored) if isinstance(stored, dict) else {}


def _save(sdk, bag: dict) -> None:
    """Persist the read map, ignoring a session that cannot hold it."""
    try:
        sdk.session.state_set(bag, namespace=PLUGIN)
    except sdk.Failed:
        pass


def _mtime_ns(sdk, path) -> int | None:
    """Current ``st_mtime_ns``, or None when unreadable.

    ``fs.list`` pointed at a file answers for that file alone, which is how
    this asks for one stat without building a glob out of a filename.
    """
    try:
        entries = sdk.fs.list(path, details=True)
    except sdk.Failed:
        return None
    for entry in entries or []:
        if not entry.get("is_dir"):
            return entry.get("mtime")
    return None


def record_read(sdk, path) -> None:
    """Mark ``path`` as seen at its current mtime."""
    bag = _bag(sdk)
    mtime = _mtime_ns(sdk, path)
    if bag is None or mtime is None:
        return
    key = sdk.path.normalize(path)
    bag.pop(key, None)  # re-insert at the end so eviction is oldest-read-first
    bag[key] = mtime
    while len(bag) > MAX_ENTRIES:
        bag.pop(next(iter(bag)))
    _save(sdk, bag)


def forget(sdk, path) -> None:
    """Drop ``path`` from the read map (after a delete)."""
    bag = _bag(sdk)
    if bag is None:
        return
    if bag.pop(sdk.path.normalize(path), None) is not None:
        _save(sdk, bag)


def check(sdk, path) -> str:
    """Classify ``path`` as fresh / stale / unread / unknown for enforcement."""
    bag = _bag(sdk)
    if bag is None:
        return UNKNOWN
    recorded = bag.get(sdk.path.normalize(path))
    if recorded is None:
        return UNREAD
    return FRESH if _mtime_ns(sdk, path) == recorded else STALE
