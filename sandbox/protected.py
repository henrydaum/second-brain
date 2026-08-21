"""Files a Request never reads, whatever the path says.

``secret.reveal`` is UNSAFE and prompts; ``config.read`` hands back an opaque
handle. Both controls describe the *front* door. The back door is that
credentials also live in ordinary files — ``config.json`` holds every
``secret_*`` setting in plaintext — and ``fs.read`` is classified safe for any
path at all. Reading the config file therefore returned every API key with no
dialog, which made the whole handle mechanism decorative.

So the file itself is the boundary. This is deliberately a *path* deny-list
rather than a policy branch: the leak is not about who is asking or how, it is
that these bytes must not cross regardless. Keeping it in the handlers means
``fs.read``, ``fs.read_bytes`` and ``fs.search`` all get the same answer from
one place — ``fs.search`` matters as much as ``fs.read``, since it returns
matching *lines* and ``pattern="secret_"`` would do the job by itself.

**Writes go through here too**, which they did not always. The old reasoning
was that ``fs.write`` outside scratch is UNSAFE and therefore asked about — but
an approval is *type-level*: a command declaring ``fs.write`` in ``requests``
carries ``fs.write`` in ``chain.approved``, and every write it then makes is
classified SAFE. One "yes" to a command is not a person agreeing to have their
database overwritten, so these paths are refused outright rather than left to a
dialog that may never be shown. The kernel edits both files through its own
code and never through a Request, so nothing legitimate loses anything.

The database is on the list for a different reason — it is reachable through
``db.query``, which scopes rows per user and refuses ``password_hash``. Reading
the file directly would walk around all of it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Reasons, kept alongside the paths so a refusal can say which control it is
# protecting rather than just "no".
CONFIG_REASON = ("it holds secret_* settings in plaintext; "
                 "read them with sdk.config.read, which returns a handle")
DB_REASON = ("it is the database; query it with sdk.db.query, "
             "which scopes rows to the current user")

# Shared by mediated text reads and the narrow ``cat`` shell recognizer. A
# shell command that can return more than ``fs.read`` is not its safe alias.
MAX_TEXT_READ_BYTES = 8 * 1024 * 1024


def _resolve(path) -> Path | None:
    """Resolve without requiring existence, or None if it cannot be read."""
    try:
        return Path(path).resolve()
    except (OSError, ValueError, TypeError):
        return None


@lru_cache(maxsize=1)
def protected_paths() -> dict:
    """Resolved path -> why it is protected.

    Cached: this sits on the hot path of every read Request, and loading the
    config to answer it each time would make reading a file cost reading two.
    ``db_path`` does not change without a restart. Call :func:`reset` if it
    ever does.

    Built from the kernel's own constants rather than a hardcoded list, so
    this follows DATA_DIR wherever it actually is. Absent kernel (tests, a
    bare container) yields an empty map, which fails open by design: there is
    nothing to protect when there is no config and no database.
    """
    found = {}
    try:
        from config.config_manager import (_DEFAULT_CONFIG_PATH,
                                           _DEFAULT_PLUGIN_CONFIG_PATH)
        for raw in (_DEFAULT_CONFIG_PATH, _DEFAULT_PLUGIN_CONFIG_PATH):
            if (resolved := _resolve(raw)) is not None:
                found[resolved] = CONFIG_REASON
    except Exception:
        pass

    try:
        from config import config_manager
        raw = (config_manager.load() or {}).get("db_path")
        if raw and (resolved := _resolve(raw)) is not None:
            # SQLite's sidecars carry the same rows mid-transaction.
            for suffix in ("", "-wal", "-shm", "-journal"):
                found[Path(str(resolved) + suffix)] = DB_REASON
    except Exception:
        pass

    return found


def reason_for(path) -> str:
    """Why this path may not be read, or "" if it may be.

    Directories are not protected — listing a folder that contains the config
    reveals nothing the path constants do not already say.
    """
    resolved = _resolve(path)
    if resolved is None:
        return ""
    return protected_paths().get(resolved, "")


def is_protected(path) -> bool:
    """Whether this path is off limits to a read Request."""
    return bool(reason_for(path))


def reset():
    """Forget the cached set — for tests, and for a moved database."""
    protected_paths.cache_clear()
