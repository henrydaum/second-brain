"""Resolve the agent memory folder and topic file paths.

Memory is a folder of markdown topic files plus a ``MEMORY.md`` index:

    DATA_DIR/memory/
        MEMORY.md       # index: one line per topic — inlined into the prompt
        <topic>.md      # full memories — read on demand via the memory tool

The layout is per-user-ready: the base user's memory lives directly in
``DATA_DIR/memory/`` and every other user gets ``memory/users/<id>/``. This
resolver is the single place that knows that mapping — the kernel prompt
builder and the store ``memory`` tool both import it, so per-user memory
lights up everywhere the day the prompt plumbing carries a user id.
"""

from __future__ import annotations

import re
from pathlib import Path

from paths import DATA_DIR
from pipeline.database import DEFAULT_USER_ID

INDEX_FILENAME = "MEMORY.md"

_TOPIC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*$")


def memory_root(user_id: int | None = None) -> Path:
    """The memory folder for ``user_id`` (the base user when None)."""
    uid = DEFAULT_USER_ID if user_id is None else int(user_id)
    if uid == DEFAULT_USER_ID:
        return DATA_DIR / "memory"
    return DATA_DIR / "memory" / "users" / str(uid)


def topic_path(topic: str, user_id: int | None = None) -> Path:
    """Validated path for one topic file inside the user's memory root.

    Raises ``ValueError`` for names that are empty, reserved, contain path
    separators, or otherwise resolve outside the memory root.
    """
    name = (topic or "").strip()
    if name.lower().endswith(".md"):
        name = name[:-3]
    if not name or not _TOPIC_RE.match(name) or name.upper() == "MEMORY":
        raise ValueError(f"Invalid memory topic name: {topic!r}")
    root = memory_root(user_id)
    path = (root / f"{name}.md").resolve()
    if path.parent != root.resolve():
        raise ValueError(f"Memory topic escapes the memory folder: {topic!r}")
    return path


def list_topics(user_id: int | None = None) -> list[Path]:
    """Topic files in the user's memory root (the index excluded)."""
    root = memory_root(user_id)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.md") if p.is_file() and p.name != INDEX_FILENAME)
