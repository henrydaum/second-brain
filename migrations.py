"""One-shot DATA_DIR layout migrations, run at boot before anything reads it.

Stdlib only and import-light: this runs ahead of the app proper, in the same
spirit as the supervisor's raw config peek in ``main.pyw``. Every step is
idempotent and guarded by "does the old thing still exist", so a second boot is
a no-op and an interrupted run resumes.

The rule throughout is **never guess**. A file that does not match what a step
expects is left where it is and reported, because the cost of leaving a stray
file behind is a log line and the cost of moving the wrong one is an agent's
work disappearing into a folder nobody looks in.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from paths import DATA_DIR

#: Old tree folder -> new. The trees were named for what they held, which
#: stopped being true once they held parsers, backends and scripts as well;
#: they are named for *where they came from* now. ``sandbox_plugins`` also
#: collided with ``sandbox/``, the security boundary, which is two unrelated
#: things sharing a word in a codebase where one of them is load-bearing.
_TREE_RENAMES = (
    ("sandbox_plugins", "workspace"),
    ("installed_plugins", "installed"),
)

#: Files that used to share one ``helpers/`` root, split out by prefix into the
#: roots their kernel registries now scan.
_HELPER_SPLITS = (
    ("parse_", "parsers"),
    ("llm_", "llm"),
)

#: Junk from features that no longer exist, and whether deleting one is
#: allowed to destroy content.
#:
#: ``heartbeat`` holds "<pid> <timestamp>" written by the stall watchdog that
#: ``main.pyw`` explains the removal of — there is no reading of those two
#: numbers that is worth keeping, so it goes whatever it contains.
#: ``memory.md`` is different: it was only ever ``touch``-ed empty at boot and
#: was never the memory system (that is ``workspace/memory/``), but it sits in a folder
#: where a person might reasonably have typed notes into a file by that name.
#: So it goes only if it is still empty.
_ORPHANS = (("memory.md", True), ("heartbeat", False))

#: The attachment cache, before and after it moved into the agent's own tree.
#: Relative to DATA_DIR, and ``_ATTACHMENTS_NEW`` must agree with
#: ``trees.attachment_cache()`` — restated rather than imported because this
#: module runs ahead of the app and stays stdlib-only, and because the whole
#: job here is to name the *old* spelling, which no live module may.
_ATTACHMENTS_OLD = ("attachment_cache",)
_ATTACHMENTS_NEW = ("workspace", "attachments")

#: The memory folder, before and after it moved into the agent's own tree.
#: Same restatement rule as the attachment cache above: this module names the
#: *old* spelling, which no live module may. The new one must agree with
#: ``agent.system_prompt._agent_memory`` and the store's memory tool.
_MEMORY_OLD = ("memory",)
_MEMORY_NEW = ("workspace", "memory")


def migrate(data_dir: Path | None = None) -> list[str]:
    """Bring an existing DATA_DIR up to the current layout.

    Returns one line per thing actually done — empty when there was nothing to
    do, which is the normal case after the first run.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    if not root.is_dir():
        return []
    done: list[str] = []
    done += _rename_trees(root)
    done += _split_helpers(root)
    # After the rename, so the destination tree is already called ``workspace``.
    done += _move_attachment_cache(root)
    done += _move_memory(root)
    done += _drop_orphans(root)
    return done


def _rename_trees(root: Path) -> list[str]:
    """Rename the two DATA_DIR trees, refusing to merge into an existing one."""
    done = []
    for old_name, new_name in _TREE_RENAMES:
        old, new = root / old_name, root / new_name
        if not old.is_dir():
            continue
        if new.exists():
            # Both present means a half-finished migration or a hand-made
            # folder. Merging could silently shadow one tree's file with
            # another's, so this stops and says so.
            done.append(f"! {old_name}/ and {new_name}/ both exist — "
                        f"left {old_name}/ alone, merge it by hand")
            continue
        old.rename(new)
        done.append(f"{old_name}/ -> {new_name}/")
    return done


def _split_helpers(root: Path) -> list[str]:
    """Move parsers and LLM backends out of each tree's old ``helpers/`` root.

    The root itself goes only if it ends up empty. Anything else in there was
    a shared library with no family, which the layout no longer has a place
    for — that is a decision for a person, not for a boot step.
    """
    done = []
    for _old, tree_name in _TREE_RENAMES:
        helpers = root / tree_name / "helpers"
        if not helpers.is_dir():
            continue
        for source in sorted(helpers.glob("*.py")):
            destination_root = next(
                (name for prefix, name in _HELPER_SPLITS
                 if source.name.startswith(prefix)), None)
            if destination_root is None:
                continue
            target_dir = root / tree_name / destination_root
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source.name
            if target.exists():
                done.append(f"! {tree_name}/{destination_root}/{source.name} "
                            f"already exists — left the copy in helpers/")
                continue
            source.rename(target)
            done.append(f"{tree_name}/helpers/{source.name} -> "
                        f"{tree_name}/{destination_root}/{source.name}")
        leftovers = [p.name for p in helpers.iterdir()
                     if p.name != "__pycache__"]
        if leftovers:
            done.append(f"! {tree_name}/helpers/ still holds {len(leftovers)} "
                        f"file(s) with no root to move to: "
                        f"{', '.join(sorted(leftovers)[:5])}")
            continue
        shutil.rmtree(helpers, ignore_errors=True)
        done.append(f"removed empty {tree_name}/helpers/")
    return done


def _move_attachment_cache(root: Path) -> list[str]:
    """Move ``attachment_cache/`` into the workspace tree, and repoint config.

    Two halves, and skipping the second would be the silent failure: the
    folder is a ``sync_directory`` by default, so a config still naming the
    old path leaves incoming attachments unindexed *and* the watcher pointed
    at a directory that no longer exists — neither of which shows up as an
    error anybody reads.

    Files move one at a time rather than as a folder rename, because the
    destination existing is the ordinary case here (anything that has already
    booted once has a ``workspace/``), and a name that collides is left where
    it is and reported. Attachment names carry a unix timestamp, so a
    collision means two files that genuinely differ.
    """
    old = root.joinpath(*_ATTACHMENTS_OLD)
    if not old.is_dir():
        return _repoint_sync_directories(root)

    new = root.joinpath(*_ATTACHMENTS_NEW)
    done = []
    moved = kept = 0
    try:
        new.mkdir(parents=True, exist_ok=True)
        for source in sorted(old.iterdir()):
            if not source.is_file():
                continue
            target = new / source.name
            if target.exists():
                kept += 1
                continue
            source.rename(target)
            moved += 1
    except OSError as exc:
        done.append(f"! could not finish moving {_ATTACHMENTS_OLD[0]}/: {exc}")
        return done + _repoint_sync_directories(root)

    where = "/".join(_ATTACHMENTS_NEW)
    if moved:
        done.append(f"{_ATTACHMENTS_OLD[0]}/ -> {where}/ ({moved} file(s))")
    if kept:
        done.append(f"! {kept} file(s) left in {_ATTACHMENTS_OLD[0]}/ — "
                    f"{where}/ already holds that name")
    else:
        try:
            leftovers = [p.name for p in old.iterdir()]
        except OSError:
            leftovers = ["?"]
        if leftovers:
            done.append(f"! {_ATTACHMENTS_OLD[0]}/ still holds "
                        f"{len(leftovers)} entry(s) that are not files")
        else:
            shutil.rmtree(old, ignore_errors=True)
            if not moved:
                done.append(f"removed empty {_ATTACHMENTS_OLD[0]}/")
    return done + _repoint_sync_directories(root)


def _move_memory(root: Path) -> list[str]:
    """Move ``memory/`` into the workspace tree.

    Memory is the one thing the agent is asked to keep current about itself,
    and it sat in a folder the agent had no standing write grant for — so
    every save went through an approval dialog for a file the system's own
    prompt tells it to maintain. Inside ``workspace/`` it inherits the
    free-write grant, and ``MEMORY.md`` travels with the topics that index it.

    Nothing in config names this path, so unlike the attachment cache there is
    no second half to get wrong. Files move one at a time for the same reason:
    ``workspace/`` already exists on anything that has booted once, and a
    colliding topic name means two files that genuinely differ.
    """
    old = root.joinpath(*_MEMORY_OLD)
    if not old.is_dir():
        return []

    new = root.joinpath(*_MEMORY_NEW)
    where = "/".join(_MEMORY_NEW)
    done: list[str] = []
    moved = kept = 0
    try:
        new.mkdir(parents=True, exist_ok=True)
        for source in sorted(old.iterdir()):
            if not source.is_file():
                continue
            target = new / source.name
            if target.exists():
                kept += 1
                continue
            source.rename(target)
            moved += 1
    except OSError as exc:
        return [f"! could not finish moving {_MEMORY_OLD[0]}/: {exc}"]

    if moved:
        done.append(f"{_MEMORY_OLD[0]}/ -> {where}/ ({moved} file(s))")
    if kept:
        done.append(f"! {kept} file(s) left in {_MEMORY_OLD[0]}/ — "
                    f"{where}/ already holds that name")
        return done
    try:
        leftovers = [p.name for p in old.iterdir()]
    except OSError:
        leftovers = ["?"]
    if leftovers:
        done.append(f"! {_MEMORY_OLD[0]}/ still holds "
                    f"{len(leftovers)} entry(s) that are not files")
    else:
        shutil.rmtree(old, ignore_errors=True)
        if not moved:
            done.append(f"removed empty {_MEMORY_OLD[0]}/")
    return done


def _repoint_sync_directories(root: Path) -> list[str]:
    """Rewrite the old attachment cache path in ``config.json``.

    Only ever an exact-match replacement of the one path this migration knows
    it moved, and only when the new path is not already listed. Everything
    else in the file is written back untouched — an unreadable or
    unrecognisable config is left entirely alone, since the app is about to
    load it properly and a boot step guessing at JSON is how a person's
    settings disappear.
    """
    config_path = root / "config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    listed = data.get("sync_directories")
    if not isinstance(listed, list):
        return []

    old = str(root.joinpath(*_ATTACHMENTS_OLD))
    new = str(root.joinpath(*_ATTACHMENTS_NEW))
    rewritten, changed = [], False
    for entry in listed:
        if isinstance(entry, str) and _same_path(entry, old):
            changed = True
            if not any(isinstance(other, str) and _same_path(other, new)
                       for other in listed):
                rewritten.append(new)
            continue
        rewritten.append(entry)
    if not changed:
        return []

    data["sync_directories"] = rewritten
    try:
        config_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    except OSError as exc:
        return [f"! could not repoint sync_directories: {exc}"]
    return ["sync_directories now points at the moved attachment cache"]


def _same_path(left: str, right: str) -> bool:
    """Whether two spellings name the same directory.

    Config paths are hand-edited and round-tripped through JSON, so the old
    entry can differ from what this module builds by a separator, a trailing
    slash, or Windows case. Compared as text rather than resolved, because a
    directory that has just been moved away no longer exists to resolve.
    """
    def key(value: str) -> str:
        return value.replace("\\", "/").rstrip("/").casefold()
    return key(left) == key(right)


def _drop_orphans(root: Path) -> list[str]:
    """Delete leftovers from removed features."""
    done = []
    for name, only_if_empty in _ORPHANS:
        path = root / name
        try:
            if not path.is_file():
                continue
            if only_if_empty and path.stat().st_size:
                done.append(f"! kept {name} — it is no longer used but is "
                            f"not empty")
                continue
            path.unlink()
            done.append(f"removed orphaned {name}")
        except OSError:
            continue
    return done
