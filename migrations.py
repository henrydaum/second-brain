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
#: was never the memory system (that is ``memory/``), but it sits in a folder
#: where a person might reasonably have typed notes into a file by that name.
#: So it goes only if it is still empty.
_ORPHANS = (("memory.md", True), ("heartbeat", False))


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
