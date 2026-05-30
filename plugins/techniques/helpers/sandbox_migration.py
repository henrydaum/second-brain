"""One-time migration of legacy sandbox *skill* plugins to *technique*.

Before the skill→technique rename, agent-authored techniques lived in
``DATA_DIR/sandbox_skills/skill_<slug>.py`` and subclassed ``BaseSkill``. This
moves them to ``DATA_DIR/sandbox_techniques/technique_<slug>.py`` and rewrites
the base-class / import name so the current loader and AST validator accept
them.

This module deliberately references the legacy ``skill`` names — that is the
whole point of a migration. It is idempotent and best-effort: anything it can't
handle is left untouched rather than lost.
"""

from __future__ import annotations

import logging
from pathlib import Path

from paths import DATA_DIR

log = logging.getLogger("technique_migration")

_OLD_DIR = DATA_DIR / "sandbox_skills"
_NEW_DIR = DATA_DIR / "sandbox_techniques"
_OLD_PREFIX = "skill_"
_NEW_PREFIX = "technique_"


def migrate_sandbox_skills() -> None:
    """Move legacy ``sandbox_skills/skill_*.py`` to ``sandbox_techniques/
    technique_*.py``, rewriting ``BaseSkill`` -> ``BaseTechnique``.

    Run once at startup before plugin discovery. Safe to call every boot.
    """
    if not _OLD_DIR.is_dir():
        return
    _NEW_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in sorted(_OLD_DIR.glob("*.py")):
        name = old_path.name
        new_name = (_NEW_PREFIX + name[len(_OLD_PREFIX):]) if name.startswith(_OLD_PREFIX) else name
        new_path = _NEW_DIR / new_name
        if new_path.exists():
            continue  # already migrated — never clobber a newer file
        try:
            src = old_path.read_text(encoding="utf-8")
        except Exception:
            log.warning("Could not read legacy technique %s; leaving in place", old_path)
            continue
        # Only the base class + its import need to change for the file to load;
        # "plugins.BaseSkill" is covered because it contains "BaseSkill".
        new_src = src.replace("BaseSkill", "BaseTechnique")
        try:
            new_path.write_text(new_src, encoding="utf-8")
            old_path.unlink()
        except Exception:
            log.warning("Could not migrate legacy technique %s", old_path, exc_info=True)
            continue
        log.info("Migrated sandbox technique %s -> %s", name, new_name)

    # Drop the old directory once it is drained.
    try:
        if not any(_OLD_DIR.iterdir()):
            _OLD_DIR.rmdir()
    except Exception:
        pass
