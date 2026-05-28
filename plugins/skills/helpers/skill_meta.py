"""Shared skill-file inspection used by skill index/embed tasks."""

from __future__ import annotations

import ast
from pathlib import Path

from paths import ROOT_DIR, SANDBOX_SKILLS
from plugins.skills.helpers.skill_store import slugify

SKILL_DIRS = ((ROOT_DIR / "plugins" / "skills").resolve(), SANDBOX_SKILLS.resolve())


def is_skill_module(path: Path) -> bool:
    return path.suffix.lower() == ".py" and path.name.startswith("skill_") and _in_skill_dir(path)


def read_skill_meta(path: Path) -> dict | None:
    """Parse a skill file and return its declared metadata.

    Returns None when the file isn't a BaseSkill module. Raises ValueError
    when the class is missing a non-empty name or description.
    """
    code = path.read_text(encoding="utf-8")
    tree = ast.parse(code)
    if not any(
        isinstance(n, ast.ImportFrom)
        and n.module == "plugins.BaseSkill"
        and any(a.name == "BaseSkill" for a in n.names)
        for n in tree.body
    ):
        return None
    cls = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef)
         and any(_base_name(b) == "BaseSkill" for b in n.bases)),
        None,
    )
    if cls is None:
        return None
    vals = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in cls.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in {"name", "description", "kind", "hidden"}
    }
    name = str(vals.get("name") or "").strip()
    desc = str(vals.get("description") or "").strip()
    if not name or not desc:
        raise ValueError("skill file must declare non-empty name and description")
    return {
        "slug": slugify(name) or path.stem.removeprefix("skill_"),
        "name": name,
        "description": desc,
        "kind": str(vals.get("kind") or "background"),
        "hidden": int(bool(vals.get("hidden", False))),
    }


def _in_skill_dir(path: Path) -> bool:
    try:
        return path.resolve().parent in SKILL_DIRS
    except Exception:
        return False


def _base_name(node) -> str:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""
