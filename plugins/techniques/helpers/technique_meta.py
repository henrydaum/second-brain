"""Shared technique-file inspection used by technique index/embed tasks."""

from __future__ import annotations

import ast
from pathlib import Path

from paths import ROOT_DIR, SANDBOX_TECHNIQUES
from plugins.techniques.helpers.technique_store import slugify

TECHNIQUE_DIRS = ((ROOT_DIR / "plugins" / "techniques").resolve(), SANDBOX_TECHNIQUES.resolve())


def is_technique_module(path: Path) -> bool:
    return path.suffix.lower() == ".py" and path.name.startswith("technique_") and _in_technique_dir(path)


def read_technique_meta(path: Path) -> dict | None:
    """Parse a technique file and return its declared metadata.

    Returns None when the file isn't a BaseTechnique module. Raises ValueError
    when the class is missing a non-empty name or description.
    """
    code = path.read_text(encoding="utf-8")
    tree = ast.parse(code)
    if not any(
        isinstance(n, ast.ImportFrom)
        and n.module == "plugins.BaseTechnique"
        and any(a.name == "BaseTechnique" for a in n.names)
        for n in tree.body
    ):
        return None
    cls = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef)
         and any(_base_name(b) == "BaseTechnique" for b in n.bases)),
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
        raise ValueError("technique file must declare non-empty name and description")
    return {
        "slug": slugify(name) or path.stem.removeprefix("technique_"),
        "name": name,
        "description": desc,
        "kind": str(vals.get("kind") or "background"),
        "hidden": int(bool(vals.get("hidden", False))),
    }


def _in_technique_dir(path: Path) -> bool:
    try:
        return path.resolve().parent in TECHNIQUE_DIRS
    except Exception:
        return False


def _base_name(node) -> str:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""
