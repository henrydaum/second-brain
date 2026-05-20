"""Migrate legacy /skills/*.skill.py files to /plugins/skills/skill_<slug>.py.

Legacy format: module-level SKILL_NAME/SKILL_DESCRIPTION/... constants and a
top-level ``def run(canvas, ...)``.

New format: a ``class <Name>(BaseSkill)`` with metadata as class attributes
and ``def run(self, canvas, ...)``. We reuse the existing
``wrap_user_code_in_class`` so the generated files match what the tools
emit when creating sandbox skills.

Run from repo root::

    python scripts/migrate_legacy_skills.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.skills.helpers.skill_store import (  # noqa: E402
    assert_valid,
    class_name_for_slug,
    slugify,
    validate_controls,
    wrap_user_code_in_class,
    extract_run_params,
    _run_param_names_in_body,
)

LEGACY_DIR = ROOT / "skills"
TARGET_DIR = ROOT / "plugins" / "skills"

META_FIELDS = {
    "SKILL_NAME": "name",
    "SKILL_DESCRIPTION": "description",
    "SKILL_KIND": "kind",
    "SKILL_OWNER": "owner",
    "SKILL_CREATED_AT": "created_at",
    "SKILL_HIDDEN": "hidden",
    "SKILL_CONTROLS": "controls",
}


def split_legacy(source: str) -> tuple[dict, str]:
    """Return (metadata, user_code). user_code keeps imports + run()."""
    tree = ast.parse(source)
    meta: dict = {}
    body_segments: list[str] = []

    for node in tree.body:
        # Pull metadata assignments out.
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in META_FIELDS
        ):
            key = META_FIELDS[node.targets[0].id]
            try:
                meta[key] = ast.literal_eval(node.value)
            except Exception as e:
                raise RuntimeError(f"could not eval {node.targets[0].id}: {e}")
            continue

        seg = ast.get_source_segment(source, node)
        if seg is None:
            continue
        body_segments.append(seg)

    user_code = "\n\n".join(body_segments) + "\n"
    return meta, user_code


def migrate_file(legacy_path: Path) -> Path:
    source = legacy_path.read_text(encoding="utf-8")
    meta, user_code = split_legacy(source)

    name = meta.get("name") or legacy_path.stem.replace(".skill", "").replace("_", " ").title()
    description = meta.get("description") or ""
    kind = meta.get("kind") or "creation"
    owner = meta.get("owner") or "library"
    created_at = float(meta.get("created_at") or 0.0)
    hidden = bool(meta.get("hidden", False))
    raw_controls = meta.get("controls") or []

    slug = slugify(name)
    target = TARGET_DIR / f"skill_{slug}.py"

    run_params = extract_run_params(user_code) or _run_param_names_in_body(user_code)
    controls = validate_controls(raw_controls, run_params, code=user_code)

    file_source = wrap_user_code_in_class(
        class_name=class_name_for_slug(slug),
        name=name,
        description=description,
        kind=kind,
        owner=owner,
        created_at=created_at,
        controls=controls,
        hidden=hidden,
        user_code=user_code,
    )
    assert_valid(file_source)
    target.write_text(file_source, encoding="utf-8")
    return target


def main() -> int:
    legacy_files = sorted(LEGACY_DIR.glob("*.skill.py"))
    if not legacy_files:
        print("no legacy skills found")
        return 0
    failed: list[tuple[Path, Exception]] = []
    for path in legacy_files:
        try:
            out = migrate_file(path)
            print(f"  {path.name:40s} -> {out.relative_to(ROOT)}")
        except Exception as e:
            failed.append((path, e))
            print(f"  FAIL {path.name}: {e}")
    print(f"\nmigrated {len(legacy_files) - len(failed)}/{len(legacy_files)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
