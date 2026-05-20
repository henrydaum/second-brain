"""Shared render-and-commit helper used by canvas actions and tools.

Wraps the boring loop: resolve the skill loader from the bound runtime,
replay the chain into a temp PNG, then commit it onto the session's
composite path via ``layered_canvas.commit_image``. Keeps the
state-machine action classes from having to import PIL or know about
the skill registry layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from plugins.helpers.palettes import get_palette
from plugins.skills.helpers.skill_runner import replay_chain, run_skill
from plugins.tools.helpers import layered_canvas as lc


def _skill_loader_from_runtime() -> Any:
    runtime = getattr(lc, "_runtime_ref", None)
    registry = getattr(runtime, "skill_registry", None) if runtime else None
    if registry is None:
        return lambda _slug: None
    return registry.get_record


def render_chain(
    session_key: str,
    chain: list[dict],
    *,
    palette_id: str,
    size: int,
    op: str,
    out_name: str = "_render.png",
    chain_entry: dict | None = None,
    on_step=None,
) -> dict:
    """Replay ``chain`` into a temp PNG, commit it to the session, and
    return the canvas snapshot dict. Raises on render failure."""
    if not chain:
        raise ValueError("nothing to render")
    out = lc.image_path(session_key).with_name(out_name)
    replay_chain(
        chain,
        palette=get_palette(palette_id),
        size=int(size),
        output_image_path=out,
        workdir=out.parent,
        skill_loader=_skill_loader_from_runtime(),
        on_step=on_step,
    )
    with Image.open(out) as img:
        lc.commit_image(session_key, img.convert("RGBA"), op, chain_entry)
    return lc.canvas(session_key) or {}


def run_one_skill(
    session_key: str,
    skill,
    *,
    params: dict,
    palette_id: str,
    size: int,
    seed: int,
    input_image_path: Path | None,
    op: str,
    chain_entry: dict | None,
    timeout_s: float = 30.0,
    memory_mb: int = 768,
) -> dict:
    """Run a single skill, commit the result, return the canvas snapshot."""
    tmp = lc.image_path(session_key).with_name(f"_skill_{skill.slug}.png")
    run_skill(
        skill,
        params=params,
        palette=get_palette(palette_id),
        size=int(size),
        seed=int(seed),
        input_image_path=input_image_path,
        output_image_path=tmp,
        timeout_s=float(timeout_s),
        memory_mb=int(memory_mb),
    )
    with Image.open(tmp) as img:
        lc.commit_image(session_key, img.convert("RGBA"), op, chain_entry)
    return lc.canvas(session_key) or {}
