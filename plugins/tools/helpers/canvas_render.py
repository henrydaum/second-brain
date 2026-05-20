"""Shared render-and-commit helper used by canvas actions and tools.

Wraps the boring loop: replay the chain into a temp image, then commit
it onto the session's composite path via ``layered_canvas.commit_image``.
Keeps the state-machine action classes from having to import PIL or
know about the skill registry layout.
"""

from __future__ import annotations

from typing import Any, Callable

from PIL import Image

from plugins.helpers.palettes import get_palette
from plugins.skills.helpers.skill_runner import replay_chain
from plugins.tools.helpers import layered_canvas as lc


def render_chain(
    session_key: str,
    chain: list[dict],
    *,
    palette_id: str,
    size: int,
    op: str,
    out_name: str = "_render.png",
    on_step=None,
    canvas: Any = None,
    skill_loader: Callable[[str], Any] | None = None,
) -> dict:
    """Replay ``chain`` into a temp image, commit it, and return the
    frontend-shaped canvas dict.

    ``canvas`` (a ``Canvas`` instance) makes this a swap-on-success
    render: the draft canvas is mutated; the live ``cs.canvas`` is not
    touched until the caller assigns the draft. ``skill_loader`` comes
    from ``cs.cache['canvas_deps']`` so we don't reach back into the
    runtime."""
    if not chain:
        raise ValueError("nothing to render")
    if skill_loader is None:
        skill_loader = lambda _slug: None
    out = lc.image_path(session_key).with_name(out_name)
    replay_chain(
        chain,
        palette=get_palette(palette_id),
        size=int(size),
        output_image_path=out,
        workdir=out.parent,
        skill_loader=skill_loader,
        on_step=on_step,
    )
    with Image.open(out) as img:
        lc.commit_image(session_key, img.convert("RGBA"), op, None, canvas=canvas)
    return lc.to_frontend_shape(canvas) if canvas is not None else (lc.canvas(session_key) or {})
