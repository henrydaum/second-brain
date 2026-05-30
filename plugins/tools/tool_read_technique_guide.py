"""read_technique_guide: technique authoring template + art_kit introspection.

Bare call →  the contents of templates/technique_template.py plus a tagline
            index of every art_kit helper (one line each, pulled live
            from the helper's docstring).
methods=…  →  for each named art_kit helper, return its signature,
            full docstring, and source code. No template, no index —
            focused follow-up reads.

The single source of truth for art_kit is plugins/techniques/helpers/art_kit.py.
This tool reflects whatever lives there; docstrings written on the helpers
flow straight to the agent.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from plugins.techniques.helpers import art_kit as _art_kit


_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "technique_template.py"


# Curated groups mirroring art_kit.build_namespace. Adding a new helper
# means adding its name here so it shows up in the index. The signature,
# docstring, and source come from art_kit itself — no duplication.
_GROUPS: list[tuple[str, list[str]]] = [
    ("Math",             ["lerp", "clamp", "smoothstep", "remap"]),
    ("Color",            ["hex_to_rgb", "rgb_to_hex", "mix_hex", "palette_color", "oklch_to_rgb"]),
    ("Composition",      ["rule_of_thirds", "vogel_spiral", "jittered_grid"]),
    ("Tiny 3D",          ["mesh", "cube_mesh", "render_3d"]),
    ("Noise",            ["value_noise", "fbm", "value_noise_grid", "fbm_grid"]),
    ("Masks",            ["radial_falloff"]),
    ("Numpy transforms", ["centered_grid", "bilinear_sample"]),
    ("Voronoi",          ["voronoi_nearest"]),
    ("Flow",             ["flow_field"]),
    ("L-systems",        ["lindenmayer", "turtle_segments"]),
    ("Waves",            ["wave_field"]),
    ("Attractors",       ["attractor_points"]),
    ("Text",             ["text", "text_bbox"]),
]


def _palette_color_inner_node() -> ast.FunctionDef | None:
    """Locate the inner `palette_color` def inside `_palette_color_fn`.

    `palette_color` is the only public art_kit helper built as a closure
    (its palette is bound at sandbox-startup time), so plain
    inspect.signature / getdoc on the namespace attribute would require a
    live canvas. Pull the metadata straight out of the source instead.
    """
    src = inspect.getsource(_art_kit._palette_color_fn)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "palette_color":
            return node
    return None


def _resolve(name: str) -> tuple[str, str, str] | None:
    """Return (signature, docstring, source) for an art_kit helper.

    ``signature`` is ``"name(args)"``; ``docstring`` is the cleaned docstring
    (``inspect.getdoc``); ``source`` is the function body (or the closure
    factory, for palette_color). Returns None if the name is unknown.
    """
    if name == "palette_color":
        node = _palette_color_inner_node()
        if node is None:
            return None
        try:
            sig = f"palette_color({ast.unparse(node.args)})"
        except Exception:
            sig = "palette_color(t, value=1.0)"
        doc = ast.get_docstring(node) or ""
        # Hand back the whole closure factory so the reader sees how the
        # palette is captured. The inner palette_color body is the
        # important part; the factory is small.
        source = inspect.getsource(_art_kit._palette_color_fn)
        return sig, doc, source

    fn = getattr(_art_kit, name, None)
    if fn is None or not callable(fn):
        return None
    try:
        sig = f"{name}{inspect.signature(fn)}"
    except (TypeError, ValueError):
        sig = f"{name}(...)"
    doc = inspect.getdoc(fn) or ""
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = ""
    return sig, doc, source


def _first_line(doc: str) -> str:
    for ln in doc.splitlines():
        s = ln.strip()
        if s:
            return s
    return ""


def _build_index() -> str:
    """Render the tagline index — one bullet per art_kit helper."""
    lines = [
        "## art_kit index",
        "",
        "Every helper below is available on the injected `art_kit` namespace "
        "(no import needed). Call `read_technique_guide(methods=[\"<name>\", ...])` "
        "to drill into any of them — you get full signature, docstring, and "
        "source. Reach for that whenever a tagline isn't enough.",
        "",
    ]
    for section, names in _GROUPS:
        lines.append(f"### {section}")
        for n in names:
            info = _resolve(n)
            if info is None:
                lines.append(f"- `art_kit.{n}` — [missing]")
                continue
            sig, doc, _ = info
            tag = _first_line(doc)
            lines.append(f"- `art_kit.{sig}` — {tag}" if tag else f"- `art_kit.{sig}`")
        lines.append("")
    lines.append("### Constants")
    lines.append("- `art_kit.pi`, `art_kit.tau` — `math.pi` / `math.tau` re-exports.")
    return "\n".join(lines)


def _build_details(names: list[str]) -> str:
    blocks: list[str] = []
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        info = _resolve(name)
        if info is None:
            blocks.append(
                f"## art_kit.{name}\n\n"
                f"[unknown helper — not exposed by art_kit]"
            )
            continue
        sig, doc, source = info
        block = [f"## art_kit.{sig}", ""]
        if doc:
            block.append(doc)
            block.append("")
        if source:
            block.append("```python")
            block.append(source.rstrip())
            block.append("```")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


class ReadTechniqueGuide(BaseTool):
    name = "read_technique_guide"
    description = (
        "Return the canvas-technique authoring template (palette discipline, "
        "technique kinds, control descriptors, composition rules, determinism "
        "contract, performance notes) plus a tagline index of every art_kit "
        "helper. Pass `methods=[\"fbm_grid\", \"vogel_spiral\", ...]` to "
        "drill into specific helpers and get full signature, docstring, and "
        "source. Call the bare form once per session before authoring a "
        "technique; call with `methods` whenever you need the exact shape of a "
        "helper while drafting."
    )
    max_calls = 8
    background_safe = True
    parameters = {
        "type": "object",
        "properties": {
            "methods": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of art_kit helper names. When provided, "
                    "the tool returns full signature + docstring + source "
                    "for each (and skips the template/index). When omitted, "
                    "returns the technique template followed by the art_kit "
                    "tagline index."
                ),
            }
        },
        "required": [],
    }

    def run(self, context, methods=None, **kwargs) -> ToolResult:
        if methods:
            text = _build_details(list(methods))
            return ToolResult(
                data={"methods": list(methods), "length": len(text)},
                llm_summary=text,
            )
        try:
            template = _TEMPLATE_PATH.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, error=f"could not read technique template: {e}")
        text = f"{template}\n\n---\n\n{_build_index()}\n"
        return ToolResult(data={"length": len(text)}, llm_summary=text)
