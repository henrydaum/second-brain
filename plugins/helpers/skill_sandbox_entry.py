"""Child process for skill execution."""

from __future__ import annotations

import importlib
import builtins as _builtins
import json
import platform
import re
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plugins.helpers.skill_store import validate_skill_code
from plugins.helpers.art_kit import build_namespace as _build_art_kit

ALLOWED = {"math", "random", "colorsys", "numpy", "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFilter", "PIL.ImageOps", "PIL.ImageEnhance", "PIL.ImageChops", "PIL.ImageColor"}

PALETTE_SLOTS = {"background", "primary", "secondary", "tertiary", "accent"}


def _limits():
    if platform.system() == "Linux":
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (35, 35))
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in ALLOWED and name.split(".")[0] not in ALLOWED:
        raise ImportError(f"import not allowed: {name}")
    for item in fromlist or ():
        full = f"{name}.{item}"
        if full in ALLOWED:
            importlib.import_module(full)
    return importlib.import_module(name)


class ColorValue(str):
    """Palette slot that works as a hex string and an RGB sequence."""

    def __new__(cls, value):
        return str.__new__(cls, value)

    @property
    def rgb(self):
        h = str(self).lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    def __getitem__(self, item):
        return self.rgb[item] if isinstance(item, int) else str.__getitem__(self, item)

    def __iter__(self):
        return iter(self.rgb)


class Canvas:
    def __init__(self, job):
        p = job["palette"]
        colors = {k: ColorValue(v) for k, v in p["colors"].items()}
        self.palette = SimpleNamespace(**colors, id=p["id"], name=p["name"], kind=p["kind"], colors=colors)
        self.size = self.width = self.height = int(job["size"])
        self.seed = int(job["seed"])
        self._image = Image.open(job["input_image_path"]).convert("RGBA") if job.get("input_image_path") else None
        self._committed = None

    @property
    def image(self):
        if self._image is None:
            raise ValueError("canvas.image is only available to transform skills")
        return self._image.copy()

    def new(self, w=None, h=None, color=None):
        if isinstance(w, str) and h is None and color is None:
            color, w = w, None
        return Image.new("RGBA", (int(w or self.size), int(h or w or self.size)), color or self.palette.background)

    def create_image(self, color=None):
        return self.new(color=color or self.palette.background)

    def commit(self, image):
        self._committed = image.convert("RGBA")


def _hint_for(error_type: str, message: str, skill_line: str) -> str | None:
    """Pattern-match common skill bugs to a one-line corrective hint."""
    msg = message or ""
    line = skill_line or ""

    if "import not allowed" in msg:
        m = re.search(r"import not allowed:\s*(\S+)", msg)
        name = m.group(1) if m else "that module"
        return f"Only math, random, colorsys, numpy, and PIL.* are importable inside a skill. Got '{name}'. Drop that import or replace it with the allowed equivalents."

    if "did not call canvas.commit" in msg:
        return "Your run() finished without calling canvas.commit(image). Every code path must end with canvas.commit(img). If you built a numpy array, wrap it: canvas.commit(Image.fromarray(arr, 'RGB').convert('RGBA'))."

    if "canvas.image is only available to transform skills" in msg:
        return "This is a creation skill — canvas.image only exists for transform skills. Start a fresh image with canvas.new(color=canvas.palette.background) or canvas.create_image()."

    if error_type == "AttributeError":
        m = re.search(r"has no attribute '([^']+)'", msg)
        attr = m.group(1) if m else ""
        if attr and attr not in PALETTE_SLOTS and ("palette" in line or "palette" in msg.lower()):
            return f"Palette slots are background, primary, secondary, tertiary, accent. '{attr}' isn't one of them — pick one of those."

    if error_type in {"ValueError", "TypeError"} and ("Image" in line or "fromarray" in line or "image data" in msg.lower() or "buffer is not large enough" in msg.lower()):
        return "Image arrays must be shape (h, w, 3) for RGB or (h, w, 4) for RGBA, with dtype=uint8. Check arr.shape and arr.dtype before Image.fromarray()."

    if error_type in {"IndexError", "ValueError"} and ("broadcast" in msg.lower() or "shape" in msg.lower() or "out of bounds" in msg.lower()):
        return f"Numpy shape/index mismatch on the failing line ({line.strip()!r}). Print .shape of each array right before this line to find the mismatch."

    if error_type == "ZeroDivisionError":
        return "A divisor reached zero — usually a normalization step where the data span collapsed. Guard with `(span or 1.0)` or fall back to a default."

    if error_type == "NameError":
        m = re.search(r"name '([^']+)' is not defined", msg)
        name = m.group(1) if m else ""
        if name == "np":
            return "You used `np` but didn't import numpy. Add `import numpy as np` at the top of the skill."
        if name == "Image":
            return "You used `Image` but didn't import it. Add `from PIL import Image` at the top of the skill."
        if name == "math":
            return "You used `math` but didn't import it. Add `import math` at the top of the skill."

    return None


def _diagnose(exc: BaseException, code: str) -> dict:
    """Walk the traceback to the innermost frame inside the skill and build a diagnostic dict."""
    tb = exc.__traceback__
    skill_lineno = None
    skill_line = ""
    while tb is not None:
        if tb.tb_frame.f_globals.get("__name__") == "__skill__":
            skill_lineno = tb.tb_lineno
            skill_line = (code.splitlines()[skill_lineno - 1] if 0 < skill_lineno <= len(code.splitlines()) else "").rstrip()
        tb = tb.tb_next

    error_type = type(exc).__name__
    message = str(exc) or error_type
    hint = _hint_for(error_type, message, skill_line)

    summary = traceback.format_exception_only(type(exc), exc)
    summary_text = "".join(summary).strip()

    return {
        "error_type": error_type,
        "message": message,
        "skill_lineno": skill_lineno,
        "skill_line": skill_line,
        "hint": hint,
        "summary": summary_text,
    }


def _blank_canvas_check(img: Image.Image) -> dict | None:
    """Cheap heuristic: did the skill commit a flat-color image?"""
    try:
        import numpy as np
        arr = np.asarray(img.convert("RGB"))
        sample = arr[::8, ::8].reshape(-1, 3)
        if sample.size == 0:
            return None
        std = float(sample.astype(np.float32).std(axis=0).max())
        # `np.unique` on rows: collapse to a uint32 key for speed.
        keys = (sample[:, 0].astype(np.uint32) << 16) | (sample[:, 1].astype(np.uint32) << 8) | sample[:, 2].astype(np.uint32)
        unique_ratio = float(np.unique(keys).size) / float(keys.size)
        if std < 4.0 and unique_ratio < 0.005:
            return {
                "warning": "blank_canvas",
                "message": "Rendered image appears to be a flat background color — the subject may not have been drawn. If the user asked for a plain background, ignore this. Otherwise adjust and re-run.",
                "std": std,
                "unique_ratio": unique_ratio,
            }
    except Exception:
        return None
    return None


def _write_sidecar(output_image_path: str, payload: dict) -> None:
    try:
        Path(output_image_path + ".err.json").write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def main():
    _limits()
    job_path = sys.argv[1]
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    output_image_path = job["output_image_path"]
    code = job["code"]

    try:
        errors = validate_skill_code(code)
        if errors:
            raise ValueError("; ".join(errors))
        canvas = Canvas(job)
        safe = {k: getattr(_builtins, k) for k in ("abs", "all", "any", "bool", "dict", "enumerate", "Exception", "filter", "float", "int", "len", "list", "map", "max", "min", "pow", "range", "round", "set", "sorted", "str", "sum", "tuple", "ValueError", "zip")}
        safe.update({"__import__": _import, "print": lambda *a, **k: None})
        art_kit = _build_art_kit(canvas.palette)
        ns = {"__builtins__": safe, "__name__": "__skill__", "art_kit": art_kit}
        exec(code, ns, ns)
        result = ns["run"](canvas, **(job.get("params") or {}))
        if canvas._committed is None and isinstance(result, Image.Image):
            canvas.commit(result)
        if canvas._committed is None:
            raise ValueError("skill did not call canvas.commit(image)")
        canvas._committed.save(output_image_path, "PNG")
    except BaseException as exc:
        diag = _diagnose(exc, code)
        _write_sidecar(output_image_path, diag)
        # Print a compact, agent-friendly summary as the last stderr line so
        # older callers that only read the last line still get useful info.
        tail = f"{diag['error_type']}: {diag['message']}"
        if diag.get("skill_lineno"):
            tail += f" (line {diag['skill_lineno']}: {diag['skill_line'].strip()!r})"
        print(tail, file=sys.stderr)
        sys.exit(1)

    warning = _blank_canvas_check(canvas._committed)
    if warning is not None:
        _write_sidecar(output_image_path, warning)


if __name__ == "__main__":
    main()
