"""Child process for technique execution.

The parent (technique_runner.py) spawns this with ``python -I -B`` and hands it
a JSON job over a temp file. We re-validate the source via AST, exec the
module under a restricted ``__builtins__``, locate the BaseTechnique subclass,
instantiate it, and call ``instance.run(canvas, **params)``.

Imports inside the user's technique go through ``_import``, which permits only
the literal entries in ``ALLOWED`` (plus the ``plugins.BaseTechnique`` import
that every technique needs). ``plugins`` is NOT admitted as a top-level alias
— that would let a technique reach into the rest of the helpers tree.
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from plugins.techniques.helpers.technique_store import validate_technique_code
from plugins.techniques.helpers.art_kit import build_namespace as _build_art_kit
from plugins.BaseTechnique import BaseTechnique

ALLOWED = {
    "math", "random", "colorsys",
    "numpy", "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFilter",
    "PIL.ImageOps", "PIL.ImageEnhance", "PIL.ImageChops", "PIL.ImageColor",
    "plugins.BaseTechnique",
}

PALETTE_SLOTS = {"background", "primary", "secondary", "tertiary", "accent"}


def _limits(memory_mb: int = 768):
    # RLIMIT_CPU works on Linux + Darwin. RLIMIT_AS is unreliable on Darwin
    # across malloc backends, so memory enforcement is the parent's psutil
    # watchdog (see technique_runner.run_technique); we only set RLIMIT_AS on Linux
    # as belt-and-braces.
    try:
        import resource
    except ImportError:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (35, 35))
    except (ValueError, OSError):
        pass
    if platform.system() == "Linux":
        try:
            cap = max(64, int(memory_mb)) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except (ValueError, OSError):
            pass


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError(f"relative import not allowed: {name}")
    if name in ALLOWED:
        for item in fromlist or ():
            full = f"{name}.{item}"
            if full in ALLOWED:
                importlib.import_module(full)
        return importlib.import_module(name)
    # Top-level fallback for things like `PIL.ImageDraw.foo`. Never for
    # `plugins.*` — only literal `plugins.BaseTechnique` is admitted above.
    top = name.split(".")[0]
    if top == "plugins" or top not in ALLOWED:
        raise ImportError(f"import not allowed: {name}")
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
        self.kind = str(job.get("kind") or "background")
        self._image = Image.open(job["input_image_path"]).convert("RGBA") if job.get("input_image_path") else None
        self._committed = None

    @property
    def image(self):
        if self._image is None:
            raise ValueError("canvas.image is only available to filter and object techniques")
        return self._image.copy()

    def new(self, w=None, h=None, color=None):
        if isinstance(w, str) and h is None and color is None:
            color, w = w, None
        return Image.new("RGBA", (int(w or self.size), int(h or w or self.size)), color or self.palette.background)

    def create_image(self, color=None):
        return self.new(color=color or self.palette.background)

    def new_layer(self, w=None, h=None):
        """Return a fully-transparent RGBA image at canvas size — the
        starting point for an object technique. The framework alpha-composites
        whatever you commit onto the prior canvas, so paint only what you
        want visible."""
        return Image.new("RGBA", (int(w or self.size), int(h or w or self.size)), (0, 0, 0, 0))

    def commit(self, image):
        self._committed = image.convert("RGBA")

    def commit_array(self, arr):
        """Commit a numpy HxWxC array. Accepts float in [0, 1] or uint8 in [0, 255];
        C may be 3 (RGB) or 4 (RGBA). Saves the numpy round-trip boilerplate that
        every numeric filter would otherwise repeat: clip, scale, dtype convert,
        Image.fromarray, convert to RGBA, then commit."""
        import numpy as _np
        a = _np.asarray(arr)
        if a.ndim != 3 or a.shape[-1] not in (3, 4):
            raise ValueError(
                f"commit_array expected an HxWx3 or HxWx4 array, got shape {a.shape}"
            )
        if a.dtype.kind == "f":
            a = _np.clip(a * 255.0, 0.0, 255.0).astype(_np.uint8)
        elif a.dtype != _np.uint8:
            a = _np.clip(a, 0, 255).astype(_np.uint8)
        mode = "RGB" if a.shape[-1] == 3 else "RGBA"
        self._committed = Image.fromarray(a, mode).convert("RGBA")

    def image_array(self, mode="RGB", dtype="float"):
        """Return the current image as a numpy array. ``mode`` is "RGB" or "RGBA"
        (passed through to PIL). ``dtype="float"`` yields float32 in [0, 1];
        ``dtype="uint8"`` yields the raw bytes. Filter and object techniques only."""
        import numpy as _np
        if self._image is None:
            raise ValueError("canvas.image_array is only available to filter and object techniques")
        img = self._image.convert(str(mode))
        arr = _np.asarray(img)
        if str(dtype) == "float":
            return arr.astype(_np.float32) / 255.0
        if str(dtype) == "uint8":
            return arr.copy()
        raise ValueError(f"dtype must be 'float' or 'uint8', got {dtype!r}")


def _hint_for(error_type: str, message: str, technique_line: str) -> str | None:
    """Pattern-match common technique bugs to a one-line corrective hint."""
    msg = message or ""
    line = technique_line or ""

    if error_type == "PermissionError" and "assigned canvas paths" in msg:
        return "Techniques can't open or save arbitrary file paths. Read the input through `canvas.image` (filter/object techniques only) and commit your result with `canvas.commit(image)`. The parent saves it."

    if "import not allowed" in msg:
        m = re.search(r"import not allowed:\s*(\S+)", msg)
        name = m.group(1) if m else "that module"
        return f"Only math, random, colorsys, numpy, PIL.*, and `from plugins.BaseTechnique import BaseTechnique` are importable inside a technique. Got '{name}'. Drop that import or replace it with the allowed equivalents."

    if "did not call canvas.commit" in msg:
        return "Your run() finished without calling canvas.commit(image). Every code path must end with canvas.commit(img). If you built a numpy array, wrap it: canvas.commit(Image.fromarray(arr, 'RGB').convert('RGBA'))."

    if "canvas.image is only available to filter" in msg:
        return "This is a background technique — canvas.image only exists for filter and object techniques. Start a fresh image with canvas.new(color=canvas.palette.background) or canvas.create_image()."

    if "no BaseTechnique subclass found" in msg or "must define" in msg:
        return "Every technique must define `class <Name>(BaseTechnique):` with a `def run(self, canvas):` method. Wrap your code accordingly."

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
            return "You used `np` but didn't import numpy. Add `import numpy as np` at the top of the technique."
        if name == "Image":
            return "You used `Image` but didn't import it. Add `from PIL import Image` at the top of the technique."
        if name == "math":
            return "You used `math` but didn't import it. Add `import math` at the top of the technique."
        if name == "BaseTechnique":
            return "You used `BaseTechnique` but didn't import it. Add `from plugins.BaseTechnique import BaseTechnique` at the top of the technique."
        if name in {"Slider", "Enum", "Bool", "Pan", "Text", "Palette"}:
            return (
                f"You used `{name}` but didn't import it. Descriptor classes live alongside BaseTechnique: "
                f"`from plugins.BaseTechnique import BaseTechnique, {name}` (combine with whatever else you already import)."
            )

    return None


def _diagnose(exc: BaseException, code: str) -> dict:
    """Walk the traceback to the innermost frame inside the technique and build a diagnostic dict."""
    tb = exc.__traceback__
    technique_lineno = None
    technique_line = ""
    while tb is not None:
        if tb.tb_frame.f_globals.get("__name__") == "__technique__":
            technique_lineno = tb.tb_lineno
            technique_line = (code.splitlines()[technique_lineno - 1] if 0 < technique_lineno <= len(code.splitlines()) else "").rstrip()
        tb = tb.tb_next

    error_type = type(exc).__name__
    message = str(exc) or error_type
    hint = _hint_for(error_type, message, technique_line)

    summary = traceback.format_exception_only(type(exc), exc)
    summary_text = "".join(summary).strip()

    return {
        "error_type": error_type,
        "message": message,
        "technique_lineno": technique_lineno,
        "technique_line": technique_line,
        "hint": hint,
        "summary": summary_text,
    }


def _blank_canvas_check(img: Image.Image, canvas: "Canvas") -> dict | None:
    try:
        import numpy as np
        arr = np.asarray(img.convert("RGB"))
        sample = arr[::8, ::8].reshape(-1, 3)
        if sample.size == 0:
            return None
        std = float(sample.astype(np.float32).std(axis=0).max())
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


def _palette_adherence_check(img: Image.Image, canvas: "Canvas") -> dict | None:
    try:
        import numpy as np
        slots = ("background", "tertiary", "secondary", "primary", "accent")
        colors: list[tuple[int, int, int]] = []
        for slot in slots:
            hexv = getattr(canvas.palette, slot, None)
            if hexv is None:
                continue
            h = str(hexv).lstrip("#")
            colors.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
        if len(colors) < 2:
            return None
        colors_arr = np.array(colors, dtype=np.float32)
        lum = 0.2126 * colors_arr[:, 0] + 0.7152 * colors_arr[:, 1] + 0.0722 * colors_arr[:, 2]
        colors_arr = colors_arr[np.argsort(lum)]
        ramp_pts = []
        for i in range(len(colors_arr) - 1):
            ts = np.linspace(0.0, 1.0, 16, endpoint=False)[:, None]
            seg = colors_arr[i] * (1 - ts) + colors_arr[i + 1] * ts
            ramp_pts.append(seg)
        ramp_pts.append(colors_arr[-1:])
        ramp = np.concatenate(ramp_pts, axis=0)

        arr = np.asarray(img.convert("RGB"))
        sample = arr[::8, ::8].reshape(-1, 3).astype(np.float32)
        if sample.size == 0:
            return None
        diff = sample[:, None, :] - ramp[None, :, :]
        min_dist = np.sqrt((diff * diff).sum(axis=2)).min(axis=1)
        off_ratio = float((min_dist > 60.0).mean())
        if off_ratio > 0.15:
            return {
                "warning": "palette_drift",
                "message": f"{int(off_ratio * 100)}% of pixels are far from the palette ramp — likely hardcoded RGB/hex values in the technique. Every color must come from canvas.palette.background/primary/secondary/tertiary/accent or art_kit.palette_color(t). Replace any literal (r,g,b) tuples or '#xxxxxx' strings.",
                "off_ratio": off_ratio,
            }
    except Exception:
        return None
    return None


def _transparent_canvas_check(img: Image.Image, canvas: "Canvas") -> dict | None:
    try:
        import numpy as np
        arr = np.asarray(img.convert("RGBA"))
        alpha = arr[::8, ::8, 3]
        if alpha.size == 0:
            return None
        transparent_ratio = float((alpha < 16).mean())
        if transparent_ratio > 0.5:
            return {
                "warning": "transparent_canvas",
                "message": f"{int(transparent_ratio * 100)}% of the canvas is transparent. Start from an opaque base (canvas.new(color=canvas.palette.background)) and ensure your draw/paste operations preserve coverage. If using Image.alpha_composite, both operands must be RGBA.",
                "transparent_ratio": transparent_ratio,
            }
    except Exception:
        return None
    return None


_VALIDATORS = (_transparent_canvas_check, _palette_adherence_check, _blank_canvas_check)


def _validate_output(img: Image.Image, canvas: "Canvas") -> dict | None:
    for check in _VALIDATORS:
        result = check(img, canvas)
        if result is not None:
            return result
    return None


def _write_sidecar(output_image_path: str, payload: dict) -> None:
    try:
        Path(output_image_path + ".err.json").write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def _install_pil_path_guard(allowed: set[str]) -> None:
    """After the parent's Image.open(input_image_path) is done, swap in
    guarded versions of Image.open and Image.Image.save that refuse any path
    outside the allowed set. File-like objects are passed through untouched
    so in-memory PIL operations keep working."""
    import os as _os
    _orig_open = Image.open
    _orig_save = Image.Image.save

    def _check(fp, mode: str) -> None:
        if isinstance(fp, (str, bytes, _os.PathLike)):
            try:
                resolved = str(Path(_os.fsdecode(fp)).resolve())
            except (OSError, ValueError):
                raise PermissionError(
                    f"technique may only {mode} its assigned canvas paths"
                )
            if resolved not in allowed:
                raise PermissionError(
                    f"technique may only {mode} its assigned canvas paths; got {resolved}"
                )

    def _guarded_open(fp, *args, **kwargs):
        _check(fp, "open")
        return _orig_open(fp, *args, **kwargs)

    def _guarded_save(self, fp, *args, **kwargs):
        _check(fp, "save")
        return _orig_save(self, fp, *args, **kwargs)

    Image.open = _guarded_open
    Image.Image.save = _guarded_save


def _find_technique_instance(ns: dict) -> object:
    """Locate the BaseTechnique subclass in the exec'd namespace, instantiate it."""
    for value in ns.values():
        if isinstance(value, type) and value is not BaseTechnique and issubclass(value, BaseTechnique):
            return value()
    raise ValueError("no BaseTechnique subclass found in technique file")


def _apply_param_bounds(instance, params: dict) -> dict:
    """Set descriptor-backed ``instance.<name>`` values and return leftovers."""
    bounds = getattr(instance, "_param_bounds", None) or {}
    if not bounds:
        return params
    residual = dict(params)
    for name, spec in bounds.items():
        if name in residual:
            value = residual.pop(name)
        else:
            value = spec.get("default")
        kind = spec.get("type")
        if kind == "slider":
            try:
                v = float(value)
            except (TypeError, ValueError):
                v = float(spec.get("default", spec["min"]))
            lo, hi = float(spec["min"]), float(spec["max"])
            value = lo if v < lo else hi if v > hi else v
        elif kind == "bool":
            value = bool(value)
        elif kind == "enum":
            allowed = spec.get("allowed") or []
            if value not in allowed:
                value = spec.get("default")
        elif kind == "text":
            s = "" if value is None else str(value)
            cap = int(spec.get("max_length", 120))
            value = s[:cap]
        setattr(instance, name, value)
    return residual


def _dispatch_run(instance, canvas, params: dict):
    """Apply descriptor params, reject stray controls, then call run(canvas)."""
    residual = _apply_param_bounds(instance, params)
    unknown = sorted(k for k in residual if k != "palette")
    if unknown:
        raise ValueError(
            "unknown control parameter(s): "
            + ", ".join(unknown)
            + "; declare controls with BaseTechnique descriptors"
        )
    return instance.run(canvas)


def _run_job(job: dict) -> int:
    _limits(memory_mb=int(job.get("memory_mb", 768)))
    output_image_path = job["output_image_path"]
    code = job["code"]

    try:
        errors = validate_technique_code(code)
        if errors:
            raise ValueError("; ".join(errors))
        canvas = Canvas(job)
        # Lock down Image.open / Image.save so technique code can only touch the
        # paths the parent assigned. The parent already did its Image.open()
        # via Canvas(job) above; the guard only affects user code below.
        allowed_paths = {str(Path(output_image_path).resolve())}
        if job.get("input_image_path"):
            try:
                allowed_paths.add(str(Path(job["input_image_path"]).resolve()))
            except (OSError, ValueError):
                pass
        _install_pil_path_guard(allowed_paths)
        safe = {k: getattr(_builtins, k) for k in (
            "abs", "all", "any", "bool", "dict", "enumerate", "Exception",
            "filter", "float", "int", "len", "list", "map", "max", "min",
            "pow", "property", "range", "round", "set", "sorted", "staticmethod",
            "classmethod", "str", "sum", "tuple", "ValueError", "zip",
            "isinstance", "issubclass", "type", "object", "super",
        )}
        safe.update({
            "__import__": _import,
            "__build_class__": _builtins.__build_class__,
            "print": lambda *a, **k: None,
        })
        # BaseTechnique is injected by name so the technique's `from plugins.BaseTechnique
        # import BaseTechnique` resolves through _import — keep that path live.
        art_kit = _build_art_kit(canvas.palette)
        ns = {"__builtins__": safe, "__name__": "__technique__", "art_kit": art_kit}
        exec(code, ns, ns)
        instance = _find_technique_instance(ns)
        result = _dispatch_run(instance, canvas, dict(job.get("params") or {}))
        if canvas._committed is None and isinstance(result, Image.Image):
            canvas.commit(result)
        if canvas._committed is None:
            raise ValueError("technique did not call canvas.commit(image)")
        if canvas.kind == "object" and canvas._image is not None:
            final_image = Image.alpha_composite(canvas._image, canvas._committed)
        else:
            final_image = canvas._committed
        final_image.save(output_image_path, "PNG", compress_level=int(job.get("png_compress_level", 1)))
    except BaseException as exc:
        diag = _diagnose(exc, code)
        _write_sidecar(output_image_path, diag)
        tail = f"{diag['error_type']}: {diag['message']}"
        if diag.get("technique_lineno"):
            tail += f" (line {diag['technique_lineno']}: {diag['technique_line'].strip()!r})"
        print(tail, file=sys.stderr)
        return 1

    # For object techniques, validate the composited final (what the viewer
    # sees) rather than the pre-composite sparse-alpha layer.
    to_validate = final_image if canvas.kind == "object" else canvas._committed
    warning = _validate_output(to_validate, canvas)
    if warning is not None:
        _write_sidecar(output_image_path, warning)
    return 0


def _run_job_path(job_path: str) -> int:
    return _run_job(json.loads(Path(job_path).read_text(encoding="utf-8")))


def _prewarm() -> None:
    import numpy  # noqa: F401
    from PIL import ImageChops, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageOps  # noqa: F401


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _prewarm()
        print("READY", flush=True)
        job_path = sys.stdin.readline().strip()
        sys.exit(_run_job_path(job_path) if job_path else 1)
    sys.exit(_run_job_path(sys.argv[1]))


if __name__ == "__main__":
    main()
