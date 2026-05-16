"""Child process for skill execution."""

from __future__ import annotations

import importlib
import builtins as _builtins
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plugins.helpers.skill_store import validate_skill_code

ALLOWED = {"math", "random", "colorsys", "numpy", "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFilter", "PIL.ImageOps", "PIL.ImageEnhance", "PIL.ImageChops", "PIL.ImageColor"}


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


def main():
    _limits()
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    errors = validate_skill_code(job["code"])
    if errors:
        raise ValueError("; ".join(errors))
    canvas = Canvas(job)
    safe = {k: getattr(_builtins, k) for k in ("abs", "all", "any", "bool", "dict", "enumerate", "Exception", "filter", "float", "int", "len", "list", "map", "max", "min", "pow", "range", "round", "set", "sorted", "str", "sum", "tuple", "ValueError", "zip")}
    safe.update({"__import__": _import, "print": lambda *a, **k: None})
    ns = {"__builtins__": safe, "__name__": "__skill__"}
    exec(job["code"], ns, ns)
    result = ns["run"](canvas, **(job.get("params") or {}))
    if canvas._committed is None and isinstance(result, Image.Image):
        canvas.commit(result)
    if canvas._committed is None:
        raise ValueError("skill did not call canvas.commit(image)")
    canvas._committed.save(job["output_image_path"], "PNG")


if __name__ == "__main__":
    main()
