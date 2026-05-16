"""Canvas transform tools for the public generative-art demo."""

from __future__ import annotations

import json, math, random, time
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

from paths import DATA_DIR
from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers.color_theory import beautify_image
from plugins.tools.helpers.fractal_gallery import current, image_stats, mark_original, set_current

try:
    import numpy as np
except Exception:
    np = None


def _num(v, d, lo, hi):
    try: return max(lo, min(hi, float(v)))
    except Exception: return d


def _save(ctx, op, img, meta):
    out = DATA_DIR / "fractals" / "canvas_ops"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"{op}-{meta['seed']}-{time.strftime('%Y%m%d-%H%M%S')}.png"
    img = beautify_image(img, meta["seed"], meta.get("mood") or meta.get("palette", "plasma"), op)
    img.convert("RGB").save(path, "PNG", optimize=True)
    meta = {**meta, "kind": op, "path": str(path), "stats": image_stats(path)}
    path.with_suffix(".json").write_text(json.dumps({**meta, "original": True}, indent=2), encoding="utf-8")
    mark_original(path, meta); set_current(getattr(ctx, "session_key", None), path, True, meta)
    s = meta["stats"]
    return ToolResult(data=meta, llm_summary=f"Applied {op}: beauty {s['beauty_score']}, brightness {s['brightness']}, contrast {s['contrast']}, detail {s['detail']}, guidance={s['guidance']}.", attachment_paths=[str(path)])


class _CanvasTool(BaseTool):
    auto_register = False; requires_services = []; max_calls = 3
    parameters = {"type": "object", "properties": {"strength": {"type": "number", "minimum": 0, "maximum": 3}, "seed": {"type": "integer"}}}
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw); cls.auto_register = bool(getattr(cls, "name", ""))
    def src(self, ctx):
        item = current(getattr(ctx, "session_key", None))
        if not item: raise ValueError("No current canvas image. Generate an image first.")
        return Image.open(item["path"]).convert("RGB")
    def seed(self, kw):
        return int(_num(kw.get("seed"), random.randint(1, 2_147_483_647), 1, 2_147_483_647))


class ApplyColorGrade(_CanvasTool):
    name = "apply_color_grade"; description = "Restyle the current canvas with punchy color, contrast, saturation, and brightness."
    parameters = {"type": "object", "properties": {**_CanvasTool.parameters["properties"], "mood": {"type": "string", "enum": ["neon", "cinematic", "ice", "sunset", "toxic", "mono"]}}}
    def run(self, context, **kw):
        try: img, seed, mood = self.src(context), self.seed(kw), kw.get("mood") or "neon"
        except ValueError as e: return ToolResult.failed(str(e))
        x = _num(kw.get("strength"), 1.0, 0, 3); img = ImageEnhance.Contrast(img).enhance(1+.32*x); img = ImageEnhance.Color(img).enhance(.15 if mood=="mono" else 1+.65*x); img = ImageEnhance.Brightness(img).enhance(1+.06*x)
        if mood in {"ice","toxic","sunset"}: img = ImageChops.screen(img, ImageOps.colorize(ImageOps.grayscale(img), {"ice":"#001a3d","toxic":"#102200","sunset":"#280014"}[mood], {"ice":"#afffff","toxic":"#baff32","sunset":"#ffbd6e"}[mood]).convert("RGB"))
        return _save(context, "color_grade", img, {"seed": seed, "mood": mood, "strength": x})


class ApplyBloom(_CanvasTool):
    name = "apply_bloom"; description = "Add luminous glow and bloom to the current canvas. Great after fractals, attractors, and flow fields."
    def run(self, context, **kw):
        try: img, seed = self.src(context), self.seed(kw)
        except ValueError as e: return ToolResult.failed(str(e))
        x = _num(kw.get("strength"), 1.2, 0, 3); glow = ImageEnhance.Brightness(img.filter(ImageFilter.GaussianBlur(4+8*x))).enhance(1+.8*x)
        return _save(context, "bloom", ImageChops.screen(img, glow), {"seed": seed, "strength": x})


class ApplyKaleidoscope(_CanvasTool):
    name = "apply_kaleidoscope"; description = "Turn the current canvas into mirrored kaleidoscopic symmetry."
    def run(self, context, **kw):
        try: img, seed = self.src(context), self.seed(kw)
        except ValueError as e: return ToolResult.failed(str(e))
        w,h=img.size; q=img.crop((0,0,w//2,h//2)); top=ImageChops.screen(q, ImageOps.mirror(q)); bottom=ImageOps.flip(top); out=Image.new("RGB",(w,h)); out.paste(top,(0,0)); out.paste(ImageOps.mirror(top),(w//2,0)); out.paste(bottom,(0,h//2)); out.paste(ImageOps.mirror(bottom),(w//2,h//2))
        return _save(context, "kaleidoscope", ImageEnhance.Color(out).enhance(1.35), {"seed": seed, "strength": _num(kw.get("strength"), 1, 0, 3)})


class ApplyMirror(_CanvasTool):
    name = "apply_mirror"; description = "Mirror the current canvas horizontally, vertically, or both for bold symmetry."
    parameters = {"type": "object", "properties": {**_CanvasTool.parameters["properties"], "axis": {"type": "string", "enum": ["horizontal", "vertical", "both"]}}}
    def run(self, context, **kw):
        try: img, seed = self.src(context), self.seed(kw)
        except ValueError as e: return ToolResult.failed(str(e))
        axis=kw.get("axis") or "horizontal"; w,h=img.size; out=img.copy()
        if axis in {"horizontal","both"}: out.paste(ImageOps.mirror(img.crop((0,0,w//2,h))), (w//2,0))
        if axis in {"vertical","both"}: out.paste(ImageOps.flip(out.crop((0,0,w,h//2))), (0,h//2))
        return _save(context, "mirror", out, {"seed": seed, "axis": axis})


class ApplyFeedback(_CanvasTool):
    name = "apply_feedback"; description = "Create recursive video-feedback trails from the current canvas."
    def run(self, context, **kw):
        try: img, seed = self.src(context), self.seed(kw)
        except ValueError as e: return ToolResult.failed(str(e))
        r=random.Random(seed); out=img.copy()
        for i in range(7):
            lay=img.rotate(r.uniform(-4,4), resample=Image.Resampling.BICUBIC).resize((round(img.width*(.96-i*.01)), round(img.height*(.96-i*.01))))
            bg=Image.new("RGB", img.size, (0,0,0)); bg.paste(lay, ((img.width-lay.width)//2, (img.height-lay.height)//2)); out=ImageChops.screen(out, ImageEnhance.Brightness(bg).enhance(.55))
        return _save(context, "feedback", out, {"seed": seed, "strength": _num(kw.get("strength"), 1, 0, 3)})


class ApplyGlitch(_CanvasTool):
    name = "apply_glitch"; description = "Slice, offset, and color-shift the current canvas into vivid digital glitch art."
    def run(self, context, **kw):
        try: img, seed = self.src(context), self.seed(kw)
        except ValueError as e: return ToolResult.failed(str(e))
        r=random.Random(seed); out=img.copy(); w,h=img.size; x=_num(kw.get("strength"),1,0,3)
        for _ in range(round(24*x)+6):
            y=r.randrange(h); band=out.crop((0,y,w,min(h,y+r.randrange(4,32)))); out.paste(ImageChops.offset(band, r.randrange(-80,80), 0), (0,y))
        a,b,c=out.split(); return _save(context, "glitch", Image.merge("RGB",(ImageChops.offset(a,round(9*x),0),b,ImageChops.offset(c,round(-9*x),0))), {"seed": seed, "strength": x})


class ApplyDisplacement(_CanvasTool):
    name = "apply_displacement"; description = "Warp the current canvas with sine-wave displacement for liquid, heat-haze, and topographic looks."
    def run(self, context, **kw):
        try: img, seed = self.src(context), self.seed(kw)
        except ValueError as e: return ToolResult.failed(str(e))
        if np is None: return _save(context, "displacement", img.filter(ImageFilter.CONTOUR), {"seed": seed, "strength": _num(kw.get("strength"), 1, 0, 3)})
        x=_num(kw.get("strength"),1,0,3); arr=np.asarray(img); yy=np.arange(arr.shape[0])[:,None]; xx=np.arange(arr.shape[1])[None,:]; out=arr[(yy+(np.sin(xx/22+seed)*10*x).astype(int))%arr.shape[0], (xx+(np.cos(yy/25+seed)*10*x).astype(int))%arr.shape[1]]
        return _save(context, "displacement", Image.fromarray(out.astype("uint8"), "RGB"), {"seed": seed, "strength": x})


class ApplySharpen(_CanvasTool):
    name = "apply_sharpen"; description = "Sharpen the current canvas and pull out crisp edge detail after soft/generative renders."
    def run(self, context, **kw):
        try: img, seed = self.src(context), self.seed(kw)
        except ValueError as e: return ToolResult.failed(str(e))
        x=_num(kw.get("strength"),1,0,3)
        return _save(context, "sharpen", img.filter(ImageFilter.UnsharpMask(radius=1.5+x, percent=100+70*x, threshold=2)), {"seed": seed, "strength": x})
