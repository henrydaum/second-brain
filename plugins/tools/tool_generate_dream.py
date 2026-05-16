"""Master dream generator for the public visual demo."""

from __future__ import annotations

import hashlib, json, random, time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from paths import DATA_DIR
from plugins.BaseTool import BaseTool, ToolResult
from plugins.services.service_deep_dream import fallback_dream
from plugins.tools.helpers.color_theory import beautify_image
from plugins.tools.helpers.fractal_gallery import image_stats, mark_original, set_current
from plugins.tools.tool_fractal_suite import RenderBurningShip, RenderJulia, RenderNewtonFractal, RenderSierpinski
from plugins.tools.tool_generative_art_suite import RenderFlowField, RenderReactionDiffusion, RenderStrangeAttractor
from plugins.tools.tool_render_mandelbrot import RenderMandelbrot

MODES = {"auto", "fractal", "organic", "chaos", "crystal", "machine", "dream_archive"}
PALETTES = ["aurora", "electric", "nebula", "gold", "plasma", "laser", "sunset", "ice", "toxic", "royal"]
DREAMS = ["classic", "organic", "crystal", "machine", "soft"]


class GenerateDream(BaseTool):
    name = "generate_dream"
    description = "Generate one polished dream image from any user input. Always use this as the first visual tool: it composes a procedural base, color harmony, one DeepDream pass, and final polish."
    config_settings = [
        ("Dream Image Width", "dream_width", "Generated dream image width in pixels.", 960, {"type": "integer"}),
        ("Dream Image Height", "dream_height", "Generated dream image height in pixels.", 720, {"type": "integer"}),
    ]
    parameters = {"type": "object", "properties": {
        "prompt": {"type": "string", "description": "Any text, including gibberish; it becomes the deterministic visual seed."},
        "mode": {"type": "string", "enum": list(MODES)},
        "intensity": {"type": "number", "minimum": 0, "maximum": 1},
    }}
    requires_services = []; max_calls = 1

    def run(self, context, **kw):
        recipe = _recipe(str(kw.get("prompt") or "untitled dream"), kw.get("mode"), kw.get("intensity"))
        recipe["width"], recipe["height"] = _dims(context)
        return render_dream(context, recipe)


def render_dream(context, recipe):
    base_path, base_meta = _base(context, recipe)
    img = Image.open(base_path).convert("RGB")
    svc = getattr(context, "services", {}).get("deep_dream") if getattr(context, "services", None) else None
    if svc and not getattr(svc, "loaded", False):
        try: svc.load()
        except Exception: pass
    dream, dream_meta = svc.dream(img, recipe["seed"], recipe["dream"], recipe["intensity"], 2, 5) if svc and getattr(svc, "loaded", False) else (fallback_dream(img, recipe["seed"], recipe["dream"], recipe["intensity"]), {"engine": "fallback", "preset": recipe["dream"]})
    dream = beautify_image(dream, recipe["seed"], recipe["palette"], "dream")
    out = DATA_DIR / "fractals" / "dreams"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"dream-{recipe['mode']}-{recipe['seed']}-{time.strftime('%Y%m%d-%H%M%S')}.png"
    dream.save(path, "PNG", optimize=True)
    meta = {"kind": "dream", "path": str(path), "base_path": str(base_path), "base": base_meta, "recipe": recipe, "dream": dream_meta, "stats": image_stats(path)}
    path.with_suffix(".json").write_text(json.dumps({**meta, "original": True}, indent=2), encoding="utf-8")
    mark_original(path, meta); set_current(getattr(context, "session_key", None), path, True, meta)
    s = meta["stats"]
    return ToolResult(data=meta, llm_summary=f"Generated dream from {recipe['mode']} / {recipe['base']} / {recipe['palette']} / {recipe['dream']}: beauty {s['beauty_score']}, guidance={s['guidance']}. The Dream Archive buttons handle sharing and remixing.", attachment_paths=[str(path)])


def refine_recipe(recipe, request):
    text = (request or "").lower(); r = {**recipe, "refinement": request}
    if any(w in text for w in ("cold", "blue", "ice")): r["palette"], r["dream"] = "ice", "soft"
    if any(w in text for w in ("warm", "gold", "sun")): r["palette"] = "gold"
    if any(w in text for w in ("organic", "living", "bio")): r["mode"], r["base"], r["dream"] = "organic", "reaction", "organic"
    if any(w in text for w in ("alien", "weird", "strange")): r["mode"], r["dream"] = "chaos", "classic"
    if any(w in text for w in ("symmetr", "crystal", "temple")): r["mode"], r["base"], r["dream"] = "crystal", "newton", "crystal"
    if any(w in text for w in ("less", "subtle", "calm")): r["intensity"] = max(.25, r.get("intensity", .55) * .72)
    if any(w in text for w in ("more", "intense", "wild")): r["intensity"] = min(.9, r.get("intensity", .55) * 1.18)
    r["seed"] = _seed(f"{r.get('prompt','')}|{request}|{r.get('seed',1)}")
    return r


def _recipe(prompt, mode=None, intensity=None):
    seed = _seed(prompt); rng = random.Random(seed); text = prompt.lower()
    mode = None if mode in {None, "", "auto"} else mode
    mode = mode if mode in MODES else "organic" if any(w in text for w in ("ocean","forest","soft","memory","bio")) else "chaos" if any(w in text for w in ("chaos","alien","weird","glitch")) else "crystal" if any(w in text for w in ("crystal","temple","symmetry","cathedral")) else rng.choice(["fractal","organic","chaos","crystal","machine","dream_archive"])
    choices = {
        "fractal": ["mandelbrot","julia","burning_ship"], "organic": ["reaction","flow"], "chaos": ["attractor","julia"],
        "crystal": ["newton","sierpinski"], "machine": ["burning_ship","newton"], "dream_archive": ["mandelbrot","attractor","reaction"],
        "auto": ["mandelbrot","julia","reaction","attractor","newton"],
    }[mode]
    palette = "ice" if any(w in text for w in ("cold","blue","ice")) else "gold" if any(w in text for w in ("gold","sun","warm")) else rng.choice(PALETTES)
    return {"prompt": prompt, "seed": seed, "mode": mode, "base": rng.choice(choices), "palette": palette, "dream": rng.choice(DREAMS), "intensity": max(.2, min(.9, float(intensity) if intensity is not None else .55 + rng.random() * .22))}


def _base(context, recipe):
    ctx = SimpleNamespace(**getattr(context, "__dict__", {})); base, seed, pal = recipe["base"], recipe["seed"], recipe["palette"]
    w, h = int(recipe.get("width") or 960), int(recipe.get("height") or 720)
    args = {"width": w, "height": h, "detail": 130, "seed": seed, "palette": pal if pal in {"plasma","laser","sunset","ice","toxic","royal"} else "plasma"}
    tools = {"mandelbrot": (RenderMandelbrot(), {"width":w,"height":h,"detail":140,"seed":seed,"palette": pal if pal in {"aurora","electric","nebula","gold"} else "aurora"}),
             "julia": (RenderJulia(), args), "burning_ship": (RenderBurningShip(), args), "newton": (RenderNewtonFractal(), args),
             "sierpinski": (RenderSierpinski(), args), "reaction": (RenderReactionDiffusion(), args), "flow": (RenderFlowField(), args), "attractor": (RenderStrangeAttractor(), args)}
    result = tools[base][0].run(ctx, **tools[base][1])
    if not result.success: raise RuntimeError(result.error)
    return Path(result.attachment_paths[0]), result.data


def _seed(text): return int(hashlib.sha256((text or "dream").encode("utf-8")).hexdigest()[:8], 16)


def _dims(context):
    cfg = getattr(context, "config", {}) or {}
    def n(k, d, lo, hi):
        try: return max(lo, min(hi, int(cfg.get(k, d))))
        except Exception: return d
    return n("dream_width", 960, 512, 1400), n("dream_height", 720, 480, 1100)
