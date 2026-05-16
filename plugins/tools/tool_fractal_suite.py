"""Flashy fractal generator suite for the public demo."""

from __future__ import annotations

import ast, cmath, colorsys, json, math, random, time
from pathlib import Path

from PIL import Image, ImageFilter

from paths import DATA_DIR
from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers.fractal_gallery import image_stats, mark_original, set_current

PALETTES = {"plasma", "laser", "sunset", "ice", "toxic", "royal"}
try:
    import numpy as np
except Exception:
    np = None


def _clamp(v, d, lo, hi, cast=float):
    try: return max(lo, min(hi, cast(v)))
    except Exception: return d


def _pick(v, xs, d): return v if v in xs else d


def _color(t, pal, seed=1, v=1):
    r = random.Random(seed); t = (t + r.random()) % 1
    base = {"plasma": .78, "laser": .52, "sunset": .02, "ice": .55, "toxic": .25, "royal": .68}.get(pal, .78)
    h = (base + .35 * math.sin(t * 5.8 + r.random() * 6) + .28 * t) % 1
    s = .65 + .35 * ((math.sin(t * 9 + seed) + 1) / 2)
    return tuple(round(255 * x * v) for x in colorsys.hsv_to_rgb(h, s, min(1, .15 + .9 * t)))


def _np_colors(t, pal, seed, mask=None):
    base = {"plasma": 4.9, "laser": 3.3, "sunset": .2, "ice": 3.7, "toxic": 1.5, "royal": 4.2}.get(pal, 4.9)
    t = (np.nan_to_num(t.astype("float32")) + random.Random(seed).random()) % 1
    arr = np.zeros((*t.shape, 3), dtype=np.uint8)
    v = np.clip(.15 + .95 * t, 0, 1)
    arr[..., 0] = (255 * v * (.5 + .5 * np.sin(base + t * 6.0))).astype("uint8")
    arr[..., 1] = (255 * v * (.5 + .5 * np.sin(base + 2.1 + t * 7.5))).astype("uint8")
    arr[..., 2] = (255 * v * (.5 + .5 * np.sin(base + 4.2 + t * 5.2))).astype("uint8")
    if mask is not None:
        arr[~mask] = (2, 3, 8)
    return arr


def _save(kind, img, meta, ctx):
    out = DATA_DIR / "fractals" / kind; out.mkdir(parents=True, exist_ok=True)
    path = out / f"{kind}-{meta['seed']}-{time.strftime('%Y%m%d-%H%M%S')}.png"
    img.save(path, "PNG", optimize=True)
    meta = {**meta, "kind": kind, "path": str(path), "stats": image_stats(path)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    mark_original(path, meta); set_current(getattr(ctx, "session_key", None), path, True, meta)
    s = meta["stats"]
    return ToolResult(data=meta, llm_summary=f"Rendered {kind.replace('_',' ')} seed {meta['seed']}: brightness {s['brightness']}, contrast {s['contrast']}, detail {s['detail']}, mostly_dark={s['mostly_dark']}. Sharing is handled by the website button.", attachment_paths=[str(path)])


def _escape(w, h, detail, seed, pal, fn, cx=0, cy=0, scale=3.0):
    img = Image.new("RGB", (w, h)); px = img.load(); aspect = h / w
    for y in range(h):
        im = cy + (y / (h - 1) - .5) * scale * aspect
        for x in range(w):
            c = complex(cx + (x / (w - 1) - .5) * scale, im); z = fn(c, None, 0, init=True)
            i = 0
            while i < detail and abs(z) <= 8:
                z = fn(c, z, i, init=False); i += 1
            px[x, y] = (2, 3, 8) if i >= detail else _color((i - math.log(max(1, math.log(abs(z) + 1))) / detail) % 1, pal, seed)
    return img


def _np_escape(w, h, detail, seed, pal, kind, cx=0, cy=0, scale=3.0, const=0j):
    if np is None:
        return None
    x = np.linspace(cx - scale / 2, cx + scale / 2, w, dtype=np.float32)
    y = np.linspace(cy - scale * h / w / 2, cy + scale * h / w / 2, h, dtype=np.float32)
    c = x[None, :] + 1j * y[:, None]
    z = c.copy() if kind == "julia" else np.zeros_like(c)
    count = np.zeros(c.shape, dtype=np.float32); live = np.ones(c.shape, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(1, detail + 1):
            if kind == "julia": z[live] = z[live] * z[live] + const
            elif kind == "burning_ship": z[live] = (np.abs(z[live].real) + 1j * np.abs(z[live].imag)) ** 2 + c[live]
            elif kind == "tricorn": z[live] = np.conj(z[live]) ** 2 + c[live]
            escaped = live & (np.abs(z) > 8)
            count[escaped] = i; live[escaped] = False
            if not live.any(): break
    return Image.fromarray(_np_colors(count / max(1, detail), pal, seed, count > 0), "RGB")


def _np_newton(w, h, detail, seed, pal, power):
    if np is None:
        return None
    x = np.linspace(-1.6, 1.6, w, dtype=np.float32); y = np.linspace(-1.25, 1.25, h, dtype=np.float32)
    z = x[None, :] + 1j * y[:, None]; count = np.zeros(z.shape, dtype=np.float32); live = np.ones(z.shape, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(1, detail + 1):
            zl = z[live]; nz = zl - (zl ** power - 1) / (power * zl ** (power - 1))
            done = np.abs(nz - zl) < 1e-5
            idx = np.flatnonzero(live)
            z.flat[idx] = nz
            if done.any():
                count.flat[idx[done]] = i; live.flat[idx[done]] = False
            if not live.any(): break
    return Image.fromarray(_np_colors((np.angle(z) / (2 * np.pi) + count / max(1, detail)) % 1, pal, seed, np.isfinite(z.real)), "RGB")


def _ifs(w, h, seed, pal, rules, n=220000):
    img = Image.new("RGB", (w, h), (2, 3, 8)); px = img.load(); r = random.Random(seed); x = y = 0.0
    for i in range(n):
        a, b, c, d, e, f = r.choices(rules, [q[-1] for q in rules])[0][:6]
        x, y = a * x + b * y + e, c * x + d * y + f
        sx, sy = int((x + 3) / 6 * w), int(h - y / 10.5 * h)
        if 0 <= sx < w and 0 <= sy < h: px[sx, sy] = _color(i / n, pal, seed, 1.15)
    return img.filter(ImageFilter.SMOOTH_MORE)


class _Base(BaseTool):
    auto_register = False; requires_services = []; max_calls = 2
    parameters = {"type": "object", "properties": {
        "width": {"type": "integer", "minimum": 512, "maximum": 1400},
        "height": {"type": "integer", "minimum": 512, "maximum": 1100},
        "detail": {"type": "integer", "minimum": 80, "maximum": 500},
        "palette": {"type": "string", "enum": list(PALETTES)},
        "seed": {"type": "integer", "description": "Optional random seed."},
    }}
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw); cls.auto_register = bool(getattr(cls, "name", ""))

    def args(self, kw):
        return (_clamp(kw.get("width"), 1100, 512, 1400, int), _clamp(kw.get("height"), 850, 512, 1100, int), _clamp(kw.get("detail"), 220, 80, 500, int), _pick(kw.get("palette"), PALETTES, "plasma"), _clamp(kw.get("seed"), random.randint(1, 2_147_483_647), 1, 2_147_483_647, int))


class RenderJulia(_Base):
    name = "render_julia"; description = "Render a gorgeous Julia set. Great for jewel-like spirals, lightning filaments, and alien glass."
    parameters = {**_Base.parameters, "properties": {**_Base.parameters["properties"], "preset": {"type": "string", "enum": ["dragon", "dendrite", "spiral", "rabbit", "storm"]}}}
    def run(self, context, **kw):
        w, h, d, p, s = self.args(kw); c = {"dragon": -.8+.156j, "dendrite": 0+.65j, "spiral": -.7269+.1889j, "rabbit": -.123+.745j, "storm": .285+.01j}[_pick(kw.get("preset"), {"dragon","dendrite","spiral","rabbit","storm"}, "spiral")]
        return _save("julia", _np_escape(w, h, d, s, p, "julia", scale=3.2, const=c) or _escape(w, h, d, s, p, lambda z0, z, i, init: z0 if init else z*z+c, scale=3.2), {"preset": kw.get("preset") or "spiral", "c": [c.real, c.imag], "seed": s, "palette": p}, context)


class RenderBurningShip(_Base):
    name = "render_burning_ship"; description = "Render the Burning Ship fractal: gothic towers, firelit coastlines, and jagged molten symmetry."
    def run(self, context, **kw):
        w, h, d, p, s = self.args(kw)
        return _save("burning_ship", _np_escape(w, h, d, s, p, "burning_ship", -.55, -.55, 2.4) or _escape(w, h, d, s, p, lambda c, z, i, init: 0 if init else complex(abs(z.real), abs(z.imag))**2+c, -.55, -.55, 2.4), {"seed": s, "palette": p}, context)


class RenderTricorn(_Base):
    name = "render_tricorn"; description = "Render the Tricorn/Mandelbar fractal: mirrored cosmic geometry with sharp ornamental bulbs."
    def run(self, context, **kw):
        w, h, d, p, s = self.args(kw)
        return _save("tricorn", _np_escape(w, h, d, s, p, "tricorn", 0, 0, 3.4) or _escape(w, h, d, s, p, lambda c, z, i, init: 0 if init else z.conjugate()**2+c, 0, 0, 3.4), {"seed": s, "palette": p}, context)


class RenderPhoenix(_Base):
    name = "render_phoenix"; description = "Render a Phoenix fractal with layered flame trails and recursive ghosting."
    def run(self, context, **kw):
        w, h, d, p, s = self.args(kw); img = Image.new("RGB", (w, h)); px = img.load()
        for y in range(h):
            for x in range(w):
                z = complex((x/w-.5)*3, (y/h-.5)*2.4); old = 0; i = 0
                while i < d and abs(z) <= 8:
                    z, old = z*z - .5 + .56667j - .42 * old, z; i += 1
                px[x, y] = (2,3,8) if i >= d else _color(i/d, p, s)
        return _save("phoenix", img, {"seed": s, "palette": p}, context)


class RenderNewtonFractal(_Base):
    name = "render_newton_fractal"; description = "Render Newton basins: stained-glass root geometry with razor-sharp mathematical borders."
    parameters = {**_Base.parameters, "properties": {**_Base.parameters["properties"], "power": {"type": "integer", "minimum": 3, "maximum": 9}}}
    def run(self, context, **kw):
        w, h, d, p, s = self.args(kw); d = min(d, 160); power = _clamp(kw.get("power"), 5, 3, 9, int); fast = _np_newton(w, h, d, s, p, power)
        if fast: return _save("newton", fast, {"seed": s, "palette": p, "power": power}, context)
        img = Image.new("RGB", (w, h)); px = img.load()
        for y in range(h):
            for x in range(w):
                z = complex((x/w-.5)*3, (y/h-.5)*2.4)
                for i in range(d):
                    if abs(z) < 1e-6: break
                    nz = z - (z**power - 1) / (power * z**(power - 1))
                    if abs(nz - z) < 1e-6: break
                    z = nz
                px[x, y] = _color(((cmath.phase(z)/(2*math.pi)) + i/d) % 1, p, s)
        return _save("newton", img, {"seed": s, "palette": p, "power": power}, context)


class RenderBarnsleyFern(_Base):
    name = "render_barnsley_fern"; description = "Render a Barnsley fern: luminous botanical fractal growth, great when the user asks for organic beauty."
    def run(self, context, **kw):
        w, h, d, p, s = self.args(kw); rules = [(0,0,0,.16,0,0,.01), (.85,.04,-.04,.85,0,1.6,.85), (.2,-.26,.23,.22,0,1.6,.07), (-.15,.28,.26,.24,0,.44,.07)]
        return _save("barnsley_fern", _ifs(w, h, s, p, rules, d * 1100), {"seed": s, "palette": p}, context)


class RenderSierpinski(_Base):
    name = "render_sierpinski"; description = "Render a Sierpinski triangle storm: iconic recursive geometry with neon dust and crystalline voids."
    def run(self, context, **kw):
        w, h, d, p, s = self.args(kw); img = Image.new("RGB", (w, h), (2,3,8)); px = img.load(); r = random.Random(s); pts = [(w/2,30),(40,h-40),(w-40,h-40)]; x,y = r.random()*w, r.random()*h
        for i in range(d * 1500):
            vx, vy = r.choice(pts); x, y = (x+vx)/2, (y+vy)/2
            px[int(x), int(y)] = _color(i/(d*1500), p, s, 1.2)
        return _save("sierpinski", img, {"seed": s, "palette": p}, context)


class RenderMandelbulb(_Base):
    name = "render_mandelbulb"; description = "Render a 3D Mandelbulb-style raymarched image. Slower, flashier, and perfect for the wow moment."
    def run(self, context, **kw):
        w, h, d, p, s = self.args(kw); w, h, d = min(w, 720), min(h, 560), min(d, 90); img = Image.new("RGB", (w, h)); px = img.load()
        def de(pos):
            x,y,z = pos; zx,zy,zz,dr,r = x,y,z,1,0
            for _ in range(8):
                r = math.sqrt(zx*zx+zy*zy+zz*zz)
                if r > 2: break
                th, ph = math.acos(zz/(r or 1)), math.atan2(zy,zx); dr = r**7*8*dr+1; zr = r**8
                th *= 8; ph *= 8; zx,zy,zz = zr*math.sin(th)*math.cos(ph)+x, zr*math.sin(th)*math.sin(ph)+y, zr*math.cos(th)+z
            return .5 * math.log(r or 1) * r / dr
        for y in range(h):
            for x in range(w):
                ox, oy, oz = 0, 0, -3.6; dx, dy, dz = (x/w-.5)*1.7, (y/h-.5)*1.3, 1; mag = math.sqrt(dx*dx+dy*dy+dz*dz); dx,dy,dz = dx/mag,dy/mag,dz/mag; t = 0
                for i in range(70):
                    dist = de((ox+dx*t, oy+dy*t, oz+dz*t)); t += dist
                    if dist < .003 or t > 7: break
                shade = 0 if t > 7 else max(.15, 1 - i / 70)
                px[x, y] = _color(shade, p, s, shade)
        return _save("mandelbulb", img, {"seed": s, "palette": p}, context)


class RenderFormulaFractal(_Base):
    name = "render_formula_fractal"; description = "Render a bounded custom escape-time fractal from a user formula for z. Safe formula variables: z, c, n, abs, conjugate, sin, cos, exp. Example: z*z+c, z*z+sin(c), conjugate(z*z)+c."
    parameters = {**_Base.parameters, "properties": {**_Base.parameters["properties"], "formula": {"type": "string", "description": "Safe expression for the next z value."}}}
    def run(self, context, **kw):
        w, h, d, p, s = self.args(kw); formula = str(kw.get("formula") or "z*z+c")[:120]
        try: code = _safe_formula(formula)
        except Exception as e: return ToolResult.failed(f"Unsupported formula: {e}")
        return _save("formula", _escape(w, h, min(d, 240), s, p, lambda c, z, i, init: 0 if init else eval(code, {"__builtins__": {}}, {"z": z, "c": c, "n": i, "abs": abs, "conjugate": lambda q: q.conjugate(), "sin": cmath.sin, "cos": cmath.cos, "exp": cmath.exp}), 0, 0, 3.0), {"seed": s, "palette": p, "formula": formula}, context)


def _safe_formula(text):
    tree = ast.parse(text, mode="eval")
    ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub)
    names = {"z", "c", "n", "abs", "conjugate", "sin", "cos", "exp"}
    for node in ast.walk(tree):
        if not isinstance(node, ok) or (isinstance(node, ast.Name) and node.id not in names) or (isinstance(node, ast.Call) and not isinstance(node.func, ast.Name)):
            raise ValueError("Formula uses unsupported syntax.")
    return compile(tree, "<formula_fractal>", "eval")
