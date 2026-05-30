from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import math
import random
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

_GRAMMARS = {
    # Open-canopy binary fan: clean trunk, splits into recognisable major
    # branches, fine halo of tip-branches. Matches the reference image feel.
    "specimen": {"axiom": "F", "rules": {"F": "F[+F][-F]F"}, "angle": 25.7, "iter": 7},
    "pure":     {"axiom": "F", "rules": {"F": "F[+F]F[-F][F]"},           "angle": 25.7, "iter": 6},
    "grove":    {"axiom": "F", "rules": {"F": "FF-[-F+F+F]+[+F-F-F]"},    "angle": 22.5, "iter": 5},
    "fern":     {"axiom": "X", "rules": {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"}, "angle": 25.0, "iter": 5},
    "coral":    {"axiom": "F", "rules": {"F": "FF+[+F-F-F]-[-F+F+F]"},    "angle": 18.0, "iter": 5},
}

def _turtle_with_depth(sentence, start, heading, step, turn):
    """Like art_kit.turtle_segments, but each segment carries its bracket depth.
    Returns list of (x1, y1, x2, y2, depth)."""
    x, y = float(start[0]), float(start[1])
    h = float(heading)
    stack = []
    depth = 0
    out = []
    for ch in sentence:
        if ch == "F" or ch == "G":
            nx = x + math.cos(h) * step
            ny = y + math.sin(h) * step
            out.append((x, y, nx, ny, depth))
            x, y = nx, ny
        elif ch == "f":
            x += math.cos(h) * step
            y += math.sin(h) * step
        elif ch == "+":
            h += turn
        elif ch == "-":
            h -= turn
        elif ch == "[":
            stack.append((x, y, h, depth))
            depth += 1
        elif ch == "]":
            if stack:
                x, y, h, depth = stack.pop()
    return out

def _bbox(segs):
    xs = [p for s in segs for p in (s[0], s[2])]
    ys = [p for s in segs for p in (s[1], s[3])]
    return min(xs), min(ys), max(xs), max(ys)

def _draw_tree(img_draw, segs, max_depth, palette_lo, palette_hi, trunk_w, tip_w, jitter_color=False, rng=None):
    """Draw segments with width and palette ramp scaled by bracket depth."""
    if not segs:
        return
    for x1, y1, x2, y2, d in segs:
        t = min(d / max(max_depth, 1), 1.0)
        ramp = palette_lo + (palette_hi - palette_lo) * t
        if jitter_color and rng is not None:
            ramp = max(0.0, min(1.0, ramp + (rng.random() - 0.5) * 0.06))
        color = art_kit.palette_color(ramp)
        width = max(1, int(round(trunk_w + (tip_w - trunk_w) * t)))
        img_draw.line(((x1, y1), (x2, y2)), fill=color, width=width)

def _render_single(canvas, s, kind, jitter):
    g = _GRAMMARS[kind]
    rng = random.Random(int(canvas.seed)) if jitter else None
    sentence = art_kit.lindenmayer(g["axiom"], g["rules"], g["iter"])
    angle = g["angle"] + (rng.uniform(-1.5, 1.5) if rng else 0.0)
    segs = _turtle_with_depth(sentence, start=(0.0, 0.0), heading=-math.pi / 2.0,
                              step=1.0, turn=math.radians(angle))
    if not segs:
        return Image.new("RGBA", (s, s), canvas.palette.background)

    # Normalize bbox into the canvas.
    x0, y0, x1, y1 = _bbox(segs)
    w = max(1e-9, x1 - x0)
    h = max(1e-9, y1 - y0)
    margin = s * 0.06
    avail = s - 2 * margin
    scale = avail / max(w, h)
    pad_x = (s - w * scale) * 0.5
    pad_y = (s - h * scale) * 0.5
    norm = [(pad_x + (a - x0) * scale, pad_y + (b - y0) * scale,
             pad_x + (c - x0) * scale, pad_y + (d - y0) * scale, dp)
            for (a, b, c, d, dp) in segs]
    max_d = max((dp for *_, dp in norm), default=1)

    img = Image.new("RGBA", (s, s), canvas.palette.background)
    draw = ImageDraw.Draw(img, "RGBA")

    trunk_w = max(4, int(s * 0.012))
    tip_w = 1
    if kind == "specimen":
        # Heavier trunk, palette ramp from dark trunk to bright tips.
        _draw_tree(draw, norm, max_d, palette_lo=0.05, palette_hi=0.92,
                   trunk_w=trunk_w, tip_w=tip_w, jitter_color=True, rng=rng)
    else:
        # Pure: keep colors uniform-ish so the math reads as math.
        _draw_tree(draw, norm, max_d, palette_lo=0.12, palette_hi=0.80,
                   trunk_w=max(3, trunk_w - 1), tip_w=tip_w,
                   jitter_color=False, rng=None)
    return img

def _render_grove(canvas, s):
    rng = random.Random(int(canvas.seed))
    img = Image.new("RGBA", (s, s), canvas.palette.background)
    draw = ImageDraw.Draw(img, "RGBA")

    # Seven silhouetted trees on a baseline, alternating rules so they look
    # like the species variety in a botanical chart.
    rule_pool = [
        {"axiom": "F", "rules": {"F": "FF-[-F+F+F]+[+F-F-F]"}, "angle": 22.5, "iter": 5},
        {"axiom": "F", "rules": {"F": "F[+FF][-FF]F[-F][+F]F"}, "angle": 25.0, "iter": 4},
        {"axiom": "F", "rules": {"F": "FF+[+F-F-F]-[-F+F+F]"}, "angle": 18.0, "iter": 5},
        {"axiom": "F", "rules": {"F": "F[+F][-F]F[-F][+F]F"},  "angle": 30.0, "iter": 5},
        {"axiom": "F", "rules": {"F": "F[+F]F[-F]F"},          "angle": 25.7, "iter": 6},
        {"axiom": "F", "rules": {"F": "FF[-F+F]F[+F-F]F"},     "angle": 20.0, "iter": 5},
        {"axiom": "F", "rules": {"F": "F[+F-F+F]F[-F+F-F]"},   "angle": 28.0, "iter": 4},
    ]
    n = len(rule_pool)
    cell_w = s / n
    baseline = s * 0.95
    cell_h = s * 0.92  # max tree height -- fill most of the canvas

    for i, g in enumerate(rule_pool):
        sentence = art_kit.lindenmayer(g["axiom"], g["rules"], g["iter"])
        segs = _turtle_with_depth(sentence, start=(0.0, 0.0), heading=-math.pi / 2.0,
                                  step=1.0, turn=math.radians(g["angle"]))
        if not segs:
            continue
        x0, y0, x1_b, y1_b = _bbox(segs)
        w = max(1e-9, x1_b - x0)
        h = max(1e-9, y1_b - y0)
        scale = min(cell_w * 0.82 / w, cell_h / h)
        cx = (i + 0.5) * cell_w
        # Anchor the tree's baseline at `baseline`.
        norm = [(cx + (a - (x0 + w / 2.0)) * scale, baseline - (y1_b - b) * scale,
                 cx + (c - (x0 + w / 2.0)) * scale, baseline - (y1_b - d) * scale, dp)
                for (a, b, c, d, dp) in segs]
        max_d = max((dp for *_, dp in norm), default=1)
        # Each tree drawn near-uniform palette (the "silhouette" look).
        ramp = 0.10 + 0.25 * (i / max(1, n - 1))
        for x1s, y1s, x2s, y2s, d in norm:
            t = min(d / max(max_d, 1), 1.0)
            color = art_kit.palette_color(ramp + 0.18 * t)
            width = max(1, int(round(3 + (1 - 3) * t)))
            draw.line(((x1s, y1s), (x2s, y2s)), fill=color, width=width)
    return img


class LSystemGroveTechnique(BaseTechnique):
    name = 'L-System Grove'
    description = 'Trees as Lindenmayer systems traced by a turtle: depth-aware line width and palette ramp give a heavy trunk fading into a fine halo of thin tip-branches. No leaves, no fbm sky -- pure branching grammar on a clean palette-background. Presets: a single big specimen at canvas center, a perfectly symmetric pure-math tree (no jitter), a grove of stylized silhouettes, a Barnsley-style fern, and a coral form. Good for "tree", "forest", "grove", "branches", "fern", "coral", "sapling", "botanical", or "L-system".'
    kind = "background"
    palette = Palette()
    shape = Enum([('specimen', 'Single Specimen'), ('pure', 'Pure Symmetric'), ('grove', 'Grove Silhouettes'), ('fern', 'Barnsley Fern'), ('coral', 'Coral')], default='coral')

    def run(self, canvas):
        s = int(canvas.size)
        kind = str(self.shape)
        if kind == "grove":
            out = _render_grove(canvas, s)
        elif kind == "pure":
            out = _render_single(canvas, s, kind="pure", jitter=False)
        else:
            out = _render_single(canvas, s, kind=kind, jitter=True)

        # Optional soft glow under the lines so the canopy gets a faint halo
        # (matches the dense-fine-branch feel of the reference). Skip for
        # the "pure" preset where crisp math is the whole point.
        if kind != "pure":
            glow = out.filter(ImageFilter.GaussianBlur(radius=s * 0.004))
            out = Image.alpha_composite(glow, out)
        canvas.commit(out)
