from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

def _dragon_segments(iters):
    sentence = art_kit.lindenmayer("FX", {"X": "X+YF+", "Y": "-FX-Y"}, iters)
    return art_kit.turtle_segments(sentence, start=(0.0, 0.0), heading=0.0,
                                   step=1.0, turn=math.radians(90.0))

def _terdragon_segments(iters):
    # axiom F, rules F -> F+F-F at 120 degrees.
    sentence = art_kit.lindenmayer("F", {"F": "F+F-F"}, iters)
    return art_kit.turtle_segments(sentence, start=(0.0, 0.0), heading=0.0,
                                   step=1.0, turn=math.radians(120.0))

def _rotate(segs, deg, ox, oy):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for x1, y1, x2, y2 in segs:
        dx1, dy1 = x1 - ox, y1 - oy
        dx2, dy2 = x2 - ox, y2 - oy
        out.append((
            ox + dx1 * ca - dy1 * sa, oy + dx1 * sa + dy1 * ca,
            ox + dx2 * ca - dy2 * sa, oy + dx2 * sa + dy2 * ca,
        ))
    return out

def _bbox(segs):
    xs = [p for s in segs for p in (s[0], s[2])]
    ys = [p for s in segs for p in (s[1], s[3])]
    return min(xs), min(ys), max(xs), max(ys)


class DragonCurveTechnique(BaseTechnique):
    name = 'Dragon Curve'
    description = 'The Heighway dragon as a Lindenmayer system: X -> X+YF+, Y -> -FX-Y at 90 degrees, rewritten 10-13 times. The walk folds back on itself again and again until it tiles its own footprint. Presets render one dragon, a twin dragon (two copies rotated 180), or a terdragon (three at 120). Palette gradient runs from the start of the walk to the end so you can see the fold order. Good for "dragon", "curve", "fold", "fractal", "path", or "labyrinth".'
    kind = "background"
    palette = Palette()
    variant = Enum([('dragon', 'Dragon'), ('twin', 'Twin Dragon'), ('terdragon', 'Terdragon (3-fold)')], default='dragon')

    def run(self, canvas):
        s = int(canvas.size)
        self.variant = str(self.variant)

        if self.variant == "twin":
            base = _dragon_segments(12)
            rotated = _rotate(base, 180.0, 0.0, 0.0)
            groups = [(base, 0.10, 0.75), (rotated, 0.55, 0.95)]
        elif self.variant == "terdragon":
            base = _terdragon_segments(8)
            groups = [
                (base, 0.10, 0.70),
                (_rotate(base, 120.0, 0.0, 0.0), 0.30, 0.85),
                (_rotate(base, 240.0, 0.0, 0.0), 0.55, 0.95),
            ]
        else:
            base = _dragon_segments(13)
            groups = [(base, 0.10, 0.92)]

        all_segs = [seg for g, _, _ in groups for seg in g]
        if not all_segs:
            canvas.commit(canvas.create_image())
            return

        x0, y0, x1, y1 = _bbox(all_segs)
        w = max(1e-9, x1 - x0)
        h = max(1e-9, y1 - y0)
        margin = s * 0.08
        avail = s - 2 * margin
        scale = avail / max(w, h)
        pad_x = (s - w * scale) * 0.5
        pad_y = (s - h * scale) * 0.5

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")

        line_w = max(2, int(s * 0.0028))
        for segs, t_lo, t_hi in groups:
            n = max(1, len(segs))
            for i, (sx, sy, ex, ey) in enumerate(segs):
                tt = t_lo + (t_hi - t_lo) * (i / n)
                color = art_kit.palette_color(tt)
                p1 = (pad_x + (sx - x0) * scale, pad_y + (sy - y0) * scale)
                p2 = (pad_x + (ex - x0) * scale, pad_y + (ey - y0) * scale)
                draw.line((p1, p2), fill=color, width=line_w)

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.005))
        out = Image.alpha_composite(glow, img)
        canvas.commit(out)
