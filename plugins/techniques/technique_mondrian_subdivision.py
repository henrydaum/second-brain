from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import random
from PIL import Image, ImageDraw

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

def _split(rect, depth, rng, stop_p=0.15, min_depth=3):
    # Force a minimum subdivision depth so no single cell ever dominates the
    # canvas; without this, an early stop at depth 1 produces a giant rect.
    if depth == 0 or (depth < (8 - min_depth) and rng.random() < stop_p):
        yield rect
        return
    x, y, w, h = rect
    if w >= h:
        t = rng.uniform(0.3, 0.7)
        cut = max(1, int(w * t))
        yield from _split((x, y, cut, h), depth - 1, rng, stop_p)
        yield from _split((x + cut, y, w - cut, h), depth - 1, rng, stop_p)
    else:
        t = rng.uniform(0.3, 0.7)
        cut = max(1, int(h * t))
        yield from _split((x, y, w, cut), depth - 1, rng, stop_p)
        yield from _split((x, y + cut, w, h - cut), depth - 1, rng, stop_p)

def _pick_color(rng, accent_quota):
    # Weighted draw: 0.6 chance light-ish ramp pos, 0.3 chance mid, accent_quota chance the top of the ramp.
    r = rng.random()
    if r < accent_quota:
        t = 0.94
    elif r < 0.5:
        t = rng.uniform(0.18, 0.42)   # near background/tertiary
    elif r < 0.85:
        t = rng.uniform(0.42, 0.7)    # secondary/primary
    else:
        t = rng.uniform(0.7, 0.88)    # bright but not accent
    return art_kit.palette_color(t)


class MondrianSubdivisionTechnique(BaseTechnique):
    name = 'Mondrian Subdivision'
    description = 'Recursive rectangular subdivision: at each step pick the longer axis, cut between 30% and 70% across, recurse with a 15% chance of stopping early. The canvas fills with axis-aligned cells whose colors come from a weighted palette draw -- background and secondary slots dominate, accent gets a tight quota. Three flavors: classic Mondrian with thick black gutters, stained-glass with thin dark seams, and low-poly which jitters each rect\'s corners into a pair of triangles. Good for "mondrian", "abstract", "modernist", "grid", "stained glass", "low poly", or any flat geometric composition.'
    kind = "background"
    palette = Palette()
    style = Enum([('mondrian', 'Classic Mondrian'), ('stained_glass', 'Stained Glass'), ('low_poly', 'Low Poly')], default='mondrian')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        rng = random.Random(seed)
        self.style = str(self.style)

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")

        rects = list(_split((0, 0, s, s), depth=8, rng=rng, stop_p=0.13))

        accent_quota = 0.06
        gutter_color = art_kit.palette_color(0.02)  # near darkest

        if self.style == "low_poly":
            for x, y, w, h in rects:
                t = _pick_color(rng, accent_quota)
                # Two triangles with corner jitter.
                jx = w * 0.06
                jy = h * 0.06
                p00 = (x + rng.uniform(-jx, jx), y + rng.uniform(-jy, jy))
                p10 = (x + w + rng.uniform(-jx, jx), y + rng.uniform(-jy, jy))
                p11 = (x + w + rng.uniform(-jx, jx), y + h + rng.uniform(-jy, jy))
                p01 = (x + rng.uniform(-jx, jx), y + h + rng.uniform(-jy, jy))
                # Diagonal direction varies for some life.
                if rng.random() < 0.5:
                    draw.polygon([p00, p10, p11], fill=t)
                    draw.polygon([p00, p11, p01], fill=_pick_color(rng, accent_quota))
                else:
                    draw.polygon([p00, p10, p01], fill=t)
                    draw.polygon([p10, p11, p01], fill=_pick_color(rng, accent_quota))
        else:
            gutter_w = 6 if self.style == "mondrian" else 2
            for x, y, w, h in rects:
                t = _pick_color(rng, accent_quota)
                draw.rectangle((x, y, x + w - 1, y + h - 1), fill=t,
                               outline=gutter_color, width=gutter_w)
            # Frame the canvas.
            draw.rectangle((0, 0, s - 1, s - 1), outline=gutter_color, width=gutter_w)

        canvas.commit(img)
