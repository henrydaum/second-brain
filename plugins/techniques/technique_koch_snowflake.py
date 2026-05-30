from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

_SHAPES = {
    "snowflake":      ("F--F--F",   {"F": "F+F--F+F"},                 60.0, 5),
    "anti_snowflake": ("F++F++F",   {"F": "F-F++F-F"},                 60.0, 5),
    "island":         ("F+F+F+F",   {"F": "F+F-F-FF+F+F-F"},           90.0, 5),
}

def _segments_bbox(segs):
    xs = [p for s in segs for p in (s[0], s[2])]
    ys = [p for s in segs for p in (s[1], s[3])]
    return min(xs), min(ys), max(xs), max(ys)


class KochSnowflakeTechnique(BaseTechnique):
    name = 'Koch Snowflake'
    description = 'The Koch curve as a Lindenmayer system: F -> F+F--F+F rewritten to depth 5-6, traced by a turtle into a single closed polyline. Three flavors -- the classic snowflake, the inverted anti-snowflake, and an island variant with thicker arms. Rendered as a palette-graded outline with soft glow. Good for "snowflake", "koch", "frost", "crystal", "winter", "ornament", or "lace".'
    kind = "background"
    palette = Palette()
    shape = Enum([('snowflake', 'Snowflake'), ('anti_snowflake', 'Anti-Snowflake'), ('island', 'Koch Island')], default='snowflake')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        rng = random.Random(seed)
        axiom, rules, angle_deg, iters = _SHAPES.get(str(self.shape), _SHAPES["snowflake"])

        sentence = art_kit.lindenmayer(axiom, rules, iters)
        # Step size will be normalized after we know the bbox; use 1.0 here.
        raw = art_kit.turtle_segments(
            sentence, start=(0.0, 0.0), heading=0.0,
            step=1.0, turn=math.radians(angle_deg),
        )
        if not raw:
            canvas.commit(canvas.create_image())
            return

        x0, y0, x1, y1 = _segments_bbox(raw)
        w = max(1e-9, x1 - x0)
        h = max(1e-9, y1 - y0)
        margin = s * 0.08
        avail = s - 2 * margin
        scale = avail / max(w, h)
        # Center inside the canvas.
        pad_x = (s - w * scale) * 0.5
        pad_y = (s - h * scale) * 0.5

        bg = canvas.palette.background
        img = Image.new("RGBA", (s, s), bg)
        draw = ImageDraw.Draw(img, "RGBA")

        n_seg = max(1, len(raw))
        line_w = max(2, int(s * 0.0035))
        for i, (sx, sy, ex, ey) in enumerate(raw):
            t = i / n_seg
            # Two-pass: a slight color jitter via seed keeps adjacent segments lively.
            ramp = 0.25 + 0.65 * t + (rng.random() - 0.5) * 0.05
            color = art_kit.palette_color(art_kit.clamp(ramp, 0.0, 1.0))
            p1 = (pad_x + (sx - x0) * scale, pad_y + (sy - y0) * scale)
            p2 = (pad_x + (ex - x0) * scale, pad_y + (ey - y0) * scale)
            draw.line((p1, p2), fill=color, width=line_w)

        # Soft glow underneath, original line over the top.
        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.006))
        out = Image.alpha_composite(glow, img)
        canvas.commit(out)
