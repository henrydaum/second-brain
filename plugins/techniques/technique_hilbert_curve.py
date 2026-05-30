from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import math
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class HilbertCurveTechnique(BaseTechnique):
    name = 'Hilbert Curve'
    description = 'The Hilbert space-filling curve: a single continuous self-similar path that visits every cell of a grid without crossing itself, drawn as one unbroken stroke whose color sweeps along the palette ramp from start to end. Higher order subdivides finer. Good for "Hilbert curve", "space-filling curve", "fractal path", "maze", or a continuous winding line.'
    kind = "background"

    palette = Palette()
    order   = Slider(3, 7, default=6, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        order = int(self.order)

        # Hilbert L-system; A/B are non-drawing rewrite symbols, F draws.
        sentence = art_kit.lindenmayer(
            "A",
            {"A": "-BF+AFA+FB-", "B": "+AF-BFB-FA+"},
            order,
        )
        segs = art_kit.turtle_segments(
            sentence, start=(0.0, 0.0), heading=0.0, step=1.0,
            turn=math.radians(90.0),
        )
        if not segs:
            canvas.commit(canvas.create_image())
            return

        xs = [p for sg in segs for p in (sg[0], sg[2])]
        ys = [p for sg in segs for p in (sg[1], sg[3])]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        margin = s * 0.07
        avail = s - 2 * margin
        scale = avail / max(x1 - x0, y1 - y0, 1e-6)
        ox = margin + (avail - (x1 - x0) * scale) * 0.5
        oy = margin + (avail - (y1 - y0) * scale) * 0.5

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")
        width = max(1, int(s * 0.10 / (2 ** order)))
        n = len(segs)
        for i, (a, b, c, d) in enumerate(segs):
            t = 0.18 + 0.78 * (i / max(1, n - 1))
            p1 = (ox + (a - x0) * scale, oy + (b - y0) * scale)
            p2 = (ox + (c - x0) * scale, oy + (d - y0) * scale)
            draw.line((p1, p2), fill=art_kit.palette_color(t), width=width)

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.003))
        canvas.commit(Image.alpha_composite(glow, img))
