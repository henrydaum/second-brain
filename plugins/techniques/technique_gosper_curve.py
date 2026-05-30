from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import math
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class GosperCurveTechnique(BaseTechnique):
    name = 'Gosper Curve'
    description = 'The Gosper curve (flowsnake): a space-filling fractal whose path tiles the plane with hexagonal Gosper islands, drawn as one continuous stroke that sweeps the palette ramp from start to finish. Each order replaces every segment with seven, coiling tighter. Good for "Gosper curve", "flowsnake", "hex fractal", "space-filling curve", or a coiling continuous line.'
    kind = "background"

    palette = Palette()
    order   = Slider(2, 5, default=4, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        order = int(self.order)

        sentence = art_kit.lindenmayer(
            "A",
            {"A": "A-B--B+A++AA+B-", "B": "+A-BB--B-A++A+B"},
            order,
        )
        # Both A and B are forward-draw moves in the Gosper grammar.
        drawable = sentence.replace("A", "F").replace("B", "F")
        segs = art_kit.turtle_segments(
            drawable, start=(0.0, 0.0), heading=0.0, step=1.0,
            turn=math.radians(60.0),
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
        width = max(1, int(s * 0.05 / (1.6 ** order)))
        n = len(segs)
        for i, (a, b, cc, d) in enumerate(segs):
            t = 0.18 + 0.78 * (i / max(1, n - 1))
            p1 = (ox + (a - x0) * scale, oy + (b - y0) * scale)
            p2 = (ox + (cc - x0) * scale, oy + (d - y0) * scale)
            draw.line((p1, p2), fill=art_kit.palette_color(t), width=width)

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.003))
        canvas.commit(Image.alpha_composite(glow, img))
