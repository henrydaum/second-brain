from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import math
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class MaurerRoseTechnique(BaseTechnique):
    name = 'Maurer Rose'
    description = 'A Maurer rose: walk a rose curve r = sin(n*theta) in fixed integer-degree steps and connect the samples with straight chords. The chords weave a luminous moire web of overlapping lines across the petals, far richer than the smooth rose alone. Good for "rose curve", "Maurer rose", "rhodonea", "string art", "moire web", or a mathematical line mandala.'
    kind = "background"

    palette = Palette()
    petals  = Slider(2, 12, default=6, step=1)
    walk    = Slider(11, 200, default=71, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        n = int(self.petals)
        d = int(self.walk)

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")
        cx = cy = s / 2.0
        R = s * 0.45

        # The straight-chord Maurer web.
        pts = []
        for k in range(361):
            theta = math.radians(k * d)
            r = math.sin(n * theta)
            pts.append((cx + R * r * math.cos(theta), cy + R * r * math.sin(theta)))
        for i in range(len(pts) - 1):
            t = 0.25 + 0.7 * (i / float(len(pts) - 1))
            draw.line((pts[i], pts[i + 1]), fill=art_kit.palette_color(t), width=max(1, int(s * 0.0015)))

        # The smooth underlying rose, brighter, on top.
        smooth = []
        for k in range(721):
            theta = math.radians(k * 0.5)
            r = math.sin(n * theta)
            smooth.append((cx + R * r * math.cos(theta), cy + R * r * math.sin(theta)))
        draw.line(smooth, fill=art_kit.palette_color(0.95), width=max(1, int(s * 0.0022)))

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.004))
        canvas.commit(Image.alpha_composite(glow, img))
