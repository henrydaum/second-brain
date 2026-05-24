from plugins.BaseSkill import BaseSkill, Enum, Palette, Pan, Slider

import math
from PIL import ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class LSystemCrestSkill(BaseSkill):
    name = "L-System Crest"
    description = "Object overlay: a compact turtle-grammar ornament, switching between fern, tree, and coral silhouettes."
    kind = "object"
    palette = Palette()
    cx = Slider(0, 1, default=0.5, step=0.04)
    cy = Slider(0, 1, default=0.55, step=0.04)
    center = Pan(x="cx", y="cy")
    scale = Slider(0.45, 1.45, default=0.9, step=0.05)
    variant = Enum([("fern", "Fern"), ("tree", "Tree"), ("coral", "Coral")], default="fern")

    def run(self, canvas):
        specs = {
            "fern": ("X", {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"}, 5, 25),
            "tree": ("F", {"F": "FF+[+F-F-F]-[-F+F+F]"}, 4, 22.5),
            "coral": ("F", {"F": "F[+F]F[-F][F]"}, 4, 20),
        }
        axiom, rules, depth, turn = specs.get(str(self.variant), specs["fern"])
        segs = art_kit.turtle_segments(art_kit.lindenmayer(axiom, rules, depth), step=1, turn=math.radians(turn))
        xs = [v for x1, y1, x2, y2 in segs for v in (x1, x2)]
        ys = [v for x1, y1, x2, y2 in segs for v in (y1, y2)]
        w, h = max(xs) - min(xs) or 1, max(ys) - min(ys) or 1
        m, sx, sy = canvas.size * 0.34 * self.scale, self.cx * canvas.size, self.cy * canvas.size
        k = m / max(w, h)
        ox, oy = sx - (min(xs) + w / 2) * k, sy - (min(ys) + h / 2) * k
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        for i, (x1, y1, x2, y2) in enumerate(segs):
            draw.line((x1 * k + ox, y1 * k + oy, x2 * k + ox, y2 * k + oy), fill=art_kit.with_alpha(art_kit.palette_color(0.25 + 0.7 * i / len(segs)), 180), width=max(1, int(canvas.size * 0.004 * self.scale)))
        canvas.commit(img)
