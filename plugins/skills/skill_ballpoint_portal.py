from plugins.BaseSkill import BaseSkill, Slider, Enum

import math
import random
import numpy as np
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class BallpointPortalSkill(BaseSkill):
    name = "Ballpoint Portal"
    description = "Red ballpoint scribble over lined notebook paper. Depth controls how strongly the effect penetrates the canvas: dark regions get dense crosshatched scribble, light regions stay as clean lined paper showing through like windows or portals."
    kind = "filter"
    depth = Slider(0.0, 1.0, default=0.6, step=0.05)
    ink = Enum([("red", "Red"), ("blue", "Blue"), ("black", "Black")], default="red")
    line_spacing = Slider(18, 48, default=28, step=2)
    stroke_density = Slider(0.3, 1.5, default=0.9, step=0.1)

    def run(self, canvas):
        s = canvas.size
        rng = random.Random(canvas.seed)
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722

        paper = "#f6f1e0"
        img = Image.new("RGBA", (s, s), paper)
        draw = ImageDraw.Draw(img, "RGBA")

        # Blue ruled lines + red margin.
        rule = "#88a8c8"
        margin = "#c84848"
        spacing = int(self.line_spacing)
        for y in range(spacing, s, spacing):
            draw.line((0, y, s, y), fill=rule, width=1)
        margin_x = int(s * 0.10)
        draw.line((margin_x, 0, margin_x, s), fill=margin, width=2)

        ink = {"red": "#c4302b", "blue": "#1b3a8a", "black": "#1a1a1a"}[str(self.ink)]
        depth = float(self.depth)
        density = float(self.stroke_density)

        # Per-pixel mask: dark regions get more scribble, scaled by depth.
        mask = np.clip((1.0 - lum) * (0.3 + 1.6 * depth) - (0.55 - 0.5 * depth), 0.0, 1.0)

        cell = 8
        for cy in range(0, s, cell):
            for cx in range(0, s, cell):
                m = float(mask[cy:cy + cell, cx:cx + cell].mean())
                if m < 0.05:
                    continue
                n_strokes = int(m * density * 4.5)
                for _ in range(n_strokes):
                    x0 = cx + rng.random() * cell
                    y0 = cy + rng.random() * cell
                    ang = rng.uniform(-0.6, 0.6) + (math.pi / 4 if rng.random() < 0.5 else -math.pi / 4)
                    L = cell * (0.6 + rng.random() * 1.3)
                    x1 = x0 + math.cos(ang) * L
                    y1 = y0 + math.sin(ang) * L
                    draw.line((x0, y0, x1, y1), fill=ink, width=1)

        canvas.commit(img)
