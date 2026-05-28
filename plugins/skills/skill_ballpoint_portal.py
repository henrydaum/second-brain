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
        base = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        paper_img = Image.new("RGBA", (s, s), paper)
        draw = ImageDraw.Draw(paper_img, "RGBA")

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

        gx = np.zeros_like(lum)
        gy = np.zeros_like(lum)
        gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
        gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
        edge = np.clip(np.hypot(gx, gy) * 8.0, 0.0, 1.0)

        # Per-pixel mask: dark/detail regions become the portal; clean regions stay original.
        mask = np.clip((1.0 - lum) * (0.3 + 1.6 * depth) - (0.55 - 0.5 * depth), 0.0, 1.0)
        paper_arr = np.asarray(paper_img.convert("RGB"), dtype=np.float32) / 255.0
        paper_alpha = np.clip((mask * 0.35 + edge * 0.55) * (0.18 + 0.72 * depth), 0.0, 0.82)
        img = Image.fromarray(np.clip((arr * (1.0 - paper_alpha[..., None]) + paper_arr * paper_alpha[..., None]) * 255, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        draw = ImageDraw.Draw(img, "RGBA")

        detail = np.clip(edge * 1.45 + mask * 0.08, 0.0, 1.0)

        cell = max(5, int(12 - min(density, 1.5) * 3))
        for cy in range(0, s, cell):
            for cx in range(0, s, cell):
                d = float(detail[cy:cy + cell, cx:cx + cell].mean())
                if d < 0.10:
                    continue
                ex = float(gx[cy:cy + cell, cx:cx + cell].mean())
                ey = float(gy[cy:cy + cell, cx:cx + cell].mean())
                base_ang = math.atan2(ey, ex) + math.pi / 2 if abs(ex) + abs(ey) > 1e-4 else rng.choice((-0.75, 0.75))
                n_strokes = max(1, int(d * density * 4.0))
                for _ in range(n_strokes):
                    x0 = cx + rng.random() * cell
                    y0 = cy + rng.random() * cell
                    ang = base_ang + rng.uniform(-0.22, 0.22)
                    L = cell * (0.9 + d * 2.2)
                    dx, dy = math.cos(ang) * L * 0.5, math.sin(ang) * L * 0.5
                    draw.line((x0 - dx, y0 - dy, x0 + dx, y0 + dy), fill=art_kit.with_alpha(ink, 0.58 + d * 0.35), width=1)

        canvas.commit(img)
