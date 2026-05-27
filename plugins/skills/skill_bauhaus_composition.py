from plugins.BaseSkill import BaseSkill, Enum, Palette, Slider
from PIL import ImageDraw

import math
import random

try:
    art_kit
except NameError:
    art_kit = None


class BauhausCompositionSkill(BaseSkill):
    name = "Bauhaus Composition"
    description = "Object overlay: a Kandinsky/early-Bauhaus primary-shape composition — circle, triangle, square — in palette colors."
    kind = "object"
    palette = Palette()
    layout = Enum([("stack", "Stack"), ("triad", "Triad"), ("orbit", "Orbit")], default="triad")
    shape_count = Slider(3, 7, default=4, step=1)
    line_weight = Slider(0, 8, default=3, step=1)

    def _polygon(self, draw, pts, fill, outline, lw):
        if lw <= 1:
            draw.polygon(pts, fill=fill, outline=outline)
        else:
            draw.polygon(pts, fill=fill)
            for i in range(len(pts)):
                draw.line([pts[i], pts[(i + 1) % len(pts)]], fill=outline, width=lw)

    def _circle(self, draw, cx, cy, r, fill, outline, lw):
        if lw <= 1:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline)
        else:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=lw)

    def _triangle(self, draw, cx, cy, r, rotation, fill, outline, lw):
        pts = art_kit.regular_polygon(cx, cy, r, 3, rotation=rotation)
        self._polygon(draw, pts, fill, outline, lw)

    def _square(self, draw, cx, cy, r, rotation, fill, outline, lw):
        pts = art_kit.regular_polygon(cx, cy, r, 4, rotation=rotation)
        self._polygon(draw, pts, fill, outline, lw)

    def run(self, canvas):
        img = canvas.new_layer()
        rng = random.Random(canvas.seed)
        size = canvas.size
        draw = ImageDraw.Draw(img, "RGBA")
        layout = str(self.layout)
        count = int(round(float(self.shape_count)))
        lw = int(round(float(self.line_weight)))
        outline_color = canvas.palette.background
        colors = [canvas.palette.primary, canvas.palette.accent, canvas.palette.secondary]
        shapes = ["circle", "triangle", "square"]
        primary_radius = size * 0.22

        if layout == "stack":
            cx, cy = size * 0.5, size * 0.5
            offsets = [(0, -size * 0.12), (0, 0), (0, size * 0.12)]
            for i, (ox, oy) in enumerate(offsets):
                shape = shapes[i]
                fill = colors[i]
                rot = rng.random() * 0.3
                if shape == "circle":
                    self._circle(draw, cx + ox, cy + oy, primary_radius * 0.9, fill, outline_color, lw)
                elif shape == "triangle":
                    self._triangle(draw, cx + ox, cy + oy, primary_radius * 1.05, rot - math.pi / 2, fill, outline_color, lw)
                else:
                    self._square(draw, cx + ox, cy + oy, primary_radius * 0.85, rot, fill, outline_color, lw)
        elif layout == "orbit":
            cx, cy = size * 0.5, size * 0.5
            self._circle(draw, cx, cy, primary_radius * 0.6, colors[0], outline_color, lw)
            ring_radius = size * 0.27
            for i, shape in enumerate(["triangle", "square"]):
                ang = math.pi / 4 + i * math.pi
                ox = math.cos(ang) * ring_radius
                oy = math.sin(ang) * ring_radius
                if shape == "triangle":
                    self._triangle(draw, cx + ox, cy + oy, primary_radius * 0.7, ang - math.pi / 2, colors[1], outline_color, lw)
                else:
                    self._square(draw, cx + ox, cy + oy, primary_radius * 0.55, math.pi / 6, colors[2], outline_color, lw)
        else:
            self._circle(draw, size * 0.32, size * 0.38, primary_radius, colors[0], outline_color, lw)
            self._triangle(draw, size * 0.52, size * 0.68, primary_radius * 1.05, -math.pi / 2 + 0.18, colors[1], outline_color, lw)
            self._square(draw, size * 0.72, size * 0.35, primary_radius * 0.75, math.pi / 9, colors[2], outline_color, lw)

        for _ in range(max(0, count - 3)):
            shape = rng.choice(shapes)
            fill = rng.choice(colors)
            cx = rng.uniform(size * 0.12, size * 0.88)
            cy = rng.uniform(size * 0.12, size * 0.88)
            r = size * rng.uniform(0.04, 0.09)
            rot = rng.random() * art_kit.tau
            if shape == "circle":
                self._circle(draw, cx, cy, r, fill, outline_color, lw)
            elif shape == "triangle":
                self._triangle(draw, cx, cy, r, rot, fill, outline_color, lw)
            else:
                self._square(draw, cx, cy, r, rot, fill, outline_color, lw)

        canvas.commit(img)
