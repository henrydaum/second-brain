from plugins.BaseTechnique import BaseTechnique, Palette, Slider
from PIL import ImageDraw

import math
import random


def _arc_points(cx, cy, radius, start_deg, end_deg, steps=24):
    a0 = math.radians(start_deg)
    a1 = math.radians(end_deg)
    return [
        (cx + radius * math.cos(a0 + (a1 - a0) * i / steps),
         cy + radius * math.sin(a0 + (a1 - a0) * i / steps))
        for i in range(steps + 1)
    ]

try:
    art_kit
except NameError:
    art_kit = None


class TruchetFieldTechnique(BaseTechnique):
    name = "Truchet Field"
    description = "Background: a grid of arc-quadrant tiles forming maze-like ribbons in palette primary on palette background."
    kind = "background"
    palette = Palette()
    tile_size = Slider(20, 80, default=40, step=2)
    curl_chance = Slider(0.0, 1.0, default=0.5, step=0.02)
    line_weight = Slider(2, 14, default=6, step=1)

    def run(self, canvas):
        img = canvas.new(color=canvas.palette.background)
        size = canvas.size
        draw = ImageDraw.Draw(img, "RGBA")
        rng = random.Random(canvas.seed)
        tile = int(round(float(self.tile_size)))
        curl_p = float(self.curl_chance)
        lw = int(round(float(self.line_weight)))
        stroke = canvas.palette.primary

        cols = (size + tile - 1) // tile + 1
        rows = (size + tile - 1) // tile + 1

        radius = tile / 2
        for row in range(rows):
            for col in range(cols):
                x0 = col * tile
                y0 = row * tile
                use_a = rng.random() < curl_p
                if use_a:
                    p1 = _arc_points(x0, y0, radius, 0, 90)
                    p2 = _arc_points(x0 + tile, y0 + tile, radius, 180, 270)
                else:
                    p1 = _arc_points(x0 + tile, y0, radius, 90, 180)
                    p2 = _arc_points(x0, y0 + tile, radius, 270, 360)
                draw.line(p1, fill=stroke, width=lw, joint="curve")
                draw.line(p2, fill=stroke, width=lw, joint="curve")

        canvas.commit(img)
