from plugins.BaseSkill import BaseSkill, Palette, Slider
from PIL import ImageDraw

import random

try:
    art_kit
except NameError:
    art_kit = None


class TruchetFieldSkill(BaseSkill):
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

        for r in range(rows):
            for c in range(cols):
                x0 = c * tile
                y0 = r * tile
                use_a = rng.random() < curl_p
                half = tile / 2
                pad = lw / 2
                if use_a:
                    cx1, cy1 = x0, y0
                    bbox1 = [cx1 - half - pad, cy1 - half - pad, cx1 + half + pad, cy1 + half + pad]
                    draw.arc(bbox1, start=0, end=90, fill=stroke, width=lw)
                    cx2, cy2 = x0 + tile, y0 + tile
                    bbox2 = [cx2 - half - pad, cy2 - half - pad, cx2 + half + pad, cy2 + half + pad]
                    draw.arc(bbox2, start=180, end=270, fill=stroke, width=lw)
                else:
                    cx1, cy1 = x0 + tile, y0
                    bbox1 = [cx1 - half - pad, cy1 - half - pad, cx1 + half + pad, cy1 + half + pad]
                    draw.arc(bbox1, start=90, end=180, fill=stroke, width=lw)
                    cx2, cy2 = x0, y0 + tile
                    bbox2 = [cx2 - half - pad, cy2 - half - pad, cx2 + half + pad, cy2 + half + pad]
                    draw.arc(bbox2, start=270, end=360, fill=stroke, width=lw)

        canvas.commit(img)
