from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import math
from PIL import Image, ImageDraw

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

def _hex_polygon(cx, cy, r):
    # Pointy-top hexagon: vertices at 30, 90, 150, 210, 270, 330 degrees.
    pts = []
    for i in range(6):
        a = math.radians(60 * i - 30)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts

def _hex_top_face(cx, cy, r):
    # Top face for iso-towers: same as hex polygon, drawn at the elevated y.
    return _hex_polygon(cx, cy, r)


class HexMosaicTechnique(BaseTechnique):
    name = 'Hex Mosaic'
    description = 'Pointy-top hexagonal tiling with column-offset coordinates. Each hex is drawn as a filled polygon, colored by one of three rules: a radial palette gradient outward from center, an fbm-perturbed mosaic so soft regions of related hue emerge across the grid, or an isometric extrusion that treats each hex\'s height as a three-shade column. Hex outlines are anti-aliased and flush so the mosaic reads clean. Good for "hex", "hexagons", "tiling", "mosaic", "honeycomb", "cells", or any modular-grid algorithmic motif.'
    kind = "background"
    palette = Palette()
    style = Enum([('gradient', 'Radial Gradient'), ('fbm', 'fBm Regions'), ('iso_towers', 'Iso Towers')], default='iso_towers')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        self.style = str(self.style)

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")

        # Hex radius: tuned so the tiling fills the canvas with ~22 hexes across.
        r = s / 24.0
        hex_w = math.sqrt(3) * r        # width of a pointy-top hex
        hex_h_step = 1.5 * r            # vertical step between row centers

        cols = int(s / hex_w) + 2
        rows = int(s / hex_h_step) + 3

        cx_center = s / 2.0
        cy_center = s / 2.0
        max_d = math.hypot(s / 2.0, s / 2.0)

        outline = art_kit.palette_color(0.05)
        outline_w = max(1, int(r * 0.06))

        if self.style == "iso_towers":
            # Render back-to-front so towers in front overlap towers behind.
            cells = []
            for row in range(-1, rows):
                for col in range(-1, cols):
                    cx = col * hex_w + (hex_w * 0.5 if row % 2 else 0.0)
                    cy = row * hex_h_step + r
                    t_height = art_kit.fbm(seed, cx * 0.012, cy * 0.012, octaves=4)
                    # Quantize to a small number of heights for the "town" feel.
                    height = (0.20 + 0.65 * t_height) * (s * 0.18)
                    cells.append((row, col, cx, cy, t_height, height))
            # Sort by cy ascending so further-back cells (smaller y) render first.
            cells.sort(key=lambda c: c[3])
            for _, _, cx, cy, t_h, height in cells:
                if not (-hex_w <= cx <= s + hex_w and -hex_h_step <= cy <= s + hex_h_step):
                    continue
                base = art_kit.palette_color(art_kit.clamp(0.30 + 0.60 * t_h, 0.0, 1.0))
                top = art_kit.palette_color(art_kit.clamp(0.50 + 0.40 * t_h, 0.0, 1.0))
                shade = art_kit.palette_color(art_kit.clamp(0.18 + 0.40 * t_h, 0.0, 1.0))
                # Base hex (footprint).
                draw.polygon(_hex_polygon(cx, cy, r), fill=base, outline=outline)
                # Side panels: connect lower 3 vertices to corresponding raised vertices.
                poly = _hex_polygon(cx, cy, r)
                poly_top = [(x, y - height) for (x, y) in poly]
                # Two visible side panels (lower-left and lower-right of the hex).
                # Vertices order: 0=right, 1=bottom-right, 2=bottom-left, 3=left, 4=top-left, 5=top-right.
                # The bottom-facing edges in a pointy-top hex viewed from above are 1-2 and 2-3.
                # We use a simpler set: edges (1,2) and (2,3).
                draw.polygon([poly[1], poly[2], poly_top[2], poly_top[1]], fill=shade, outline=outline)
                draw.polygon([poly[2], poly[3], poly_top[3], poly_top[2]], fill=shade, outline=outline)
                # Top.
                draw.polygon(poly_top, fill=top, outline=outline)
            canvas.commit(img)
            return

        for row in range(-1, rows):
            for col in range(-1, cols):
                cx = col * hex_w + (hex_w * 0.5 if row % 2 else 0.0)
                cy = row * hex_h_step + r
                if not (-hex_w <= cx <= s + hex_w and -hex_h_step <= cy <= s + hex_h_step):
                    continue
                if self.style == "gradient":
                    d = math.hypot(cx - cx_center, cy - cy_center) / max_d
                    t = art_kit.clamp(0.18 + 0.78 * d, 0.0, 1.0)
                else:  # fbm
                    f = art_kit.fbm(seed, cx * 0.010, cy * 0.010, octaves=4)
                    t = art_kit.clamp(0.20 + 0.75 * f, 0.0, 1.0)
                fill = art_kit.palette_color(t)
                draw.polygon(_hex_polygon(cx, cy, r), fill=fill, outline=outline, width=outline_w)

        canvas.commit(img)
