from plugins.BaseSkill import BaseSkill, Slider, Enum, Palette

import math
import random
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class IsoHexCubesSkill(BaseSkill):
    name = "Iso Hex Cubes"
    description = "Tessellated isometric cubes on a hexagonal grid. Each cube shows three visible faces (top diamond + two side rhombi) shaded from a base palette color. Optional per-cube height jitter raises some cubes for a stacked-blocks look."
    kind = "background"
    palette = Palette()
    cube_size = Slider(20, 90, default=46, step=2)
    height_jitter = Slider(0.0, 1.0, default=0.35, step=0.05)
    top_shade = Enum([("light", "Light"), ("accent", "Accent"), ("random", "Random")], default="random")
    gap = Slider(0, 4, default=0, step=1)

    def run(self, canvas):
        s = canvas.size
        rng = random.Random(canvas.seed)
        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")

        size = float(self.cube_size)
        gap = float(self.gap)
        h_jit = float(self.height_jitter)

        cos30 = math.cos(math.radians(30.0))
        sin30 = math.sin(math.radians(30.0))
        hw = size * cos30  # half-width of diamond
        hh = size * sin30  # half-height of diamond top
        # Hex tiling spacing.
        dx = 2.0 * hw + gap
        dy = (hh + size * 0.5) * 2.0 + gap * 0.5  # vertical step between rows of two interlocking hex columns

        # Generate cube centers via row/col with offset rows.
        cubes = []
        rows = int(s / dy) + 4
        cols = int(s / dx) + 4
        for r in range(-2, rows):
            offset = (r % 2) * hw
            for c in range(-2, cols):
                cx = c * dx + offset + hw
                # Top of the cube's diamond apex sits at cy; vertical position:
                cy = r * (hh + size * 0.5) + hh
                if cx < -size or cx > s + size or cy < -size or cy > s + size:
                    continue
                lift = (rng.random() * 2.0 - 1.0) * h_jit * size * 0.5
                cubes.append((cx, cy + lift, rng.random()))

        # Back-to-front: smaller cy first.
        cubes.sort(key=lambda c: c[1])

        bg_hex = canvas.palette.background
        accent_hex = canvas.palette.accent
        top_mode = str(self.top_shade)

        for cx, cy, jitter in cubes:
            if top_mode == "light":
                top_color = art_kit.palette_color(0.85)
            elif top_mode == "accent":
                top_color = accent_hex
            else:
                top_color = art_kit.palette_color(0.4 + 0.55 * jitter)
            left_color = art_kit.mix_hex(top_color, bg_hex, 0.45)
            right_color = art_kit.mix_hex(top_color, bg_hex, 0.65)

            # Top diamond around (cx, cy).
            top = [
                (cx, cy - hh),
                (cx + hw, cy),
                (cx, cy + hh),
                (cx - hw, cy),
            ]
            # Side faces drop by `size` in screen y.
            side_h = size
            left = [
                (cx - hw, cy),
                (cx, cy + hh),
                (cx, cy + hh + side_h),
                (cx - hw, cy + side_h),
            ]
            right = [
                (cx, cy + hh),
                (cx + hw, cy),
                (cx + hw, cy + side_h),
                (cx, cy + hh + side_h),
            ]
            draw.polygon(left, fill=left_color)
            draw.polygon(right, fill=right_color)
            draw.polygon(top, fill=top_color)

        canvas.commit(img)
