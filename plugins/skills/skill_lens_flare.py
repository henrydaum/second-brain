from plugins.BaseSkill import BaseSkill

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class LensFlareSkill(BaseSkill):
    name = 'Lens Flare'
    description = 'Palette-tinted lens flare: a glowing source point with a string of secondary ghost discs along the line from the source through the canvas center. Pan to place the light source. Params: brightness (0.0-1.5, default 0.85), ghosts (0-8, default 5).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'palette', 'name': 'palette', 'label': 'Palette'},
        {'type': 'slider', 'name': 'brightness', 'label': 'Brightness', 'min': 0.0, 'max': 1.5, 'step': 0.05, 'default': 0.85},
        {'type': 'pan', 'name': 'source', 'label': 'Source', 'x_param': 'sx', 'y_param': 'sy', 'step': 0.05, 'x_default': 0.25, 'y_default': 0.25},
    ]

    def run(self, canvas, brightness=0.85, sx=0.25, sy=0.25, ghosts=5):
        s = canvas.size
        b = float(art_kit.clamp(brightness, 0.0, 2.0))
        sx = float(art_kit.clamp(sx, 0.0, 1.0))
        sy = float(art_kit.clamp(sy, 0.0, 1.0))
        n = int(art_kit.clamp(ghosts, 0, 12))

        base = canvas.image.convert("RGBA")
        flare = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        draw = ImageDraw.Draw(flare, "RGBA")

        px = sx * s
        py = sy * s
        cx = s / 2.0
        cy = s / 2.0

        # Main glow
        main_color = art_kit.hex_to_rgb(canvas.palette.accent)
        main_r = s * 0.20
        for i in range(8, 0, -1):
            r = main_r * (i / 8.0)
            alpha = int(min(255, 30 * i * b))
            draw.ellipse((px - r, py - r, px + r, py + r), fill=(*main_color, alpha))

        # Ghost discs along the line through center
        dx = cx - px
        dy = cy - py
        for i in range(1, n + 1):
            t = i / float(n + 1) * 1.8  # extend past center
            gx = px + dx * t * 2.0
            gy = py + dy * t * 2.0
            ratio = i / max(1, n)
            color = art_kit.hex_to_rgb(art_kit.palette_color(ratio))
            r = s * (0.015 + 0.05 * (1.0 - ratio))
            alpha = int(120 * b * (1.0 - 0.4 * ratio))
            draw.ellipse((gx - r, gy - r, gx + r, gy + r), fill=(*color, alpha))

        flare = flare.filter(ImageFilter.GaussianBlur(s * 0.008))

        # Streaks: long horizontal/vertical lines from source
        streak = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(streak, "RGBA")
        streak_color = (*art_kit.hex_to_rgb(canvas.palette.primary), int(140 * b))
        sdraw.line((px - s, py, px + s, py), fill=streak_color, width=max(1, int(s * 0.004)))
        sdraw.line((px, py - s, px, py + s), fill=streak_color, width=max(1, int(s * 0.004)))
        streak = streak.filter(ImageFilter.GaussianBlur(s * 0.006))

        out = Image.alpha_composite(base, flare)
        out = Image.alpha_composite(out, streak)
        canvas.commit(out)
