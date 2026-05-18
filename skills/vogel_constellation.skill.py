SKILL_NAME = "Vogel Constellation"
SKILL_DESCRIPTION = "A starfield laid out with Vogel-spiral seed positions, connected into a constellation graph by nearest-neighbor links. fbm sky behind, palette-bright stars in front. No literal constellations -- the structure emerges from the spiral. Good for \"stars\", \"night sky\", \"constellation\", \"galaxy\", or \"cosmos\"."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1779667200.0
SKILL_CONTROLS = [
    {"type": "enum", "name": "depth", "label": "Depth",
     "options": [
         {"value": "shallow", "label": "Shallow"},
         {"value": "deep", "label": "Deep"},
         {"value": "abyss", "label": "Abyss"},
     ], "default": "deep"},
    {"type": "palette", "name": "palette", "label": "Palette"},
]

import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def run(canvas, depth="deep", **_):
    s = int(canvas.size)
    seed = int(canvas.seed)
    rng = random.Random(seed)
    n_stars = {"shallow": 80, "deep": 180, "abyss": 320}.get(str(depth), 180)

    # Background: fbm sky, sampled coarse then upscaled.
    LOW = 96
    low = np.array(
        [[art_kit.fbm(seed + 31, x * 0.04, y * 0.04, octaves=4) for x in range(LOW)]
         for y in range(LOW)],
        dtype=np.float32,
    )
    low_img = Image.fromarray(np.clip(low * 255.0, 0, 255).astype(np.uint8), "L").resize((s, s), Image.BICUBIC)
    field = np.asarray(low_img, dtype=np.float32) / 255.0
    LUT = 256
    sky_lut = np.array(
        [art_kit.hex_to_rgb(art_kit.mix_hex(
            art_kit.palette_color(0.05), art_kit.palette_color(0.45),
            k / (LUT - 1))) for k in range(LUT)],
        dtype=np.uint8,
    )
    rgb = sky_lut[np.clip((field * (LUT - 1)).astype(np.int32), 0, LUT - 1)]
    img = Image.fromarray(rgb, "RGB").convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    scale = s * 0.48
    points_raw = art_kit.vogel_spiral(n_stars, scale=scale)
    cx, cy = s * 0.5, s * 0.5
    rot = rng.uniform(0.0, math.tau)
    cr, sr = math.cos(rot), math.sin(rot)
    points = [(cx + (x * cr - y * sr), cy + (x * sr + y * cr)) for (x, y) in points_raw]

    # Faint constellation lines: connect each star to its nearest-by-index neighbor (close pairs).
    line_color = art_kit.palette_color(0.55, value=0.6)
    for i in range(len(points) - 1):
        if rng.random() < 0.35:
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            if (x2 - x1) ** 2 + (y2 - y1) ** 2 < (s * 0.18) ** 2:
                draw.line((x1, y1, x2, y2), fill=line_color, width=1)

    # Stars: brighter, varied radius.
    for i, (x, y) in enumerate(points):
        t = i / max(1, n_stars - 1)
        brightness = 0.7 + 0.3 * rng.random()
        color = art_kit.palette_color(0.85 + 0.13 * brightness, value=1.0 + 0.1 * brightness)
        r = 0.8 + (1.0 - t) * 2.6 + rng.random() * 0.8
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.006))
    out = Image.alpha_composite(glow, img)
    canvas.commit(out)
