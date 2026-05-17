SKILL_NAME = "Sunflower Field"
SKILL_DESCRIPTION = "Vogel/golden-angle spiral of palette-graded seeds. Reliable choice for sunflowers, daisies, dandelions, dense radial fields. Params: count (50-3000, default 1100), seed_size (0.005-0.04, default 0.018), inner_falloff (0.0-1.0, default 0.5)."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

import math, random
from PIL import Image, ImageDraw


def run(canvas, count=1100, seed_size=0.018, inner_falloff=0.5):
    rng = random.Random(canvas.seed)
    s = canvas.size
    img = canvas.create_image()
    draw = ImageDraw.Draw(img, "RGBA")
    n = int(art_kit.clamp(count, 50, 3000))
    size_frac = float(art_kit.clamp(seed_size, 0.003, 0.05))
    falloff = float(art_kit.clamp(inner_falloff, 0.0, 1.0))
    cx, cy = s / 2.0, s / 2.0
    for i, (nx, ny) in enumerate(art_kit.vogel_spiral(n, scale=0.46)):
        x = cx + nx * s
        y = cy + ny * s
        t = i / max(1, n - 1)
        color = art_kit.palette_color(0.18 + t * 0.78)
        r = s * size_frac * (1.0 + falloff * t)
        jitter = rng.uniform(-r * 0.15, r * 0.15)
        draw.ellipse((x - r + jitter, y - r, x + r + jitter, y + r), fill=color)
    canvas.commit(img)
