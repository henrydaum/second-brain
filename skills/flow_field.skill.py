SKILL_NAME = "Flow Field"
SKILL_DESCRIPTION = "Streamlines drifting along an fbm vector field. Reliable choice for wind, water, hair, organic curves, abstract ribbons. Params: streamlines (20-300, default 130), step_length (0.5-5.0, default 2.0), max_steps (40-300, default 160), noise_scale (0.001-0.02, default 0.0045)."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

import math, random
from PIL import ImageDraw


def run(canvas, streamlines=130, step_length=2.0, max_steps=160, noise_scale=0.0045):
    rng = random.Random(canvas.seed)
    s = canvas.size
    img = canvas.create_image()
    draw = ImageDraw.Draw(img, "RGBA")
    n = int(art_kit.clamp(streamlines, 20, 400))
    steps = int(art_kit.clamp(max_steps, 30, 400))
    sl = float(art_kit.clamp(step_length, 0.4, 6.0))
    scale = float(art_kit.clamp(noise_scale, 0.0005, 0.04))
    for i in range(n):
        x = rng.uniform(0, s)
        y = rng.uniform(0, s)
        color = art_kit.palette_color(0.18 + (i / max(1, n - 1)) * 0.78)
        cr, cg, cb = art_kit.hex_to_rgb(color)
        prev_x, prev_y = x, y
        width = max(1, int(s * 0.0015))
        for _ in range(steps):
            angle = art_kit.fbm(canvas.seed, x * scale, y * scale, octaves=3) * math.tau * 1.6
            x += math.cos(angle) * sl
            y += math.sin(angle) * sl
            if x < 0 or x >= s or y < 0 or y >= s:
                break
            draw.line((prev_x, prev_y, x, y), fill=(cr, cg, cb, 130), width=width)
            prev_x, prev_y = x, y
    canvas.commit(img)
