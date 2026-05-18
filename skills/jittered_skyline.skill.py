SKILL_NAME = "Jittered Skyline"
SKILL_DESCRIPTION = "A city as silhouettes of varying-height rectangles on a jittered grid, with scattered palette-bright window pixels and an fbm sky. No skyscraper details, no street -- the city emerges from the rectangle distribution. Good for \"city\", \"skyline\", \"buildings\", \"urban\", or \"downtown\"."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1779667200.0
SKILL_CONTROLS = [
    {"type": "enum", "name": "time", "label": "Time",
     "options": [
         {"value": "dusk", "label": "Dusk"},
         {"value": "night", "label": "Night"},
         {"value": "dawn", "label": "Dawn"},
     ], "default": "dusk"},
    {"type": "palette", "name": "palette", "label": "Palette"},
]

import math
import random
import numpy as np
from PIL import Image, ImageDraw


def run(canvas, time="dusk", **_):
    s = int(canvas.size)
    seed = int(canvas.seed)
    rng = random.Random(seed)

    horizon = int(s * 0.62)

    # Sky gradient via fbm.
    y_idx, x_idx = np.mgrid[0:s, 0:s].astype(np.float32)
    sky_t = np.clip(y_idx / horizon, 0.0, 1.0)
    sky_t = sky_t * sky_t * (3.0 - 2.0 * sky_t)
    LUT = 256
    top = art_kit.palette_color(0.85 if str(time) != "night" else 0.7)
    bot = art_kit.palette_color(0.35 if str(time) != "night" else 0.15)
    sky_lut = np.array(
        [art_kit.hex_to_rgb(art_kit.mix_hex(top, bot, k / (LUT - 1))) for k in range(LUT)],
        dtype=np.uint8,
    )
    rgb = sky_lut[np.clip((sky_t * (LUT - 1)).astype(np.int32), 0, LUT - 1)]
    img = Image.fromarray(rgb, "RGB").convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    # Two layers of buildings (back faded, front dark).
    for layer in range(2):
        depth = 1 - layer
        n_cols = 22 + layer * 14
        col_w = s / n_cols
        layer_base = horizon + int(layer * s * 0.05)
        b_color_t = 0.05 + 0.25 * depth
        b_color = art_kit.palette_color(b_color_t)
        for ci in range(n_cols):
            cx_left = ci * col_w + rng.uniform(-col_w * 0.15, col_w * 0.15)
            w = col_w * (0.7 + rng.random() * 0.5)
            h_factor = 0.25 + rng.random() ** 1.6 * 0.7
            h = (s * 0.45) * h_factor * (0.6 + 0.5 * depth)
            top_y = layer_base - h
            draw.rectangle((cx_left, top_y, cx_left + w, s), fill=b_color)

            # Windows: only on front layer, sparse.
            if layer == 1 and h > 30:
                win_color = art_kit.palette_color(0.92, value=1.1)
                step_y = max(6, int(h * 0.06))
                step_x = max(4, int(w * 0.18))
                y = top_y + step_y
                while y < layer_base - step_y:
                    x_w = cx_left + step_x * 0.5
                    while x_w < cx_left + w - step_x * 0.5:
                        if rng.random() < 0.22:
                            draw.rectangle((x_w, y, x_w + 2, y + 3), fill=win_color)
                        x_w += step_x
                    y += step_y

    canvas.commit(img)
