SKILL_NAME = "Fbm Horizon"
SKILL_DESCRIPTION = "An abstract landscape built from layered fbm hill silhouettes against a graded sky. Atmospheric perspective lightens distant ridges toward the sky color. No literal trees, no clouds -- the terrain emerges from noise. Good for \"landscape\", \"mountains\", \"hills\", \"horizon\", or \"valley\"."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1779667200.0
SKILL_CONTROLS = [
    {"type": "enum", "name": "mood", "label": "Mood",
     "options": [
         {"value": "dawn", "label": "Dawn"},
         {"value": "haze", "label": "Haze"},
         {"value": "stark", "label": "Stark"},
     ], "default": "dawn"},
    {"type": "palette", "name": "palette", "label": "Palette"},
]

import math
import numpy as np
from PIL import Image


def run(canvas, mood="dawn", **_):
    s = int(canvas.size)
    seed = int(canvas.seed)
    layers = {"dawn": 5, "haze": 6, "stark": 4}.get(str(mood), 5)
    haze_strength = {"dawn": 0.55, "haze": 0.8, "stark": 0.2}.get(str(mood), 0.55)

    horizon_y = int(s * 0.55)

    y_idx, x_idx = np.mgrid[0:s, 0:s].astype(np.float32)
    sky_t = np.clip(y_idx / horizon_y, 0.0, 1.0)
    sky_t = sky_t * sky_t * (3.0 - 2.0 * sky_t)

    LUT = 256
    sky_lut = np.array(
        [art_kit.hex_to_rgb(art_kit.mix_hex(
            art_kit.palette_color(0.85),
            art_kit.palette_color(0.4),
            k / (LUT - 1))) for k in range(LUT)],
        dtype=np.uint8,
    )
    sky_idx = np.clip((sky_t * (LUT - 1)).astype(np.int32), 0, LUT - 1)
    rgb = sky_lut[sky_idx]

    xs = np.arange(s, dtype=np.float32)
    for li in range(layers):
        depth = li / max(1, layers - 1)
        amp = (s * 0.10) + depth * (s * 0.18)
        base_y = horizon_y + int(depth * s * 0.22)
        scale = 0.004 + depth * 0.006
        heights = np.array(
            [art_kit.fbm(seed + li * 1009, x * scale * s, depth * 7.0, octaves=4) for x in xs],
            dtype=np.float32,
        )
        ridge = base_y - (heights - 0.5) * amp

        # Per-column fill below ridge -> shade tone for this layer.
        shade_t = 0.05 + (1.0 - depth) * 0.45
        base_rgb = np.array(art_kit.hex_to_rgb(art_kit.palette_color(shade_t)), dtype=np.float32)
        sky_blend = np.array(art_kit.hex_to_rgb(art_kit.palette_color(0.9)), dtype=np.float32)
        haze = depth * haze_strength
        layer_rgb = (base_rgb * (1.0 - haze) + sky_blend * haze).astype(np.uint8)

        mask = y_idx >= ridge[np.newaxis, :]
        rgb[mask] = layer_rgb

    canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
