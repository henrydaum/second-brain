SKILL_NAME = "Wave Sea"
SKILL_DESCRIPTION = "Water as interference: several point sources sum into a wave field, palette-mapped from troughs to crests. No literal waves drawn -- the surface emerges from sin(2*pi*d/lambda) sums. Good for \"ocean\", \"water\", \"ripples\", \"pond\", \"reflection\", or \"sound\"."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1779667200.0
SKILL_CONTROLS = [
    {"type": "enum", "name": "weather", "label": "Weather",
     "options": [
         {"value": "calm", "label": "Calm"},
         {"value": "choppy", "label": "Choppy"},
         {"value": "storm", "label": "Storm"},
     ], "default": "calm"},
    {"type": "palette", "name": "palette", "label": "Palette"},
]

import math
import random
import numpy as np
from PIL import Image


def run(canvas, weather="calm", **_):
    s = int(canvas.size)
    seed = int(canvas.seed)
    rng = random.Random(seed)

    n_sources = {"calm": 3, "choppy": 6, "storm": 10}.get(str(weather), 3)
    wl_min, wl_max = {
        "calm": (s * 0.18, s * 0.35),
        "choppy": (s * 0.08, s * 0.22),
        "storm": (s * 0.05, s * 0.18),
    }.get(str(weather), (s * 0.18, s * 0.35))

    sources = []
    for _ in range(n_sources):
        cx = rng.uniform(-s * 0.3, s * 1.3)
        cy = rng.uniform(-s * 0.3, s * 1.3)
        wl = rng.uniform(wl_min, wl_max)
        ph = rng.random()
        sources.append((cx, cy, wl, ph))

    wf = art_kit.wave_field(sources)

    y_idx, x_idx = np.mgrid[0:s, 0:s].astype(np.float32)
    field = np.zeros((s, s), dtype=np.float32)
    for cx, cy, wl, ph in sources:
        d = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        field += np.sin(2.0 * math.pi * (d / max(wl, 1e-6) + ph))
    field /= max(1, n_sources)
    field = (field + 1.0) * 0.5
    field = field * field * (3.0 - 2.0 * field)

    LUT = 256
    lut = np.array(
        [art_kit.hex_to_rgb(art_kit.palette_color(0.15 + 0.7 * (k / (LUT - 1))))
         for k in range(LUT)],
        dtype=np.uint8,
    )
    idx = np.clip((field * (LUT - 1)).astype(np.int32), 0, LUT - 1)
    rgb = lut[idx]

    canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
