SKILL_NAME = "Voronoi Cells"
SKILL_DESCRIPTION = "Voronoi tiling of jittered grid seeds, each cell filled from a palette ramp. Reliable choice for cellular tissue, basalt columns, cracked earth, stained glass, mosaic abstracts. Params: cells (16-300, default 90), jitter (0.0-1.0, default 0.65)."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

import random
import numpy as np
from PIL import Image


def run(canvas, cells=90, jitter=0.65):
    rng = random.Random(canvas.seed)
    rng_np = np.random.default_rng(canvas.seed)
    s = canvas.size
    n = int(art_kit.clamp(cells, 12, 400))
    j = float(art_kit.clamp(jitter, 0.0, 1.0))

    # Seed points on a jittered grid for roughly even spacing.
    side = max(2, int(round(n ** 0.5)))
    cell_w = s / side
    seeds = []
    for r in range(side):
        for c in range(side):
            cx = (c + 0.5) * cell_w + (rng.random() - 0.5) * cell_w * j
            cy = (r + 0.5) * cell_w + (rng.random() - 0.5) * cell_w * j
            seeds.append((cx, cy))
    seeds = np.asarray(seeds[:n], dtype=np.float32)

    # Compute at reduced resolution for speed, then NEAREST-resize to full size.
    res = min(s, 360)
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    sx = xx * (s / res)
    sy = yy * (s / res)
    dx = sx[..., None] - seeds[:, 0]
    dy = sy[..., None] - seeds[:, 1]
    idx = np.argmin(dx * dx + dy * dy, axis=-1)

    # Palette colors per cell index.
    palette = np.zeros((len(seeds), 3), dtype=np.uint8)
    for i in range(len(seeds)):
        t = (i / max(1, len(seeds) - 1) + rng.random() * 0.06) % 1.0
        palette[i] = art_kit.hex_to_rgb(art_kit.palette_color(t))
    out = palette[idx]
    img = Image.fromarray(out, "RGB").resize((s, s), Image.NEAREST).convert("RGBA")
    canvas.commit(img)
