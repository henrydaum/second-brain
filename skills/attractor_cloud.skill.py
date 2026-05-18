SKILL_NAME = "Attractor Cloud"
SKILL_DESCRIPTION = "A strange-attractor point cloud (de Jong or Clifford) accumulated into a palette-graded density image. Each pass adds organic, smoke-like structure that fills the canvas with mathematical residue. No literal subject -- pure algorithmic form. Good for \"abstract\", \"organic\", \"smoke\", \"dust\", \"swarm\", or any prompt without a clear technique."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1779667200.0
SKILL_CONTROLS = [
    {"type": "enum", "name": "kind", "label": "Kind",
     "options": [
         {"value": "de_jong", "label": "De Jong"},
         {"value": "clifford", "label": "Clifford"},
     ], "default": "de_jong"},
    {"type": "palette", "name": "palette", "label": "Palette"},
]

import math
import numpy as np
from PIL import Image


def run(canvas, kind="de_jong", **_):
    s = int(canvas.size)
    seed = int(canvas.seed)
    n_points = 220_000

    pts = art_kit.attractor_points(str(kind), n_points, seed)
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)

    margin = s * 0.06
    span = s - 2 * margin
    cx = (xs * 0.5 + 0.5) * span + margin
    cy = (ys * 0.5 + 0.5) * span + margin
    ix = np.clip(cx.astype(np.int32), 0, s - 1)
    iy = np.clip(cy.astype(np.int32), 0, s - 1)

    density = np.zeros((s, s), dtype=np.float32)
    np.add.at(density, (iy, ix), 1.0)
    # Compress dynamic range.
    density = np.log1p(density)
    dmax = float(density.max()) or 1.0
    density = density / dmax
    density = density ** 0.7

    LUT = 256
    lut = np.array(
        [art_kit.hex_to_rgb(art_kit.palette_color(0.1 + 0.85 * (k / (LUT - 1))))
         for k in range(LUT)],
        dtype=np.uint8,
    )
    bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
    idx = np.clip((density * (LUT - 1)).astype(np.int32), 0, LUT - 1)
    rgb = lut[idx]
    mask = density < 0.02
    rgb[mask] = bg

    canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
