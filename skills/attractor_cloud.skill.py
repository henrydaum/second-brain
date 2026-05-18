SKILL_NAME = "Attractor Cloud"
SKILL_DESCRIPTION = "Strange-attractor point cloud (de Jong or Clifford) accumulated into a palette-graded density image. Organic, smoke-like structure fills the canvas with mathematical residue."
SKILL_KIND = "creation"
SKILL_OWNER = "web:0e0c7c0c-92af-46ef-bb48-69154d2c9f44"
SKILL_CREATED_AT = 1779071212.705876
SKILL_CONTROLS = [{'type': 'enum', 'name': 'kind', 'label': 'Attractor Type', 'options': [{'value': 'de_jong', 'label': 'de Jong'}, {'value': 'clifford', 'label': 'Clifford'}], 'default': 'de_jong'}, {'type': 'slider', 'name': 'density_boost', 'label': 'Density', 'min': 0.5, 'max': 3.0, 'step': 0.1, 'default': 1.0}, {'type': 'palette', 'name': 'palette', 'label': 'Palette'}]

# Attractor Cloud (creation)
# Strange-attractor point cloud (de Jong or Clifford) accumulated into a palette-graded density image.
# Each pass adds organic, smoke-like structure that fills the canvas with mathematical residue.
# No literal subject -- pure algorithmic form.

import math
import numpy as np
from PIL import Image

def run(canvas, kind="de_jong", density_boost=1.0, **_):
    s = int(canvas.size)
    seed = int(canvas.seed)
    n_points = int(220_000 * density_boost)

    if kind == "de_jong":
        # Known stable de Jong attractor presets
        presets = [
            (1.7, 1.8, 1.9, 0.4),
            (1.5, 2.8, 2.0, 0.8),
            (2.0, 2.0, 1.5, 0.4),
            (-1.2, -2.1, -1.2, 2.0),
            (-1.7, -2.1, -1.8, -1.9),
        ]
        a, b, c, d = presets[seed % len(presets)]
        x, y = 0.1, 0.1
    else:
        presets = [
            (-1.2, -1.1, -1.0, 0.7),
            (-2.0, -2.0, -1.0, 0.8),
            (-2.5, -2.5, 0.5, 0.9),
            (-1.7, -1.8, -1.9, 0.5),
        ]
        a, b, c, d = presets[seed % len(presets)]
        x, y = 0.1, 0.1

    pts = []
    for _ in range(n_points):
        if kind == "de_jong":
            xn = math.sin(a * y) - math.cos(b * x)
            yn = math.sin(c * x) - math.cos(d * y)
        else:
            xn = math.sin(a * y) + c * math.cos(a * x)
            yn = math.sin(b * x) + d * math.cos(b * y)
        x, y = xn, yn
        pts.append((x, y))

    pts = np.array(pts, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # Robust normalization: use actual percentile bounds to handle outliers
    margin = s * 0.06
    span = s - 2 * margin

    px_lo, px_hi = float(np.percentile(xs, 2)), float(np.percentile(xs, 98))
    py_lo, py_hi = float(np.percentile(ys, 2)), float(np.percentile(ys, 98))
    px_spread = px_hi - px_lo or 1.0
    py_spread = py_hi - py_lo or 1.0

    cx = (xs - px_lo) / px_spread * span + margin
    cy = (ys - py_lo) / py_spread * span + margin
    ix = np.clip(cx.astype(np.int32), 0, s - 1)
    iy = np.clip(cy.astype(np.int32), 0, s - 1)

    density = np.zeros((s, s), dtype=np.float32)
    np.add.at(density, (iy, ix), 1.0)
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
