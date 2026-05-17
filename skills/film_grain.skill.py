SKILL_NAME = "Film Grain"
SKILL_DESCRIPTION = "Deterministic per-pixel noise overlay seeded from canvas.seed. Adds tactile texture; great over flat palette grades. Params: intensity (0.0-0.3, default 0.07), monochrome (bool, default True)."
SKILL_KIND = "transform"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

import numpy as np
from PIL import Image


def run(canvas, intensity=0.07, monochrome=True):
    img = canvas.image.convert("RGB")
    s = canvas.size
    intensity = float(art_kit.clamp(intensity, 0.0, 0.4))
    rng = np.random.default_rng(canvas.seed)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if bool(monochrome):
        noise = rng.standard_normal((s, s, 1)).astype(np.float32) * intensity
    else:
        noise = rng.standard_normal((s, s, 3)).astype(np.float32) * intensity
    out = np.clip(arr + noise, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
