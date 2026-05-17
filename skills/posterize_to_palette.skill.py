SKILL_NAME = "Posterize To Palette"
SKILL_DESCRIPTION = "Quantize the image to N palette anchors via nearest-color in RGB. Produces a flat, screen-printed look strongly tied to the palette. Params: anchors (3-12, default 6)."
SKILL_KIND = "transform"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

import numpy as np
from PIL import Image


def run(canvas, anchors=6):
    img = canvas.image.convert("RGB")
    n = int(art_kit.clamp(anchors, 3, 14))
    arr = np.asarray(img).astype(np.float32) / 255.0
    pal = np.array(
        [art_kit.hex_to_rgb(art_kit.palette_color(i / max(1, n - 1))) for i in range(n)],
        dtype=np.float32,
    ) / 255.0
    # Distance from each pixel to each anchor in RGB.
    d = np.sum((arr[..., None, :] - pal[None, None, :, :]) ** 2, axis=-1)
    idx = np.argmin(d, axis=-1)
    out = (pal[idx] * 255.0).astype(np.uint8)
    canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
