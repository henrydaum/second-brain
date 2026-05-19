SKILL_NAME = "Depth Fog"
SKILL_DESCRIPTION = "Blend the current canvas toward the palette background by a depth mask: linear top-fade (skies blend into the background atmosphere), bottom-fade (foregrounds dissolve), or radial edge-fade (a circular subject in center stays sharp, edges blur into background). Pseudo-atmospheric perspective from §11 of the encyclopedia. Strength slider, gentle by default."
SKILL_KIND = "transform"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1779667200.0
SKILL_HIDDEN = False
SKILL_CONTROLS = [
    {"type": "enum", "name": "direction", "label": "Direction",
     "options": [
         {"value": "top",    "label": "Top Fade"},
         {"value": "bottom", "label": "Bottom Fade"},
         {"value": "radial", "label": "Radial Edge"},
     ],
     "default": "top"},
    {"type": "slider", "name": "strength", "label": "Strength",
     "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.55},
]

import numpy as np
from PIL import Image


def run(canvas, direction="top", strength=0.55, **_):
    img = canvas.image.convert("RGB")
    s = img.size[0]
    strength = float(art_kit.clamp(strength, 0.0, 1.0))
    direction = str(direction)

    arr = np.asarray(img, dtype=np.float32)
    bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32)

    ys, xs = np.mgrid[0:s, 0:s].astype(np.float32)
    if direction == "top":
        mask = 1.0 - (ys / max(s - 1, 1))
    elif direction == "bottom":
        mask = ys / max(s - 1, 1)
    else:  # radial -- 0 at center, 1 at corners
        cx = s / 2.0
        cy = s / 2.0
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        d_max = np.sqrt(2.0) * cx
        mask = np.clip(d / d_max, 0.0, 1.0)

    # Smoothstep so the fade isn't a hard linear ramp.
    mask = mask * mask * (3.0 - 2.0 * mask)
    mask = mask * strength
    mask = mask[..., None]
    out = arr * (1.0 - mask) + bg[None, None, :] * mask
    out = np.clip(out, 0, 255).astype(np.uint8)
    canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
