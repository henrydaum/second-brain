SKILL_NAME = "Tilt Shift"
SKILL_DESCRIPTION = "Blur the top and bottom of the image while keeping a horizontal focus band sharp — the 'miniature' look. Params: focus_y (0.1-0.9, default 0.55), focus_band (0.05-0.6, default 0.22), max_blur (1-30, default 12)."
SKILL_KIND = "transform"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

import numpy as np
from PIL import Image, ImageFilter


def run(canvas, focus_y=0.55, focus_band=0.22, max_blur=12):
    img = canvas.image.convert("RGB")
    s = canvas.size
    fy = float(art_kit.clamp(focus_y, 0.05, 0.95))
    fb = float(art_kit.clamp(focus_band, 0.05, 0.6))
    mb = float(art_kit.clamp(max_blur, 1, 40))

    blurred = img.filter(ImageFilter.GaussianBlur(mb))
    yy = np.arange(s).astype(np.float32) / max(1, s - 1)
    d = np.abs(yy - fy)
    edge0 = fb / 2.0
    edge1 = edge0 + 0.15
    t = np.clip((d - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
    smooth = (t * t * (3.0 - 2.0 * t))
    mask_strip = (smooth * 255.0).astype(np.uint8).reshape(s, 1)
    mask = Image.fromarray(mask_strip, "L").resize((s, s), Image.NEAREST)
    out = Image.composite(blurred, img, mask)
    canvas.commit(out.convert("RGBA"))
