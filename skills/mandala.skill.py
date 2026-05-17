SKILL_NAME = "Mandala"
SKILL_DESCRIPTION = "N-fold rotationally symmetric ornament built from concentric rings of stamped shapes. Reliable choice for mandalas, snowflakes, kaleidoscopes, decorative seals. Params: symmetry (3-16, default 8), layers (2-10, default 6)."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

import math, random
from PIL import Image, ImageDraw


def run(canvas, symmetry=8, layers=6):
    rng = random.Random(canvas.seed)
    s = canvas.size
    sym = int(art_kit.clamp(symmetry, 3, 16))
    n_layers = int(art_kit.clamp(layers, 2, 12))
    img = canvas.create_image()
    cx, cy = s / 2, s / 2

    # Build a single radial spoke; rotate-composite to make it symmetric.
    spoke = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spoke, "RGBA")
    max_r = s * 0.46
    for li in range(n_layers):
        r_inner = (li / n_layers) * max_r + s * 0.02
        r_outer = ((li + 1) / n_layers) * max_r + s * 0.02
        r_mid = (r_inner + r_outer) / 2
        thickness = (r_outer - r_inner) * 0.62
        color = art_kit.palette_color(0.15 + (li / max(1, n_layers - 1)) * 0.8)
        x = cx + r_mid
        y = cy
        choice = (li + rng.randint(0, 1)) % 3
        if choice == 0:
            sd.ellipse((x - thickness / 2, y - thickness / 2, x + thickness / 2, y + thickness / 2), fill=color)
        elif choice == 1:
            sd.polygon([
                (x, y - thickness / 2),
                (x + thickness / 2, y),
                (x, y + thickness / 2),
                (x - thickness / 2, y),
            ], fill=color)
        else:
            sd.regular_polygon((x, y, thickness / 2), n_sides=6, rotation=0, fill=color)

    composite = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    for k in range(sym):
        rotated = spoke.rotate(360.0 * k / sym, resample=Image.BICUBIC, center=(cx, cy))
        composite.alpha_composite(rotated)
    img.alpha_composite(composite)

    # Centerpiece.
    draw = ImageDraw.Draw(img, "RGBA")
    r0 = s * 0.045
    draw.ellipse((cx - r0, cy - r0, cx + r0, cy + r0), fill=art_kit.palette_color(0.96))

    canvas.commit(img)
