SKILL_NAME = "Fractal Tree"
SKILL_DESCRIPTION = "Recursive branching L-system tree, palette ramp from trunk to tips. Reliable choice for trees, coral, lightning, river deltas, capillary networks. Params: depth (4-11, default 9), branch_ratio (0.55-0.85, default 0.72), spread (0.3-1.0, default 0.55)."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

import math, random
from PIL import ImageDraw


def run(canvas, depth=9, branch_ratio=0.72, spread=0.55):
    rng = random.Random(canvas.seed)
    s = canvas.size
    img = canvas.create_image()
    draw = ImageDraw.Draw(img, "RGBA")
    d = int(art_kit.clamp(depth, 3, 11))
    ratio = float(art_kit.clamp(branch_ratio, 0.5, 0.9))
    sp = float(art_kit.clamp(spread, 0.2, 1.2))

    def branch(x, y, angle, length, level):
        if level <= 0 or length < 1.2:
            return
        x2 = x + math.cos(angle) * length
        y2 = y + math.sin(angle) * length
        t = 1.0 - level / d
        cr, cg, cb = art_kit.hex_to_rgb(art_kit.palette_color(0.1 + t * 0.85))
        width = max(1, int(length * 0.11))
        draw.line((x, y, x2, y2), fill=(cr, cg, cb, 235), width=width)
        a1 = angle - sp * 0.55 + rng.uniform(-0.1, 0.1)
        a2 = angle + sp * 0.55 + rng.uniform(-0.1, 0.1)
        branch(x2, y2, a1, length * ratio, level - 1)
        branch(x2, y2, a2, length * ratio, level - 1)

    branch(s / 2, s * 0.96, -math.pi / 2, s * 0.27, d)
    canvas.commit(img)
