SKILL_NAME = "Dawn Sun"
SKILL_DESCRIPTION = "A sun anchored on a rule-of-thirds intersection over a palette-graded sky, with a glow halo and optional radial rays. Reliable choice for sunrises, sunsets, lone-sun landscapes. Params: rays (0-48, default 22), sun_size (0.05-0.25, default 0.13), low_horizon (bool, default True)."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

import math, random
from PIL import Image, ImageDraw, ImageFilter


def run(canvas, rays=22, sun_size=0.13, low_horizon=True):
    rng = random.Random(canvas.seed)
    s = canvas.size
    img = Image.new("RGBA", (s, s), canvas.palette.background)

    # Sky gradient: palette ramp top->bottom via a 1-pixel column resized to full width.
    column = Image.new("RGBA", (1, s))
    cdraw = ImageDraw.Draw(column)
    for y in range(s):
        t = y / max(1, s - 1)
        cdraw.point((0, y), fill=art_kit.palette_color(t))
    img.paste(column.resize((s, s), Image.BILINEAR))

    # Pick a rule-of-thirds anchor for the sun.
    thirds = art_kit.rule_of_thirds(s)
    horizon_y = thirds.horizons[1] if bool(low_horizon) else thirds.horizons[0]
    cx = thirds.verticals[1] if rng.random() > 0.4 else thirds.verticals[0]
    cy = horizon_y - int(s * 0.06)
    radius = int(s * art_kit.clamp(float(sun_size), 0.03, 0.3))

    # Halo (large blurred bright disc).
    halo = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo, "RGBA")
    hr, hg, hb = art_kit.hex_to_rgb(art_kit.palette_color(0.92))
    hd.ellipse((cx - radius * 3, cy - radius * 3, cx + radius * 3, cy + radius * 3), fill=(hr, hg, hb, 110))
    halo = halo.filter(ImageFilter.GaussianBlur(radius * 1.4))
    img.alpha_composite(halo)

    # Radial rays.
    n_rays = int(art_kit.clamp(rays, 0, 60))
    if n_rays > 0:
        ray_layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        rd = ImageDraw.Draw(ray_layer, "RGBA")
        ar, ag, ab = art_kit.hex_to_rgb(art_kit.palette_color(0.98))
        for i in range(n_rays):
            theta = (i / n_rays) * math.tau + rng.uniform(-0.03, 0.03)
            x2 = cx + math.cos(theta) * s
            y2 = cy + math.sin(theta) * s
            rd.line((cx, cy, x2, y2), fill=(ar, ag, ab, 44), width=max(2, int(s * 0.0035)))
        ray_layer = ray_layer.filter(ImageFilter.GaussianBlur(max(1.0, radius * 0.25)))
        img.alpha_composite(ray_layer)

    # Sun disc.
    draw = ImageDraw.Draw(img, "RGBA")
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=art_kit.palette_color(0.98))

    canvas.commit(img)
