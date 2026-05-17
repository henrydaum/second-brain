SKILL_NAME = "Wave Strata"
SKILL_DESCRIPTION = "Stacked horizontal sediment bands with sine-modulated edges, palette-graded top to bottom. Reliable choice for landscapes, sunsets, geological strata, layered abstracts. Params: bands (4-30, default 14), amplitude (0.0-0.12, default 0.035), frequency (0.5-6.0, default 2.4)."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

import math, random
from PIL import ImageDraw


def run(canvas, bands=14, amplitude=0.035, frequency=2.4):
    rng = random.Random(canvas.seed)
    s = canvas.size
    img = canvas.create_image()
    draw = ImageDraw.Draw(img, "RGBA")
    n = int(art_kit.clamp(bands, 3, 40))
    amp = float(art_kit.clamp(amplitude, 0.0, 0.2)) * s
    freq = float(art_kit.clamp(frequency, 0.3, 8.0))
    band_h = s / n
    step = max(2, s // 240)
    phases = [rng.uniform(0, math.tau) for _ in range(n + 1)]
    # Pre-compute edge curves (n+1 of them) so adjacent bands share an edge.
    edges = []
    for ei in range(n + 1):
        baseline = ei * band_h
        curve = []
        for x in range(0, s + step, step):
            y = baseline + math.sin((x / s) * math.tau * freq + phases[ei]) * amp
            curve.append((x, y))
        edges.append(curve)
    for i in range(n):
        t = i / max(1, n - 1)
        color = art_kit.palette_color(t)
        top = edges[i]
        bot = edges[i + 1]
        poly = top + list(reversed(bot))
        draw.polygon(poly, fill=color)
    canvas.commit(img)
