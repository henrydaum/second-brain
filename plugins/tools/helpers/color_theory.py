"""Seeded color-theory polish for generative canvas images."""

from __future__ import annotations

import colorsys, math, random

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps, ImageStat

try:
    import numpy as np
except Exception:
    np = None

FAMILIES = {
    "plasma": (.78, "analogous"), "aurora": (.46, "analogous"),
    "laser": (.55, "split"), "electric": (.55, "split"),
    "sunset": .04, "gold": .12, "inferno": .06,
    "ice": (.55, "duotone"), "toxic": (.29, "split"),
    "royal": (.70, "accent"), "nebula": (.72, "accent"),
}


def oklch_to_rgb(l, c, h):
    a, b = c * math.cos(h * math.tau), c * math.sin(h * math.tau)
    L, M, S = l + .3963377774*a + .2158037573*b, l - .1055613458*a - .0638541728*b, l - .0894841775*a - 1.291485548*b
    L, M, S = L**3, M**3, S**3
    r, g, bl = 4.0767416621*L - 3.3077115913*M + .2309699292*S, -1.2684380046*L + 2.6097574011*M - .3413193965*S, -.0041960863*L - .7034186147*M + 1.707614701*S
    enc = lambda x: 12.92*x if x <= .0031308 else 1.055*(max(0, x)**(1/2.4)) - .055
    return tuple(round(max(0, min(1, enc(x))) * 255) for x in (r, g, bl))


def harmony(name="plasma", seed=1):
    r = random.Random(f"{name}:{seed}"); spec = FAMILIES.get(name, (.78, "analogous"))
    base, mode = spec if isinstance(spec, tuple) else (spec, "warm")
    base = (base + r.uniform(-.025, .025)) % 1
    offs = {"analogous": (-.055, .02, .095), "split": (0, .42, .58), "duotone": (0, .50, .08), "accent": (0, .08, .38), "warm": (0, .07, .54)}[mode]
    dark = oklch_to_rgb(.13, .045, base + offs[0])
    mid = oklch_to_rgb(.55, .16, base + offs[1])
    hi = oklch_to_rgb(.82, .18, base + offs[2])
    accent = oklch_to_rgb(.74, .25, base + offs[-1] + .11)
    return [dark, mid, hi, accent]


def palette_color(t, name="plasma", seed=1, value=1.0):
    pal = harmony(name, seed); t = max(0, min(1, (t + random.Random(seed).random() * .07) % 1))
    i = min(len(pal) - 2, int(t * (len(pal) - 1))); f = t * (len(pal) - 1) - i
    return tuple(round((pal[i][j] * (1 - f) + pal[i + 1][j] * f) * value) for j in range(3))


def beautify_image(img, seed=1, palette="plasma", kind="image"):
    img = img.convert("RGB")
    before = visual_stats(img)
    if np is not None:
        arr = np.asarray(img).astype("float32") / 255
        lum = (arr[..., 0]*.2126 + arr[..., 1]*.7152 + arr[..., 2]*.0722)
        lo, hi = np.percentile(lum, (2, 99)); lum = np.clip((lum - lo) / ((hi - lo) or 1), 0, 1)
        pal = np.array(harmony(palette, seed), dtype="float32") / 255
        x = lum * (len(pal) - 1); i = np.clip(x.astype(int), 0, len(pal)-2); f = x - i
        mapped = pal[i] * (1 - f[..., None]) + pal[i + 1] * f[..., None]
        mix = .52 if kind in {"pixel_mosaic", "color_grade"} else .66
        arr = arr * (1 - mix) + mapped * mix
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"), "RGB")
    else:
        p = harmony(palette, seed)
        img = ImageChops.screen(img, ImageOps.colorize(ImageOps.grayscale(img), p[0], p[2]).convert("RGB"))
    b = before["brightness"]
    img = ImageEnhance.Contrast(img).enhance(1.18 if before["contrast"] < .28 else 1.06)
    img = ImageEnhance.Color(img).enhance(.92 if before["saturation"] > .72 else 1.18)
    if b < .16: img = ImageEnhance.Brightness(img).enhance(1.22)
    gray = ImageOps.grayscale(img)
    glow = ImageEnhance.Brightness(img.filter(ImageFilter.GaussianBlur(5))).enhance(.65)
    img = Image.composite(ImageChops.screen(img, glow), img, gray.point(lambda x: 90 if x > 175 else 0))
    if before["detail"] < .055: img = img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=115, threshold=3))
    for _ in range(3):
        after = visual_stats(img)
        changed = False
        if after["saturation"] > .76: img = ImageEnhance.Color(img).enhance(.72); changed = True
        if after["brightness"] < .18: img = ImageEnhance.Brightness(img).enhance(1.35); changed = True
        if after["contrast"] < .25: img = ImageEnhance.Contrast(img).enhance(1.16); changed = True
        if after["detail"] < .045: img = img.filter(ImageFilter.UnsharpMask(radius=1.8, percent=135, threshold=2)); changed = True
        if not changed: break
    return img


def visual_stats(img):
    img = img.convert("RGB"); img.thumbnail((180, 180))
    stat, edge = ImageStat.Stat(img), ImageStat.Stat(img.filter(ImageFilter.FIND_EDGES))
    means, stds = stat.mean, stat.stddev
    hsv = [colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in img.getdata()]
    sat = sum(s for _, s, _ in hsv) / len(hsv)
    hues = sorted(h for h, s, v in hsv if s > .18 and v > .08)
    spread = 0 if not hues else min(1, (hues[-1] - hues[0]) * 1.35)
    brightness, contrast, detail = sum(means) / 765, sum(stds) / 382.5, sum(edge.mean) / 765
    score = max(0, min(1, .28*min(1, brightness/.34) + .27*min(1, contrast/.42) + .2*min(1, detail/.12) + .17*(1 - abs(sat-.52)) + .08*min(spread, .9)/.9))
    guidance = []
    if brightness < .16: guidance.append("needs_light")
    if contrast < .25: guidance.append("needs_contrast")
    if detail < .045: guidance.append("low_detail")
    if sat < .22: guidance.append("muted_palette")
    if sat > .78: guidance.append("too_saturated")
    if not guidance: guidance.append("strong_palette")
    return {"brightness": round(brightness, 3), "contrast": round(contrast, 3), "detail": round(detail, 3), "dominant_rgb": [round(x) for x in means], "mostly_dark": sum(means)/3 < 20, "saturation": round(sat, 3), "hue_spread": round(spread, 3), "beauty_score": round(score, 3), "guidance": guidance}
