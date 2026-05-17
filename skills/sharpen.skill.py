SKILL_NAME = "Sharpen"
SKILL_DESCRIPTION = "Crisp up edge detail with an unsharp mask. Good final-pass after a creation skill or any softening transform. Params: radius (1.0-3.0, default 1.5), percent (50-200, default 140), threshold (0-10, default 2)."
SKILL_KIND = "transform"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

from PIL import ImageFilter


def run(canvas, radius=1.5, percent=140, threshold=2):
    radius = float(art_kit.clamp(radius, 0.2, 6.0))
    percent = int(art_kit.clamp(percent, 0, 400))
    threshold = int(art_kit.clamp(threshold, 0, 20))
    sharpened = canvas.image.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )
    canvas.commit(sharpened)
