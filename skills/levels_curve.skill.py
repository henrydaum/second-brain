SKILL_NAME = "Levels Curve"
SKILL_DESCRIPTION = "Tune contrast, brightness, and saturation in one pass. Good for rescuing a flat or muted output. Params: contrast (0.3-2.5, default 1.10), brightness (0.3-2.5, default 1.0), saturation (0.0-2.5, default 1.18)."
SKILL_KIND = "transform"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

from PIL import ImageEnhance


def run(canvas, contrast=1.10, brightness=1.0, saturation=1.18):
    img = canvas.image.convert("RGB")
    contrast = float(art_kit.clamp(contrast, 0.2, 3.0))
    brightness = float(art_kit.clamp(brightness, 0.2, 3.0))
    saturation = float(art_kit.clamp(saturation, 0.0, 3.0))
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Color(img).enhance(saturation)
    canvas.commit(img.convert("RGBA"))
