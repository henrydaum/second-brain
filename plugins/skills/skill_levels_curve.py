from plugins.BaseSkill import BaseSkill

from PIL import ImageEnhance

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class LevelsCurveSkill(BaseSkill):
    name = 'Levels Curve'
    description = 'Tune contrast, brightness, and saturation in one pass. Good for rescuing a flat or muted output. Params: contrast (0.3-2.5, default 1.10), brightness (0.3-2.5, default 1.0), saturation (0.0-2.5, default 1.18).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'contrast', 'label': 'Contrast', 'min': 0.2, 'max': 3.0, 'step': 0.05, 'default': 1.10},
        {'type': 'slider', 'name': 'brightness', 'label': 'Brightness', 'min': 0.2, 'max': 3.0, 'step': 0.05, 'default': 1.0},
        {'type': 'slider', 'name': 'saturation', 'label': 'Saturation', 'min': 0.0, 'max': 3.0, 'step': 0.05, 'default': 1.18},
    ]

    def run(self, canvas, contrast=1.10, brightness=1.0, saturation=1.18):
        img = canvas.image.convert("RGB")
        contrast = float(art_kit.clamp(contrast, 0.2, 3.0))
        brightness = float(art_kit.clamp(brightness, 0.2, 3.0))
        saturation = float(art_kit.clamp(saturation, 0.0, 3.0))
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Color(img).enhance(saturation)
        canvas.commit(img.convert("RGBA"))
