from plugins.BaseSkill import BaseSkill

from PIL import ImageOps

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class SolarizeSkill(BaseSkill):
    name = 'Solarize'
    description = 'Invert all pixel values above a threshold — the classic darkroom solarization look. Bright regions flip to dark, midtones get weird. Param: threshold (0-255, default 128).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'threshold', 'label': 'Threshold', 'min': 0, 'max': 255, 'step': 1, 'default': 128},
    ]

    def run(self, canvas, threshold=128):
        t = int(art_kit.clamp(threshold, 0, 255))
        img = canvas.image.convert("RGB")
        out = ImageOps.solarize(img, threshold=t)
        canvas.commit(out.convert("RGBA"))
