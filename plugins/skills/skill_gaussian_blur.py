from plugins.BaseSkill import BaseSkill

from PIL import ImageFilter

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class GaussianBlurSkill(BaseSkill):
    name = 'Gaussian Blur'
    description = 'Standard Gaussian blur. The workhorse softening pass — use before sharpen for a "dreamy" look or to smooth high-frequency noise. Params: radius (0.5-40, default 3.0).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'radius', 'label': 'Radius', 'min': 0.5, 'max': 40.0, 'step': 0.5, 'default': 3.0},
    ]

    def run(self, canvas, radius=3.0):
        radius = float(art_kit.clamp(radius, 0.0, 60.0))
        out = canvas.image.filter(ImageFilter.GaussianBlur(radius))
        canvas.commit(out.convert("RGBA"))
