from plugins.BaseTechnique import BaseTechnique, Slider

from PIL import ImageEnhance

try:
    art_kit
except NameError:
    art_kit = None


class LevelsCurveTechnique(BaseTechnique):
    name = 'Levels Curve'
    description = 'Tune contrast, brightness, and saturation in one pass. Good for rescuing a flat or muted output.'
    kind = "filter"

    contrast   = Slider(0.2, 3.0, default=1.10, step=0.05)
    brightness = Slider(0.2, 3.0, default=1.0, step=0.05)
    saturation = Slider(0.0, 3.0, default=1.18, step=0.05)

    def run(self, canvas):
        img = canvas.image.convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(float(self.contrast))
        img = ImageEnhance.Brightness(img).enhance(float(self.brightness))
        img = ImageEnhance.Color(img).enhance(float(self.saturation))
        canvas.commit(img.convert("RGBA"))
