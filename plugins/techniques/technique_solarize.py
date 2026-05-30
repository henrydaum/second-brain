from plugins.BaseTechnique import BaseTechnique, Slider

from PIL import ImageOps

try:
    art_kit
except NameError:
    art_kit = None


class SolarizeTechnique(BaseTechnique):
    name = 'Solarize'
    description = 'Invert all pixel values above a threshold — the classic darkroom solarization look. Bright regions flip to dark, midtones get weird.'
    kind = "filter"

    threshold = Slider(0, 255, default=128, step=1)

    def run(self, canvas):
        out = ImageOps.solarize(canvas.image.convert("RGB"), threshold=int(self.threshold))
        canvas.commit(out.convert("RGBA"))
