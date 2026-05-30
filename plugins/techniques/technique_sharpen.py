from plugins.BaseTechnique import BaseTechnique, Slider

from PIL import ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class SharpenTechnique(BaseTechnique):
    name = 'Sharpen'
    description = 'Crisp up edge detail with an unsharp mask. Good final-pass after a background technique or any softening filter.'
    kind = "filter"

    radius    = Slider(0.2, 6.0, default=1.5, step=0.1)
    percent   = Slider(0, 400, default=140, step=5)
    threshold = Slider(0, 20, default=2, step=1)

    def run(self, canvas):
        out = canvas.image.filter(ImageFilter.UnsharpMask(
            radius=float(self.radius),
            percent=int(self.percent),
            threshold=int(self.threshold),
        ))
        canvas.commit(out.convert("RGBA"))
