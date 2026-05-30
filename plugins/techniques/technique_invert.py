from plugins.BaseTechnique import BaseTechnique, Slider

from PIL import Image, ImageOps

try:
    art_kit
except NameError:
    art_kit = None


class InvertTechnique(BaseTechnique):
    name = 'Invert'
    description = 'Photographic negative — invert RGB channels. Blend amount controls how far toward the negative we go (full=true negative, half=ghostly mid-tone).'
    kind = "filter"

    amount = Slider(0.0, 1.0, default=1.0)

    def run(self, canvas):
        img = canvas.image.convert("RGB")
        inv = ImageOps.invert(img)
        out = Image.blend(img, inv, self.amount)
        canvas.commit(out.convert("RGBA"))
