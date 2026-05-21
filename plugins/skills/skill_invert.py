from plugins.BaseSkill import BaseSkill

from PIL import Image, ImageOps

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class InvertSkill(BaseSkill):
    name = 'Invert'
    description = 'Photographic negative — invert RGB channels. Blend amount controls how far toward the negative we go (full=true negative, half=ghostly mid-tone). Param: amount (0.0-1.0, default 1.0).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'amount', 'label': 'Amount', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'default': 1.0},
    ]

    def run(self, canvas, amount=1.0):
        amt = float(art_kit.clamp(amount, 0.0, 1.0))
        img = canvas.image.convert("RGB")
        inv = ImageOps.invert(img)
        out = Image.blend(img, inv, amt)
        canvas.commit(out.convert("RGBA"))
