from plugins.BaseSkill import BaseSkill

from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class PixelateSkill(BaseSkill):
    name = 'Pixelate'
    description = 'Block-mean downsample then nearest-neighbour upsample. Classic chunky-pixel look. Bigger block_size = bigger pixels. Param: block_size (2-80, default 12).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'block_size', 'label': 'Block Size', 'min': 2, 'max': 80, 'step': 1, 'default': 12},
    ]

    def run(self, canvas, block_size=12):
        b = int(art_kit.clamp(block_size, 2, 200))
        s = canvas.size
        small_w = max(1, s // b)
        img = canvas.image.convert("RGBA")
        tiny = img.resize((small_w, small_w), Image.BILINEAR)
        out = tiny.resize((s, s), Image.NEAREST)
        canvas.commit(out)
