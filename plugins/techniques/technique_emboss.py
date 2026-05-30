from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import math
import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class EmbossTechnique(BaseTechnique):
    name = 'Emboss'
    description = 'Bas-relief emboss: lights the canvas from a chosen angle so brightness gradients become raised and recessed ridges, like an image pressed into metal or stone. The grey relief is remapped onto the palette luminance ramp so it stays on-palette instead of flat grey. Good for "emboss", "bas relief", "engraved metal", "carved", "embossed", or a directional ridge effect.'
    kind = "filter"

    palette = Palette()
    angle   = Slider(0, 360, default=135, step=5)
    depth   = Slider(0.5, 5.0, default=2.0, step=0.1)

    def run(self, canvas):
        s = int(canvas.size)
        a = math.radians(float(self.angle))
        depth = float(self.depth)
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)

        k = 3
        sx = int(round(math.cos(a) * k))
        sy = int(round(math.sin(a) * k))
        shifted = np.roll(np.roll(lum, sy, axis=0), sx, axis=1)
        grad = lum - shifted
        # Centre on a metallic mid-tone: strongly amplified directional ridges
        # plus a touch of the original tone so flat areas don't collapse to one
        # solid colour.
        relief = np.clip(0.5 + (lum - 0.5) * 0.35 + grad * depth * 10.0, 0.0, 1.0)

        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(k2 / 255.0)) for k2 in range(256)],
            dtype=np.uint8,
        )
        idx = np.clip((relief * 255).astype(np.int32), 0, 255)
        canvas.commit_array(lut[idx])
