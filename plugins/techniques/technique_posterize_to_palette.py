from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class PosterizeToPaletteTechnique(BaseTechnique):
    name = 'Posterize To Palette'
    description = 'Quantize the image to N palette anchors via nearest-color in RGB. Produces a flat, screen-printed look strongly tied to the palette.'
    kind = "filter"

    palette = Palette()
    anchors = Slider(3, 14, default=6, step=1)

    def run(self, canvas):
        n = int(self.anchors)
        arr = canvas.image_array(mode="RGB", dtype="float")
        pal = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(i / max(1, n - 1))) for i in range(n)],
            dtype=np.float32,
        ) / 255.0
        d = np.sum((arr[..., None, :] - pal[None, None, :, :]) ** 2, axis=-1)
        idx = np.argmin(d, axis=-1)
        canvas.commit_array(pal[idx])
