from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class ScanlinesTechnique(BaseTechnique):
    name = 'Scanlines'
    description = 'CRT-style horizontal scanlines: darken every Nth row toward palette.background. Tactile retro overlay.'
    kind = "filter"

    palette   = Palette()
    intensity = Slider(0.0, 1.0, default=0.45, step=0.05)
    thickness = Slider(1, 6, default=1, step=1)

    def run(self, canvas):
        amt = float(self.intensity)
        thk = int(self.thickness)
        arr = canvas.image_array(mode="RGB", dtype="float")
        s = arr.shape[0]
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        rows = np.arange(s)
        mask = ((rows // thk) % 2 == 0).astype(np.float32) * amt
        m = mask[:, None, None]
        canvas.commit_array(arr * (1.0 - m) + bg[None, None, :] * m)
