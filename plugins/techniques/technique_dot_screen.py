from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class DotScreenTechnique(BaseTechnique):
    name = 'Dot Screen'
    description = 'Overlay a regular dot pattern at canvas resolution, with dot alpha controlled by local luminance — keeps original colors but adds a printed-screen texture.'
    kind = "filter"

    palette   = Palette()
    cell_size = Slider(4, 30, default=10, step=1)
    strength  = Slider(0.0, 1.0, default=0.5, step=0.05)

    def run(self, canvas):
        c = int(self.cell_size)
        amt = float(self.strength)
        s = canvas.size
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        gx = xx % c - (c / 2.0)
        gy = yy % c - (c / 2.0)
        d = np.sqrt(gx * gx + gy * gy)
        radius = (1.0 - lum) * (c * 0.45)
        mask = np.clip((radius - d) / 1.5, 0.0, 1.0) * amt
        m = mask[..., None]
        dot_color = np.array(art_kit.hex_to_rgb(canvas.palette.accent), dtype=np.float32) / 255.0
        canvas.commit_array(arr * (1.0 - m) + dot_color[None, None, :] * m)
