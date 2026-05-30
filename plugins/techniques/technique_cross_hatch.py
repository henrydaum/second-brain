from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import math
import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class CrossHatchTechnique(BaseTechnique):
    name = 'Cross-Hatch Engraving'
    description = 'Render the canvas as a pen-and-ink engraving: tones are built from layers of parallel hatch lines, with darker areas accumulating more crossing line sets at new angles. The ink is a bright palette tone laid over the palette background, like an etched plate or banknote portrait. Good for "cross-hatch", "engraving", "etching", "pen and ink", "line shading", or a hatched drawing.'
    kind = "filter"

    palette = Palette()
    spacing = Slider(3, 16, default=6, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        sp = float(self.spacing)
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        darkness = np.clip(1.0 - lum, 0.0, 1.0)

        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        lw = max(1.0, sp * 0.32)
        ink = np.zeros((s, s), dtype=np.float32)

        # Progressively add hatch layers; each kicks in for darker tones.
        passes = [(15.0, 0.20), (75.0, 0.40), (135.0, 0.60), (45.0, 0.80)]
        for ang_deg, thresh in passes:
            a = math.radians(ang_deg)
            proj = xx * math.cos(a) + yy * math.sin(a)
            line = (np.mod(proj, sp) < lw).astype(np.float32)
            ink = np.maximum(ink, line * (darkness > thresh).astype(np.float32))

        # Soften the line edges a touch toward the tone they represent.
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        ink_col = np.array(art_kit.hex_to_rgb(art_kit.palette_color(0.9)), dtype=np.float32) / 255.0
        a = ink[..., None]
        out = bg[None, None, :] * (1.0 - a) + ink_col[None, None, :] * a
        canvas.commit_array(out)
