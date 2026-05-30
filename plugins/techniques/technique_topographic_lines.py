from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


class TopographicLinesTechnique(BaseTechnique):
    name = "Topographic Lines"
    description = "Contour-line topographic map on aged paper. An fbm height field is quantized into elevation bands; band boundaries are drawn as thin black lines, with every fifth line as a darker index contour."
    kind = "background"
    palette = Palette()
    octaves = Slider(2, 6, default=4, step=1)
    band_count = Slider(10, 60, default=28, step=1)
    line_weight = Slider(0.5, 2.5, default=1.0, step=0.1)
    paper = Enum([("aged_yellow", "Aged Yellow"), ("ivory", "Ivory"), ("palette_bg", "Palette BG")], default="aged_yellow")

    def run(self, canvas):
        s = canvas.size
        seed = canvas.seed

        paper = {
            "aged_yellow": (0.96, 0.92, 0.73),
            "ivory": (0.96, 0.94, 0.88),
            "palette_bg": None,
        }[str(self.paper)]

        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        freq = 0.004
        h = art_kit.fbm_grid(seed, xx * freq, yy * freq, octaves=int(self.octaves))
        h = (h - h.min()) / max(float(h.max() - h.min()), 1e-6)

        bands = int(self.band_count)
        quantized = np.floor(h * bands).astype(np.int32)
        # Detect band-boundary pixels via gradient on the quantized field.
        dx = np.zeros_like(quantized)
        dy = np.zeros_like(quantized)
        dx[:, 1:] = quantized[:, 1:] - quantized[:, :-1]
        dy[1:, :] = quantized[1:, :] - quantized[:-1, :]
        boundary = (dx != 0) | (dy != 0)
        # Index contours every 5th band.
        index = (quantized % 5 == 0) & boundary

        if paper is None:
            from PIL import ImageColor
            bg_rgb = np.array(ImageColor.getrgb(canvas.palette.background), dtype=np.float32) / 255.0
        else:
            bg_rgb = np.array(paper, dtype=np.float32)

        out = np.broadcast_to(bg_rgb, (s, s, 3)).copy()
        line = np.array([0.18, 0.13, 0.08], dtype=np.float32)
        # Line weight: dilate boundaries by widening with shifted ORs.
        lw = float(self.line_weight)
        if lw >= 1.5:
            b2 = boundary.copy()
            b2[:-1, :] |= boundary[1:, :]
            b2[:, :-1] |= boundary[:, 1:]
            boundary = b2
        out[boundary] = line
        if lw >= 1.0:
            idx2 = index.copy()
            if lw >= 1.5:
                idx2[:-1, :] |= index[1:, :]
                idx2[:, :-1] |= index[:, 1:]
            out[idx2] = line * 0.5

        canvas.commit_array(out)
