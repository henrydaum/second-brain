from plugins.BaseTechnique import BaseTechnique, Pan, Slider

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class LiquidBulgeTechnique(BaseTechnique):
    name = "Liquid Bulge"
    description = "Filter: local pinch or bulge centered on a pan-driven point with smooth radial falloff — distinct from full-frame fisheye."
    kind = "filter"
    cx = Slider(0, 1, default=0.5, step=0.02)
    cy = Slider(0, 1, default=0.5, step=0.02)
    center = Pan(x='cx', y='cy')
    strength = Slider(-1.0, 1.0, default=0.55, step=0.05)
    radius = Slider(0.1, 0.9, default=0.45, step=0.02)

    def run(self, canvas):
        cx_n = float(self.cx)
        cy_n = float(self.cy)
        strength = float(self.strength)
        radius_n = float(self.radius)

        arr = canvas.image_array(mode="RGB", dtype="float")
        H, W = arr.shape[:2]
        cx_px = cx_n * W
        cy_px = cy_n * H
        radius_px = max(radius_n * min(W, H), 1e-6)

        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        dx = xs - cx_px
        dy = ys - cy_px
        r_norm = np.clip(np.sqrt(dx * dx + dy * dy) / radius_px, 0.0, 1.0)

        s = 1.0 - r_norm
        influence = s * s * (3.0 - 2.0 * s)
        factor = 1.0 - strength * influence

        src_x = cx_px + dx * factor
        src_y = cy_px + dy * factor

        out = art_kit.bilinear_sample(arr, src_x, src_y)
        canvas.commit_array(out)
