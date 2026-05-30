from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class CrtTechnique(BaseTechnique):
    name = 'Crt'
    description = 'Combo CRT-monitor effect: gentle barrel distortion + chromatic aberration + scanlines + vignette toward palette.background. One-click "old TV" look.'
    kind = "filter"

    palette            = Palette()
    amount             = Slider(0.0, 1.5, default=0.7, step=0.05)
    scanline_intensity = Slider(0.0, 1.0, default=0.4, step=0.05)

    def run(self, canvas):
        amt = float(self.amount)
        sl = float(self.scanline_intensity)
        arr = canvas.image_array(mode="RGB", dtype="float") * 255.0
        s = canvas.size
        xx, yy, nx, ny = art_kit.centered_grid(s)
        cx = (s - 1) / 2.0
        r2 = nx * nx + ny * ny
        scale = 1.0 + 0.18 * amt * r2
        sx = cx + nx * scale * cx
        sy = cx + ny * scale * cx
        ca = 4.0 * amt
        length = np.sqrt(r2) + 1e-6
        ux = nx / length
        uy = ny / length
        r_plane = art_kit.bilinear_sample(arr[..., 0], sx + ux * ca, sy + uy * ca)
        g_plane = art_kit.bilinear_sample(arr[..., 1], sx, sy)
        b_plane = art_kit.bilinear_sample(arr[..., 2], sx - ux * ca, sy - uy * ca)
        out = np.stack([r_plane, g_plane, b_plane], axis=-1) / 255.0

        rows = np.arange(s)
        scan = (rows % 2 == 0).astype(np.float32) * sl
        out = out * (1.0 - scan[:, None, None])

        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        vign = np.clip(np.sqrt(r2) * 0.85, 0.0, 1.0)
        vign = vign * vign * (3.0 - 2.0 * vign) * (0.5 * amt)
        v = vign[..., None]
        out = out * (1.0 - v) + bg[None, None, :] * v

        outside = (r2 > 1.05).astype(np.float32)[..., None]
        out = out * (1.0 - outside) + bg[None, None, :] * outside
        canvas.commit_array(out)
