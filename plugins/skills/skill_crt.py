from plugins.BaseSkill import BaseSkill

import numpy as np

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class CrtSkill(BaseSkill):
    name = 'Crt'
    description = 'Combo CRT-monitor effect: gentle barrel distortion + chromatic aberration + scanlines + vignette toward palette.background. One-click "old TV" look. Params: amount (0.0-1.5, default 0.7) — global dial; scanline_intensity (0.0-1.0, default 0.4).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'palette', 'name': 'palette', 'label': 'Palette'},
        {'type': 'slider', 'name': 'amount', 'label': 'Amount', 'min': 0.0, 'max': 1.5, 'step': 0.05, 'default': 0.7},
        {'type': 'slider', 'name': 'scanline_intensity', 'label': 'Scanlines', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'default': 0.4},
    ]

    def run(self, canvas, amount=0.7, scanline_intensity=0.4):
        amt = float(art_kit.clamp(amount, 0.0, 2.0))
        sl = float(art_kit.clamp(scanline_intensity, 0.0, 1.0))
        arr = canvas.image_array(mode="RGB", dtype="float")
        s = canvas.size
        xx, yy, nx, ny = art_kit.centered_grid(s)
        cx = (s - 1) / 2.0
        r2 = nx * nx + ny * ny
        # Barrel distortion sampling map.
        scale = 1.0 + 0.18 * amt * r2
        sx = cx + nx * scale * cx
        sy = cx + ny * scale * cx
        # Chromatic aberration: per-channel slightly different sample positions.
        ca = 4.0 * amt
        length = np.sqrt(r2) + 1e-6
        ux = nx / length
        uy = ny / length
        r_plane = art_kit.bilinear_sample(arr[..., 0], sx + ux * ca, sy + uy * ca)
        g_plane = art_kit.bilinear_sample(arr[..., 1], sx, sy)
        b_plane = art_kit.bilinear_sample(arr[..., 2], sx - ux * ca, sy - uy * ca)
        out = np.stack([r_plane, g_plane, b_plane], axis=-1)

        # Scanlines.
        rows = np.arange(s)
        scan = (rows % 2 == 0).astype(np.float32) * sl
        out = out * (1.0 - scan[:, None, None])

        # Vignette toward palette.background.
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        vign = np.clip(np.sqrt(r2) * 0.85, 0.0, 1.0)
        vign = vign * vign * (3.0 - 2.0 * vign) * (0.5 * amt)
        v = vign[..., None]
        out = out * (1.0 - v) + bg[None, None, :] * v

        # Mask off the area outside the curved screen so corners are background.
        outside = (r2 > 1.05).astype(np.float32)[..., None]
        out = out * (1.0 - outside) + bg[None, None, :] * outside

        canvas.commit_array(out)
