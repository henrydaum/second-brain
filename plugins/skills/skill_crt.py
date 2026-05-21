from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


def _sample_plane(plane, fx, fy):
    h, w = plane.shape
    fx = np.clip(fx, 0, w - 1)
    fy = np.clip(fy, 0, h - 1)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = fx - x0
    wy = fy - y0
    a = plane[y0, x0]
    b = plane[y0, x1]
    c = plane[y1, x0]
    d = plane[y1, x1]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy


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
        img = canvas.image.convert("RGB")
        s = canvas.size
        arr = np.asarray(img, dtype=np.float32)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        cx = cy = (s - 1) / 2.0
        nx = (xx - cx) / max(cx, 1.0)
        ny = (yy - cy) / max(cy, 1.0)
        # Barrel distortion sampling map.
        r2 = nx * nx + ny * ny
        scale = 1.0 + 0.18 * amt * r2
        sx = cx + nx * scale * cx
        sy = cy + ny * scale * cy
        # Chromatic aberration: per-channel slightly different scale.
        ca = 4.0 * amt
        length = np.sqrt(r2) + 1e-6
        ux = nx / length
        uy = ny / length
        r_sx = sx + ux * ca
        r_sy = sy + uy * ca
        b_sx = sx - ux * ca
        b_sy = sy - uy * ca
        r_plane = _sample_plane(arr[..., 0], r_sx, r_sy)
        g_plane = _sample_plane(arr[..., 1], sx, sy)
        b_plane = _sample_plane(arr[..., 2], b_sx, b_sy)
        out = np.stack([r_plane, g_plane, b_plane], axis=-1) / 255.0

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

        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
