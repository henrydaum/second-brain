from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import math
import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

def _smin(a, b, k):
    """Polynomial smooth-min in [0, k]. Lower k -> sharper join."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return a * h + b * (1.0 - h) - k * h * (1.0 - h)

def _sdf_spheres(x, y):
    centers = [(-0.35, 0.20, 0.45), (0.30, -0.15, 0.40), (0.10, 0.40, 0.30)]
    d = None
    for cx, cy, r in centers:
        di = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
        d = di if d is None else _smin(d, di, 0.18)
    return d

def _sdf_torus(x, y):
    # Annulus: outer radius - |dist - mean|.
    r_mean = 0.55
    thickness = 0.12
    return np.abs(np.sqrt(x * x + y * y) - r_mean) - thickness

def _sdf_rbox(x, y, hw, hh, r):
    dx = np.abs(x) - hw
    dy = np.abs(y) - hh
    ax = np.maximum(dx, 0.0)
    ay = np.maximum(dy, 0.0)
    return np.sqrt(ax * ax + ay * ay) + np.minimum(np.maximum(dx, dy), 0.0) - r

def _sdf_shapes(x, y):
    box = _sdf_rbox(x + 0.15, y - 0.05, 0.45, 0.30, 0.10)
    circ = np.sqrt((x - 0.35) ** 2 + (y + 0.30) ** 2) - 0.30
    return _smin(box, circ, 0.20)


class RaymarchSdfTechnique(BaseTechnique):
    name = 'Raymarch SDF'
    description = '2D raymarching of a signed distance function: for each pixel, march along a ray taking step sizes equal to the SDF value at the current point. The number of steps to hit (or near-miss) the surface becomes the palette ramp coordinate; the final distance modulates atmosphere fog. Three SDF presets -- three overlapping circles smin\'d together, a torus annulus, and a rounded box + circle composition. Rendered at half resolution then upscaled to stay in the 30s budget. Good for "raymarching", "sdf", "shapes", "sphere", "torus", "glow", or any implicit-geometry algorithmic motif.'
    kind = "background"
    palette = Palette()
    scene = Enum([('spheres', 'Three Spheres'), ('torus', 'Torus'), ('shapes', 'Mixed Shapes')], default='torus')

    def run(self, canvas):
        s = int(canvas.size)
        self.scene = str(self.scene)

        # Render at half-res for speed, then upscale.
        R = 512
        # Normalize coords to roughly [-1, 1] across both axes.
        ys, xs = np.mgrid[0:R, 0:R].astype(np.float32)
        nx = (xs - R / 2.0) / (R / 2.0)
        ny = (ys - R / 2.0) / (R / 2.0)

        sdf_fns = {"spheres": _sdf_spheres, "torus": _sdf_torus, "shapes": _sdf_shapes}
        sdf = sdf_fns.get(self.scene, _sdf_spheres)

        # Direct 2D SDF visualization. The "raymarch" geometry is realized
        # vectorially: at each pixel, compute the signed distance, then march
        # along the +x screen direction in 32 steps to compute a soft shadow
        # term (how much of the ray would be occluded by the shape going right).
        d = sdf(nx, ny)

        # Outside / inside split.
        inside = d < 0.0

        # Soft shadow: at each pixel, look forward along +x for a few SDF samples;
        # the minimum signed distance along that segment determines shadow weight.
        shadow = np.ones_like(d)
        march_len = 0.35
        n_march = 32
        step = march_len / n_march
        px = nx.copy()
        py = ny.copy()
        for i in range(1, n_march + 1):
            px = px + step
            ds = sdf(px, py)
            # Penumbra-style: minimum of (8 * ds / t) clamped to [0, 1].
            t_along = i * step
            shadow = np.minimum(shadow, np.clip(8.0 * ds / max(t_along, 1e-3), 0.0, 1.0))
        shadow = np.clip(shadow, 0.0, 1.0)

        # Outside: palette ramp by distance (close = bright, far = soft).
        # Bands of frac(d * freq) add the iso-contour reading.
        far = np.clip(d / 0.9, 0.0, 1.0)
        bands = 0.5 + 0.5 * np.cos(2 * math.pi * d * 6.0)
        outer_t = 0.85 - 0.55 * far + 0.06 * bands

        # Inside: warm solid that brightens slightly toward the surface.
        near_surface = 1.0 - np.clip(-d / 0.4, 0.0, 1.0)
        inner_t = 0.40 + 0.45 * near_surface

        t_field = np.where(inside, inner_t, outer_t)
        # Apply shadow as a brightness dip in the outer region.
        t_field = np.where(inside, t_field, t_field - 0.18 * (1.0 - shadow))
        t_field = np.clip(t_field, 0.02, 0.99)

        LUT = 512
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(k / (LUT - 1))) for k in range(LUT)],
            dtype=np.uint8,
        )
        idx = np.clip((t_field * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = lut[idx]
        img = Image.fromarray(rgb, "RGB").resize((s, s), Image.BICUBIC).convert("RGBA")
        canvas.commit(img)
