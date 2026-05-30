from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

def _fbm_grid(seed, grid_size, freq, octaves):
    """Vectorized fbm field on a `grid_size x grid_size` lattice."""
    yy, xx = np.mgrid[0:grid_size, 0:grid_size].astype(np.float32)
    field = art_kit.fbm_grid(seed, xx * freq, yy * freq, octaves=octaves)
    return field.astype(np.float32)


class FbmLandscapeTechnique(BaseTechnique):
    name = 'fBm Landscape'
    description = 'A pure noise field rendered straight to canvas: fractional Brownian motion sampled at a coarse grid, smoothed back up to full resolution, then mapped through the palette ramp. Five regimes shift the personality completely -- soft cloud strata, rolling terrain with a warmer horizon, high-contrast magma, a layered nebula with secondary color modulation, and ridged turbulence. The universal fallback background: looks composed on every palette. Good for "clouds", "terrain", "mist", "fog", "atmosphere", "magma", "nebula", "texture", or any abstract organic background.'
    kind = "background"
    palette = Palette()
    regime = Enum([('clouds', 'Clouds'), ('terrain', 'Terrain'), ('magma', 'Magma'), ('nebula', 'Nebula'), ('ridges', 'Ridged Turbulence')], default='nebula')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        self.regime = str(self.regime)

        # Each regime: (grid_size, frequency, octaves, gamma, t_lo, t_hi, special).
        # Grid sampled in pure-Python fbm and BICUBIC-upscaled to canvas. 160 is the
        # sweet spot: coarse enough to stay well inside the budget at 1024, fine
        # enough that BICUBIC produces a clean field (each cell becomes ~6.4 px).
        cfg = {
            "clouds":  (160, 0.031, 3, 1.1, 0.40, 0.92, None),
            "terrain": (160, 0.025, 4, 0.9,  0.10, 0.95, "horizon"),
            "magma":   (160, 0.042, 6, 1.5,  0.05, 1.00, None),
            "nebula":  (160, 0.028, 5, 1.0,  0.20, 0.95, "nebula"),
            "ridges":  (160, 0.035, 5, 1.1,  0.15, 0.95, "ridged"),
        }
        grid_size, freq, octaves, gamma, t_lo, t_hi, special = cfg.get(self.regime, cfg["nebula"])

        base = _fbm_grid(seed, grid_size, freq, octaves)
        if special == "ridged":
            base = 1.0 - np.abs(base - 0.5) * 2.0  # ridges along the half-value contour
        base = base - float(base.min())
        bmax = float(base.max()) or 1.0
        base = base / bmax

        if special == "horizon":
            ys = np.linspace(1.0, 0.0, grid_size, dtype=np.float32)[:, None]
            base = np.clip(base * 0.5 + ys * 0.55, 0.0, 1.0)

        # Apply gamma to push contrast into the right curve.
        base = np.clip(base ** gamma, 0.0, 1.0)

        # Optional nebula color modulation: a second slow fbm tints palette ramp.
        if special == "nebula":
            warp = _fbm_grid(seed ^ 0x9E3779B9, grid_size, freq * 0.45, 3)
            warp = warp - float(warp.min())
            warp = warp / (float(warp.max()) or 1.0)
            # Pull t toward (lo) where warp is low and (hi) where warp is high.
            t_field = t_lo + (t_hi - t_lo) * (0.55 * base + 0.45 * warp)
        else:
            t_field = t_lo + (t_hi - t_lo) * base

        LUT = 512
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(k / (LUT - 1))) for k in range(LUT)],
            dtype=np.uint8,
        )
        idx = np.clip((t_field * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = lut[idx]
        img = Image.fromarray(rgb, "RGB").resize((s, s), Image.BICUBIC).convert("RGBA")
        canvas.commit(img)
