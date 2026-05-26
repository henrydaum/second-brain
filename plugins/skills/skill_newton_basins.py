from plugins.BaseSkill import BaseSkill, Enum, Palette

import math
import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

def _roots_of_unity(n):
    return [math.cos(2 * math.pi * k / n) + 1j * math.sin(2 * math.pi * k / n) for k in range(n)]

def _polys():
    return {
        "cubic":     (lambda z: z**3 - 1,        lambda z: 3 * z**2,           _roots_of_unity(3), 1.6),
        "perturbed": (lambda z: z**3 - 2*z + 2,  lambda z: 3 * z**2 - 2,       None,               1.8),
        "quartic":   (lambda z: z**4 - 1,        lambda z: 4 * z**3,           _roots_of_unity(4), 1.6),
        "quintic":   (lambda z: z**5 - 1,        lambda z: 5 * z**4,           _roots_of_unity(5), 1.6),
        "sextic":    (lambda z: z**6 - 1,        lambda z: 6 * z**5,           _roots_of_unity(6), 1.6),
        "octic":     (lambda z: z**8 - 1,        lambda z: 8 * z**7,           _roots_of_unity(8), 1.6),
    }


class NewtonBasinsSkill(BaseSkill):
    name = 'Newton Basins'
    description = 'Newton\'s method on a complex polynomial, colored by basin of attraction. Each pixel iterates z = z - f(z)/f\'(z); the root it converges to picks the palette band, the iteration count modulates brightness within the band. Produces lacy, intricate boundaries between basins -- a different geometry than escape-time fractals. Good for "fractal", "newton", "basins", "lace", "stained glass", or any mathematically-elaborate algorithmic motif.'
    kind = "background"
    palette = Palette()
    polynomial = Enum([('cubic', 'z^3 - 1 (three roots)'), ('perturbed', 'z^3 - 2z + 2 (cycles)'), ('quartic', 'z^4 - 1 (four roots)'), ('quintic', 'z^5 - 1 (five roots)'), ('sextic', 'z^6 - 1 (six roots)'), ('octic', 'z^8 - 1 (eight roots)')], default='cubic')

    def run(self, canvas):
        s = int(canvas.size)
        key = str(self.polynomial)
        polys = _polys()
        f, fp, roots, view_half = polys.get(key, polys["cubic"])

        re = np.linspace(-view_half, view_half, s, dtype=np.float32)
        im = np.linspace(-view_half, view_half, s, dtype=np.float32)
        R, I = np.meshgrid(re, im)

        n_iter = 40
        tol = 1e-4  # relaxed for complex64

        # Live-buffer compaction: drop pixels from the working set as they
        # converge. Most pixels finish in <10 iters; only basin boundaries
        # need the full 40. Without compaction we kept doing the full grid
        # of complex128 arithmetic every iteration and timed out at s=1024.
        N = s * s
        iters_flat = np.full(N, n_iter, dtype=np.int32)
        converged_flat = np.zeros(N, dtype=bool)
        Z_final = (R + 1j * I).astype(np.complex64).ravel()
        live_idx = np.arange(N, dtype=np.int64)
        Z_live = Z_final.copy()

        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(n_iter):
                denom = fp(Z_live)
                denom = np.where(np.abs(denom) < 1e-7, np.complex64(1e-7), denom)
                dZ = f(Z_live) / denom
                Z_live = Z_live - dZ
                done = np.abs(dZ) < tol
                if done.any():
                    done_global = live_idx[done]
                    iters_flat[done_global] = i + 1
                    converged_flat[done_global] = True
                    Z_final[done_global] = Z_live[done]
                    keep = ~done
                    live_idx = live_idx[keep]
                    Z_live = Z_live[keep]
                    if live_idx.size == 0:
                        break
        if live_idx.size:
            Z_final[live_idx] = Z_live

        Z = Z_final.reshape(s, s)
        iters = iters_flat.reshape(s, s).astype(np.float32)
        converged = converged_flat.reshape(s, s)

        # For polynomials with known roots, classify by nearest root.
        # For "perturbed" (which has three roots we'd need to solve for), classify
        # by final Z's argument -- still produces a clean basin map.
        if roots is not None:
            root_arr = np.array(roots, dtype=np.complex64)
            d = np.abs(Z[..., None] - root_arr[None, None, :])
            basin = np.argmin(d, axis=-1).astype(np.int32)
            n_basins = len(roots)
        else:
            # Quantize argument into 6 bins; pretty and stable for cycling polys.
            n_basins = 6
            ang = (np.angle(Z) + math.pi) / (2 * math.pi)  # [0,1)
            basin = np.clip((ang * n_basins).astype(np.int32), 0, n_basins - 1)

        # Brightness from iteration count -- few iters = sharp center, many = boundary haze.
        brightness = 1.0 - np.clip(iters / float(n_iter), 0.0, 1.0)
        brightness = 0.25 + 0.75 * brightness  # keep some color in the slow regions

        # Build a per-basin base color from the palette ramp, then attenuate by brightness.
        base = np.zeros((n_basins, 3), dtype=np.float32)
        for b in range(n_basins):
            t = b / max(1, n_basins - 1)
            # Pull each basin toward a distinct palette slot, biased away from 0 (background).
            base[b] = np.array(art_kit.hex_to_rgb(art_kit.palette_color(0.18 + 0.78 * t)), dtype=np.float32)
        rgb = base[basin] * brightness[..., None]

        # The deeply-unconverged pixels go to background -- those are the fractal seams.
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32)
        unstable = ~converged
        rgb[unstable] = bg

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
