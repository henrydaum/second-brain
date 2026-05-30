from plugins.BaseTechnique import BaseTechnique, Enum, Palette, Slider, Pan

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


class NewtonBasinsExplorerTechnique(BaseTechnique):
    name = 'Newton Basins Explorer'
    description = 'Newton\'s method on a complex polynomial, colored by basin of attraction. Each pixel iterates z = z - f(z)/f\'(z); the root it converges to picks the palette band, the iteration count modulates brightness within the band. Produces lacy, intricate boundaries between basins -- a different geometry than escape-time fractals. Pan, zoom, and iteration controls let you dive into the lace between roots and chase your own detail. Good for "fractal", "newton", "basins", "lace", "stained glass", or any mathematically-elaborate algorithmic motif.'
    kind = "background"
    palette = Palette()
    polynomial = Enum([('cubic', 'z^3 - 1 (three roots)'), ('perturbed', 'z^3 - 2z + 2 (cycles)'), ('quartic', 'z^4 - 1 (four roots)'), ('quintic', 'z^5 - 1 (five roots)'), ('sextic', 'z^6 - 1 (six roots)'), ('octic', 'z^8 - 1 (eight roots)')], default='cubic')
    # Pan range ±3 (= 1.5 view-widths to either side) so you can navigate
    # between adjacent features at deep zoom; capping at ±1 only let you
    # move by half a screen, which felt like nothing once zoomed in.
    pan_x = Slider(-3.0, 3.0, default=0.0, step=0.05)
    pan_y = Slider(-3.0, 3.0, default=0.0, step=0.05)
    pan = Pan(x='pan_x', y='pan_y', label='Pan')
    zoom_extra = Slider(0.5, 32.0, default=1.0, step=0.05, label='Zoom')
    iterations = Slider(0, 1500, default=1500, step=10, label='Iterations')

    def run(self, canvas):
        s = int(canvas.size)
        key = str(self.polynomial)
        polys = _polys()
        f, fp, roots, view_half = polys.get(key, polys["cubic"])

        zoom_extra = float(self.zoom_extra)
        iter_override = int(self.iterations)
        n_iter = iter_override if iter_override >= 50 else 40

        # Step up to f64 once total zoom passes ~2^10. The polynomial roots
        # live on (or near) the unit circle, so the "total zoom" here is the
        # zoom slider directly -- view_half is just framing margin.
        use_f64 = zoom_extra > 1024.0
        cplx = np.complex128 if use_f64 else np.complex64
        real = np.float64 if use_f64 else np.float32

        half = view_half / zoom_extra
        # Pan offsets the view center; ±1 on the slider moves by a full half-view.
        offset_x = float(self.pan_x) * half
        offset_y = float(self.pan_y) * half
        re = np.linspace(-half + offset_x, half + offset_x, s, dtype=real)
        im = np.linspace(-half + offset_y, half + offset_y, s, dtype=real)
        R, I = np.meshgrid(re, im)

        tol = 1e-4 if not use_f64 else 1e-6

        # Live-buffer compaction: drop pixels from the working set as they
        # converge. Newton converges quadratically -- most pixels finish in
        # <10 iters; only basin boundaries iterate to the cap. Without
        # compaction the full grid of complex arithmetic ran every iter and
        # timed out at s=1024.
        N = s * s
        iters_flat = np.full(N, n_iter, dtype=np.int32)
        converged_flat = np.zeros(N, dtype=bool)
        Z_final = (R + 1j * I).astype(cplx).ravel()
        live_idx = np.arange(N, dtype=np.int64)
        Z_live = Z_final.copy()
        tiny = cplx(1e-7) if not use_f64 else cplx(1e-12)

        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(n_iter):
                denom = fp(Z_live)
                denom = np.where(np.abs(denom) < abs(tiny), tiny, denom)
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
            root_arr = np.array(roots, dtype=cplx)
            d = np.abs(Z[..., None] - root_arr[None, None, :])
            basin = np.argmin(d, axis=-1).astype(np.int32)
            n_basins = len(roots)
        else:
            # Quantize argument into 6 bins; pretty and stable for cycling polys.
            n_basins = 6
            ang = (np.angle(Z) + math.pi) / (2 * math.pi)  # [0,1)
            basin = np.clip((ang * n_basins).astype(np.int32), 0, n_basins - 1)

        # Brightness from iteration count -- few iters = sharp center, many =
        # boundary haze. Reference is a fixed 40 (not n_iter) so the haze look
        # survives when the user dials iterations way up; otherwise a 30-iter
        # pixel reads as brightness 0.98 instead of 0.25 and the lace washes out.
        brightness = 1.0 - np.clip(iters / 40.0, 0.0, 1.0)
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
