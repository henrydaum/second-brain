from plugins.BaseTechnique import BaseTechnique, Enum, Palette, Slider, Pan

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

_SPOTS = {
    "full":          (-0.5,      0.0,       0.0, 200),  # the whole set
    "seahorse":      (-0.7453,   0.1127,    9.0, 400),  # west "valley" between cardioid and period-2 bulb
    "elephant":      ( 0.2825,   0.01,      8.0, 400),  # east valley off the cardioid
    "triple_spiral": (-0.088,    0.654,     8.0, 400),  # upper antenna, triple-armed spiral
    "lightning":          (0,   1,       45, 250),  # Mandelbrot edge shows lighting-like dendritic patterns
    "spiral_galaxy": (-0.74543,  0.11301,  12.0, 600),  # deep seahorse: galactic spirals
}


class MandelbrotExplorerTechnique(BaseTechnique):
    name = 'Mandelbrot Explorer'
    description = "A guided tour of the Mandelbrot set's most famous landmarks. Pick a spot -- Seahorse Valley, Elephant Valley, dendritic lightning patterns, deep spiral galaxies -- and the view, zoom, and iteration depth are all dialed in for you. Pair with any palette to taste. Pan, zoom, and iteration controls let you wander off the preset and chase your own detail. Optimized for M1: complex64 working set, cardioid + period-2 bulb early-exit, and live-buffer compaction every 3 iterations."
    kind = "background"
    palette = Palette()
    spot = Enum([('full', 'Full Set'), ('seahorse', 'Seahorse Valley'), ('elephant', 'Elephant Valley'), ('triple_spiral', 'Triple Spiral'), ('lightning', 'Lightning'), ('spiral_galaxy', 'Spiral Galaxy')], default='full')
    # Pan range ±3 (= 1.5 view-widths to either side) so you can navigate
    # between adjacent features at deep zoom; capping at ±1 only let you
    # move by half a screen, which felt like nothing once zoomed in.
    pan_x = Slider(-3.0, 3.0, default=0.0, step=0.05)
    pan_y = Slider(-3.0, 3.0, default=0.0, step=0.05)
    pan = Pan(x='pan_x', y='pan_y', label='Pan')
    zoom_extra = Slider(0.5, 32.0, default=1.0, step=0.05, label='Zoom')
    iterations = Slider(0, 1500, default=1500, step=10, label='Iterations')

    def run(self, canvas):
        cx, cy, zoom_exp, detail = _SPOTS.get(str(self.spot), _SPOTS["full"])
        s = int(canvas.size)
        zoom = float(2.0 ** zoom_exp) * float(self.zoom_extra)
        iter_override = int(self.iterations)
        n_iter = iter_override if iter_override >= 50 else int(detail)

        # Step up to f64 once total zoom passes ~2^10, regardless of whether
        # the extra zoom came from the preset or the user slider.
        use_f64 = zoom > 1024.0
        cplx = np.complex128 if use_f64 else np.complex64
        real = np.float64 if use_f64 else np.float32

        view = 3.0 / zoom
        half = view * 0.5
        # Pan offsets the view center; ±1 on the slider moves by a full half-view.
        cx_eff = cx + float(self.pan_x) * half
        cy_eff = cy + float(self.pan_y) * half
        re = np.linspace(cx_eff - half, cx_eff + half, s, dtype=real)
        im = np.linspace(cy_eff - half, cy_eff + half, s, dtype=real)
        R, I = np.meshgrid(re, im)

        # Cardioid + period-2 bulb tests skip the largest interior regions
        # before any iteration. This is what makes the "full" preset instant.
        q = (R - 0.25) ** 2 + I * I
        in_cardioid = q * (q + (R - 0.25)) <= 0.25 * I * I
        in_bulb = (R + 1.0) ** 2 + I * I <= 0.0625
        inside_known = (in_cardioid | in_bulb).ravel()

        N = s * s
        out_flat = np.zeros(N, dtype=np.float64)
        inside_flat = inside_known.copy()

        live_idx = np.flatnonzero(~inside_known)
        C_live = (R + 1j * I).ravel()[live_idx].astype(cplx)
        Z_live = np.zeros_like(C_live)

        # Bailout 16 (vs 256) keeps |z|^2 inside float32 across the 3-iter
        # compaction window. The log-log smoothing is still well-defined.
        bailout2 = float(16 * 16) if not use_f64 else float(1 << 16)
        inv_log2 = 1.0 / np.log(2.0)
        compact_every = 3

        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(n_iter):
                Z_live = Z_live * Z_live + C_live
                absZ2 = Z_live.real * Z_live.real + Z_live.imag * Z_live.imag
                if (i + 1) % compact_every == 0 or i == n_iter - 1:
                    esc = absZ2 > bailout2
                    if esc.any():
                        log_mod = 0.5 * np.log(absZ2[esc].astype(np.float64))
                        nu = np.log(log_mod * inv_log2) * inv_log2
                        out_flat[live_idx[esc]] = (i + 1) - nu
                        keep = ~esc
                        live_idx = live_idx[keep]
                        Z_live = Z_live[keep]
                        C_live = C_live[keep]
                        if live_idx.size == 0:
                            break

        inside_flat[live_idx] = True

        valid = ~inside_flat
        t = np.zeros_like(out_flat)
        if valid.any():
            v = np.log(out_flat[valid] + 1.0)
            vmin = float(v.min())
            vmax = float(v.max())
            if vmax - vmin > 1e-9:
                v = (v - vmin) / (vmax - vmin)
            else:
                v = np.zeros_like(v)
            t[valid] = v

        LUT_SIZE = 512
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(k / (LUT_SIZE - 1))) for k in range(LUT_SIZE)],
            dtype=np.uint8,
        )
        idx_lut = np.clip((t * (LUT_SIZE - 1)).astype(np.int32), 0, LUT_SIZE - 1)
        rgb = lut[idx_lut].reshape(s, s, 3)

        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
        rgb[inside_flat.reshape(s, s)] = bg

        canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
