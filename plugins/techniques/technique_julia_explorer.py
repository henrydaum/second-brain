from plugins.BaseTechnique import BaseTechnique, Enum, Palette, Slider, Pan

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

_SPOTS = {
    "dendrite":  ( 0.0,     1.0,     0.0, 250),   # c = i; thin tree-like fractal
    "rabbit":    (-0.123,   0.745,   0.0, 250),   # the Douady rabbit
    "san_marco": (-0.75,    0.0,     0.0, 250),   # symmetric basin shaped like the cathedral
    "siegel":    (-0.391,  -0.587,   0.0, 280),   # quasi-circle Siegel disk
    "dragon":    (-0.8,     0.156,   0.0, 250),   # curling dragon-like arms
    "spiral":    (-0.7269,  0.1889,  0.0, 300),   # tight twin-spiral
    "airplane":  (-1.755,   0.0,     0.0, 250),   # real-axis cantor-like airplane
    "dust":      ( 0.45,    0.1428,  0.0, 220),   # totally disconnected dust
}


class JuliaExplorerTechnique(BaseTechnique):
    name = 'Julia Explorer'
    description = 'A guided tour of the most beloved filled Julia sets. Pick a constant -- the spindly Dendrite, the Douady Rabbit, the curling Dragon, the Siegel Disk, the cantor-dust airplane -- and the right framing and iteration depth come along for free. Pair with any palette. Pan, zoom, and iteration controls let you wander off the preset and chase your own detail. Optimized for M1: complex64 working set, |z0| pre-escape filter, and live-buffer compaction every 3 iterations.'
    kind = "background"
    palette = Palette()
    spot = Enum([('dendrite', 'Dendrite'), ('rabbit', 'Douady Rabbit'), ('san_marco', 'San Marco'), ('siegel', 'Siegel Disk'), ('dragon', 'Dragon'), ('spiral', 'Spiral'), ('airplane', 'Airplane'), ('dust', 'Cantor Dust')], default='dragon')
    # Pan range ±3 (= 1.5 view-widths to either side) so you can navigate
    # between adjacent features at deep zoom; capping at ±1 only let you
    # move by half a screen, which felt like nothing once zoomed in.
    pan_x = Slider(-3.0, 3.0, default=0.0, step=0.05)
    pan_y = Slider(-3.0, 3.0, default=0.0, step=0.05)
    pan = Pan(x='pan_x', y='pan_y', label='Pan')
    zoom_extra = Slider(0.5, 32.0, default=1.0, step=0.05, label='Zoom')
    iterations = Slider(0, 1500, default=1500, step=10, label='Iterations')

    def run(self, canvas):
        jx, jy, zoom_exp, detail = _SPOTS.get(str(self.spot), _SPOTS["dragon"])
        s = int(canvas.size)
        zoom = float(2.0 ** zoom_exp) * float(self.zoom_extra)
        iter_override = int(self.iterations)
        n_iter = iter_override if iter_override >= 50 else int(detail)

        # Step up to f64 once total zoom passes ~2^10, regardless of whether
        # the extra zoom came from the preset or the user slider.
        use_f64 = zoom > 1024.0
        cplx = np.complex128 if use_f64 else np.complex64
        real = np.float64 if use_f64 else np.float32

        c_scalar = float(jx) + 1j * float(jy)
        c = cplx(c_scalar)

        # 4-unit window centered at origin frames every filled Julia with |c| <= 2.
        view = 4.0 / zoom
        half = view * 0.5
        # Pan offsets the view center; ±1 on the slider moves by a full half-view.
        offset_x = float(self.pan_x) * half
        offset_y = float(self.pan_y) * half
        re = np.linspace(-half + offset_x, half + offset_x, s, dtype=real)
        im = np.linspace(-half + offset_y, half + offset_y, s, dtype=real)
        R, I = np.meshgrid(re, im)

        er2 = max(abs(c_scalar), 2.0) ** 2
        initial_abs2 = (R * R + I * I).ravel()
        escapes_now = initial_abs2 > er2

        N = s * s
        out_flat = np.zeros(N, dtype=np.float64)
        inv_log2 = 1.0 / np.log(2.0)
        if escapes_now.any():
            ae = initial_abs2[escapes_now].astype(np.float64)
            log_mod = 0.5 * np.log(ae)
            out_flat[escapes_now] = -(np.log(log_mod * inv_log2) * inv_log2)

        inside_flat = np.zeros(N, dtype=bool)
        live_idx = np.flatnonzero(~escapes_now)
        Z_live = (R + 1j * I).ravel()[live_idx].astype(cplx)

        bailout2 = float(16 * 16) if not use_f64 else float(1 << 16)
        if bailout2 < er2:
            bailout2 = er2
        compact_every = 3

        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(n_iter):
                Z_live = Z_live * Z_live + c
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
                        if live_idx.size == 0:
                            break

        inside_flat[live_idx] = True

        valid = ~inside_flat
        t = np.zeros_like(out_flat)
        if valid.any():
            v = out_flat[valid]
            v = v - float(v.min())
            v = np.log(v + 1.0)
            vmax = float(v.max())
            if vmax > 1e-9:
                v = v / vmax
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
