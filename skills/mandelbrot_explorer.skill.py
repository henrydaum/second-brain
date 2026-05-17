SKILL_NAME = "Mandelbrot Explorer"
SKILL_DESCRIPTION = "Interactive Mandelbrot set, palette-graded via continuous escape-time smoothing. Pan the complex plane, dial zoom up to ~260,000x, and tune iteration depth. Optimized for M1: cardioid + period-2 bulb early-exit skips most interior pixels, and a flat live-index buffer is compacted every iteration so escaped points stop costing work. Params: cx (-2..1, default -0.5), cy (-1.5..1.5, default 0.0), zoom_exp (-1..18, default 0.0 -> 1x), detail (30..800 iterations, default 220)."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0
SKILL_CONTROLS = [
    {"type": "pan", "name": "center", "label": "Pan",
     "x_param": "cx", "y_param": "cy", "step": 0.05,
     "x_default": -0.5, "y_default": 0.0},
    {"type": "slider", "name": "zoom_exp", "label": "Zoom",
     "min": -1.0, "max": 18.0, "step": 0.25, "default": 0.0},
    {"type": "slider", "name": "detail", "label": "Detail",
     "min": 30, "max": 800, "step": 10, "default": 220},
    {"type": "palette"},
]

import numpy as np
from PIL import Image


def run(canvas, cx=-0.5, cy=0.0, zoom_exp=0.0, detail=220):
    s = int(canvas.size)
    zoom = float(2.0 ** float(zoom_exp))
    n_iter = int(art_kit.clamp(detail, 30, 800))

    # View width: at zoom_exp=0 the canonical set (~3.0 wide) fills the canvas.
    view = 3.0 / zoom
    half = view * 0.5

    # At deep zooms we need float64 precision; complex128 is just two float64.
    re = np.linspace(cx - half, cx + half, s, dtype=np.float64)
    im = np.linspace(cy - half, cy + half, s, dtype=np.float64)
    R, I = np.meshgrid(re, im)

    # Cardioid + period-2 bulb tests catch the largest interior regions
    # cheaply -- skipping them is what makes shallow renders feel instant.
    q = (R - 0.25) ** 2 + I * I
    in_cardioid = q * (q + (R - 0.25)) <= 0.25 * I * I
    in_bulb = (R + 1.0) ** 2 + I * I <= 0.0625
    inside_known = (in_cardioid | in_bulb).ravel()

    N = s * s
    out_flat = np.zeros(N, dtype=np.float64)
    inside_flat = inside_known.copy()

    # Flat live buffer: as pixels escape we shrink the working arrays. By
    # iter ~50 the live count is usually <20% of N, so per-iter cost drops
    # quickly even though detail is high.
    live_idx = np.flatnonzero(~inside_known)
    C_live = (R + 1j * I).ravel()[live_idx]
    Z_live = np.zeros_like(C_live)

    bailout2 = float(1 << 16)  # 256^2 -- large enough for stable log-log smoothing
    inv_log2 = 1.0 / np.log(2.0)

    for i in range(n_iter):
        Z_live = Z_live * Z_live + C_live
        absZ2 = Z_live.real * Z_live.real + Z_live.imag * Z_live.imag
        esc = absZ2 > bailout2
        if esc.any():
            # Smooth escape value: i + 1 - log2(log2(|z|))  with bailout >> 1
            # gives continuous, banding-free coloring.
            log_mod = 0.5 * np.log(absZ2[esc])
            nu = np.log(log_mod * inv_log2) * inv_log2
            out_flat[live_idx[esc]] = (i + 1) - nu
            keep = ~esc
            live_idx = live_idx[keep]
            Z_live = Z_live[keep]
            C_live = C_live[keep]
            if live_idx.size == 0:
                break

    # Anything still alive after n_iter is treated as interior.
    inside_flat[live_idx] = True

    # Perceptual log remap so the gradient stays vivid at deep zooms where
    # raw escape counts cluster near the iteration cap.
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

    # Palette LUT: one hex_to_rgb pass, then vectorized lookup -- avoids
    # calling palette_color() once per pixel.
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
