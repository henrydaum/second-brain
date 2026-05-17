SKILL_NAME = "Mandelbrot Explorer"
SKILL_DESCRIPTION = "Interactive Mandelbrot set, palette-graded via continuous escape-time smoothing. Pan scales with zoom -- one click always moves the view by the same fraction of what you're looking at. Optimized for M1: complex64 working set, cardioid + period-2 bulb early-exit, and live-buffer compaction every 3 iterations. Params: cx (-2..1, default -0.5), cy (-1.5..1.5, default 0.0), zoom_exp (-1..18, default 0.0 -> 1x), detail (30..800 iterations, default 220)."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0
SKILL_CONTROLS = [
    {"type": "pan", "name": "center", "label": "Pan",
     "x_param": "cx", "y_param": "cy", "step": 0.3,
     "x_default": -0.5, "y_default": 0.0,
     "step_scale_param": "zoom_exp"},
    {"type": "slider", "name": "zoom_exp", "label": "Zoom",
     "min": -1.0, "max": 18.0, "step": 0.25, "default": 0.0},
    {"type": "slider", "name": "detail", "label": "Detail",
     "min": 30, "max": 800, "step": 10, "default": 220},
    {"type": "palette", "name": "palette", "label": "Palette"},
]

import numpy as np
from PIL import Image


def run(canvas, cx=-0.5, cy=0.0, zoom_exp=0.0, detail=220, **_):
    s = int(canvas.size)
    zoom_exp = float(zoom_exp)
    zoom = float(2.0 ** zoom_exp)
    n_iter = int(art_kit.clamp(detail, 30, 800))

    # Past ~zoom_exp 10 float32 can't resolve a pixel, so step up to float64.
    use_f64 = zoom_exp > 10.0
    cplx = np.complex128 if use_f64 else np.complex64
    real = np.float64 if use_f64 else np.float32

    # 3-unit-wide window at zoom_exp=0 frames the canonical set.
    view = 3.0 / zoom
    half = view * 0.5
    re = np.linspace(cx - half, cx + half, s, dtype=real)
    im = np.linspace(cy - half, cy + half, s, dtype=real)
    R, I = np.meshgrid(re, im)

    # Cardioid + period-2 bulb tests skip the largest interior regions
    # before any iteration -- this is what makes shallow renders feel instant.
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

    # Smaller bailout (16) keeps |z|^2 well inside float32 range even with
    # compact-every-3, while staying large enough for stable log-log smoothing.
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

    # Anything still alive after n_iter is interior.
    inside_flat[live_idx] = True

    # Perceptual log remap keeps the gradient vivid at deep zooms where raw
    # escape counts cluster near the iteration cap.
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
