SKILL_NAME = "Julia Explorer"
SKILL_DESCRIPTION = "Filled Julia set z -> z^2 + c with continuous escape-time smoothing and palette grading. The jx and jy sliders tune the constant c -- each tiny nudge yields a completely different fractal (dendrites, dragons, rabbits, dust). View stays centered at the origin. Optimized for M1: complex64 working set, |z0| pre-escape filter, and live-buffer compaction every 3 iterations. Iteration depth auto-scales with zoom so deep zooms keep their detail. Params: jx (-1.5..1.5, default -0.7), jy (-1.5..1.5, default 0.27015), zoom_exp (-1..18, default 0.0 -> 1x)."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0
SKILL_CONTROLS = [
    {"type": "slider", "name": "jx", "label": "c.real (jx)",
     "min": -1.5, "max": 1.5, "step": 0.005, "default": -0.7},
    {"type": "slider", "name": "jy", "label": "c.imag (jy)",
     "min": -1.5, "max": 1.5, "step": 0.005, "default": 0.27015},
    {"type": "slider", "name": "zoom_exp", "label": "Zoom",
     "min": -1.0, "max": 18.0, "step": 0.25, "default": 0.0},
    {"type": "palette", "name": "palette", "label": "Palette"},
]

import numpy as np
from PIL import Image


def run(canvas, jx=-0.7, jy=0.27015, zoom_exp=0.0, **_):
    s = int(canvas.size)
    zoom_exp = float(zoom_exp)
    zoom = float(2.0 ** zoom_exp)

    # Iteration depth scales with zoom so the deep-zoom edges keep resolving
    # without the user having to tune a separate knob.
    n_iter = int(art_kit.clamp(120 + 30 * max(0.0, zoom_exp), 120, 700))

    use_f64 = zoom_exp > 10.0
    cplx = np.complex128 if use_f64 else np.complex64
    real = np.float64 if use_f64 else np.float32

    c_scalar = float(jx) + 1j * float(jy)
    c = cplx(c_scalar)

    # 4-unit window centered at origin frames every filled Julia with |c| <= 2.
    view = 4.0 / zoom
    half = view * 0.5
    re = np.linspace(-half, half, s, dtype=real)
    im = np.linspace(-half, half, s, dtype=real)
    R, I = np.meshgrid(re, im)

    # Any pixel past the escape radius escapes on the next step, so we can
    # finalize them at iter 0 without iterating.
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

    # Small bailout (16) keeps |z|^2 inside float32 range across the 3-iter
    # compaction window while remaining valid for the log-log smoothing.
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
        v = v - float(v.min())  # pre-escape pixels can produce small negatives
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
