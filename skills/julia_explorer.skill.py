SKILL_NAME = "Julia Explorer"
SKILL_DESCRIPTION = "A guided tour of the most beloved filled Julia sets. Pick a constant -- the spindly Dendrite, the Douady Rabbit, the curling Dragon, the Siegel Disk, the cantor-dust airplane -- and the right framing and iteration depth come along for free. Pair with any palette. Optimized for M1: complex64 working set, |z0| pre-escape filter, and live-buffer compaction every 3 iterations."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0
SKILL_CONTROLS = [
    {"type": "enum", "name": "spot", "label": "Spot",
     "options": [
         {"value": "dendrite", "label": "Dendrite"},
         {"value": "rabbit",   "label": "Douady Rabbit"},
         {"value": "san_marco","label": "San Marco"},
         {"value": "siegel",   "label": "Siegel Disk"},
         {"value": "dragon",   "label": "Dragon"},
         {"value": "spiral",   "label": "Spiral"},
         {"value": "airplane", "label": "Airplane"},
         {"value": "dust",     "label": "Cantor Dust"},
     ],
     "default": "dragon"},
    {"type": "palette", "name": "palette", "label": "Palette"},
]

import numpy as np
from PIL import Image

# Curated Julia constants. Each entry: (jx, jy, zoom_exp, detail).
# These c values are the canonical, named landmarks of Julia-set theory --
# each gives a visually distinct filled set centered on the origin.
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


def run(canvas, spot="dragon", **_):
    jx, jy, zoom_exp, detail = _SPOTS.get(str(spot), _SPOTS["dragon"])
    s = int(canvas.size)
    zoom = float(2.0 ** zoom_exp)
    n_iter = int(detail)

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
