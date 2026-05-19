SKILL_NAME = "Gray-Scott"
SKILL_DESCRIPTION = "Reaction-diffusion as Gray-Scott PDE on a 256x256 grid, integrated for ~3500 steps then upscaled. Two species A and B diffuse and react; the steady-state texture depends entirely on the feed rate f and kill rate k. Five named presets explore the parameter landscape: spots, mazes, worms, coral, and U-skate (moving solitons). Final B concentration mapped through palette LUT -- the patterns emerge from chemistry, not drawing. Good for \"reaction diffusion\", \"gray scott\", \"texture\", \"spots\", \"stripes\", \"coral\", \"organic\", or any biology-flavored algorithmic motif."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1779667200.0
SKILL_HIDDEN = False
SKILL_CONTROLS = [
    {"type": "enum", "name": "regime", "label": "Regime",
     "options": [
         {"value": "spots",  "label": "Spots"},
         {"value": "maze",   "label": "Maze"},
         {"value": "worms",  "label": "Worms"},
         {"value": "coral",  "label": "Coral"},
         {"value": "uskate", "label": "U-skate (solitons)"},
     ],
     "default": "coral"},
    {"type": "palette", "name": "palette", "label": "Palette"},
]

import numpy as np
from PIL import Image


_PRESETS = {
    # f, k, steps -- chosen so the resulting B texture is mature, not noisy.
    # Native grid is 384x384; values picked from the canonical mitchell map
    # and verified to produce the named texture at this scale.
    "spots":  (0.0540, 0.0620, 5000),
    "maze":   (0.0290, 0.0570, 5000),
    "worms":  (0.0780, 0.0610, 5000),
    "coral":  (0.0620, 0.0620, 5000),
    "uskate": (0.0620, 0.0609, 6000),
}


def _laplacian(z):
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0)
        + np.roll(z, 1, 1) + np.roll(z, -1, 1)
        - 4.0 * z
    )


def run(canvas, regime="coral", **_):
    s = int(canvas.size)
    seed = int(canvas.seed)
    f, k, n_steps = _PRESETS.get(str(regime), _PRESETS["coral"])
    rng = np.random.default_rng(seed)

    N = 384
    A = np.ones((N, N), dtype=np.float32)
    B = np.zeros((N, N), dtype=np.float32)
    # Seed many small noisy patches of B distributed across the canvas so the
    # pattern develops from multiple nucleation sites and fills the frame.
    n_seeds = 14
    for _ in range(n_seeds):
        r0 = int(rng.integers(N // 8, N - N // 8))
        c0 = int(rng.integers(N // 8, N - N // 8))
        rad = int(rng.integers(8, 18))
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if dr * dr + dc * dc <= rad * rad:
                    rr, cc = (r0 + dr) % N, (c0 + dc) % N
                    A[rr, cc] = 0.5
                    B[rr, cc] = 0.25 + 0.4 * float(rng.random())

    Du, Dv = 0.16, 0.08
    dt = 1.0

    for _ in range(n_steps):
        La = _laplacian(A)
        Lb = _laplacian(B)
        ABB = A * B * B
        A = A + (Du * La - ABB + f * (1.0 - A)) * dt
        B = B + (Dv * Lb + ABB - (k + f) * B) * dt
        np.clip(A, 0.0, 1.0, out=A)
        np.clip(B, 0.0, 1.0, out=B)

    # Stretch B's actual range to [0,1] so the palette ramp uses the full LUT.
    bmin = float(B.min())
    bmax = float(B.max())
    if bmax - bmin > 1e-6:
        t_field = (B - bmin) / (bmax - bmin)
    else:
        t_field = np.zeros_like(B)
    t_field = t_field ** 0.85  # gentle gamma to lift mid-range

    LUT = 256
    lut = np.array(
        [art_kit.hex_to_rgb(art_kit.palette_color(k_ / (LUT - 1))) for k_ in range(LUT)],
        dtype=np.uint8,
    )
    idx = np.clip((t_field * (LUT - 1)).astype(np.int32), 0, LUT - 1)
    rgb = lut[idx]
    img = Image.fromarray(rgb, "RGB").resize((s, s), Image.LANCZOS).convert("RGBA")
    canvas.commit(img)
