from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None

_R_PENTOMINO = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]

_ACORN = [(0, 1), (1, 3), (2, 0), (2, 1), (2, 4), (2, 5), (2, 6)]

_REPLICATOR = [
    (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 4),
    (2, 0), (2, 4),
    (3, 0), (3, 4),
    (4, 0), (4, 1), (4, 2),
]

_GLIDER_GUN = [
    (5, 1), (5, 2), (6, 1), (6, 2),
    (3, 13), (3, 14), (4, 12), (4, 16), (5, 11), (5, 17), (6, 11), (6, 15),
    (6, 17), (6, 18), (7, 11), (7, 17), (8, 12), (8, 16), (9, 13), (9, 14),
    (1, 25), (2, 23), (2, 25), (3, 21), (3, 22), (4, 21), (4, 22),
    (5, 21), (5, 22), (6, 23), (6, 25), (7, 25),
    (3, 35), (3, 36), (4, 35), (4, 36),
]

def _seed_grid(N, kind, seed_int):
    g = np.zeros((N, N), dtype=np.uint8)
    rng = np.random.default_rng(seed_int)
    if kind == "soup":
        # Cover most of the grid with a dense random patch; Conway stabilizes
        # to ~3% live density after a few hundred steps but the *trails* it
        # leaves behind track every cell that was alive recently, so we want
        # a big initial footprint.
        margin = N // 20
        size = N - 2 * margin
        density = 0.38
        block = (rng.random((size, size)) < density).astype(np.uint8)
        g[margin:margin + size, margin:margin + size] = block
        return g
    if kind == "r_pentomino":
        cells = _R_PENTOMINO
        anchor = (N // 2 - 1, N // 2 - 1)
    elif kind == "glider_gun":
        cells = _GLIDER_GUN
        anchor = (N // 2 - 5, N // 2 - 18)
    elif kind == "acorn":
        cells = _ACORN
        anchor = (N // 2 - 1, N // 2 - 3)
    else:  # replicator
        cells = _REPLICATOR
        anchor = (N // 2 - 2, N // 2 - 2)
    for r, c in cells:
        rr, cc = anchor[0] + r, anchor[1] + c
        if 0 <= rr < N and 0 <= cc < N:
            g[rr, cc] = 1
    return g

def _step(g):
    n = sum(
        np.roll(np.roll(g, dy, 0), dx, 1)
        for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        if (dy or dx)
    )
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)


class ConwayLifeTechnique(BaseTechnique):
    name = 'Conway Life'
    description = 'Conway\'s Game of Life, rendered with a decay trail so the final frame shows where life recently lived as well as where it lives now. Born on 3 neighbors, survives on 2 or 3 -- the same B3/S23 rule, fed by five named initial conditions: random soup, R-pentomino (chaotic for 1103 generations), Gosper glider gun, acorn (a 5-cell methuselah), and replicator. Palette gradient runs from background (long dead) through warm tones (recently active) to accent (currently alive). Good for "cellular automata", "conway", "life", "gliders", "emergence", or any organic-grid algorithmic motif.'
    kind = "background"
    palette = Palette()
    seed_pattern = Enum([('soup', 'Random Soup'), ('r_pentomino', 'R-pentomino'), ('glider_gun', 'Glider Gun'), ('acorn', 'Acorn'), ('replicator', 'Replicator')], default='soup', label='Seed')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        kind = str(self.seed_pattern)

        # Grid resolution: 256 cells across; upscale to canvas size at the end.
        N = 256
        n_steps = {"soup": 600, "r_pentomino": 700, "glider_gun": 400,
                   "acorn": 800, "replicator": 240}.get(kind, 500)

        g = _seed_grid(N, kind, seed)
        decay = np.zeros((N, N), dtype=np.float32)
        # Slow decay so trails persist long enough to fill the canvas.
        decay_rate = 0.985
        for _ in range(n_steps):
            decay *= decay_rate
            decay[g == 1] = 1.0  # refresh on every live cell
            g = _step(g)
        # One more update so the "currently alive" pass reflects the latest state.
        decay *= decay_rate
        decay[g == 1] = 1.0

        # Palette mapping: 0.0 -> background slot, 0.4-0.85 -> trail ramp,
        # 1.0 currently-alive -> high accent slot.
        LUT = 256
        ramp_lo, ramp_hi = 0.35, 0.85
        ramp = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(ramp_lo + (ramp_hi - ramp_lo) * (k / (LUT - 1))))
             for k in range(LUT)],
            dtype=np.uint8,
        )
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
        alive_color = np.array(art_kit.hex_to_rgb(art_kit.palette_color(0.98)), dtype=np.uint8)

        trail_idx = np.clip((decay * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = ramp[trail_idx]
        # Cells with negligible decay (never lived recently) fall back to bg.
        rgb[decay < 0.04] = bg
        # Currently-alive cells overwrite to the accent color.
        rgb[g == 1] = alive_color

        img = Image.fromarray(rgb, "RGB").resize((s, s), Image.NEAREST).convert("RGBA")
        canvas.commit(img)
