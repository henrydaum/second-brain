from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class ElementaryCaTechnique(BaseTechnique):
    name = 'Elementary CA'
    description = 'Wolfram\'s 1D cellular automata: each row is the next time step of the row above, computed from a 2-state, 3-cell neighborhood according to a numbered rule. Rule 30 is chaos; rule 90 grows the Sierpinski triangle from a single cell; rule 110 is Turing-complete and weaves tangled gliders; rule 184 models traffic flow. The full evolution is rendered as a palette-graded image -- live cells warm, dead cells background. Good for "cellular automata", "wolfram", "rule 30", "rule 90", "emergence", or any algorithmic-evolution motif.'
    kind = "background"
    palette = Palette()
    rule = Enum([('30', 'Rule 30 (chaos)'), ('90', 'Rule 90 (Sierpinski)'), ('110', 'Rule 110 (complex)'), ('73', 'Rule 73 (crystal)'), ('184', 'Rule 184 (traffic)'), ('150', 'Rule 150 (XOR weave)')], default='110')

    def run(self, canvas):
        s = int(canvas.size)
        try:
            rule_num = int(str(self.rule))
        except (TypeError, ValueError):
            rule_num = 110
        rule_num &= 0xFF

        bits = np.array([(rule_num >> i) & 1 for i in range(8)], dtype=np.uint8)

        # One row per pixel row. width = canvas size so each cell maps to one pixel.
        width = s
        steps = s
        grid = np.zeros((steps, width), dtype=np.uint8)
        # Start from a single center cell for the rules whose canonical images
        # rely on it (90, 30, 150). Rule 110 and rule 184 also look fine from a
        # single seed at this resolution.
        grid[0, width // 2] = 1
        # For chaotic rules, a tiny noisy seed at the top edges gives more visual
        # variety. Use a deterministic per-seed RNG.
        if rule_num in (110, 184, 73):
            rng = np.random.default_rng(int(canvas.seed))
            edge = max(1, width // 30)
            grid[0, :edge] = rng.integers(0, 2, size=edge)
            grid[0, -edge:] = rng.integers(0, 2, size=edge)

        for t in range(1, steps):
            row = grid[t - 1]
            # neighborhood index: left<<2 | center<<1 | right.
            n = (np.roll(row, 1) << 2) | (row << 1) | np.roll(row, -1)
            grid[t] = bits[n]

        # Map cells through the palette ramp. We modulate intensity slightly by
        # local neighbor density so big solid blocks vary subtly instead of being
        # flat color -- avoids the "flat poster" look.
        # Local density: 5-cell wide horizontal average, kept cheap with numpy.
        pad = np.pad(grid.astype(np.float32), ((0, 0), (2, 2)), mode="wrap")
        density = (pad[:, :-4] + pad[:, 1:-3] + pad[:, 2:-2] + pad[:, 3:-1] + pad[:, 4:]) / 5.0

        LUT = 256
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(k / (LUT - 1))) for k in range(LUT)],
            dtype=np.uint8,
        )
        bg_t = 0.05
        fg_lo, fg_hi = 0.45, 0.95
        t_field = np.where(grid > 0, fg_lo + (fg_hi - fg_lo) * density, bg_t).astype(np.float32)
        idx = np.clip((t_field * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = lut[idx]

        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
        rgb[grid == 0] = bg

        canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
