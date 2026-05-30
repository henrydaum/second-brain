from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import math
import random
import numpy as np
from PIL import Image, ImageDraw

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

def _render_lsystem(canvas, s):
    sentence = art_kit.lindenmayer("F-G-G", {"F": "F-G+F+G-F", "G": "GG"}, 6)
    raw = art_kit.turtle_segments(sentence, start=(0.0, 0.0), heading=0.0,
                                  step=1.0, turn=math.radians(120.0))
    if not raw:
        return canvas.create_image()
    xs = [p for r in raw for p in (r[0], r[2])]
    ys = [p for r in raw for p in (r[1], r[3])]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    w = max(1e-9, x1 - x0)
    h = max(1e-9, y1 - y0)
    margin = s * 0.08
    avail = s - 2 * margin
    scale = avail / max(w, h)
    pad_x = (s - w * scale) * 0.5
    pad_y = (s - h * scale) * 0.5

    img = Image.new("RGBA", (s, s), canvas.palette.background)
    draw = ImageDraw.Draw(img, "RGBA")
    n = max(1, len(raw))
    line_w = max(2, int(s * 0.0025))
    for i, (sx, sy, ex, ey) in enumerate(raw):
        t = i / n
        color = art_kit.palette_color(0.2 + 0.7 * t)
        p1 = (pad_x + (sx - x0) * scale, pad_y + (sy - y0) * scale)
        p2 = (pad_x + (ex - x0) * scale, pad_y + (ey - y0) * scale)
        draw.line((p1, p2), fill=color, width=line_w)
    return img

def _render_chaos(canvas, s, seed):
    margin = s * 0.08
    inner = s - 2 * margin
    # Equilateral triangle vertices, pointing up.
    h = inner * math.sqrt(3) / 2.0
    cx = s / 2.0
    cy = (s + h) / 2.0  # baseline center y
    verts = np.array([
        [cx, cy - h],
        [cx - inner / 2.0, cy],
        [cx + inner / 2.0, cy],
    ], dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_points = 200_000
    choices = rng.integers(0, 3, size=n_points)
    px = np.empty(n_points, dtype=np.float64)
    py = np.empty(n_points, dtype=np.float64)
    x, y = float(rng.uniform(0, s)), float(rng.uniform(0, s))
    # 20-step burn-in.
    for i in range(20):
        v = verts[int(rng.integers(0, 3))]
        x, y = 0.5 * (x + v[0]), 0.5 * (y + v[1])
    for i in range(n_points):
        v = verts[choices[i]]
        x = 0.5 * (x + v[0])
        y = 0.5 * (y + v[1])
        px[i] = x
        py[i] = y

    ix = np.clip(px.astype(np.int32), 0, s - 1)
    iy = np.clip(py.astype(np.int32), 0, s - 1)
    density = np.zeros((s, s), dtype=np.float32)
    np.add.at(density, (iy, ix), 1.0)
    density = np.log1p(density)
    dmax = float(density.max()) or 1.0
    density = (density / dmax) ** 0.65

    LUT = 256
    lut = np.array(
        [art_kit.hex_to_rgb(art_kit.palette_color(0.15 + 0.8 * (k / (LUT - 1))))
         for k in range(LUT)],
        dtype=np.uint8,
    )
    bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
    idx = np.clip((density * (LUT - 1)).astype(np.int32), 0, LUT - 1)
    rgb = lut[idx]
    mask = density < 0.02
    rgb[mask] = bg
    return Image.fromarray(rgb, "RGB").convert("RGBA")

def _render_carpet(canvas, s):
    # Render at a smaller integer grid then upscale, so depth-5 (243 cells)
    # maps cleanly onto pixels.
    depth = 5
    g = 3 ** depth  # 243
    grid = np.ones((g, g), dtype=np.float32)
    # For each level, blank out the center third of each remaining solid block.
    size = g
    for d in range(depth):
        step = 3 ** (depth - d - 1)
        # Walk every cell at this level: a cell at (r, c) corresponds to
        # the block [r*step:(r+1)*step, c*step:(c+1)*step] in grid.
        n_cells = 3 ** (d + 1)
        # The "punched" cells in this level are those where both r%3==1 and c%3==1.
        for r in range(n_cells):
            if r % 3 != 1:
                continue
            for c in range(n_cells):
                if c % 3 != 1:
                    continue
                # But only punch if the surrounding 3x3 block at the previous
                # level was still solid -- i.e. its parent cell was solid.
                pr, pc = r // 3, c // 3
                parent_step = 3 ** (depth - d)
                if grid[pr * parent_step, pc * parent_step] > 0:
                    grid[r * step:(r + 1) * step, c * step:(c + 1) * step] = 0.0

    # Add a soft palette ramp by radius so the carpet isn't single-tone.
    ys, xs = np.mgrid[0:g, 0:g].astype(np.float32)
    rad = np.sqrt((xs - g / 2.0) ** 2 + (ys - g / 2.0) ** 2)
    rad /= rad.max() or 1.0
    t_field = 0.25 + 0.65 * rad
    LUT = 256
    lut = np.array(
        [art_kit.hex_to_rgb(art_kit.palette_color(k / (LUT - 1))) for k in range(LUT)],
        dtype=np.uint8,
    )
    rgb = lut[np.clip((t_field * (LUT - 1)).astype(np.int32), 0, LUT - 1)]
    bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
    rgb[grid < 0.5] = bg

    img = Image.fromarray(rgb, "RGB").resize((s, s), Image.NEAREST).convert("RGBA")
    return img


class SierpinskiTriangleTechnique(BaseTechnique):
    name = 'Sierpinski Triangle'
    description = 'The Sierpinski gasket -- the same self-similar shape rendered two completely different ways, both demonstrated side by side as named presets. The L-system preset uses F->F-G+F+G-F at 120 degrees to trace the gasket as a single turtle path. The chaos-game preset throws 200,000 random points at the midpoint-toward-a-random-vertex rule and accumulates them into a palette-graded density buffer. The carpet preset trades triangles for squares (the 2D Cantor carpet, also Sierpinski). Good for "sierpinski", "gasket", "triangle", "chaos game", "carpet", or "self-similar".'
    kind = "background"
    palette = Palette()
    method = Enum([('lsystem', 'L-system Turtle'), ('chaos_game', 'Chaos Game'), ('carpet', 'Sierpinski Carpet')], default='chaos_game')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        self.method = str(self.method)
        if self.method == "lsystem":
            out = _render_lsystem(canvas, s)
        elif self.method == "carpet":
            out = _render_carpet(canvas, s)
        else:
            out = _render_chaos(canvas, s, seed)
        canvas.commit(out)
