SKILL_NAME = "L-System Grove"
SKILL_DESCRIPTION = "Trees as Lindenmayer systems: a stochastic rule rewrites a string into branching turtle paths, drawn with palette-graded line thickness. Several seeded trees clustered into a grove with fbm sky behind. No leaves, no trunks -- just branching grammar. Good for \"tree\", \"forest\", \"grove\", \"branches\", \"fern\", \"coral\", or \"lightning\"."
SKILL_KIND = "creation"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1779667200.0
SKILL_HIDDEN = False
SKILL_CONTROLS = [
    {"type": "enum", "name": "shape", "label": "Shape",
     "options": [
         {"value": "tree", "label": "Tree"},
         {"value": "fern", "label": "Fern"},
         {"value": "coral", "label": "Coral"},
     ], "default": "tree"},
    {"type": "palette", "name": "palette", "label": "Palette"},
]

import math
import random
import numpy as np
from PIL import Image, ImageDraw


_GRAMMARS = {
    "tree":  {"axiom": "F", "rules": {"F": "FF+[+F-F-F]-[-F+F+F]"}, "iter": 4, "angle": 22.5},
    "fern":  {"axiom": "X", "rules": {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"}, "iter": 5, "angle": 25.0},
    "coral": {"axiom": "F", "rules": {"F": "F[+F]F[-F][F]"}, "iter": 4, "angle": 28.0},
}


def run(canvas, shape="tree", **_):
    s = int(canvas.size)
    seed = int(canvas.seed)
    rng = random.Random(seed)
    g = _GRAMMARS.get(str(shape), _GRAMMARS["tree"])

    # Sky background: fbm sampled on a coarse grid then upscaled.
    LOW = 96
    low = np.array(
        [[art_kit.fbm(seed + 17, x * 0.06, y * 0.06, octaves=3) for x in range(LOW)]
         for y in range(LOW)],
        dtype=np.float32,
    )
    low_img = Image.fromarray(np.clip(low * 255.0, 0, 255).astype(np.uint8), "L").resize((s, s), Image.BICUBIC)
    noise = np.asarray(low_img, dtype=np.float32) / 255.0
    y_idx, _ = np.mgrid[0:s, 0:s].astype(np.float32)
    t = y_idx / s
    field = np.clip(t * 0.85 + (noise - 0.5) * 0.25, 0.0, 1.0)
    LUT = 256
    sky_lut = np.array(
        [art_kit.hex_to_rgb(art_kit.mix_hex(
            art_kit.palette_color(0.9), art_kit.palette_color(0.45),
            k / (LUT - 1))) for k in range(LUT)],
        dtype=np.uint8,
    )
    rgb = sky_lut[np.clip((field * (LUT - 1)).astype(np.int32), 0, LUT - 1)]
    img = Image.fromarray(rgb, "RGB").convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    n_trees = 5
    for ti in range(n_trees):
        sentence = art_kit.lindenmayer(g["axiom"], g["rules"], g["iter"])
        scale = 0.55 + rng.random() * 0.55
        base_x = s * (0.12 + 0.18 * ti) + rng.uniform(-s * 0.04, s * 0.04)
        base_y = s * (0.92 + rng.uniform(-0.03, 0.02))
        step = (s * 0.012) * scale
        turn = math.radians(g["angle"] + rng.uniform(-3.0, 3.0))
        segments = art_kit.turtle_segments(
            sentence, start=(base_x, base_y), heading=-math.pi / 2.0,
            step=step, turn=turn,
        )
        # Color: trunks darker (palette_color near 0.1), tips brighter.
        n_seg = max(1, len(segments))
        for si, (x1, y1, x2, y2) in enumerate(segments):
            t_seg = si / n_seg
            color = art_kit.palette_color(0.1 + 0.45 * t_seg)
            width = max(1, int((1.0 - t_seg) * (4.0 * scale) + 1.0))
            draw.line((x1, y1, x2, y2), fill=color, width=width)

    canvas.commit(img)
