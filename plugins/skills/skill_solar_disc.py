from plugins.BaseSkill import BaseSkill

import math
import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class SolarDiscSkill(BaseSkill):
    name = 'Solar Disc'
    description = 'An abstract sun -- a radial-falloff core over an fbm field of palette-warmed rays. No literal circle, no line rays: a luminous gradient in the palette\'s brightest tones. Good for any "sun", "star", "radiant", "dawn", or "sunset" request.'
    kind = 'creation'
    owner = 'library'
    created_at = 1779667200.0
    hidden = False
    controls = [{'type': 'enum', 'name': 'mood', 'label': 'Mood', 'options': [{'value': 'calm', 'label': 'Calm'}, {'value': 'fierce', 'label': 'Fierce'}, {'value': 'eclipse', 'label': 'Eclipse'}], 'default': 'calm'}, {'type': 'palette', 'name': 'palette', 'label': 'Palette'}]

    def run(self, canvas, mood="calm", **_):
        s = int(canvas.size)
        seed = int(canvas.seed)
        contrast = {"calm": 0.45, "fierce": 0.95, "eclipse": 1.25}.get(str(mood), 0.45)
        core = {"calm": 0.55, "fierce": 0.45, "eclipse": 0.28}.get(str(mood), 0.55)

        cx, cy = s / 2.0, s / 2.0
        y_idx, x_idx = np.mgrid[0:s, 0:s].astype(np.float32)
        dx = x_idx - cx
        dy = y_idx - cy
        r = np.sqrt(dx * dx + dy * dy) / (s * 0.5)
        theta = np.arctan2(dy, dx)

        ANG = 720
        ang_lut = np.array(
            [art_kit.fbm(seed, math.cos(a) * 2.5, math.sin(a) * 2.5, octaves=5)
             for a in np.linspace(-math.pi, math.pi, ANG, endpoint=False)],
            dtype=np.float32,
        )
        ang_idx = ((theta + math.pi) / (2.0 * math.pi) * ANG).astype(np.int32) % ANG
        rays = ang_lut[ang_idx]

        falloff = np.clip(1.0 - r / core, 0.0, 1.0)
        falloff = falloff * falloff * (3.0 - 2.0 * falloff)

        outer_span = max(1e-6, 1.6 - core)
        corona = np.clip(1.0 - (r - core) / outer_span, 0.0, 1.0)
        corona = corona * (0.35 + contrast * rays)

        field = np.clip(falloff + corona, 0.0, 1.5)

        LUT = 256
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(k / (LUT - 1))) for k in range(LUT)],
            dtype=np.uint8,
        )
        idx = np.clip((field / 1.5 * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = lut[idx]

        canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
