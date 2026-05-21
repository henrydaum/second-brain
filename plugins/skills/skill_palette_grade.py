from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class PaletteGradeSkill(BaseSkill):
    name = 'Palette Grade'
    description = 'Map luminance through the canvas palette ramp for a cohesive tonal feel. The single most-useful post-process — call after any creation skill. Params: mix (0.0-1.0, default 0.66) - how strongly to push toward the palette.'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'palette', 'name': 'palette', 'label': 'Palette'},
        {'type': 'slider', 'name': 'mix', 'label': 'Mix', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'default': 0.66},
    ]

    def run(self, canvas, mix=0.66):
        mix = float(art_kit.clamp(mix, 0.0, 1.0))
        img = canvas.image.convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
        lo = float(np.percentile(lum, 2))
        hi = float(np.percentile(lum, 99))
        lum = np.clip((lum - lo) / max(1e-6, hi - lo), 0.0, 1.0)

        # Sample five stops along the palette ramp.
        stops = np.array([
            art_kit.hex_to_rgb(art_kit.palette_color(i / 4.0)) for i in range(5)
        ], dtype=np.float32) / 255.0

        x = lum * (len(stops) - 1)
        i = np.clip(x.astype(np.int32), 0, len(stops) - 2)
        f = (x - i)[..., None]
        mapped = stops[i] * (1.0 - f) + stops[i + 1] * f
        out = arr * (1.0 - mix) + mapped * mix
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
