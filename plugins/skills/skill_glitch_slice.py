from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class GlitchSliceSkill(BaseSkill):
    name = 'Glitch Slice'
    description = 'Split the image into horizontal bands and shift each by a random offset — the classic data-mosh glitch. Determinism comes from canvas.seed so the glitch is the same across palette swaps. Params: intensity (0.0-1.0, default 0.4), bands (4-80, default 24).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'intensity', 'label': 'Intensity', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'default': 0.4},
        {'type': 'slider', 'name': 'bands', 'label': 'Bands', 'min': 4, 'max': 80, 'step': 1, 'default': 24},
    ]

    def run(self, canvas, intensity=0.4, bands=24):
        intensity = float(art_kit.clamp(intensity, 0.0, 1.5))
        n_bands = int(art_kit.clamp(bands, 2, 200))
        img = canvas.image.convert("RGB")
        s = canvas.size
        arr = np.asarray(img, dtype=np.uint8).copy()
        rng = np.random.default_rng(canvas.seed)
        max_shift = int(s * 0.5 * intensity)
        edges = np.linspace(0, s, n_bands + 1, dtype=np.int32)
        for i in range(n_bands):
            y0, y1 = edges[i], edges[i + 1]
            if y1 <= y0:
                continue
            shift = int(rng.integers(-max_shift, max_shift + 1)) if max_shift > 0 else 0
            arr[y0:y1] = np.roll(arr[y0:y1], shift=shift, axis=1)
            # Occasional channel break: swap R/B on this band.
            if rng.random() < 0.10 * intensity:
                arr[y0:y1, :, [0, 2]] = arr[y0:y1, :, [2, 0]]
        canvas.commit(Image.fromarray(arr, "RGB").convert("RGBA"))
