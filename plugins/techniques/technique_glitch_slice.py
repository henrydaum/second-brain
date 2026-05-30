from plugins.BaseTechnique import BaseTechnique, Slider

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


class GlitchSliceTechnique(BaseTechnique):
    name = 'Glitch Slice'
    description = 'Split the image into horizontal bands and shift each by a random offset — the classic data-mosh glitch. Deterministic from canvas.seed.'
    kind = "filter"

    intensity = Slider(0.0, 1.0, default=0.4, step=0.05)
    bands     = Slider(4, 80, default=24, step=1)

    def run(self, canvas):
        intensity = float(self.intensity)
        n_bands = int(self.bands)
        s = canvas.size
        arr = canvas.image_array(mode="RGB", dtype="uint8")
        rng = np.random.default_rng(canvas.seed)
        max_shift = int(s * 0.5 * intensity)
        edges = np.linspace(0, s, n_bands + 1, dtype=np.int32)
        for i in range(n_bands):
            y0, y1 = edges[i], edges[i + 1]
            if y1 <= y0:
                continue
            shift = int(rng.integers(-max_shift, max_shift + 1)) if max_shift > 0 else 0
            arr[y0:y1] = np.roll(arr[y0:y1], shift=shift, axis=1)
            if rng.random() < 0.10 * intensity:
                arr[y0:y1, :, [0, 2]] = arr[y0:y1, :, [2, 0]]
        canvas.commit(Image.fromarray(arr, "RGB").convert("RGBA"))
