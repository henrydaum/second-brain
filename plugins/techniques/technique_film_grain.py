from plugins.BaseTechnique import BaseTechnique, Slider, Bool

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


class FilmGrainTechnique(BaseTechnique):
    name = 'Film Grain'
    description = 'Deterministic per-pixel noise overlay seeded from canvas.seed. Adds tactile texture; great over flat palette grades.'
    kind = "filter"

    intensity  = Slider(0.0, 0.4, default=0.07, step=0.005)
    monochrome = Bool(default=True)

    def run(self, canvas):
        s = canvas.size
        rng = np.random.default_rng(canvas.seed)
        arr = canvas.image_array(mode="RGB", dtype="float")
        if self.monochrome:
            noise = rng.standard_normal((s, s, 1)).astype(np.float32) * float(self.intensity)
        else:
            noise = rng.standard_normal((s, s, 3)).astype(np.float32) * float(self.intensity)
        canvas.commit_array(np.clip(arr + noise, 0.0, 1.0))
