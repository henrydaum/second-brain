from plugins.BaseTechnique import BaseTechnique, Slider

import math
import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class MotionBlurTechnique(BaseTechnique):
    name = 'Motion Blur'
    description = 'Directional blur along an angle — fakes camera shake or fast motion.'
    kind = "filter"

    length = Slider(3, 60, default=18, step=1)
    angle  = Slider(0, 360, default=0, step=5)

    def run(self, canvas):
        n = int(self.length)
        a = math.radians(float(self.angle))
        arr = canvas.image_array(mode="RGB", dtype="float") * 255.0
        dx = math.cos(a)
        dy = math.sin(a)
        acc = np.zeros_like(arr)
        for i in range(n):
            t = i - (n - 1) / 2.0
            sx = int(round(t * dx))
            sy = int(round(t * dy))
            acc += np.roll(arr, shift=(sy, sx), axis=(0, 1))
        canvas.commit_array(acc / max(n, 1) / 255.0)
