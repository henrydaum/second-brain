from plugins.BaseTechnique import BaseTechnique, Slider, Enum

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class Anaglyph3dTechnique(BaseTechnique):
    name = 'Anaglyph 3d'
    description = '3D-glasses look. Takes two horizontally shifted copies of the image and combines them into one anaglyph — wear red/cyan glasses to see (pseudo-)depth.'
    kind = "filter"

    offset = Slider(1, 40, default=10, step=1)
    mode   = Enum([
        ('red_cyan',     'Red / Cyan'),
        ('red_blue',     'Red / Blue'),
        ('green_magenta','Green / Magenta'),
    ], default='red_cyan')

    def run(self, canvas):
        d = int(self.offset)
        arr = canvas.image_array(mode="RGB", dtype="float")
        left = np.roll(arr, shift=-d, axis=1)
        right = np.roll(arr, shift=d, axis=1)
        out = np.zeros_like(arr)
        if self.mode == 'red_cyan':
            out[..., 0] = left[..., 0]
            out[..., 1] = right[..., 1]
            out[..., 2] = right[..., 2]
        elif self.mode == 'red_blue':
            out[..., 0] = left[..., 0]
            out[..., 1] = (left[..., 1] + right[..., 1]) * 0.5
            out[..., 2] = right[..., 2]
        else:
            out[..., 0] = right[..., 0]
            out[..., 1] = left[..., 1]
            out[..., 2] = right[..., 2]
        canvas.commit_array(out)
