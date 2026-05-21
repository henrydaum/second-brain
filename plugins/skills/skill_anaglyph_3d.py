from plugins.BaseSkill import BaseSkill

import numpy as np

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class Anaglyph3dSkill(BaseSkill):
    name = 'Anaglyph 3d'
    description = '3D-glasses look. Takes two horizontally shifted copies of the image and combines them into one anaglyph: one channel from the left-eye shift, the other channels from the right-eye shift. Wear red/cyan glasses to see (pseudo-)depth. Params: offset (1-40 px, default 10), mode enum (red/cyan, red/blue, green/magenta).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'offset', 'label': 'Offset', 'min': 1, 'max': 40, 'step': 1, 'default': 10},
        {'type': 'enum', 'name': 'mode', 'label': 'Mode', 'options': [
            {'value': 'red_cyan', 'label': 'Red / Cyan'},
            {'value': 'red_blue', 'label': 'Red / Blue'},
            {'value': 'green_magenta', 'label': 'Green / Magenta'},
        ], 'default': 'red_cyan'},
    ]

    def run(self, canvas, offset=10, mode='red_cyan'):
        d = int(art_kit.clamp(offset, 0, 80))
        mode = str(mode)
        arr = canvas.image_array(mode="RGB", dtype="float")
        left = np.roll(arr, shift=-d, axis=1)
        right = np.roll(arr, shift=d, axis=1)
        out = np.zeros_like(arr)
        if mode == 'red_cyan':
            out[..., 0] = left[..., 0]
            out[..., 1] = right[..., 1]
            out[..., 2] = right[..., 2]
        elif mode == 'red_blue':
            out[..., 0] = left[..., 0]
            out[..., 1] = (left[..., 1] + right[..., 1]) * 0.5
            out[..., 2] = right[..., 2]
        else:  # green_magenta
            out[..., 0] = right[..., 0]
            out[..., 1] = left[..., 1]
            out[..., 2] = right[..., 2]
        canvas.commit_array(out)
