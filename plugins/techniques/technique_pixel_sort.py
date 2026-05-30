from plugins.BaseTechnique import BaseTechnique, Slider, Enum

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


class PixelSortTechnique(BaseTechnique):
    name = 'Pixel Sort'
    description = 'Sort pixels by luminance along rows or columns within luminance-threshold bands — the iconic Kim-Asendorf glitch.'
    kind = "filter"

    threshold = Slider(0.0, 1.0, default=0.45, step=0.05)
    direction = Enum([('row', 'Rows'), ('col', 'Columns')], default='row')
    mode      = Enum([('bright', 'Sort Bright'), ('dark', 'Sort Dark')], default='bright')

    def run(self, canvas):
        th = float(self.threshold)
        arr = canvas.image_array(mode="RGB", dtype="uint8")
        if self.direction == 'col':
            arr = arr.transpose(1, 0, 2)
        h, w, _ = arr.shape
        lum = (arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722) / 255.0
        mask = lum > th if self.mode == 'bright' else lum < th
        for y in range(h):
            row_mask = mask[y]
            if not row_mask.any():
                continue
            diff = np.diff(row_mask.astype(np.int8))
            starts = list(np.where(diff == 1)[0] + 1)
            ends = list(np.where(diff == -1)[0] + 1)
            if row_mask[0]:
                starts.insert(0, 0)
            if row_mask[-1]:
                ends.append(w)
            for s0, e0 in zip(starts, ends):
                if e0 - s0 < 2:
                    continue
                seg = arr[y, s0:e0]
                seg_lum = seg[:, 0] * 0.2126 + seg[:, 1] * 0.7152 + seg[:, 2] * 0.0722
                arr[y, s0:e0] = seg[np.argsort(seg_lum)]
        if self.direction == 'col':
            arr = arr.transpose(1, 0, 2)
        canvas.commit(Image.fromarray(arr, "RGB").convert("RGBA"))
