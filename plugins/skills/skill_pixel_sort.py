from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class PixelSortSkill(BaseSkill):
    name = 'Pixel Sort'
    description = 'Sort pixels by luminance along rows or columns within luminance-threshold bands — the iconic Kim-Asendorf glitch. Bright streaks rearrange into smooth gradients, dark regions stay intact. Params: threshold (0.0-1.0, default 0.45), direction enum (row/col), mode enum (sort bright pixels vs dark).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'threshold', 'label': 'Threshold', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'default': 0.45},
        {'type': 'enum', 'name': 'direction', 'label': 'Direction', 'options': [
            {'value': 'row', 'label': 'Rows'},
            {'value': 'col', 'label': 'Columns'},
        ], 'default': 'row'},
        {'type': 'enum', 'name': 'mode', 'label': 'Mode', 'options': [
            {'value': 'bright', 'label': 'Sort Bright'},
            {'value': 'dark', 'label': 'Sort Dark'},
        ], 'default': 'bright'},
    ]

    def run(self, canvas, threshold=0.45, direction='row', mode='bright'):
        th = float(art_kit.clamp(threshold, 0.0, 1.0))
        img = canvas.image.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8).copy()
        if str(direction) == 'col':
            arr = arr.transpose(1, 0, 2)
        h, w, _ = arr.shape
        lum = (arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722) / 255.0
        if str(mode) == 'bright':
            mask = lum > th
        else:
            mask = lum < th
        for y in range(h):
            row_mask = mask[y]
            if not row_mask.any():
                continue
            # Find contiguous runs.
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
                order = np.argsort(seg_lum)
                arr[y, s0:e0] = seg[order]
        if str(direction) == 'col':
            arr = arr.transpose(1, 0, 2)
        canvas.commit(Image.fromarray(arr, "RGB").convert("RGBA"))
