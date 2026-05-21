from plugins.BaseSkill import BaseSkill

import math
import numpy as np

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class KaleidoscopeSkill(BaseSkill):
    name = 'Kaleidoscope'
    description = 'Fold N angular wedges around the center; the result is N-fold rotational symmetry from a single source slice. Classic toy. Params: segments (3-24, default 8), rotation (0-360, default 0).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'segments', 'label': 'Segments', 'min': 3, 'max': 24, 'step': 1, 'default': 8},
        {'type': 'slider', 'name': 'rotation', 'label': 'Rotation', 'min': 0, 'max': 360, 'step': 5, 'default': 0},
    ]

    def run(self, canvas, segments=8, rotation=0):
        n = int(art_kit.clamp(segments, 2, 36))
        rot = math.radians(float(art_kit.clamp(rotation, 0, 360)))
        arr = canvas.image_array(mode="RGB", dtype="float")
        s = canvas.size
        xx, yy, _, _ = art_kit.centered_grid(s)
        cx = (s - 1) / 2.0
        dx = xx - cx
        dy = yy - cx
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx) - rot
        wedge = 2.0 * math.pi / n
        t = np.mod(theta, 2.0 * wedge)
        t = np.where(t > wedge, 2.0 * wedge - t, t)
        sx = cx + np.cos(t + rot) * r
        sy = cx + np.sin(t + rot) * r
        canvas.commit_array(art_kit.bilinear_sample(arr, sx, sy))
