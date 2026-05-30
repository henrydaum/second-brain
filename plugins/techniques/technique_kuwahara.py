from plugins.BaseTechnique import BaseTechnique, Slider

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class KuwaharaTechnique(BaseTechnique):
    name = 'Kuwahara Oil Paint'
    description = 'Kuwahara filter: an edge-preserving smoothing that turns the canvas painterly. Each pixel looks at four overlapping quadrant windows and adopts the average color of whichever is flattest (lowest variance), so flat areas blur into oil-paint blobs while edges stay crisp. Good for "oil painting", "Kuwahara", "painterly", "smudge", "brushstrokes", or a soft posterized repaint.'
    kind = "filter"

    radius = Slider(2, 9, default=4, step=1)

    def run(self, canvas):
        r = int(self.radius)
        arr = canvas.image_array(mode="RGB", dtype="float")   # (H, W, 3) in [0, 1]
        H, W = arr.shape[:2]
        lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

        def integ(a):
            S = np.zeros((H + 1, W + 1), dtype=np.float64)
            S[1:, 1:] = a.cumsum(0).cumsum(1)
            return S

        S_ch = [integ(arr[..., c]) for c in range(3)]
        S_l = integ(lum)
        S_l2 = integ(lum * lum)

        yy, xx = np.mgrid[0:H, 0:W]

        def boxsum(S, r0, r1, c0, c1):
            return (S[r1 + 1, c1 + 1] - S[r0, c1 + 1] - S[r1 + 1, c0] + S[r0, c0])

        def quadrant(r0, r1, c0, c1):
            count = ((r1 - r0 + 1) * (c1 - c0 + 1)).astype(np.float64)
            ml = boxsum(S_l, r0, r1, c0, c1) / count
            ml2 = boxsum(S_l2, r0, r1, c0, c1) / count
            var = np.maximum(ml2 - ml * ml, 0.0)
            means = np.stack([boxsum(S_ch[c], r0, r1, c0, c1) / count for c in range(3)], axis=-1)
            return var, means

        y0 = np.clip(yy - r, 0, H - 1)
        y1 = np.clip(yy + r, 0, H - 1)
        x0 = np.clip(xx - r, 0, W - 1)
        x1 = np.clip(xx + r, 0, W - 1)

        quads = [
            quadrant(y0, yy, x0, xx),   # top-left
            quadrant(y0, yy, xx, x1),   # top-right
            quadrant(yy, y1, x0, xx),   # bottom-left
            quadrant(yy, y1, xx, x1),   # bottom-right
        ]
        var_stack = np.stack([q[0] for q in quads], axis=0)     # (4, H, W)
        mean_stack = np.stack([q[1] for q in quads], axis=0)    # (4, H, W, 3)
        pick = np.argmin(var_stack, axis=0)                      # (H, W)
        out = np.take_along_axis(mean_stack, pick[None, ..., None], axis=0)[0]

        canvas.commit_array(np.clip(out, 0.0, 1.0))
