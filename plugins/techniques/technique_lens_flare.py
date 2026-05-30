from plugins.BaseTechnique import BaseTechnique, Slider, Pan, Palette

import numpy as np
from PIL import Image, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class LensFlareTechnique(BaseTechnique):
    name = 'Lens Flare'
    description = 'Photographic lens flare built from real light: a tight palette-tinted core with a 1/r-squared falloff and broad halo, an anamorphic horizontal streak, and a string of soft ghost discs and iris rings marching along the source-through-center axis with a chromatic fringe. All screen-blended (additive light) onto the canvas so it glows instead of painting over. Good for "lens flare", "sun glare", "bloom", "light leak", or a bright focal accent.'
    kind = "filter"

    palette    = Palette()
    brightness = Slider(0.0, 1.5, default=0.9, step=0.05)
    sx         = Slider(0.0, 1.0, default=0.30, step=0.05)
    sy         = Slider(0.0, 1.0, default=0.28, step=0.05)
    source     = Pan(x='sx', y='sy')
    ghosts     = Slider(0, 8, default=5, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        b = float(self.brightness)
        base = canvas.image_array(mode="RGB", dtype="float")  # (s, s, 3) in [0, 1]

        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        px, py = float(self.sx) * s, float(self.sy) * s
        cx = cy = s / 2.0

        def tint(t, value=1.0):
            return np.array(
                art_kit.hex_to_rgb(art_kit.palette_color(t, value)), dtype=np.float32
            ) / 255.0

        flare = np.zeros((s, s, 3), dtype=np.float32)

        # --- Core glow: a very tight core, a medium pool, and a wide soft halo,
        #     each tinted at a different point on the bright end of the ramp.
        r = np.hypot(xx - px, yy - py) / s
        core = 1.0 / (1.0 + 1400.0 * r * r)
        pool = 1.0 / (1.0 + 90.0 * r * r)
        halo = np.exp(-((r / 0.30) ** 2))
        flare += core[..., None] * tint(0.98) * (1.2 * b)
        flare += pool[..., None] * tint(0.82) * (0.55 * b)
        flare += halo[..., None] * tint(0.60) * (0.45 * b)

        # --- Anamorphic streak: bright horizontal bar (+ faint vertical cross),
        #     a smooth gaussian falloff rather than a flat line.
        dx = (xx - px) / s
        dy = (yy - py) / s
        h_streak = np.exp(-((dy / 0.006) ** 2)) * np.exp(-((dx / 0.55) ** 2))
        v_streak = np.exp(-((dx / 0.004) ** 2)) * np.exp(-((dy / 0.40) ** 2))
        flare += (h_streak + 0.5 * v_streak)[..., None] * tint(0.92) * (0.8 * b)

        # --- Ghosts: soft discs / iris rings spaced along the source -> center ->
        #     beyond axis, each channel nudged for a spectral fringe.
        n = int(self.ghosts)
        vx, vy = (cx - px), (cy - py)
        axis_len = (vx * vx + vy * vy) ** 0.5 or 1.0
        ax, ay = vx / axis_len, vy / axis_len
        for i in range(1, n + 1):
            ratio = i / float(n + 1)
            k = ratio * 2.6 - 0.3                       # spread before & past center
            gx, gy = px + vx * k * 2.0, py + vy * k * 2.0
            gr = s * (0.012 + 0.055 * (2.0 * abs(0.5 - ratio)))
            col = tint(0.2 + 0.7 * ((i * 0.37) % 1.0))
            is_ring = (i % 2 == 0)
            for ch, off in enumerate((-1.0, 0.0, 1.0)):
                ox = gx + ax * off * (gr * 0.18)
                oy = gy + ay * off * (gr * 0.18)
                dd = np.hypot(xx - ox, yy - oy)
                shape = np.exp(-((dd / gr) ** 2))
                if is_ring:
                    ring = np.exp(-(((dd - gr) / (gr * 0.30)) ** 2))
                    shape = shape * 0.25 + ring
                flare[..., ch] += shape * col[ch] * (0.5 * b * (1.0 - 0.35 * ratio))

        # Mild blur on the flare layer only, then screen-blend over the base.
        flare_u8 = np.clip(flare * 255.0, 0, 255).astype(np.uint8)
        flare_img = Image.fromarray(flare_u8, "RGB").filter(
            ImageFilter.GaussianBlur(max(1.0, s * 0.0016))
        )
        flare = np.asarray(flare_img, dtype=np.float32) / 255.0

        out = 1.0 - (1.0 - base) * (1.0 - np.clip(flare, 0.0, 1.0))
        canvas.commit_array(out)
