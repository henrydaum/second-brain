from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


class DitherTechnique(BaseTechnique):
    name = 'Dither'
    description = 'Quantize the canvas to a few palette tones with classic dithering: ordered Bayer crosshatch, or Floyd-Steinberg and Atkinson error diffusion for that 1-bit / retro print look. Luminance is reduced to a handful of levels and remapped onto the palette ramp, so the result stays on-palette. Good for "dither", "1-bit", "retro", "newsprint", "Atkinson", "Floyd-Steinberg", or a low-color crosshatch.'
    kind = "filter"

    palette = Palette()
    mode    = Enum([('bayer', 'Ordered (Bayer)'), ('floyd', 'Floyd-Steinberg'), ('atkinson', 'Atkinson')], default='bayer')
    levels  = Slider(2, 8, default=4, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        L = int(self.levels)
        mode = str(self.mode)
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)

        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(k / 255.0)) for k in range(256)],
            dtype=np.uint8,
        )

        if mode == "bayer":
            def bayer(n):
                if n == 1:
                    return np.array([[0.0]], dtype=np.float32)
                m = bayer(n // 2)
                return np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
            b = bayer(8) / 64.0
            tile = np.tile(b, (s // 8 + 1, s // 8 + 1))[:s, :s]
            v = lum + (tile - 0.5) / L
            q = np.clip(np.round(v * (L - 1)) / (L - 1), 0.0, 1.0)
        else:
            D = min(s, 320)
            small = np.asarray(
                Image.fromarray((lum * 255).astype(np.uint8)).resize((D, D), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            q_small = np.zeros((D, D), dtype=np.float32)
            for y in range(D):
                for x in range(D):
                    old = small[y, x]
                    new = round(old * (L - 1)) / (L - 1)
                    q_small[y, x] = new
                    err = old - new
                    if mode == "floyd":
                        if x + 1 < D: small[y, x + 1] += err * 7 / 16
                        if y + 1 < D:
                            if x > 0: small[y + 1, x - 1] += err * 3 / 16
                            small[y + 1, x] += err * 5 / 16
                            if x + 1 < D: small[y + 1, x + 1] += err * 1 / 16
                    else:  # atkinson
                        e = err / 8.0
                        if x + 1 < D: small[y, x + 1] += e
                        if x + 2 < D: small[y, x + 2] += e
                        if y + 1 < D:
                            if x > 0: small[y + 1, x - 1] += e
                            small[y + 1, x] += e
                            if x + 1 < D: small[y + 1, x + 1] += e
                        if y + 2 < D: small[y + 2, x] += e
            q = np.asarray(
                Image.fromarray(np.clip(q_small * 255, 0, 255).astype(np.uint8)).resize((s, s), Image.NEAREST),
                dtype=np.float32,
            ) / 255.0

        idx = np.clip((q * 255).astype(np.int32), 0, 255)
        canvas.commit_array(lut[idx])
