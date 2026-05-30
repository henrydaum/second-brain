from plugins.BaseTechnique import BaseTechnique, Slider, Enum

import numpy as np
from PIL import Image, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class PencilSketchTechnique(BaseTechnique):
    name = "Pencil Sketch"
    description = "Classic dodge-and-burn pencil sketch filter. Inverts and blurs luminance, color-dodges against the original to extract pencil-like lines, then composites over a paper tint with optional grain."
    kind = "filter"
    strength = Slider(0.3, 1.0, default=0.85, step=0.05)
    edge_weight = Slider(0.0, 1.0, default=0.45, step=0.05)
    grain = Slider(0.0, 1.0, default=0.35, step=0.05)
    paper = Enum([("white", "White"), ("cream", "Cream"), ("gray", "Soft Gray")], default="cream")

    def run(self, canvas):
        s = canvas.size
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722

        # Color dodge: inverted + blurred / original.
        inv = 1.0 - lum
        inv_img = Image.fromarray(np.clip(inv * 255, 0, 255).astype(np.uint8), "L")
        blur_radius = max(2.0, s * 0.012)
        blurred = np.asarray(
            inv_img.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
            dtype=np.float32,
        ) / 255.0
        denom = np.clip(1.0 - blurred, 1e-3, 1.0)
        sketch = np.clip(lum / denom, 0.0, 1.0)

        # Sobel edges for emphasis.
        gx = np.zeros_like(lum)
        gy = np.zeros_like(lum)
        gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
        gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
        edge = np.sqrt(gx * gx + gy * gy)
        edge = np.clip(edge * 2.0, 0.0, 1.0)
        edge_dark = 1.0 - edge * float(self.edge_weight)

        out = sketch * edge_dark
        # Blend with original luminance by strength (low strength = more original).
        st = float(self.strength)
        out = out * st + lum * (1.0 - st)

        # Grain.
        if float(self.grain) > 0.01:
            yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
            n = art_kit.value_noise_grid(canvas.seed, xx * 0.5, yy * 0.5).astype(np.float32)
            out = out + (n - 0.5) * float(self.grain) * 0.15
            out = np.clip(out, 0.0, 1.0)

        paper = {"white": (1.0, 1.0, 1.0), "cream": (0.96, 0.93, 0.85), "gray": (0.88, 0.88, 0.88)}[str(self.paper)]
        rgb = np.stack([out * paper[0], out * paper[1], out * paper[2]], axis=-1)
        canvas.commit_array(rgb)
