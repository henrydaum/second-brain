from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette

import random
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class LetterFieldTechnique(BaseTechnique):
    name = "Letter Field"
    description = "Homage to Judson Rosebush's 1978 Letter Field: large overlapping colored capitals with airy background letters on a warm paper ground. Built-in Jost typeface."
    kind = "background"
    palette = Palette()
    front_count = Slider(3, 14, default=7, step=1, label="Front Letters")
    back_count = Slider(0, 60, default=22, step=1, label="Back Letters")
    bg = Enum([("ivory", "Ivory"), ("palette_bg", "Soft Palette BG"), ("black", "Soft Black")], default="ivory")

    def run(self, canvas):
        s = canvas.size
        rng = random.Random(canvas.seed)

        def lum(hex_color):
            r, g, b = art_kit.hex_to_rgb(hex_color)
            return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        bg = {"ivory": "#f5efe2", "black": "#24201c", "palette_bg": canvas.palette.background}[str(self.bg)]
        if str(self.bg) == "palette_bg" and lum(bg) < 0.68:
            bg = art_kit.mix_hex(bg, "#f5efe2", 0.72)
        img = Image.new("RGBA", (s, s), bg)

        # Back layer: small faded letters on a loose jittered grid.
        n_back = int(self.back_count)
        cols = max(2, int(n_back ** 0.5) + 1)
        back_pts = art_kit.jittered_grid(rng, cols, max(2, (n_back + cols - 1) // cols), jitter=0.75)
        for i, (px, py) in enumerate(back_pts[:n_back]):
            ch = rng.choice(_LETTERS)
            size = int(rng.uniform(s * 0.05, s * 0.18))
            x = int(s * (0.08 + 0.84 * px))
            y = int(s * (0.08 + 0.84 * py))
            t = rng.random()
            base = art_kit.palette_color(0.25 + 0.7 * t)
            color = art_kit.with_alpha(base, 0.22)
            art_kit.text(
                img, (x, y), ch,
                size=size, weight="bold", color=color, anchor="mm",
            )

        # Front layer: large letters distributed around the field, with overlap but not a center pile.
        n_front = int(self.front_count)
        front_pts = art_kit.jittered_grid(rng, max(2, int(n_front ** 0.5)), max(2, (n_front + 1) // 2), jitter=0.9)
        rng.shuffle(front_pts)
        for i, (px, py) in enumerate(front_pts[:n_front]):
            ch = rng.choice(_LETTERS)
            size = int(rng.uniform(s * 0.24, s * 0.50))
            x = s * (0.12 + 0.76 * px)
            y = s * (0.12 + 0.76 * py)
            t = i / max(1, n_front - 1)
            color = art_kit.palette_color(0.2 + 0.75 * t)
            art_kit.text(
                img, (x, y), ch,
                size=size, weight="black", color=color, anchor="mm",
            )

        canvas.commit(img)
