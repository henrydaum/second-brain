from plugins.BaseSkill import BaseSkill, Slider, Enum, Palette

import random
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class LetterFieldSkill(BaseSkill):
    name = "Letter Field"
    description = "Homage to Judson Rosebush's 1978 Letter Field: a stack of large overlapping colored capital letters on a dark ground, with smaller faded letters scattered behind. Built-in Jost typeface."
    kind = "background"
    palette = Palette()
    front_count = Slider(3, 14, default=7, step=1, label="Front Letters")
    back_count = Slider(0, 60, default=22, step=1, label="Back Letters")
    bg = Enum([("palette_bg", "Palette BG"), ("black", "Black"), ("ivory", "Ivory")], default="black")

    def run(self, canvas):
        s = canvas.size
        rng = random.Random(canvas.seed)

        bg = {
            "palette_bg": canvas.palette.background,
            "black": "#0a0a0a",
            "ivory": "#f5efe2",
        }[str(self.bg)]
        img = Image.new("RGBA", (s, s), bg)

        # Back layer: small faded letters scattered randomly.
        n_back = int(self.back_count)
        for i in range(n_back):
            ch = rng.choice(_LETTERS)
            size = int(rng.uniform(s * 0.05, s * 0.18))
            x = int(rng.uniform(s * 0.05, s * 0.95))
            y = int(rng.uniform(s * 0.05, s * 0.95))
            t = rng.random()
            base = art_kit.palette_color(0.25 + 0.7 * t)
            color = art_kit.with_alpha(base, 0.32)
            art_kit.text(
                img, (x, y), ch,
                size=size, weight="bold", color=color, anchor="mm",
            )

        # Front layer: large bold/black letters stacked at the center.
        n_front = int(self.front_count)
        cx, cy = s / 2.0, s / 2.0
        for i in range(n_front):
            ch = rng.choice(_LETTERS)
            size = int(rng.uniform(s * 0.32, s * 0.62))
            jx = (rng.random() - 0.5) * s * 0.35
            jy = (rng.random() - 0.5) * s * 0.35
            t = i / max(1, n_front - 1)
            color = art_kit.palette_color(0.2 + 0.75 * t)
            art_kit.text(
                img, (cx + jx, cy + jy), ch,
                size=size, weight="black", color=color, anchor="mm",
            )

        canvas.commit(img)
