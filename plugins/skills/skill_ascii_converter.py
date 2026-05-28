from plugins.BaseSkill import BaseSkill, Slider, Enum

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


_RAMPS = {
    "classic": " .:-=+*#%@",
    "dense":   " .'`^,:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "blocks":  " ░▒▓█",
    "binary":  "01",
    "letters": " AEIOUNTSRLHCDMPBKGFVWYJZQX",
}


class AsciiConverterSkill(BaseSkill):
    name = "ASCII Converter"
    description = "Re-render the canvas as a grid of Jost ASCII glyphs whose density tracks local luminance. Pick the ramp, cell size, and color mode (palette / mono / inverted)."
    kind = "filter"
    cell_px = Slider(8, 32, default=14, step=1)
    ramp = Enum([("classic", "Classic"), ("dense", "Dense"), ("blocks", "Blocks"), ("binary", "Binary"), ("letters", "Letters")], default="classic")
    color = Enum([("palette", "Palette"), ("mono", "Mono"), ("invert", "Invert")], default="palette")
    bg = Enum([("black", "Black"), ("white", "White"), ("palette_bg", "Palette BG")], default="black")

    def run(self, canvas):
        s = canvas.size
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722

        cell = int(self.cell_px)
        ramp = _RAMPS[str(self.ramp)]
        nramp = len(ramp)

        bg_hex = {"black": "#0a0a0a", "white": "#fafafa", "palette_bg": canvas.palette.background}[str(self.bg)]
        img = Image.new("RGBA", (s, s), bg_hex)

        cols = s // cell
        rows = s // cell
        for r in range(rows):
            y0 = r * cell
            y1 = y0 + cell
            for c in range(cols):
                x0 = c * cell
                x1 = x0 + cell
                L = float(lum[y0:y1, x0:x1].mean())
                gi = int(L * (nramp - 1))
                # Higher luminance -> denser glyph for "classic" reads light-on-dark.
                if str(self.bg) == "white":
                    gi = (nramp - 1) - gi
                glyph = ramp[gi]
                if glyph == " ":
                    continue
                cx = x0 + cell / 2
                cy = y0 + cell / 2
                mode = str(self.color)
                if mode == "palette":
                    color = art_kit.palette_color(L)
                elif mode == "invert":
                    color = art_kit.palette_color(1.0 - L)
                else:
                    color = "#fafafa" if str(self.bg) != "white" else "#0a0a0a"
                art_kit.text(
                    img, (cx, cy), glyph,
                    size=cell, weight="bold", color=color, anchor="mm",
                )

        canvas.commit(img)
