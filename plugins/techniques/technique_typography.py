from plugins.BaseTechnique import BaseTechnique, Text, Slider, Enum, Palette, Pan

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class TypographyTechnique(BaseTechnique):
    name = "Typography"
    description = (
        "Overlay a phrase in Jost, positioned anywhere on the canvas, in the "
        "palette's accent color. Composites on top of the existing layer; "
        "tweak the words, size, style, and position live."
    )
    kind = "object"
    palette = Palette()
    phrase = Text(default="hello", max_length=120, placeholder="Type something…")
    size_pct = Slider(2, 30, default=12, step=0.5, label="Size (% of canvas)")
    style = Enum(
        [("regular", "Regular"), ("italic", "Italic"),
         ("bold", "Bold"), ("bold_italic", "Bold Italic"),
         ("black", "Black")],
        default="bold",
        label="Style",
    )
    pos_x = Slider(0.0, 1.0, default=0.5, step=0.02)
    pos_y = Slider(0.0, 1.0, default=0.5, step=0.02)
    position = Pan(x="pos_x", y="pos_y", label="Position")

    def run(self, canvas):
        img = canvas.new_layer()
        s = canvas.size
        size_px = max(8, int(s * float(self.size_pct) / 100.0))
        weight, italic = {
            "regular":     ("regular", False),
            "italic":      ("regular", True),
            "bold":        ("bold", False),
            "bold_italic": ("bold", True),
            "black":       ("black", False),
        }[str(self.style)]
        px = int(s * float(self.pos_x))
        py = int(s * float(self.pos_y))
        art_kit.text(
            img, (px, py), str(self.phrase),
            size=size_px, weight=weight, italic=italic,
            color=canvas.palette.accent, anchor="mm", align="center",
            max_width=int(s * 0.9),
        )
        canvas.commit(img)
