from plugins.BaseSkill import BaseSkill, Text, Slider, Enum, Palette

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class TypographySkill(BaseSkill):
    name = "Typography"
    description = (
        "Render a phrase in Jost, centered on the canvas, in the palette's "
        "accent color. Tweak the words, size, and style live."
    )
    kind = "creation"
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

    def run(self, canvas):
        img = canvas.create_image()
        s = canvas.size
        size_px = max(8, int(s * float(self.size_pct) / 100.0))
        weight, italic = {
            "regular":     ("regular", False),
            "italic":      ("regular", True),
            "bold":        ("bold", False),
            "bold_italic": ("bold", True),
            "black":       ("black", False),
        }[str(self.style)]
        art_kit.text(
            img, (s // 2, s // 2), str(self.phrase),
            size=size_px, weight=weight, italic=italic,
            color=canvas.palette.accent, anchor="mm", align="center",
            max_width=int(s * 0.9),
        )
        canvas.commit(img)
