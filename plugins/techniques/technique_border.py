from plugins.BaseTechnique import BaseTechnique, Enum, Palette, Slider

from PIL import ImageDraw


class BorderTechnique(BaseTechnique):
    name = "Border"
    description = "Object overlay: a plain palette-color border with adjustable width."
    kind = "object"
    palette = Palette()
    width = Slider(1, 80, default=18, step=1)
    color = Enum([("background", "Background"), ("primary", "Primary"), ("secondary", "Secondary"), ("tertiary", "Tertiary"), ("accent", "Accent")], default="accent")

    def run(self, canvas):
        colors = {
            "background": canvas.palette.background,
            "primary": canvas.palette.primary,
            "secondary": canvas.palette.secondary,
            "tertiary": canvas.palette.tertiary,
            "accent": canvas.palette.accent,
        }
        img = canvas.new_layer()
        w = max(1, int(self.width))
        draw = ImageDraw.Draw(img, "RGBA")
        color = colors.get(str(self.color), canvas.palette.accent)
        s = canvas.size
        draw.rectangle((0, 0, s - 1, w - 1), fill=color)
        draw.rectangle((0, s - w, s - 1, s - 1), fill=color)
        draw.rectangle((0, 0, w - 1, s - 1), fill=color)
        draw.rectangle((s - w, 0, s - 1, s - 1), fill=color)
        canvas.commit(img)
