from plugins.BaseSkill import BaseSkill, Enum, Palette, Slider

from PIL import ImageDraw


class BorderSkill(BaseSkill):
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
        ImageDraw.Draw(img, "RGBA").rectangle((w // 2, w // 2, canvas.size - w // 2 - 1, canvas.size - w // 2 - 1), outline=colors.get(str(self.color), canvas.palette.accent), width=w)
        canvas.commit(img)
