from plugins.BaseTechnique import BaseTechnique, Enum, Palette, Slider

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class ColorFieldTechnique(BaseTechnique):
    name = "Color Field"
    description = "Background: a quick palette-bound solid, linear gradient, or radial gradient color field."
    kind = "background"
    palette = Palette()
    mode = Enum([("solid", "Solid"), ("linear", "Linear Gradient"), ("radial", "Radial Gradient")], default="linear")
    tone = Enum([("background", "Background"), ("primary", "Primary"), ("secondary", "Secondary"), ("accent", "Accent")], default="background")
    angle = Slider(0, 360, default=35, step=5)

    def run(self, canvas):
        s = canvas.size
        if self.mode == "solid":
            colors = {
                "background": canvas.palette.background,
                "primary": canvas.palette.primary,
                "secondary": canvas.palette.secondary,
                "accent": canvas.palette.accent,
            }
            canvas.commit(canvas.new(color=colors.get(str(self.tone), canvas.palette.background)))
            return
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        if self.mode == "radial":
            t = np.hypot(xx - s * 0.5, yy - s * 0.5) / (s * 0.707)
        else:
            a = np.deg2rad(float(self.angle))
            t = ((xx - s * 0.5) * np.cos(a) + (yy - s * 0.5) * np.sin(a)) / s + 0.5
        idx = np.clip(t, 0, 1) * 255
        ramp = np.array([art_kit.hex_to_rgb(art_kit.palette_color(i / 255)) for i in range(256)], dtype=np.uint8)
        canvas.commit_array(ramp[idx.astype(np.uint8)])
