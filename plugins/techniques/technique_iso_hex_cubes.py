from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette

import math
from PIL import Image, ImageDraw


class IsoHexCubesTechnique(BaseTechnique):
    name = "Iso Hex Cubes"
    description = "Evenly spaced isometric wire cubes: only borders, no shaded faces, on a plain light ground."
    kind = "background"
    palette = Palette()
    cube_size = Slider(20, 90, default=48, step=2)
    height_jitter = Slider(0.8, 3.0, default=1.2, step=0.1, label="Line Weight")
    top_shade = Enum([("dark", "Dark"), ("palette", "Palette"), ("accent", "Accent")], default="dark", label="Ink")
    gap = Slider(8, 80, default=24, step=2)

    def run(self, canvas):
        s = canvas.size
        img = Image.new("RGBA", (s, s), "#eee8d6")
        draw = ImageDraw.Draw(img, "RGBA")
        size, gap = float(self.cube_size), float(self.gap)
        hw, hh, side = size * math.cos(math.radians(30)), size * 0.5, size * 0.9
        dx, dy = 2 * hw + gap, hh + side + gap
        ink = {"dark": "#202126", "palette": canvas.palette.primary, "accent": canvas.palette.accent}.get(str(self.top_shade), "#202126")
        lw = max(1, int(round(float(self.height_jitter))))

        for r in range(-2, int(s / dy) + 4):
            for c in range(-2, int(s / dx) + 4):
                cx = c * dx + (r % 2) * dx * 0.5 + hw
                cy = r * dy + hh
                pts = [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]
                down = [(pts[i][0], pts[i][1] + side) for i in (3, 2, 1)]
                for line in (pts + [pts[0]], [pts[3], down[0], down[1], pts[2], pts[3]], [pts[1], down[2], down[1], pts[2], pts[1]], down):
                    draw.line(line, fill=ink, width=lw, joint="curve")
        canvas.commit(img)
