from plugins.BaseSkill import BaseSkill, Slider, Enum, Palette

import math
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class SpirographSkill(BaseSkill):
    name = 'Spirograph'
    description = 'A spirograph: the curve traced by a pen fixed in a small gear rolling inside (hypotrochoid) or outside (epitrochoid) a larger ring. The tooth ratio sets the petal count and how the loops nest; the pen offset opens or tightens them. The palette ramps along the single closed trace. Good for "spirograph", "hypotrochoid", "epitrochoid", "roulette curve", "gear art", or a looping geometric flower.'
    kind = "background"

    palette   = Palette()
    mode      = Enum([('hypo', 'Inner (hypotrochoid)'), ('epi', 'Outer (epitrochoid)')], default='hypo')
    teeth     = Slider(3, 24, default=7, step=1)
    pen       = Slider(0.2, 1.0, default=0.75, step=0.05)

    def run(self, canvas):
        s = int(canvas.size)
        R = 1.0
        r = float(int(self.teeth)) / 24.0      # rolling-gear radius as fraction of ring
        r = max(0.05, min(0.95, r))
        rho = float(self.pen) * r               # pen distance from gear center
        epi = (str(self.mode) == 'epi')

        # Closed after teeth/gcd turns; use the integer tooth count for that.
        teeth = int(self.teeth)
        turns = teeth // math.gcd(teeth, 24) * 2 + 2
        steps = 4000
        pts = []
        for i in range(steps + 1):
            t = (i / steps) * math.tau * turns
            if epi:
                k = (R + r) / r
                x = (R + r) * math.cos(t) - rho * math.cos(k * t)
                y = (R + r) * math.sin(t) - rho * math.sin(k * t)
            else:
                k = (R - r) / r
                x = (R - r) * math.cos(t) + rho * math.cos(k * t)
                y = (R - r) * math.sin(t) - rho * math.sin(k * t)
            pts.append((x, y))

        rmax = max((px * px + py * py) ** 0.5 for px, py in pts) or 1.0
        scale = s * 0.45 / rmax
        cx = cy = s / 2.0
        scaled = [(cx + px * scale, cy + py * scale) for px, py in pts]

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")
        w = max(1, int(s * 0.0016))
        runs = 200
        per = max(1, len(scaled) // runs)
        for j in range(0, len(scaled) - 1, per):
            t = 0.2 + 0.75 * (j / float(len(scaled)))
            draw.line(scaled[j:j + per + 1], fill=art_kit.palette_color(t), width=w, joint="curve")

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.004))
        canvas.commit(Image.alpha_composite(glow, img))
