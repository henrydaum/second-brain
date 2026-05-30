from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette

import math
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class SpirographTechnique(BaseTechnique):
    name = 'Spirograph'
    description = 'A spirograph: the curve traced by a pen fixed in a small gear rolling inside (hypotrochoid) or outside (epitrochoid) a larger ring. The tooth ratio sets the petal count and how the loops nest; the pen offset opens or tightens them. The palette ramps along the single closed trace. Good for "spirograph", "hypotrochoid", "epitrochoid", "roulette curve", "gear art", or a looping geometric flower.'
    kind = "background"

    palette   = Palette()
    mode      = Enum([('hypo', 'Inner (hypotrochoid)'), ('epi', 'Outer (epitrochoid)')], default='hypo')
    teeth     = Slider(3, 24, default=7, step=1)
    pen       = Slider(0.2, 1.0, default=0.85, step=0.05)
    layers    = Slider(1, 8, default=5, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        R = 1.0
        epi = (str(self.mode) == 'epi')
        n_layers = int(self.layers)
        cx = cy = s / 2.0

        def curve(teeth, rho_frac):
            r = max(0.05, min(0.95, teeth / 24.0))
            rho = rho_frac * r
            turns = teeth // math.gcd(teeth, 24) * 2 + 2
            steps = 3200
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
            return pts

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")
        w = max(1, int(s * 0.0014))

        # Stack several curves, each a slightly different gear / pen, rotated so
        # the rosettes interleave into a dense spirograph drawing.
        base_teeth = int(self.teeth)
        base_pen = float(self.pen)
        for L in range(n_layers):
            teeth = base_teeth + L * 2
            rho_frac = base_pen * (0.55 + 0.45 * (L / max(1, n_layers - 1))) if n_layers > 1 else base_pen
            pts = curve(teeth, rho_frac)
            rmax = max((px * px + py * py) ** 0.5 for px, py in pts) or 1.0
            scale = s * 0.45 / rmax
            phi = L * (math.pi / max(1, n_layers)) * 0.7
            ca, sa = math.cos(phi), math.sin(phi)
            scaled = [(cx + (px * ca - py * sa) * scale, cy + (px * sa + py * ca) * scale)
                      for px, py in pts]
            t = 0.25 + 0.65 * (L / max(1, n_layers - 1)) if n_layers > 1 else 0.7
            draw.line(scaled, fill=art_kit.palette_color(t), width=w, joint="curve")

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.004))
        canvas.commit(Image.alpha_composite(glow, img))
