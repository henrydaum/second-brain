from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import math
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class SuperformulaTechnique(BaseTechnique):
    name = 'Superformula'
    description = "Gielis' superformula: a single polar equation that morphs between circles, polygons, stars, flowers, and organic blobs as its symmetry and exponents change. Nested, slightly rotated copies ramp from a dark outer shell to a bright core, finished with a soft bloom. Good for \"superformula\", \"supershape\", \"organic geometry\", \"abstract flower\", \"starburst\", or a parametric bloom."
    kind = "background"

    palette = Palette()
    symmetry = Slider(2, 20, default=7, step=1)
    pinch    = Slider(0.3, 8.0, default=1.0, step=0.1)
    layers   = Slider(1, 6, default=4, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        m = int(self.symmetry)
        n1 = float(self.pinch)
        n_layers = int(self.layers)

        def supershape(n2, n3, steps=720):
            pts = []
            for i in range(steps):
                phi = math.tau * i / steps
                t1 = abs(math.cos(m * phi / 4.0)) ** n2
                t2 = abs(math.sin(m * phi / 4.0)) ** n3
                denom = t1 + t2
                if denom <= 1e-9:
                    r = 0.0
                else:
                    r = denom ** (-1.0 / n1)
                pts.append((r * math.cos(phi), r * math.sin(phi)))
            return pts

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")
        cx = cy = s / 2.0
        base_r = s * 0.42

        for L in range(n_layers):
            f = 1.0 - L / float(n_layers)               # 1 (outer) -> small (inner)
            n2 = n1 * (0.6 + 0.5 * L)
            n3 = n1 * (0.6 + 0.5 * L)
            pts = supershape(n2, n3)
            rmax = max((px * px + py * py) ** 0.5 for px, py in pts) or 1.0
            scale = base_r * f / rmax
            rot = L * (math.pi / max(1, m) * 0.25)
            ca, sa = math.cos(rot), math.sin(rot)
            poly = [(cx + (px * ca - py * sa) * scale, cy + (px * sa + py * ca) * scale)
                    for px, py in pts]
            t = 0.18 + 0.78 * (L / max(1, n_layers - 1)) if n_layers > 1 else 0.85
            draw.polygon(poly, fill=art_kit.palette_color(t))

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.01))
        canvas.commit(Image.alpha_composite(glow, img))
