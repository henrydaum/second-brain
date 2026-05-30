from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import math
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class PythagorasTreeTechnique(BaseTechnique):
    name = 'Pythagoras Tree'
    description = 'The Pythagoras tree: a square sprouts two smaller squares on its top edge, meeting at a right angle, recursively forming a fractal canopy. The lean angle skews it from symmetric to windswept; depth controls how fine the foliage gets, with the palette ramping from a dark trunk to bright leaf tips. Good for "fractal tree", "recursive squares", "Pythagoras tree", or a geometric canopy.'
    kind = "background"

    palette = Palette()
    lean    = Slider(15, 75, default=45, step=1)
    depth   = Slider(6, 13, default=11, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        max_depth = int(self.depth)
        alpha = math.radians(float(self.lean))

        # Work in math coords (y up); convert to image coords (y down) on draw.
        polys = []  # (depth, [(x, y) x4])

        def rot(vx, vy, ang):
            ca, sa = math.cos(ang), math.sin(ang)
            return (vx * ca - vy * sa, vx * sa + vy * ca)

        def grow(ax, ay, bx, by, depth):
            ux, uy = bx - ax, by - ay
            # Upward normal (90 deg CCW of the base vector).
            nx, ny = -uy, ux
            d_pt = (ax + nx, ay + ny)        # top-left of square (above A)
            c_pt = (bx + nx, by + ny)        # top-right of square (above B)
            polys.append((depth, [(ax, ay), (bx, by), c_pt, d_pt]))
            if depth >= max_depth:
                return
            # Apex on the top edge D->C forming a right triangle, angle `alpha` at D.
            wx, wy = c_pt[0] - d_pt[0], c_pt[1] - d_pt[1]
            rx, ry = rot(wx, wy, alpha)
            ca = math.cos(alpha)
            tx, ty = d_pt[0] + rx * ca, d_pt[1] + ry * ca
            grow(d_pt[0], d_pt[1], tx, ty, depth + 1)      # left child on D->T
            grow(tx, ty, c_pt[0], c_pt[1], depth + 1)      # right child on T->C

        base = s * 0.13
        grow(s * 0.5 - base / 2.0, s * 0.10, s * 0.5 + base / 2.0, s * 0.10, 0)

        # Fit everything into the canvas with a margin.
        xs = [p[0] for _, pts in polys for p in pts]
        ys = [p[1] for _, pts in polys for p in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        margin = s * 0.05
        avail = s - 2 * margin
        scale = avail / max(x1 - x0, y1 - y0, 1e-6)
        ox = (s - (x1 - x0) * scale) * 0.5
        oy = (s - (y1 - y0) * scale) * 0.5

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")
        for depth, pts in polys:
            t = 0.12 + 0.82 * (depth / max(1, max_depth))
            mapped = [(ox + (px - x0) * scale, s - (oy + (py - y0) * scale)) for px, py in pts]
            draw.polygon(mapped, fill=art_kit.palette_color(t))

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.003))
        canvas.commit(Image.alpha_composite(glow, img))
