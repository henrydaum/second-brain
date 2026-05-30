from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import math
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


def _csqrt(w):
    """Principal complex square root (cmath isn't on the technique allowlist)."""
    a, b = w.real, w.imag
    if b == 0.0:
        return math.sqrt(a) + 0j if a >= 0 else math.sqrt(-a) * 1j
    m = math.hypot(a, b)
    re = math.sqrt((m + a) / 2.0)
    im = math.copysign(math.sqrt((m - a) / 2.0), b)
    return re + im * 1j


class ApollonianGasketTechnique(BaseTechnique):
    name = 'Apollonian Gasket'
    description = 'The Apollonian gasket: a fractal packing of mutually tangent circles, every gap recursively filled with the largest circle that fits, sized by Descartes\' circle theorem. Curvature drives the palette ramp so tiny deep circles glow toward the accent. Good for "circle packing", "tangent circles", "gasket", "bubbles", or a recursive geometric fractal.'
    kind = "background"

    palette = Palette()
    depth   = Slider(3, 10, default=7, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        max_depth = int(self.depth)

        # Each circle is (curvature k, complex center z) in a unit-radius frame.
        c_outer = (-1.0 + 0j, 0 + 0j)
        c_left = (2.0 + 0j, -0.5 + 0j)
        c_right = (2.0 + 0j, 0.5 + 0j)
        c_top = (3.0 + 0j, 0 + (2.0 / 3.0) * 1j)
        c_bot = (3.0 + 0j, 0 - (2.0 / 3.0) * 1j)
        seeds = [c_outer, c_left, c_right, c_top, c_bot]

        circles = []           # (k, z) with k > 0 only (drawable)
        seen = set()
        K_MAX = float(s)        # smallest circle ~ 1px

        def key(c):
            return (round(c[0].real, 2), round(c[1].real, 3), round(c[1].imag, 3))

        def add(c):
            if c[0].real <= 0 or abs(c[0]) > K_MAX:
                return False
            kk = key(c)
            if kk in seen:
                return False
            seen.add(kk)
            circles.append(c)
            return True

        def tangent(c4, c1, c2, c3):
            """Keep only genuine solutions: tangent to all three parents."""
            r4 = 1.0 / c4[0].real
            for ci in (c1, c2, c3):
                ri = 1.0 / ci[0].real
                d = abs(c4[1] - ci[1])
                if min(abs(d - abs(r4 - ri)), abs(d - abs(r4 + ri))) > 2e-3:
                    return False
            return True

        def descartes(c1, c2, c3):
            k1, k2, k3 = c1[0], c2[0], c3[0]
            z1, z2, z3 = c1[1], c2[1], c3[1]
            ksum = k1 + k2 + k3
            kroot = 2.0 * _csqrt(k1 * k2 + k2 * k3 + k3 * k1)
            zsum = k1 * z1 + k2 * z2 + k3 * z3
            zroot = 2.0 * _csqrt(k1 * z1 * k2 * z2 + k2 * z2 * k3 * z3 + k3 * z3 * k1 * z1)
            out = []
            for k4 in (ksum + kroot, ksum - kroot):
                k4 = k4.real + 0j
                if abs(k4) < 1e-9:
                    continue
                for zr in (zroot, -zroot):
                    out.append((k4, (zsum + zr) / k4))
            return out

        def recurse(c1, c2, c3, depth):
            if depth > max_depth:
                return
            for c4 in descartes(c1, c2, c3):
                if any(key(c4) == key(c) for c in (c1, c2, c3)):
                    continue
                if not tangent(c4, c1, c2, c3):
                    continue
                if add(c4):
                    recurse(c1, c2, c4, depth + 1)
                    recurse(c1, c3, c4, depth + 1)
                    recurse(c2, c3, c4, depth + 1)

        for c in seeds:
            add(c)
        # Bootstrap from every mutually tangent seed triple.
        recurse(c_left, c_right, c_top, 1)
        recurse(c_left, c_right, c_bot, 1)
        recurse(c_outer, c_left, c_top, 1)
        recurse(c_outer, c_right, c_top, 1)
        recurse(c_outer, c_left, c_bot, 1)
        recurse(c_outer, c_right, c_bot, 1)

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")
        R = s * 0.47
        cx = cy = s / 2.0

        # Sort big-to-small so small bright circles land on top.
        circles.sort(key=lambda c: abs(c[0]), reverse=False)
        kmin, kmax = 1.0, max(2.0, max(abs(c[0]) for c in circles))
        for k, z in circles:
            rad = R / abs(k)
            if rad < 0.8:
                continue
            px = cx + z.real * R
            py = cy + z.imag * R
            t = (math.log(abs(k) + 1.0) - math.log(kmin + 1.0)) / (math.log(kmax + 1.0) - math.log(kmin + 1.0) or 1.0)
            t = 0.2 + 0.75 * min(max(t, 0.0), 1.0)
            draw.ellipse((px - rad, py - rad, px + rad, py + rad), fill=art_kit.palette_color(t))

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.004))
        canvas.commit(Image.alpha_composite(glow, img))
