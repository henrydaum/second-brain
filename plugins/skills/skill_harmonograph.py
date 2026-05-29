from plugins.BaseSkill import BaseSkill, Slider, Palette

import math
import random
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class HarmonographSkill(BaseSkill):
    name = 'Harmonograph'
    description = 'A harmonograph: the looping trace drawn by coupled, slowly-decaying pendulums. Two damped sinusoids per axis beat against each other into spirograph-like Lissajous knots that gently unwind as the swing dies away. Seed and frequency spread make each one unique; the palette ramps along the path. Good for "harmonograph", "Lissajous", "pendulum art", "spirograph", or a damped looping curve.'
    kind = "background"

    palette    = Palette()
    decay      = Slider(0.002, 0.04, default=0.012, step=0.001)
    complexity = Slider(0.0, 1.0, default=0.5, step=0.05)

    def run(self, canvas):
        s = int(canvas.size)
        rng = random.Random(int(canvas.seed))
        spread = float(self.complexity)
        dd = float(self.decay)

        # Two oscillators per axis; frequencies sit near small integers and
        # detune by an amount the complexity dial controls.
        def freq(base):
            return base + (rng.random() - 0.5) * (0.6 + 4.0 * spread)

        fx1, fx2 = freq(2.0), freq(3.0)
        fy1, fy2 = freq(2.0), freq(3.0)
        px1, px2 = rng.uniform(0, math.tau), rng.uniform(0, math.tau)
        py1, py2 = rng.uniform(0, math.tau), rng.uniform(0, math.tau)
        d1, d2, d3, d4 = (dd * rng.uniform(0.6, 1.4) for _ in range(4))
        a1, a2 = rng.uniform(0.6, 1.0), rng.uniform(0.4, 0.8)

        steps = 9000
        amp = s * 0.40
        cx = cy = s / 2.0
        pts = []
        for i in range(steps):
            t = i * 0.05
            x = (a1 * math.sin(fx1 * t + px1) * math.exp(-d1 * t)
                 + a2 * math.sin(fx2 * t + px2) * math.exp(-d2 * t))
            y = (a1 * math.sin(fy1 * t + py1) * math.exp(-d3 * t)
                 + a2 * math.sin(fy2 * t + py2) * math.exp(-d4 * t))
            pts.append((cx + x * amp * 0.5, cy + y * amp * 0.5))

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")
        w = max(1, int(s * 0.0014))
        # Draw in short colored runs so the palette sweeps along the trace.
        runs = 240
        per = max(1, len(pts) // runs)
        for j in range(0, len(pts) - 1, per):
            t = 0.2 + 0.75 * (j / float(len(pts)))
            draw.line(pts[j:j + per + 1], fill=art_kit.palette_color(t), width=w, joint="curve")

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.004))
        canvas.commit(Image.alpha_composite(glow, img))
