from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import math
import random
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class FlowStreamlinesTechnique(BaseTechnique):
    name = 'Flow Streamlines'
    description = 'Particles advected through an fbm-driven flow field, leaving palette-graded streamlines. The field swirls smoothly across the canvas -- streamlines bend with it. Good for any "wind", "hair", "current", "smoke", "weather", "motion", or "abstract" request. Also a strong default when the subject doesn\'t fit any other technique.'
    kind = "background"
    palette = Palette()
    swirl = Enum([('loose', 'Loose'), ('tight', 'Tight'), ('turbulent', 'Turbulent')], default='loose')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        rng = random.Random(seed)

        scale = {"loose": 0.0035, "tight": 0.008, "turbulent": 0.013}.get(str(self.swirl), 0.0035)
        octaves = {"loose": 3, "tight": 4, "turbulent": 6}.get(str(self.swirl), 3)

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")

        field = art_kit.flow_field(seed, scale=scale, octaves=octaves)

        n_particles = 220
        step_len = max(2.0, s * 0.004)
        n_steps = 160

        for pi in range(n_particles):
            x = rng.uniform(-s * 0.1, s * 1.1)
            y = rng.uniform(-s * 0.1, s * 1.1)
            ramp = 0.15 + 0.75 * rng.random()
            color = art_kit.palette_color(ramp)
            for si in range(n_steps):
                ang = field(x, y)
                nx = x + math.cos(ang) * step_len
                ny = y + math.sin(ang) * step_len
                if nx < -s * 0.1 or nx > s * 1.1 or ny < -s * 0.1 or ny > s * 1.1:
                    break
                draw.line((x, y, nx, ny), fill=color, width=1)
                x, y = nx, ny

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.004))
        out = Image.alpha_composite(glow, img)
        canvas.commit(out)
