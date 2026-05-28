from plugins.BaseSkill import BaseSkill, Slider, Enum, Palette

import math
import random
import numpy as np
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class GeometricAvoidanceSkill(BaseSkill):
    name = "Geometric Avoidance"
    description = "Particles march in straight lines, picking a new random direction whenever they hit any existing trail (theirs or another's). A configurable fraction of particles bend their direction with an underlying noise field for organic swirl mixed into the geometry."
    kind = "background"
    palette = Palette()
    particles = Slider(20, 220, default=110, step=5)
    noise_fraction = Slider(0.0, 1.0, default=0.35, step=0.05)
    step_size = Slider(1, 6, default=2, step=1)
    palette_mode = Enum([("per_particle", "Per Particle"), ("gradient", "Gradient")], default="per_particle")

    def run(self, canvas):
        s = canvas.size
        seed = canvas.seed
        rng = random.Random(seed)

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")

        occ = np.zeros((s, s), dtype=bool)
        step = int(self.step_size)
        n = int(self.particles)
        noise_n = int(n * float(self.noise_fraction))

        field = art_kit.flow_field(seed, scale=0.003, octaves=3)

        # Spawn particles.
        particles = []
        for i in range(n):
            x = rng.uniform(s * 0.05, s * 0.95)
            y = rng.uniform(s * 0.05, s * 0.95)
            ang = rng.uniform(0, math.tau)
            t = rng.random()
            if str(self.palette_mode) == "gradient":
                t = i / max(1, n - 1)
            color = art_kit.palette_color(0.15 + 0.8 * t)
            noisy = i < noise_n
            particles.append({"x": x, "y": y, "ang": ang, "color": color, "alive": True, "noisy": noisy})

        max_steps = 1200
        for _ in range(max_steps):
            any_alive = False
            for p in particles:
                if not p["alive"]:
                    continue
                any_alive = True
                if p["noisy"]:
                    target_ang = field(p["x"], p["y"])
                    # Lerp toward the field direction so noise bends but doesn't dominate.
                    d = math.atan2(math.sin(target_ang - p["ang"]), math.cos(target_ang - p["ang"]))
                    p["ang"] += 0.18 * d
                nx = p["x"] + math.cos(p["ang"]) * step
                ny = p["y"] + math.sin(p["ang"]) * step
                ix, iy = int(nx), int(ny)
                if not (0 <= ix < s and 0 <= iy < s) or occ[iy, ix]:
                    # Try a few new headings before giving up.
                    rerouted = False
                    for _try in range(8):
                        new_ang = rng.uniform(0, math.tau)
                        tx = p["x"] + math.cos(new_ang) * step
                        ty = p["y"] + math.sin(new_ang) * step
                        ix2, iy2 = int(tx), int(ty)
                        if 0 <= ix2 < s and 0 <= iy2 < s and not occ[iy2, ix2]:
                            p["ang"] = new_ang
                            rerouted = True
                            break
                    if not rerouted:
                        p["alive"] = False
                        continue
                    nx = p["x"] + math.cos(p["ang"]) * step
                    ny = p["y"] + math.sin(p["ang"]) * step
                draw.line((p["x"], p["y"], nx, ny), fill=p["color"], width=1)
                # Mark occupancy along the segment (cheap: just endpoints).
                ix, iy = int(nx), int(ny)
                if 0 <= ix < s and 0 <= iy < s:
                    occ[iy, ix] = True
                p["x"], p["y"] = nx, ny
            if not any_alive:
                break

        canvas.commit(img)
