from plugins.BaseTechnique import BaseTechnique, Enum, Palette, Slider

import random

try:
    art_kit
except NameError:
    art_kit = None


class CubeStack3DTechnique(BaseTechnique):
    name = "3D Cube Stack"
    description = "Object overlay: a compact isometric stack of palette-lit cubes, rendered by art_kit's tiny 3D painter."
    kind = "object"
    palette = Palette()
    density = Enum([("low", "Low"), ("mid", "Mid"), ("high", "High")], default="mid")
    scale = Slider(0.45, 1.2, default=0.82, step=0.03)

    def run(self, canvas):
        rng = random.Random(canvas.seed)
        img = canvas.new_layer()
        n = {"low": 5, "mid": 9, "high": 14}.get(str(self.density), 9)
        meshes = []
        for i in range(n):
            x = (rng.random() - 0.5) * 1.7
            z = (rng.random() - 0.5) * 1.2
            y = -0.45 + i * 0.09 + rng.random() * 0.18
            size = float(self.scale) * (0.38 + rng.random() * 0.28)
            color = art_kit.palette_color(0.35 + 0.55 * i / max(1, n - 1))
            meshes.append(art_kit.cube_mesh(size=size, center=(x, y, z), color=color))
        art_kit.render_3d(img, meshes, camera=(2.8, 2.1, 3.4), target=(0, 0, 0), fov=36, outline=canvas.palette.background)
        canvas.commit(img)
