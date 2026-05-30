from plugins.BaseTechnique import BaseTechnique, Enum, Palette, Slider

import math
import random

try:
    art_kit
except NameError:
    art_kit = None


class CrystalCluster3DTechnique(BaseTechnique):
    name = "3D Crystal Cluster"
    description = "Object overlay: low-poly crystal shards with painterly 3D projection, palette facets, and sharp silhouettes."
    kind = "object"
    palette = Palette()
    count = Enum([("few", "Few"), ("cluster", "Cluster"), ("field", "Field")], default="cluster")
    spread = Slider(0.4, 1.4, default=0.9, step=0.05)

    def _crystal(self, center, radius, height, angle, color):
        base = [(math.cos(a) * radius, -height * 0.35, math.sin(a) * radius) for a in (0, math.pi / 2, math.pi, math.pi * 1.5)]
        verts = [(0, height * 0.65, 0), *base, (0, -height * 0.65, 0)]
        ca, sa = math.cos(angle), math.sin(angle)
        verts = [(center[0] + x * ca - z * sa, center[1] + y, center[2] + x * sa + z * ca) for x, y, z in verts]
        return art_kit.mesh(verts, [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4)], color=color)

    def run(self, canvas):
        rng = random.Random(canvas.seed)
        img = canvas.new_layer()
        n = {"few": 4, "cluster": 7, "field": 11}.get(str(self.count), 7)
        meshes = []
        for i in range(n):
            t = i / max(1, n - 1)
            center = ((rng.random() - 0.5) * float(self.spread), -0.2 + t * 0.25, (rng.random() - 0.5) * 0.9)
            meshes.append(self._crystal(center, 0.16 + rng.random() * 0.11, 0.8 + rng.random() * 0.75, rng.random() * art_kit.tau, art_kit.palette_color(0.25 + 0.7 * t)))
        art_kit.render_3d(img, meshes, camera=(2.4, 1.7, 3.0), target=(0, 0.1, 0), fov=34, outline=canvas.palette.background, ambient=0.45)
        canvas.commit(img)
