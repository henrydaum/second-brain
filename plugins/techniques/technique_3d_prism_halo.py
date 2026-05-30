from plugins.BaseTechnique import BaseTechnique, Enum, Palette, Slider

import math

try:
    art_kit
except NameError:
    art_kit = None


class PrismHalo3DTechnique(BaseTechnique):
    name = "3D Prism Halo"
    description = "Object overlay: a hovering ring of triangular prisms, like a small architectural crown around the center."
    kind = "object"
    palette = Palette()
    count = Enum([("six", "Six"), ("nine", "Nine"), ("twelve", "Twelve")], default="nine")
    radius = Slider(0.55, 1.35, default=0.9, step=0.04)

    def _prism(self, center, length, width, height, angle, colors):
        pts = [(-width, -height, -length), (width, -height, -length), (0, height, -length), (-width, -height, length), (width, -height, length), (0, height, length)]
        ca, sa = math.cos(angle), math.sin(angle)
        verts = [(center[0] + x * ca - z * sa, center[1] + y, center[2] + x * sa + z * ca) for x, y, z in pts]
        return art_kit.mesh(verts, [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)], colors=colors)

    def run(self, canvas):
        img = canvas.new_layer()
        n = {"six": 6, "nine": 9, "twelve": 12}.get(str(self.count), 9)
        meshes = []
        for i in range(n):
            a = art_kit.tau * i / n
            r = float(self.radius)
            t = i / max(1, n - 1)
            center = (math.cos(a) * r, 0.08 * math.sin(a * 2), math.sin(a) * r)
            colors = [art_kit.palette_color((t + j * 0.12) % 1.0) for j in range(5)]
            meshes.append(self._prism(center, 0.18, 0.13, 0.34, a + math.pi / 2, colors))
        art_kit.render_3d(img, meshes, camera=(2.7, 2.2, 3.2), target=(0, 0, 0), fov=38, outline=canvas.palette.background, ambient=0.5, cull=False)
        canvas.commit(img)
