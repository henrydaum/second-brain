from plugins.BaseTechnique import BaseTechnique, Palette, Slider

import math

try:
    art_kit
except NameError:
    art_kit = None


class MobiusRibbon3DTechnique(BaseTechnique):
    name = "3D Möbius Ribbon"
    description = "Object overlay: a Möbius strip swept around the origin, palette-graded along arc length."
    kind = "object"
    palette = Palette()
    twists = Slider(1, 3, default=1, step=1)
    width = Slider(0.18, 0.55, default=0.32, step=0.02)
    height_scale = Slider(0.6, 1.3, default=1.0, step=0.05)

    def run(self, canvas):
        img = canvas.new_layer()
        N = int(round(float(self.twists)))
        w = float(self.width)
        h_scale = float(self.height_scale)

        R = 0.95
        U = 200
        V = 6

        verts = []
        for i in range(U + 1):
            u = art_kit.tau * i / U
            cu, su = math.cos(u), math.sin(u)
            ht = N * u / 2.0
            cth, sth = math.cos(ht), math.sin(ht)
            for j in range(V + 1):
                v = (j / V - 0.5) * 2 * w
                rad = R + v * cth
                verts.append((rad * cu, v * sth * h_scale, rad * su))

        def vidx(i, j):
            return i * (V + 1) + j

        faces = []
        colors = []
        for i in range(U):
            t = i / U
            color = art_kit.palette_color(0.2 + 0.75 * t)
            for j in range(V):
                faces.append((vidx(i, j), vidx(i + 1, j), vidx(i + 1, j + 1), vidx(i, j + 1)))
                colors.append(color)

        ribbon = art_kit.mesh(verts, faces, colors=colors)
        art_kit.render_3d(
            img, [ribbon],
            camera=(2.7, 2.0, 3.0),
            target=(0, 0.0, 0),
            fov=40,
            outline=canvas.palette.background,
            cull=False,
            ambient=0.45,
        )
        canvas.commit(img)
