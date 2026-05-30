from plugins.BaseTechnique import BaseTechnique, Palette, Slider

import math

try:
    art_kit
except NameError:
    art_kit = None


class TorusKnot3DTechnique(BaseTechnique):
    name = "3D Torus Knot"
    description = "Object overlay: a parametric (p, q) torus-knot rendered as a palette-gradient swept tube."
    kind = "object"
    palette = Palette()
    p = Slider(2, 5, default=2, step=1)
    q = Slider(2, 7, default=3, step=1)
    thickness = Slider(0.04, 0.18, default=0.09, step=0.005)

    def run(self, canvas):
        img = canvas.new_layer()
        p = int(round(float(self.p)))
        q = int(round(float(self.q)))
        thickness = float(self.thickness)

        R = 1.0
        r = 0.42
        steps = 220
        sides = 7

        def curve(t):
            ct, st = math.cos(p * t), math.sin(p * t)
            cq, sq = math.cos(q * t), math.sin(q * t)
            rad = R + r * cq
            return (rad * ct, rad * st, r * sq)

        def tangent(t):
            ct, st = math.cos(p * t), math.sin(p * t)
            cq, sq = math.cos(q * t), math.sin(q * t)
            rad = R + r * cq
            return (
                -r * q * sq * ct - p * rad * st,
                -r * q * sq * st + p * rad * ct,
                r * q * cq,
            )

        def norm(v):
            d = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
            return (v[0] / d, v[1] / d, v[2] / d)

        def cross(a, b):
            return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

        verts = []
        for i in range(steps):
            t = art_kit.tau * i / steps
            c = curve(t)
            tan = norm(tangent(t))
            n = cross(tan, (0.0, 0.0, 1.0))
            if n[0] * n[0] + n[1] * n[1] + n[2] * n[2] < 1e-6:
                n = cross(tan, (0.0, 1.0, 0.0))
            n = norm(n)
            b = norm(cross(tan, n))
            for j in range(sides):
                a = art_kit.tau * j / sides
                cosa, sina = math.cos(a), math.sin(a)
                verts.append((
                    c[0] + thickness * (cosa * n[0] + sina * b[0]),
                    c[1] + thickness * (cosa * n[1] + sina * b[1]),
                    c[2] + thickness * (cosa * n[2] + sina * b[2]),
                ))

        def vidx(i, j):
            return (i % steps) * sides + (j % sides)

        faces = []
        colors = []
        for i in range(steps):
            t = i / steps
            face_color = art_kit.palette_color(0.2 + 0.75 * t)
            for j in range(sides):
                faces.append((vidx(i, j), vidx(i + 1, j), vidx(i + 1, j + 1), vidx(i, j + 1)))
                colors.append(face_color)

        knot = art_kit.mesh(verts, faces, colors=colors)
        art_kit.render_3d(
            img, [knot],
            camera=(2.6, 1.9, 3.0),
            target=(0, 0, 0),
            fov=38,
            outline=canvas.palette.background,
            cull=False,
            ambient=0.42,
        )
        canvas.commit(img)
