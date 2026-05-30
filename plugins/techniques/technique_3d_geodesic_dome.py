from plugins.BaseTechnique import BaseTechnique, Bool, Enum, Palette

import math

try:
    art_kit
except NameError:
    art_kit = None


class GeodesicDome3DTechnique(BaseTechnique):
    name = "3D Geodesic Dome"
    description = "Object overlay: an icosahedron-subdivided geodesic dome with strut outlines, half or full coverage."
    kind = "object"
    palette = Palette()
    subdivisions = Enum([("1", "1"), ("2", "2"), ("3", "3")], default="2")
    coverage = Enum([("half", "Half"), ("full", "Full")], default="half")
    strut_only = Bool(default=True)

    def run(self, canvas):
        img = canvas.new_layer()
        n_sub = int(str(self.subdivisions))
        half = str(self.coverage) == "half"
        strut = bool(self.strut_only)

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        scale = 1.0 / math.sqrt(1.0 + phi * phi)
        verts = [
            (x * scale, y * scale, z * scale) for (x, y, z) in [
                (-1,  phi,  0), (1,  phi,  0),
                (-1, -phi,  0), (1, -phi,  0),
                ( 0,  -1,  phi), (0,   1,  phi),
                ( 0,  -1, -phi), (0,   1, -phi),
                ( phi,  0, -1), (phi,  0,  1),
                (-phi,  0, -1), (-phi, 0,  1),
            ]
        ]
        triangles = [
            (0, 11, 5),  (0, 5, 1),   (0, 1, 7),   (0, 7, 10),  (0, 10, 11),
            (1, 5, 9),   (5, 11, 4),  (11, 10, 2), (10, 7, 6),  (7, 1, 8),
            (3, 9, 4),   (3, 4, 2),   (3, 2, 6),   (3, 6, 8),   (3, 8, 9),
            (4, 9, 5),   (2, 4, 11),  (6, 2, 10),  (8, 6, 7),   (9, 8, 1),
        ]

        for _ in range(n_sub):
            cache = {}

            def midpoint(a, b):
                key = (min(a, b), max(a, b))
                if key in cache:
                    return cache[key]
                va, vb = verts[a], verts[b]
                m = ((va[0] + vb[0]) / 2.0, (va[1] + vb[1]) / 2.0, (va[2] + vb[2]) / 2.0)
                d = math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]) or 1.0
                verts.append((m[0] / d, m[1] / d, m[2] / d))
                idx = len(verts) - 1
                cache[key] = idx
                return idx

            new_tris = []
            for (a, b, c) in triangles:
                ab = midpoint(a, b)
                bc = midpoint(b, c)
                ca = midpoint(c, a)
                new_tris.append((a, ab, ca))
                new_tris.append((b, bc, ab))
                new_tris.append((c, ca, bc))
                new_tris.append((ab, bc, ca))
            triangles = new_tris

        if half:
            triangles = [
                t for t in triangles
                if (verts[t[0]][1] + verts[t[1]][1] + verts[t[2]][1]) / 3.0 >= -0.02
            ]

        colors = []
        for (a, b, c) in triangles:
            if strut:
                colors.append((0, 0, 0, 0))
            else:
                cx = (verts[a][0] + verts[b][0] + verts[c][0]) / 3.0
                cz = (verts[a][2] + verts[b][2] + verts[c][2]) / 3.0
                theta = math.atan2(cz, cx)
                t_val = (theta / art_kit.tau) % 1.0
                colors.append(art_kit.palette_color(0.25 + 0.6 * t_val))

        dome = art_kit.mesh(verts, triangles, colors=colors)
        art_kit.render_3d(
            img, [dome],
            camera=(2.4, 1.5, 3.0),
            target=(0, 0.0, 0),
            fov=40,
            outline=canvas.palette.accent if strut else canvas.palette.background,
            cull=False,
            ambient=0.55,
        )
        canvas.commit(img)
