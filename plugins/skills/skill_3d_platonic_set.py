from plugins.BaseSkill import BaseSkill, Enum, Palette, Slider

import math

try:
    art_kit
except NameError:
    art_kit = None


PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _tetrahedron():
    return (
        [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)],
        [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)],
        math.sqrt(3.0),
    )


def _cube():
    v = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    return (
        v,
        [(0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4), (2, 6, 7, 3), (0, 2, 3, 1), (4, 5, 7, 6)],
        math.sqrt(3.0),
    )


def _octahedron():
    return (
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        [(4, 0, 2), (4, 2, 1), (4, 1, 3), (4, 3, 0), (5, 2, 0), (5, 1, 2), (5, 3, 1), (5, 0, 3)],
        1.0,
    )


_ICOSA_VERTS = [
    (-1,  PHI,   0),  (1,  PHI,   0),
    (-1, -PHI,   0),  (1, -PHI,   0),
    ( 0,   -1,  PHI), (0,    1,  PHI),
    ( 0,   -1, -PHI), (0,    1, -PHI),
    ( PHI,  0,  -1),  (PHI,  0,   1),
    (-PHI,  0,  -1),  (-PHI, 0,   1),
]
_ICOSA_FACES = [
    (0, 11, 5),  (0, 5, 1),   (0, 1, 7),   (0, 7, 10),  (0, 10, 11),
    (1, 5, 9),   (5, 11, 4),  (11, 10, 2), (10, 7, 6),  (7, 1, 8),
    (3, 9, 4),   (3, 4, 2),   (3, 2, 6),   (3, 6, 8),   (3, 8, 9),
    (4, 9, 5),   (2, 4, 11),  (6, 2, 10),  (8, 6, 7),   (9, 8, 1),
]


def _icosahedron():
    return (list(_ICOSA_VERTS), list(_ICOSA_FACES), math.sqrt(1.0 + PHI * PHI))


def _dodecahedron():
    iv, ifaces = _ICOSA_VERTS, _ICOSA_FACES
    dverts = [
        (
            (iv[a][0] + iv[b][0] + iv[c][0]) / 3.0,
            (iv[a][1] + iv[b][1] + iv[c][1]) / 3.0,
            (iv[a][2] + iv[b][2] + iv[c][2]) / 3.0,
        )
        for (a, b, c) in ifaces
    ]
    vert_to_faces = [[] for _ in range(len(iv))]
    for fi, face in enumerate(ifaces):
        for v in face:
            vert_to_faces[v].append(fi)
    dfaces = []
    for fis in vert_to_faces:
        if len(fis) != 5:
            continue
        ordered = [fis[0]]
        remaining = set(fis[1:])
        while remaining:
            last = ifaces[ordered[-1]]
            advanced = False
            for cand in list(remaining):
                if len(set(last) & set(ifaces[cand])) >= 2:
                    ordered.append(cand)
                    remaining.remove(cand)
                    advanced = True
                    break
            if not advanced:
                break
        dfaces.append(tuple(ordered))
    cr = math.sqrt(sum(c * c for c in dverts[0]))
    return dverts, dfaces, cr


_SOLID_BUILDERS = {
    "tetra": _tetrahedron,
    "cube": _cube,
    "octa": _octahedron,
    "dodeca": _dodecahedron,
    "icosa": _icosahedron,
}


class PlatonicSet3DSkill(BaseSkill):
    name = "3D Platonic Set"
    description = "Object overlay: a row of platonic solids (tetra/cube/octa/dodeca/icosa) on a palette gradient, gently rotated."
    kind = "object"
    palette = Palette()
    solids = Enum(
        [("pair", "Pair"), ("triplet", "Triplet"), ("all_five", "All Five")],
        default="triplet",
    )
    rotation_phase = Slider(0, 6.28, default=0.6, step=0.05)
    spacing = Slider(0.6, 1.4, default=1.0, step=0.05)

    def _names(self, layout):
        if layout == "pair":
            return ["octa", "icosa"]
        if layout == "all_five":
            return ["tetra", "cube", "octa", "dodeca", "icosa"]
        return ["tetra", "octa", "icosa"]

    def run(self, canvas):
        img = canvas.new_layer()
        names = self._names(str(self.solids))
        n = len(names)
        spacing = float(self.spacing)
        phase = float(self.rotation_phase)
        size_target = 0.5

        meshes = []
        x_extent = (n - 1) * spacing
        for i, name in enumerate(names):
            verts, faces, cr = _SOLID_BUILDERS[name]()
            scale = size_target / cr
            angle = phase + i * 0.35
            ca, sa = math.cos(angle), math.sin(angle)
            tilt = 0.32 * math.sin(angle * 1.7)
            ct, st = math.cos(tilt), math.sin(tilt)
            x_off = -x_extent / 2.0 + i * spacing
            t = i / max(1, n - 1)
            transformed = []
            for (x, y, z) in verts:
                rx = x * ca - z * sa
                rz = x * sa + z * ca
                ry = y * ct - rz * st
                rz2 = y * st + rz * ct
                transformed.append((x_off + scale * rx, scale * ry, scale * rz2))
            color = art_kit.palette_color(0.25 + 0.65 * t)
            meshes.append(art_kit.mesh(transformed, faces, color=color))

        art_kit.render_3d(
            img, meshes,
            camera=(0.7, 1.8, 5.2),
            target=(0, 0, 0),
            fov=44,
            outline=canvas.palette.background,
            cull=False,
            ambient=0.42,
        )
        canvas.commit(img)
