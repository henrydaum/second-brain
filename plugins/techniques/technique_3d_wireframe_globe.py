from plugins.BaseTechnique import BaseTechnique, Bool, Enum, Palette, Slider

import math

try:
    art_kit
except NameError:
    art_kit = None


class WireframeGlobe3DTechnique(BaseTechnique):
    name = "3D Wireframe Globe"
    description = "Object overlay: a tilted latitude/longitude wire-sphere drawn as palette-accent struts, with optional polar caps."
    kind = "object"
    palette = Palette()
    tilt = Slider(-0.6, 0.6, default=0.22, step=0.02)
    line_density = Enum(
        [("sparse", "Sparse"), ("medium", "Medium"), ("dense", "Dense")],
        default="medium",
    )
    pole_caps = Bool(default=False)

    def run(self, canvas):
        img = canvas.new_layer()
        density = {"sparse": (8, 12), "medium": (12, 18), "dense": (18, 28)}
        lat_n, lon_n = density.get(str(self.line_density), (12, 18))
        radius = 1.0
        tilt = float(self.tilt)
        ca, sa = math.cos(tilt), math.sin(tilt)

        verts = []
        for i in range(lat_n + 1):
            phi = math.pi * i / lat_n - math.pi / 2
            cp, sp = math.cos(phi), math.sin(phi)
            for j in range(lon_n):
                theta = art_kit.tau * j / lon_n
                x = radius * cp * math.cos(theta)
                y = radius * sp
                z = radius * cp * math.sin(theta)
                y_t = y * ca - z * sa
                z_t = y * sa + z * ca
                verts.append((x, y_t, z_t))

        def vidx(i, j):
            return i * lon_n + (j % lon_n)

        keep_caps = bool(self.pole_caps)
        faces = []
        for i in range(lat_n):
            if not keep_caps and (i == 0 or i == lat_n - 1):
                continue
            for j in range(lon_n):
                faces.append((vidx(i, j), vidx(i + 1, j), vidx(i + 1, j + 1), vidx(i, j + 1)))

        wire = art_kit.mesh(verts, faces, color=(0, 0, 0, 0))
        art_kit.render_3d(
            img, [wire],
            camera=(2.4, 1.7, 3.0),
            target=(0, 0, 0),
            fov=36,
            outline=canvas.palette.accent,
            cull=False,
            ambient=1.0,
        )
        canvas.commit(img)
