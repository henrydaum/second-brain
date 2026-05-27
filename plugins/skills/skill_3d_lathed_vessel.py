from plugins.BaseSkill import BaseSkill, Bool, Enum, Palette, Slider

import math

try:
    art_kit
except NameError:
    art_kit = None


_PROFILES = {
    "amphora": [
        (0.0, -0.85), (0.32, -0.75), (0.40, -0.55), (0.42, -0.32),
        (0.38, -0.10), (0.30,  0.12), (0.20,  0.34), (0.16,  0.50),
        (0.17,  0.62), (0.24,  0.74), (0.0,   0.78),
    ],
    "lamp": [
        (0.0, -0.90), (0.45, -0.85), (0.55, -0.70), (0.48, -0.45),
        (0.32, -0.30), (0.16, -0.10), (0.13,  0.10), (0.16,  0.28),
        (0.52,  0.52), (0.55,  0.62), (0.0,   0.66),
    ],
    "column": [
        (0.0, -0.92), (0.48, -0.88), (0.50, -0.82), (0.36, -0.76),
        (0.32, -0.68), (0.32,  0.66), (0.36,  0.74), (0.50,  0.82),
        (0.48,  0.88), (0.0,   0.92),
    ],
    "bell": [
        (0.0, -0.72), (0.58, -0.65), (0.50, -0.50), (0.40, -0.30),
        (0.30, -0.08), (0.22,  0.18), (0.16,  0.38), (0.13,  0.50),
        (0.17,  0.56), (0.0,   0.62),
    ],
}


class LathedVessel3DSkill(BaseSkill):
    name = "3D Lathed Vessel"
    description = "Object overlay: a surface-of-revolution vessel — amphora, lamp, column, or bell — in palette tones with optional banding."
    kind = "object"
    palette = Palette()
    profile = Enum(
        [("amphora", "Amphora"), ("lamp", "Lamp"), ("column", "Column"), ("bell", "Bell")],
        default="amphora",
    )
    height_scale = Slider(0.7, 1.4, default=1.0, step=0.05)
    banding = Bool(default=True)

    def run(self, canvas):
        img = canvas.new_layer()
        profile = _PROFILES[str(self.profile)]
        h_scale = float(self.height_scale)
        do_bands = bool(self.banding)
        M = 36
        rings = len(profile)

        verts = []
        for (r, y) in profile:
            for j in range(M):
                theta = art_kit.tau * j / M
                verts.append((r * math.cos(theta), y * h_scale, r * math.sin(theta)))

        def vidx(i, j):
            return i * M + (j % M)

        y_min = min(p[1] for p in profile) * h_scale
        y_max = max(p[1] for p in profile) * h_scale
        y_span = (y_max - y_min) or 1e-6
        base_color = art_kit.palette_color(0.55)

        faces = []
        colors = []
        for i in range(rings - 1):
            if do_bands:
                y_mid = (profile[i][1] + profile[i + 1][1]) / 2.0 * h_scale
                t = (y_mid - y_min) / y_span
                row_color = art_kit.palette_color(0.2 + 0.7 * t)
            else:
                row_color = base_color
            for j in range(M):
                faces.append((vidx(i, j), vidx(i + 1, j), vidx(i + 1, j + 1), vidx(i, j + 1)))
                colors.append(row_color)

        vessel = art_kit.mesh(verts, faces, colors=colors)
        art_kit.render_3d(
            img, [vessel],
            camera=(2.4, 1.4, 3.0),
            target=(0, 0.0, 0),
            fov=38,
            outline=canvas.palette.background,
            ambient=0.4,
        )
        canvas.commit(img)
