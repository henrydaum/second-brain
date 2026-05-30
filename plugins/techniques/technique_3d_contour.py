from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette, Pan

import math
from PIL import ImageDraw


class ContourObject3DTechnique(BaseTechnique):
    name = "3D Contour Object"
    description = "Transparent object overlay: a tilted sphere, torus, cylinder, or blob drawn as depth contour curves."
    kind = "object"
    palette = Palette()
    shape = Enum([("sphere", "Sphere"), ("torus", "Torus"), ("cylinder", "Cylinder"), ("blob", "Blob")], default="sphere")
    line_spacing = Slider(4, 24, default=11, step=1)
    line_weight = Slider(1, 10, default=1.2, step=1)
    yaw = Slider(-1.0, 1.0, default=0.35, step=0.05)
    pitch = Slider(-1.0, 1.0, default=0.25, step=0.05)
    view = Pan(x="yaw", y="pitch", label="View")

    def run(self, canvas):
        s = canvas.size
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        scale = s * 0.33
        cyaw, syaw = math.cos(float(self.yaw) * 1.1), math.sin(float(self.yaw) * 1.1)
        cpit, spit = math.cos(float(self.pitch) * 0.9), math.sin(float(self.pitch) * 0.9)

        def project(p):
            x, y, z = p
            y, z = y * cpit - z * spit, y * spit + z * cpit
            x, z = x * cyaw + z * syaw, -x * syaw + z * cyaw
            return (s * 0.5 + x * scale, s * 0.52 - y * scale, z)

        def add_path(paths, pts):
            pp = [project(p) for p in pts]
            paths.append((sum(p[2] for p in pp) / len(pp), [(p[0], p[1]) for p in pp]))

        rings = max(5, int(120 / float(self.line_spacing)))
        steps = 160
        paths = []
        shape = str(self.shape)
        for i in range(rings):
            v = -0.9 + 1.8 * i / max(1, rings - 1)
            if shape == "torus":
                tube, rad = math.sin(v * math.pi) * 0.28, 0.66 + math.cos(v * math.pi) * 0.28
                pts = [(rad * math.cos(t), tube, rad * math.sin(t)) for t in [math.tau * k / steps for k in range(steps + 1)]]
            elif shape == "cylinder":
                pts = [(math.cos(t), v, math.sin(t)) for t in [math.tau * k / steps for k in range(steps + 1)]]
            else:
                r = math.sqrt(max(0.0, 1.0 - v * v))
                pts = []
                for k in range(steps + 1):
                    t = math.tau * k / steps
                    wobble = 1.0 + (0.12 * math.sin(3 * t + canvas.seed) * math.cos(4 * v + canvas.seed) if shape == "blob" else 0.0)
                    pts.append((r * math.cos(t) * wobble, v, r * math.sin(t) * wobble))
            add_path(paths, pts)
        if shape == "cylinder":
            for t in (0, math.pi * 0.5, math.pi, math.pi * 1.5):
                add_path(paths, [(math.cos(t), -0.9 + 1.8 * k / 80, math.sin(t)) for k in range(81)])

        ink = canvas.palette.primary
        shadow = art_kit.with_alpha(canvas.palette.background, 0.55)
        width = max(1, int(round(float(self.line_weight))))
        for _, path in sorted(paths, key=lambda p: p[0]):
            draw.line(path, fill=shadow, width=width + 2, joint="curve")
            draw.line(path, fill=ink, width=width, joint="curve")
        canvas.commit(img)


ContourObject3DTechnique._param_bounds["fill"] = {"type": "enum", "default": "transparent", "allowed": ["transparent", "palette_bg", "primary"]}
