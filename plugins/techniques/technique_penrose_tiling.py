from plugins.BaseTechnique import BaseTechnique, Bool, Enum, Palette, Slider
from PIL import ImageDraw

import math

try:
    art_kit
except NameError:
    art_kit = None


PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _initial_sun(R):
    triangles = []
    for i in range(10):
        a1 = math.pi / 2.0 - i * (math.pi / 5.0)
        a2 = a1 - math.pi / 5.0
        v0 = (0.0, 0.0)
        v1 = (R * math.cos(a1), R * math.sin(a1))
        v2 = (R * math.cos(a2), R * math.sin(a2))
        if i % 2 == 0:
            triangles.append((0, v0, v1, v2))
        else:
            triangles.append((0, v0, v2, v1))
    return triangles


def _subdivide_p2(triangles):
    out = []
    for (color, A, B, C) in triangles:
        if color == 0:
            P = (A[0] + (B[0] - A[0]) / PHI, A[1] + (B[1] - A[1]) / PHI)
            out.append((0, C, P, B))
            out.append((1, P, C, A))
        else:
            Q = (B[0] + (A[0] - B[0]) / PHI, B[1] + (A[1] - B[1]) / PHI)
            R = (B[0] + (C[0] - B[0]) / PHI, B[1] + (C[1] - B[1]) / PHI)
            out.append((1, R, C, A))
            out.append((1, Q, R, B))
            out.append((0, R, Q, A))
    return out


def _subdivide_p3(triangles):
    out = []
    for (color, A, B, C) in triangles:
        if color == 0:
            P = (A[0] + (B[0] - A[0]) / PHI, A[1] + (B[1] - A[1]) / PHI)
            out.append((0, C, P, B))
            out.append((1, P, C, A))
        else:
            Q = (B[0] + (A[0] - B[0]) / PHI, B[1] + (A[1] - B[1]) / PHI)
            out.append((1, Q, C, A))
            out.append((0, C, Q, B))
    return out


class PenroseTilingTechnique(BaseTechnique):
    name = "Penrose Tiling"
    description = "Background: aperiodic Penrose tiling via deflation — kites/darts or thick/thin rhombs — palette-mapped by tile type."
    kind = "background"
    palette = Palette()
    tiling = Enum([("kites_darts", "Kites & Darts"), ("rhombs", "Rhombs")], default="rhombs")
    iterations = Slider(3, 6, default=5, step=1)
    edge_emphasis = Bool(default=True)

    def run(self, canvas):
        img = canvas.new(color=canvas.palette.background)
        size = canvas.size
        draw = ImageDraw.Draw(img, "RGBA")
        iterations = int(round(float(self.iterations)))
        emphasis = bool(self.edge_emphasis)
        tiling = str(self.tiling)

        R = size * 0.85
        triangles = _initial_sun(R)
        sub = _subdivide_p2 if tiling == "kites_darts" else _subdivide_p3
        for _ in range(iterations):
            triangles = sub(triangles)

        cx, cy = size / 2.0, size / 2.0
        thick_color = canvas.palette.primary
        thin_color = canvas.palette.secondary
        outline = canvas.palette.background if emphasis else None
        line_w = max(1, int(round(size / 480)))

        for (color, A, B, C) in triangles:
            pts = [
                (cx + A[0], cy + A[1]),
                (cx + B[0], cy + B[1]),
                (cx + C[0], cy + C[1]),
            ]
            fill = thick_color if color == 0 else thin_color
            if outline is not None and line_w > 1:
                draw.polygon(pts, fill=fill)
                for i in range(3):
                    draw.line([pts[i], pts[(i + 1) % 3]], fill=outline, width=line_w)
            else:
                draw.polygon(pts, fill=fill, outline=outline)

        canvas.commit(img)
