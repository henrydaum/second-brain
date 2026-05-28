from plugins.BaseSkill import BaseSkill, Slider, Enum, Palette, Pan

import math
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class PenroseTriangleSkill(BaseSkill):
    name = "Penrose Triangle"
    description = "Impossible tribar overlay. 2D mode draws three palette-colored beams that interlock into Roger Penrose's classic impossible triangle. 3D mode renders an L-tripod of cubes via the tiny 3D painter for a chunky shaded version."
    kind = "object"
    palette = Palette()
    mode = Enum([("2d", "2D Flat"), ("3d", "3D Shaded")], default="2d")
    size = Slider(0.2, 0.9, default=0.6, step=0.02)
    pos_x = Slider(0.0, 1.0, default=0.5, step=0.02)
    pos_y = Slider(0.0, 1.0, default=0.5, step=0.02)
    position = Pan(x="pos_x", y="pos_y", label="Position")

    def run(self, canvas):
        s = canvas.size
        img = canvas.new_layer()
        cx = float(self.pos_x) * s
        cy = float(self.pos_y) * s
        R = float(self.size) * s * 0.45

        if str(self.mode) == "3d":
            self._render_3d(canvas, img, cx, cy, R)
        else:
            self._render_2d(canvas, img, cx, cy, R)
        canvas.commit(img)

    def _render_2d(self, canvas, img, cx, cy, R):
        draw = ImageDraw.Draw(img, "RGBA")
        t = 0.22  # beam thickness as fraction of R
        verts = []
        for i in range(3):
            a = math.radians(90 + i * 120)
            verts.append((cx + R * math.cos(a), cy - R * math.sin(a)))

        def sub(p, q): return (p[0] - q[0], p[1] - q[1])
        def add(p, q): return (p[0] + q[0], p[1] + q[1])
        def mul(p, s): return (p[0] * s, p[1] * s)
        def norm(p):
            d = math.hypot(p[0], p[1]) or 1.0
            return (p[0] / d, p[1] / d)
        def dot(p, q): return p[0] * q[0] + p[1] * q[1]

        bars = []
        for i in range(3):
            A = verts[i]
            B = verts[(i + 1) % 3]
            C = verts[(i + 2) % 3]
            eAB = norm(sub(B, A))
            nAB = (-eAB[1], eAB[0])
            if dot(nAB, sub(C, A)) < 0:
                nAB = (-nAB[0], -nAB[1])
            Ai = add(A, mul(nAB, t * R))
            Bi = add(B, mul(nAB, t * R))
            eBC = norm(sub(C, B))
            B_ext_outer = add(B, mul(eBC, t * R))
            B_ext_inner = add(B_ext_outer, mul(nAB, t * R))
            poly = [A, B, B_ext_outer, B_ext_inner, Bi, Ai]
            bars.append(poly)

        colors = [canvas.palette.primary, canvas.palette.secondary, canvas.palette.tertiary]
        outline = canvas.palette.background
        # Draw order chosen so each bar's extension covers the next one's corner,
        # producing the classic impossible-figure overlap.
        order = [2, 0, 1]
        for k in order:
            draw.polygon(bars[k], fill=colors[k], outline=outline)

    def _render_3d(self, canvas, img, cx, cy, R):
        # Three cube-beam arms meeting near origin, rendered from the classic
        # 30-degree iso angle. Not topologically impossible (3D can't be), but
        # reads as a shaded tribar.
        arm = float(self.size) * 1.6
        thick = arm * 0.22
        meshes = []
        # Three arms along x, y, z axes, each ending near the origin.
        # Each arm: a long cuboid built as one cube_mesh scaled by axis ratio.
        # cube_mesh only supports uniform size; emulate cuboids via stacked cubes.
        steps = 7
        for axis, slot in enumerate(["primary", "secondary", "tertiary"]):
            color = getattr(canvas.palette, slot)
            for i in range(steps):
                t = (i + 0.5) / steps
                offset = -arm * 0.5 + arm * t
                center = [0.0, 0.0, 0.0]
                center[axis] = offset
                meshes.append(art_kit.cube_mesh(size=thick, center=tuple(center), color=color))
        # Project into a temporary square image, then composite at (cx, cy).
        side = int(R * 2.8)
        if side < 16:
            return
        tmp = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        art_kit.render_3d(
            tmp, meshes,
            camera=(2.6, 2.0, 2.6), target=(0, 0, 0), fov=38,
            outline=canvas.palette.background,
        )
        img.alpha_composite(tmp, (int(cx - side / 2), int(cy - side / 2)))
