from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette, Pan

import math
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class PenroseTriangleTechnique(BaseTechnique):
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
        w = max(10, int(R * 0.18))
        A = (cx, cy - R)
        B = (cx + R * 0.87, cy + R * 0.50)
        C = (cx - R * 0.87, cy + R * 0.50)
        colors = [canvas.palette.primary, canvas.palette.secondary, canvas.palette.tertiary]
        outline = canvas.palette.background

        def beam(p, q, color, a=0.0, b=1.0):
            x0, y0 = p[0] + (q[0] - p[0]) * a, p[1] + (q[1] - p[1]) * a
            x1, y1 = p[0] + (q[0] - p[0]) * b, p[1] + (q[1] - p[1]) * b
            dx, dy = x1 - x0, y1 - y0
            d = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / d * w * 0.5, dx / d * w * 0.5
            poly = [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)]
            draw.polygon(poly, fill=outline)
            poly = [(x0 + nx * 0.92, y0 + ny * 0.92), (x1 + nx * 0.92, y1 + ny * 0.92), (x1 - nx * 0.92, y1 - ny * 0.92), (x0 - nx * 0.92, y0 - ny * 0.92)]
            draw.polygon(poly, fill=color)

        beam(C, A, colors[2])
        beam(B, C, colors[1])
        beam(A, B, colors[0])
        beam(C, A, colors[2], 0.76, 1.0)

    def _render_3d(self, canvas, img, cx, cy, R):
        # Three cube-beam arms meeting near origin, rendered from the classic
        # 30-degree iso angle. Not topologically impossible (3D can't be), but
        # reads as a shaded tribar.
        arm = 2.25
        thick = arm * 0.18
        meshes = []
        # Three arms along x, y, z axes, each ending near the origin.
        # Each arm: a long cuboid built as one cube_mesh scaled by axis ratio.
        # cube_mesh only supports uniform size; emulate cuboids via stacked cubes.
        steps = 7
        colors = [canvas.palette.primary, canvas.palette.secondary, canvas.palette.tertiary]
        for axis, color in enumerate(colors):
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
