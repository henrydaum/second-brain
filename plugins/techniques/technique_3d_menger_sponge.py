from plugins.BaseTechnique import BaseTechnique, Bool, Palette, Pan, Slider

import math

try:
    art_kit
except NameError:
    art_kit = None


class MengerSponge3DTechnique(BaseTechnique):
    name = "3D Menger Sponge"
    description = "Object overlay: a palette-lit Menger sponge fractal, with pan controlling yaw and pitch."
    kind = "object"
    palette = Palette()
    yaw = Slider(0, 1, default=0.58, step=0.03)
    pitch = Slider(0, 1, default=0.36, step=0.03)
    rotation = Pan(x="yaw", y="pitch")
    scale = Slider(0.55, 1.35, default=0.95, step=0.05)
    depth = Slider(1, 3, default=2, step=1, label="Depth")
    outline = Bool(default=False)

    def _centers(self, depth):
        cubes = [(0.0, 0.0, 0.0, 1.55 * float(self.scale))]
        for _ in range(depth):
            next_cubes = []
            for cx, cy, cz, size in cubes:
                step = size / 3
                for x in (-1, 0, 1):
                    for y in (-1, 0, 1):
                        for z in (-1, 0, 1):
                            if (x == 0) + (y == 0) + (z == 0) < 2:
                                next_cubes.append((cx + x * step, cy + y * step, cz + z * step, step))
            cubes = next_cubes
        return cubes

    def run(self, canvas):
        yaw = (float(self.yaw) - 0.5) * art_kit.tau
        pitch = (float(self.pitch) - 0.5) * 1.5
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)

        def rot(p):
            x, y, z = p
            x, z = x * cy - z * sy, x * sy + z * cy
            y, z = y * cp - z * sp, y * sp + z * cp
            return (x, y, z)

        meshes = []
        for cx, y, cz, size in self._centers(int(round(float(self.depth)))):
            cube = art_kit.cube_mesh(size=size, center=(cx, y, cz), color=art_kit.palette_color(0.25 + 0.55 * ((y / (1.55 * self.scale)) + 0.5)))
            meshes.append(art_kit.mesh([rot(v) for v in cube.vertices], cube.faces, color=cube.color))
        img = canvas.new_layer()
        art_kit.render_3d(img, meshes, camera=(2.8, 2.2, 3.7), target=(0, 0, 0), fov=33, outline=canvas.palette.background if self.outline else None, ambient=0.42)
        canvas.commit(img)
