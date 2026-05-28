from plugins.BaseSkill import BaseSkill, Slider, Enum, Palette

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


class ContourObject3DSkill(BaseSkill):
    name = "3D Contour Object"
    description = "Object overlay: a 3D primitive (sphere, torus, cylinder, or noise-modulated blob) drawn purely as horizontal depth contour lines instead of shaded faces. Like the topographic map of a single 3D form."
    kind = "object"
    palette = Palette()
    shape = Enum([("sphere", "Sphere"), ("torus", "Torus"), ("cylinder", "Cylinder"), ("blob", "Blob")], default="sphere")
    line_spacing = Slider(4, 24, default=11, step=1)
    line_weight = Slider(0.5, 2.5, default=1.0, step=0.1)
    fill = Enum([("transparent", "Transparent"), ("palette_bg", "Palette BG"), ("primary", "Primary")], default="transparent")

    def run(self, canvas):
        s = canvas.size
        seed = canvas.seed
        _, _, nx, ny = art_kit.centered_grid(s)
        # Use ~70% radius to leave margin.
        nx = nx / 0.7
        ny = ny / 0.7

        shape = str(self.shape)
        if shape == "sphere":
            r2 = nx * nx + ny * ny
            mask = r2 <= 1.0
            z = np.zeros_like(nx)
            z[mask] = np.sqrt(np.clip(1.0 - r2[mask], 0.0, 1.0))
        elif shape == "cylinder":
            mask = (np.abs(nx) <= 1.0) & (np.abs(ny) <= 0.8)
            z = np.zeros_like(nx)
            z[mask] = np.sqrt(np.clip(1.0 - nx[mask] ** 2, 0.0, 1.0))
        elif shape == "torus":
            R = 0.65
            r = 0.30
            d = np.sqrt(nx * nx + ny * ny) - R
            inside = (r * r) - (d * d)
            mask = inside > 0
            z = np.zeros_like(nx)
            z[mask] = np.sqrt(np.clip(inside[mask], 0.0, 1.0))
        else:  # blob
            yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
            h = art_kit.fbm_grid(seed, xx * 0.012, yy * 0.012, octaves=4).astype(np.float32)
            r2 = nx * nx + ny * ny
            falloff = np.clip(1.0 - r2 * 0.9, 0.0, 1.0)
            z = h * falloff * 1.2
            mask = z > 0.02

        # Quantize depth into bands.
        spacing = float(self.line_spacing) / 100.0
        z_norm = z.copy()
        if z_norm[mask].size > 0:
            zmin = float(z_norm[mask].min())
            zmax = float(z_norm[mask].max())
            span = max(zmax - zmin, 1e-6)
            z_norm = (z_norm - zmin) / span
        bands = np.floor(z_norm / max(spacing, 1e-3)).astype(np.int32)
        bands[~mask] = -1

        # Boundaries.
        dx = np.zeros_like(bands)
        dy = np.zeros_like(bands)
        dx[:, 1:] = bands[:, 1:] - bands[:, :-1]
        dy[1:, :] = bands[1:, :] - bands[:-1, :]
        boundary = ((dx != 0) | (dy != 0)) & mask
        if float(self.line_weight) >= 1.5:
            b2 = boundary.copy()
            b2[:-1, :] |= boundary[1:, :]
            b2[:, :-1] |= boundary[:, 1:]
            boundary = b2

        # Compose RGBA output.
        out = np.zeros((s, s, 4), dtype=np.float32)
        fill_mode = str(self.fill)
        from PIL import ImageColor
        if fill_mode == "palette_bg":
            fill_rgb = np.array(ImageColor.getrgb(canvas.palette.background), dtype=np.float32) / 255.0
            out[mask, :3] = fill_rgb
            out[mask, 3] = 1.0
        elif fill_mode == "primary":
            fill_rgb = np.array(ImageColor.getrgb(canvas.palette.primary), dtype=np.float32) / 255.0
            out[mask, :3] = fill_rgb
            out[mask, 3] = 0.85
        # contour ink: silhouette outline + bands
        line_rgb = np.array([0.08, 0.08, 0.10], dtype=np.float32)
        # Silhouette edge
        sil = mask.copy()
        sil_x = np.zeros_like(mask)
        sil_y = np.zeros_like(mask)
        sil_x[:, 1:] = mask[:, 1:] != mask[:, :-1]
        sil_y[1:, :] = mask[1:, :] != mask[:-1, :]
        silhouette = sil_x | sil_y
        out[silhouette, :3] = line_rgb
        out[silhouette, 3] = 1.0
        out[boundary, :3] = line_rgb
        out[boundary, 3] = 1.0

        canvas.commit_array(out)
