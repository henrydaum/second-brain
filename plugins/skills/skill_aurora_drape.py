from plugins.BaseSkill import BaseSkill, Palette, Slider

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class AuroraDrapeSkill(BaseSkill):
    name = "Aurora Drape"
    description = "Object overlay: vertical luminous curtains — aurora-style — driven by FBM noise with palette gradient and soft vertical falloff."
    kind = "object"
    palette = Palette()
    drape_count = Slider(1, 4, default=2, step=1)
    intensity = Slider(0.3, 1.0, default=0.75, step=0.05)
    sway = Slider(0.0, 1.0, default=0.5, step=0.05)

    def run(self, canvas):
        size = canvas.size
        n_drapes = int(round(float(self.drape_count)))
        intensity = float(self.intensity)
        sway = float(self.sway)
        seed = canvas.seed

        H = W = size
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        u = xs / W
        v = ys / H

        if n_drapes == 1:
            centers = [0.5]
        else:
            centers = list(np.linspace(0.18, 0.82, n_drapes))
        drape_width = 0.20 if n_drapes <= 2 else 0.14

        sway_field = art_kit.fbm_grid(seed, u * 1.5, v * 0.8, octaves=3) - 0.5
        sway_offset = (sway_field * sway * 0.12).astype(np.float32)

        drape_mask = np.zeros_like(u)
        color_pick = np.zeros_like(u)
        denom = max(1, len(centers) - 1)
        for i, c in enumerate(centers):
            c_eff = c + sway_offset
            dx = u - c_eff
            mask = np.exp(-(dx / drape_width) ** 2)
            stronger = mask > drape_mask
            drape_mask = np.where(stronger, mask, drape_mask)
            color_pick = np.where(stronger, np.float32(i / denom), color_pick)

        v_in = np.clip(v / 0.30, 0, 1)
        v_in = v_in * v_in * (3 - 2 * v_in)
        v_out = 1.0 - np.clip((v - 0.65) / 0.30, 0, 1)
        v_out = v_out * v_out * (3 - 2 * v_out)
        v_alpha = v_in * v_out

        noise = art_kit.fbm_grid(seed * 13 + 7, u * 6.0, v * 9.0, octaves=4).astype(np.float32)
        alpha = drape_mask * v_alpha * (0.4 + 0.6 * noise) * intensity

        lut_n = 32
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(t)) for t in np.linspace(0.2, 0.95, lut_n)],
            dtype=np.float32,
        ) / 255.0

        t_pix = np.clip(0.2 + 0.5 * color_pick + 0.25 * noise, 0, 0.999) * (lut_n - 1)
        i0 = t_pix.astype(np.int32)
        i1 = np.minimum(i0 + 1, lut_n - 1)
        f = (t_pix - i0)[..., None]
        color_rgb = lut[i0] * (1 - f) + lut[i1] * f

        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[..., :3] = color_rgb
        rgba[..., 3] = np.clip(alpha, 0, 1)

        canvas.commit_array(rgba)
