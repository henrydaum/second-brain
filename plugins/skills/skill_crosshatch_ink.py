from plugins.BaseSkill import BaseSkill, Enum, Palette, Slider

import math
import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class CrosshatchInkSkill(BaseSkill):
    name = "Crosshatch Ink"
    description = "Filter: luminance-driven pen-and-ink crosshatch overlay — line-based shading on a paper-tinted base."
    kind = "filter"
    palette = Palette()
    density = Slider(0.2, 1.0, default=0.65, step=0.02)
    paper_tint = Slider(0.0, 0.3, default=0.08, step=0.01)
    angle_set = Enum(
        [("classic_4", "Classic 4 angles"), ("tight_2", "Tight 2 angles")],
        default="classic_4",
    )

    def run(self, canvas):
        size = canvas.size
        arr = canvas.image_array(mode="RGB", dtype="float")
        L = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

        density = float(self.density)
        paper_tint = float(self.paper_tint)
        angle_choice = str(self.angle_set)

        base_spacing = max(4, int(round(20.0 / (0.4 + density))))
        thickness = max(1.0, base_spacing * 0.18)

        bg_rgb = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        paper = bg_rgb * (1 - paper_tint) + np.array([1.0, 1.0, 1.0], dtype=np.float32) * paper_tint
        ink_rgb = np.array(art_kit.hex_to_rgb(canvas.palette.primary), dtype=np.float32) / 255.0

        H = W = size
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

        if angle_choice == "tight_2":
            layers = [(45.0, 0.65), (135.0, 0.45)]
        else:
            layers = [(0.0, 0.70), (45.0, 0.55), (135.0, 0.40), (90.0, 0.25)]

        ink_mask = np.zeros((H, W), dtype=bool)
        for angle_deg, threshold in layers:
            angle = math.radians(angle_deg)
            perp = xs * math.cos(angle) + ys * math.sin(angle)
            mod = perp % base_spacing
            dist = np.minimum(mod, base_spacing - mod)
            on_line = dist < thickness / 2.0
            ink_mask |= on_line & (L < threshold)

        paper_full = np.broadcast_to(paper, (H, W, 3))
        out_arr = np.where(ink_mask[..., None], ink_rgb, paper_full)
        canvas.commit_array(out_arr.astype(np.float32))
