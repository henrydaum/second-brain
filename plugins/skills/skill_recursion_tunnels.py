from plugins.BaseSkill import BaseSkill, Slider, Enum, Palette

import math
import numpy as np
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


_SLOT_NAMES = ["background", "tertiary", "secondary", "primary", "accent"]


class RecursionTunnelsSkill(BaseSkill):
    name = "Recursion Tunnels"
    description = "Psychedelic recursion: posterize the canvas to the palette, then replace selected palette slots with nested rotating polygons that read as inward-spiraling tunnels. Untouched slots stay flat, so the original image silhouette remains readable."
    kind = "filter"
    palette = Palette()
    tunnel_slots = Enum([
        ("primary", "Primary"),
        ("accent", "Accent"),
        ("primary_accent", "Primary + Accent"),
        ("secondary_tertiary", "Secondary + Tertiary"),
    ], default="primary_accent")
    depth = Slider(6, 28, default=16, step=1)
    twist_deg = Slider(2, 20, default=9, step=1)
    polygon_sides = Slider(3, 8, default=4, step=1)

    def run(self, canvas):
        s = canvas.size
        arr = canvas.image_array(mode="RGB", dtype="uint8")

        # Build palette RGB table.
        slot_rgbs = []
        for slot in _SLOT_NAMES:
            hex_color = getattr(canvas.palette, slot)
            slot_rgbs.append(art_kit.hex_to_rgb(hex_color))
        pal = np.array(slot_rgbs, dtype=np.float32)

        # Quantize each pixel to the nearest palette slot.
        flat = arr.reshape(-1, 3).astype(np.float32)
        d = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(axis=-1)
        idx = np.argmin(d, axis=-1).reshape(s, s)

        # Flat posterized output.
        out_arr = pal[idx].astype(np.uint8)
        out = Image.fromarray(out_arr, "RGB").convert("RGBA")

        which = str(self.tunnel_slots)
        tunnel_set = {
            "primary": ["primary"],
            "accent": ["accent"],
            "primary_accent": ["primary", "accent"],
            "secondary_tertiary": ["secondary", "tertiary"],
        }[which]

        sides = int(self.polygon_sides)
        depth = int(self.depth)
        twist = math.radians(float(self.twist_deg))

        for slot in tunnel_set:
            slot_idx = _SLOT_NAMES.index(slot)
            mask_np = (idx == slot_idx)
            if not mask_np.any():
                continue
            ys, xs = np.where(mask_np)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            R = 0.5 * math.hypot(x1 - x0, y1 - y0)
            if R < 8:
                continue

            mask_img = Image.fromarray((mask_np.astype(np.uint8) * 255), "L")
            tunnel_img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
            tdraw = ImageDraw.Draw(tunnel_img, "RGBA")

            base_hex = getattr(canvas.palette, slot)
            bg_hex = canvas.palette.background
            for i in range(depth):
                t = i / max(1, depth - 1)
                r = R * (1.0 - 0.92 * t)
                if r < 1.5:
                    break
                rot = twist * i
                color = art_kit.mix_hex(base_hex, bg_hex, t * 0.85)
                pts = art_kit.regular_polygon(cx, cy, r, sides, rotation=rot)
                tdraw.polygon(pts, fill=color, outline=art_kit.with_alpha(bg_hex, 0.5))

            # Composite tunnel only inside the slot mask.
            out.paste(tunnel_img, (0, 0), mask_img)

        canvas.commit(out)
