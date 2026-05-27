from plugins.BaseSkill import BaseSkill, Enum, Palette, Slider
from PIL import ImageDraw

import math

try:
    art_kit
except NameError:
    art_kit = None


class ArchWindowFrameSkill(BaseSkill):
    name = "Architectural Window Frame"
    description = "Object overlay: an arched window — round, lancet, gothic, or ogee — with palette-tinted glass and mullion dividers."
    kind = "object"
    palette = Palette()
    arch_style = Enum(
        [("round", "Round"), ("lancet", "Lancet"), ("gothic", "Gothic"), ("ogee", "Ogee")],
        default="gothic",
    )
    mullion_count = Slider(0, 6, default=2, step=1)
    glass_alpha = Slider(0.0, 0.55, default=0.18, step=0.02)

    def _arch_points(self, style, left, right, baseline, n=64):
        width = right - left
        cx = (left + right) / 2.0
        pts = []
        if style == "round":
            R = width / 2.0
            for i in range(n + 1):
                theta = math.pi - math.pi * i / n
                pts.append((cx + R * math.cos(theta), baseline - R * math.sin(theta)))
        elif style == "ogee":
            R = width / 2.0
            peak_y = baseline - width * 0.78
            for i in range(n + 1):
                t = i / n
                x = left + t * width
                base = baseline - math.sqrt(max(0.0, R * R - (x - cx) * (x - cx)))
                proximity = 1.0 - abs(t - 0.5) * 2.0
                peak_pull = max(0.0, proximity - 0.6) * 2.5
                pts.append((x, base + (peak_y - base) * peak_pull))
        else:
            delta = -width * 0.20 if style == "lancet" else 0.0
            R = width - delta
            for i in range(n + 1):
                t = i / n
                x = left + t * width
                cxa = (right - delta) if t < 0.5 else (left + delta)
                y_below = math.sqrt(max(0.0, R * R - (x - cxa) * (x - cxa)))
                pts.append((x, baseline - y_below))
        return pts

    def run(self, canvas):
        img = canvas.new_layer()
        size = canvas.size
        draw = ImageDraw.Draw(img, "RGBA")

        margin_x = size * 0.16
        margin_top = size * 0.06
        left = margin_x
        right = size - margin_x
        width = right - left
        body_top = size * 0.50
        body_bottom = size * 0.90

        style = str(self.arch_style)
        mullions = int(round(float(self.mullion_count)))
        glass_alpha = float(self.glass_alpha)
        outline = canvas.palette.background

        arch_pts = self._arch_points(style, left, right, body_top, n=64)
        max_arch_height = body_top - margin_top
        actual = body_top - min(p[1] for p in arch_pts)
        if actual > max_arch_height and actual > 0:
            scale = max_arch_height / actual
            arch_pts = [(p[0], body_top - (body_top - p[1]) * scale) for p in arch_pts]

        full_polygon = list(arch_pts) + [(right, body_bottom), (left, body_bottom)]

        if glass_alpha > 0.001:
            glass_color = art_kit.with_alpha(canvas.palette.primary, int(255 * glass_alpha))
            draw.polygon(full_polygon, fill=glass_color)
            if mullions > 0:
                for m in range(mullions):
                    t = (m + 1) / (mullions + 1)
                    color_t = 0.2 + 0.6 * t
                    tint = art_kit.with_alpha(art_kit.palette_color(color_t), int(255 * glass_alpha * 0.7))
                    x_band_left = left + width * t - width * 0.04
                    x_band_right = left + width * t + width * 0.04
                    draw.rectangle([x_band_left, body_top, x_band_right, body_bottom], fill=tint)

        frame_w = max(3, int(round(size / 90)))
        for i in range(len(arch_pts) - 1):
            draw.line([arch_pts[i], arch_pts[i + 1]], fill=outline, width=frame_w)
        draw.line([(left, body_top), (left, body_bottom)], fill=outline, width=frame_w)
        draw.line([(right, body_top), (right, body_bottom)], fill=outline, width=frame_w)
        draw.line([(left, body_bottom), (right, body_bottom)], fill=outline, width=frame_w)

        if mullions > 0:
            mul_w = max(2, frame_w // 2)
            for m in range(mullions):
                t = (m + 1) / (mullions + 1)
                x = left + width * t
                idx = int(round(t * (len(arch_pts) - 1)))
                arch_y = arch_pts[idx][1]
                draw.line([(x, arch_y + frame_w), (x, body_bottom)], fill=outline, width=mul_w)
            tr_w = max(2, frame_w // 2)
            draw.line([(left, body_top), (right, body_top)], fill=outline, width=tr_w)

        canvas.commit(img)
