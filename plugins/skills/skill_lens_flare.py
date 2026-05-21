from plugins.BaseSkill import BaseSkill, Slider, Pan, Palette

from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class LensFlareSkill(BaseSkill):
    name = 'Lens Flare'
    description = 'Palette-tinted lens flare: a glowing source point with a string of secondary ghost discs along the line from the source through the canvas center.'
    kind = 'transform'

    palette    = Palette()
    brightness = Slider(0.0, 1.5, default=0.85, step=0.05)
    sx         = Slider(0.0, 1.0, default=0.25, step=0.05)
    sy         = Slider(0.0, 1.0, default=0.25, step=0.05)
    source     = Pan(x='sx', y='sy')

    def run(self, canvas):
        s = canvas.size
        b = float(self.brightness)
        n = 5

        base = canvas.image.convert("RGBA")
        flare = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        draw = ImageDraw.Draw(flare, "RGBA")

        px = self.sx * s
        py = self.sy * s
        cx = s / 2.0
        cy = s / 2.0

        main_color = art_kit.hex_to_rgb(canvas.palette.accent)
        main_r = s * 0.20
        for i in range(8, 0, -1):
            r = main_r * (i / 8.0)
            alpha = int(min(255, 30 * i * b))
            draw.ellipse((px - r, py - r, px + r, py + r), fill=(*main_color, alpha))

        dx = cx - px
        dy = cy - py
        for i in range(1, n + 1):
            t = i / float(n + 1) * 1.8
            gx = px + dx * t * 2.0
            gy = py + dy * t * 2.0
            ratio = i / max(1, n)
            color = art_kit.hex_to_rgb(art_kit.palette_color(ratio))
            r = s * (0.015 + 0.05 * (1.0 - ratio))
            alpha = int(120 * b * (1.0 - 0.4 * ratio))
            draw.ellipse((gx - r, gy - r, gx + r, gy + r), fill=(*color, alpha))

        flare = flare.filter(ImageFilter.GaussianBlur(s * 0.008))

        streak = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(streak, "RGBA")
        streak_color = (*art_kit.hex_to_rgb(canvas.palette.primary), int(140 * b))
        sdraw.line((px - s, py, px + s, py), fill=streak_color, width=max(1, int(s * 0.004)))
        sdraw.line((px, py - s, px, py + s), fill=streak_color, width=max(1, int(s * 0.004)))
        streak = streak.filter(ImageFilter.GaussianBlur(s * 0.006))

        out = Image.alpha_composite(base, flare)
        out = Image.alpha_composite(out, streak)
        canvas.commit(out)
