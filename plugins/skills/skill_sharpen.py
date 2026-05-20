from plugins.BaseSkill import BaseSkill

from PIL import ImageFilter

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class SharpenSkill(BaseSkill):
    name = 'Sharpen'
    description = 'Crisp up edge detail with an unsharp mask. Good final-pass after a creation skill or any softening transform. Params: radius (1.0-3.0, default 1.5), percent (50-200, default 140), threshold (0-10, default 2).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False

    def run(self, canvas, radius=1.5, percent=140, threshold=2):
        radius = float(art_kit.clamp(radius, 0.2, 6.0))
        percent = int(art_kit.clamp(percent, 0, 400))
        threshold = int(art_kit.clamp(threshold, 0, 20))
        sharpened = canvas.image.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
        )
        canvas.commit(sharpened)
