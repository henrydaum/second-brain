from plugins.BaseSkill import BaseSkill, Slider

from PIL import ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class GaussianBlurSkill(BaseSkill):
    name = 'Gaussian Blur'
    description = 'Standard Gaussian blur. The workhorse softening pass — use before sharpen for a "dreamy" look or to smooth high-frequency noise.'
    kind = "filter"

    radius = Slider(0.5, 40.0, default=3.0, step=0.5)

    def run(self, canvas):
        out = canvas.image.filter(ImageFilter.GaussianBlur(self.radius))
        canvas.commit(out.convert("RGBA"))
