from plugins.BaseTechnique import BaseTechnique, Slider

from PIL import Image, ImageChops, ImageFilter, ImageOps

try:
    art_kit
except NameError:
    art_kit = None


class BloomGlowTechnique(BaseTechnique):
    name = 'Bloom Glow'
    description = 'Highlight bloom: extract bright pixels, blur them, screen-blend back over the image. Adds atmosphere to suns, lights, glowing edges.'
    kind = "filter"

    radius    = Slider(1, 80, default=18, step=1)
    strength  = Slider(0.0, 1.5, default=0.75, step=0.05)
    threshold = Slider(0, 255, default=165, step=5)

    def run(self, canvas):
        img = canvas.image.convert("RGB")
        gray = ImageOps.grayscale(img)
        mask = gray.point(lambda v, t=int(self.threshold): 255 if v > t else 0)
        blurred = img.filter(ImageFilter.GaussianBlur(float(self.radius)))
        glowed = ImageChops.screen(img, blurred)
        composite = Image.composite(glowed, img, mask)
        out = Image.blend(img, composite, float(self.strength))
        canvas.commit(out.convert("RGBA"))
