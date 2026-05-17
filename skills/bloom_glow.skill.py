SKILL_NAME = "Bloom Glow"
SKILL_DESCRIPTION = "Highlight bloom: extract bright pixels, blur them, screen-blend back over the image. Adds atmosphere to suns, lights, glowing edges. Params: radius (1-60, default 18), strength (0.0-1.5, default 0.75), threshold (0-255, default 165)."
SKILL_KIND = "transform"
SKILL_OWNER = "library"
SKILL_CREATED_AT = 1730000000.0

from PIL import Image, ImageChops, ImageFilter, ImageOps


def run(canvas, radius=18, strength=0.75, threshold=165):
    img = canvas.image.convert("RGB")
    r = float(art_kit.clamp(radius, 1, 80))
    strength = float(art_kit.clamp(strength, 0.0, 1.5))
    th = int(art_kit.clamp(threshold, 0, 255))

    gray = ImageOps.grayscale(img)
    mask = gray.point(lambda v, t=th: 255 if v > t else 0)
    blurred = img.filter(ImageFilter.GaussianBlur(r))
    glowed = ImageChops.screen(img, blurred)
    composite = Image.composite(glowed, img, mask)
    out = Image.blend(img, composite, strength)
    canvas.commit(out.convert("RGBA"))
