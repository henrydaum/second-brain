"""Classic pixel operations. All functions return RGBA; none mutate the input.

Geometry changes pixel dimensions. Colour adjustments preserve alpha. Objects
contain only their own pixels. File reads use SDK helpers, never raw paths in PIL.
"""
import math
from .art_kit import read_image, write_png, resize_image, crop_image, load_font, text
from .canvas_catalog import prepare

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]


def rgba(value, palette):
    """Literal CSS/hex colour or a live @role, never an implicit photo recolour."""
    from PIL import ImageColor
    if value.startswith("@"):
        role = value[1:]
        colors = palette.get("colors", {})
        if role not in colors:
            raise ValueError(f"unknown palette role {value}; available: {list(colors)}")
        value = colors[role]
    if value == "transparent":
        return (0, 0, 0, 0)
    return ImageColor.getcolor(value, "RGBA")


def execute(sdk, script, operation, kind, input_path, output_path,
            width, height, seed, palette, controls):
    from PIL import Image
    controls = prepare(script, controls, kind)["controls"]
    source = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    image = operation(sdk, source, controls, palette)
    return {"path": write_png(sdk, output_path, image), "width": image.width, "height": image.height}


def load_image(sdk, image, c, palette):
    loaded = read_image(sdk, c["path"])
    return loaded if c["fit"] == "native" else resize_image(loaded, image.size, c["fit"])


def crop(sdk, image, c, palette):
    return crop_image(image, (c["left"], c["top"], c["right"], c["bottom"]))


def resize(sdk, image, c, palette):
    return resize_image(image, (c["width"], c["height"]), c["fit"])


def rotate(sdk, image, c, palette):
    from PIL import Image
    angle = c["angle"] % 360
    if angle == 0:
        return image.copy()
    if angle in (90, 180, 270) and (c["expand"] or angle == 180 or image.width == image.height):
        method = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
                  270: Image.Transpose.ROTATE_270}[angle]
        return image.transpose(method)
    return image.convert("RGBa").rotate(angle, Image.Resampling.BICUBIC,
                                        expand=c["expand"], fillcolor=(0, 0, 0, 0)).convert("RGBA")


def flip(sdk, image, c, palette):
    from PIL import ImageOps
    result = ImageOps.mirror(image) if c["horizontal"] else image.copy()
    return ImageOps.flip(result) if c["vertical"] else result


def _rgb_map(image, operation):
    import numpy as np
    from PIL import Image
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    output = np.uint8(np.clip(operation(rgb) * 255 + .5, 0, 255))
    result = Image.fromarray(output).convert("RGBA")
    result.putalpha(image.getchannel("A"))
    return result


def brightness(sdk, image, c, palette):
    return _rgb_map(image, lambda rgb: rgb * c["factor"])


def contrast(sdk, image, c, palette):
    import numpy as np
    weights = np.asarray(image.getchannel("A"), dtype=np.float32) / 255
    def adjust(rgb):
        lum = rgb @ np.array([.299, .587, .114], dtype=np.float32)
        total = float(weights.sum())
        pivot = float((lum * weights).sum()) / total if total else 0
        return (rgb - pivot) * c["factor"] + pivot
    return _rgb_map(image, adjust)


def saturation(sdk, image, c, palette):
    import numpy as np
    def adjust(rgb):
        gray = (rgb @ np.array([.299, .587, .114], dtype=np.float32))[..., None]
        return gray + (rgb - gray) * c["factor"]
    return _rgb_map(image, adjust)


def exposure(sdk, image, c, palette):
    import numpy as np
    def adjust(rgb):
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        linear = np.clip(linear * 2 ** c["stops"], 0, 1)
        return np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
    return _rgb_map(image, adjust)


def gamma(sdk, image, c, palette):
    return _rgb_map(image, lambda rgb: rgb ** (1 / c["gamma"]))


def _weighted_blur(image, radius):
    """Float premultiplication avoids 8-bit alpha rounding before filtering."""
    import numpy as np
    if radius == 0:
        return np.asarray(image, dtype=np.float32) / 255
    arr = np.asarray(image, dtype=np.float32) / 255
    arr[..., :3] *= arr[..., 3:]
    blurred = np.empty_like(arr)
    for channel in range(4):
        blurred[..., channel] = _gaussian_plane(arr[..., channel], radius)
    alpha = blurred[..., 3:]
    blurred[..., :3] = np.divide(blurred[..., :3], alpha,
                                 out=np.zeros_like(blurred[..., :3]), where=alpha > 1e-8)
    return blurred


def _gaussian_plane(plane, radius):
    """Separable Gaussian convolution in float, with replicated border pixels.

    Small radii use the sampled kernel directly; large radii use three box
    passes matched to its variance so cost does not grow with kernel area.
    """
    import numpy as np
    if radius <= 2:
        extent = max(1, math.ceil(radius * 3))
        positions = np.arange(-extent, extent + 1, dtype=np.float32)
        kernel = np.exp(-positions ** 2 / (2 * radius ** 2))
        kernel /= kernel.sum()
        for axis in (0, 1):
            pads = [(0, 0), (0, 0)]
            pads[axis] = (extent, extent)
            padded = np.pad(plane, pads, mode="edge")
            result = np.zeros_like(plane)
            for i, weight in enumerate(kernel):
                slices = [slice(None), slice(None)]
                slices[axis] = slice(i, i + plane.shape[axis])
                result += padded[tuple(slices)] * weight
            plane = result
        return plane
    lower = int(math.sqrt(4 * radius * radius + 1))
    if lower % 2 == 0:
        lower -= 1
    # Blend adjacent odd box widths to match variance continuously. Choosing
    # only integer widths would make small radius edits do nothing, then jump.
    low_variance = (lower * lower - 1) / 12
    high_variance = ((lower + 2) ** 2 - 1) / 12
    mix = (radius * radius / 3 - low_variance) / (high_variance - low_variance)
    for _ in range(3):
        for axis in (0, 1):
            plane = _box_plane(plane, axis, lower) * (1 - mix) + _box_plane(plane, axis, lower + 2) * mix
    return plane


def _box_plane(plane, axis, width):
    import numpy as np
    extent = width // 2
    pads = [(0, 0), (0, 0)]
    pads[axis] = (extent, extent)
    sums = np.cumsum(np.pad(plane, pads, mode="edge"), axis=axis, dtype=np.float64)
    pads[axis] = (1, 0)
    sums = np.pad(sums, pads)
    first, last = [slice(None), slice(None)], [slice(None), slice(None)]
    first[axis], last[axis] = slice(width, None), slice(None, -width)
    return ((sums[tuple(first)] - sums[tuple(last)]) / width).astype(np.float32)


def blur(sdk, image, c, palette):
    import numpy as np
    from PIL import Image
    if c["radius"] == 0:
        return image.copy()
    return Image.fromarray(np.uint8(np.clip(_weighted_blur(image, c["radius"]) * 255 + .5, 0, 255)))


def sharpen(sdk, image, c, palette):
    import numpy as np
    if c["amount"] == 0 or c["radius"] == 0:
        return image.copy()
    soft = _weighted_blur(image, c["radius"])[..., :3]
    def adjust(rgb):
        detail = rgb - soft
        detail = np.where(np.abs(detail) * 255 >= c["threshold"], detail, 0)
        return rgb + detail * c["amount"] / 100
    return _rgb_map(image, adjust)


def grayscale(sdk, image, c, palette):
    return saturation(sdk, image, {"factor": 0}, palette)


def invert(sdk, image, c, palette):
    return _rgb_map(image, lambda rgb: 1 - rgb)


def solid(sdk, image, c, palette):
    from PIL import Image
    return Image.new("RGBA", image.size, rgba(c["color"], palette))


def gradient(sdk, image, c, palette):
    import numpy as np
    from PIL import Image
    angle = math.radians(c["angle"])
    dx, dy = math.cos(angle), math.sin(angle)
    x = np.arange(image.width, dtype=np.float32) * dx
    y = np.arange(image.height, dtype=np.float32) * dy
    values = y[:, None] + x[None, :]
    lo, hi = float(values.min()), float(values.max())
    t = ((values - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(values))[..., None]
    a, b = (np.array(rgba(c[key], palette), dtype=np.float32) / 255 for key in ("start", "end"))
    a[:3] *= a[3]
    b[:3] *= b[3]
    result = a * (1 - t) + b * t
    result[..., :3] = np.divide(result[..., :3], result[..., 3:],
                                out=np.zeros_like(result[..., :3]), where=result[..., 3:] > 0)
    return Image.fromarray(np.uint8(np.clip(result * 255 + .5, 0, 255)))


def _overlay(size, paint):
    """Supersampling for deterministic antialiasing of shape boundaries."""
    from PIL import Image, ImageDraw
    scale = 2
    overlay = Image.new("RGBA", (size[0] * scale, size[1] * scale))
    paint(ImageDraw.Draw(overlay), scale)
    return resize_image(overlay, size)


def line(sdk, image, c, palette):
    def paint(draw, scale):
        points = [(x * scale, y * scale) for x, y in c["points"]]
        draw.line(points, fill=rgba(c["color"], palette), width=max(1, round(c["width"] * scale)), joint="curve")
    return _overlay(image.size, paint)


def shape(sdk, image, c, palette):
    def paint(draw, scale):
        bounds = (c["left"] * scale, c["top"] * scale,
                  c["right"] * scale - 1, c["bottom"] * scale - 1)
        kwargs = {"fill": rgba(c["fill"], palette)}
        if c["stroke_width"] > 0 and rgba(c["stroke"], palette)[3] > 0:
            kwargs.update(outline=rgba(c["stroke"], palette), width=max(1, round(c["stroke_width"] * scale)))
        if c["shape"] == "rectangle":
            draw.rectangle(bounds, **kwargs)
        else:
            draw.ellipse(bounds, **kwargs)
    return _overlay(image.size, paint)


def draw_text(sdk, image, c, palette):
    from PIL import Image
    result = Image.new("RGBA", image.size)
    font = load_font(sdk, c["font_path"], c["size"]) if c["font_path"] else None
    text(result, (c["x"], c["y"]), c["content"], size=c["size"], color=rgba(c["color"], palette),
         max_width=c["max_width"] or None, align=c["align"], font=font)
    return result


def duotone(sdk, image, c, palette):
    import numpy as np
    shadows = np.array(rgba(c["shadows"], palette)[:3], dtype=np.float32) / 255
    highlights = np.array(rgba(c["highlights"], palette)[:3], dtype=np.float32) / 255
    def adjust(rgb):
        lum = (rgb @ np.array([.299, .587, .114], dtype=np.float32))[..., None]
        mapped = shadows * (1 - lum) + highlights * lum
        return rgb * (1 - c["amount"]) + mapped * c["amount"]
    return _rgb_map(image, adjust)


def vignette(sdk, image, c, palette):
    import numpy as np
    x = (np.arange(image.width, dtype=np.float32) + .5) / image.width * 2 - 1
    y = (np.arange(image.height, dtype=np.float32) + .5) / image.height * 2 - 1
    distance = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2) / math.sqrt(2)
    t = np.clip((distance - c["radius"]) / max(1e-6, 1 - c["radius"]), 0, 1)
    weight = (1 - c["amount"] * t * t * (3 - 2 * t))[..., None]
    return _rgb_map(image, lambda rgb: rgb * weight)
