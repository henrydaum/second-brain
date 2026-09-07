"""Grayscale. Metadata and implementation live together.

Convert RGB to luminance while preserving alpha.
"""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Grayscale',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Convert RGB to luminance while preserving alpha.',
 'controls': {},
 'example': {},
 'tags': 'black white monochrome greyscale',
 'aliases': ['canvas_grayscale']}


def apply(sdk, image, controls, palette):
    import numpy as np
    return map_rgb(image, lambda rgb: np.repeat(
        (rgb @ np.array([.299, .587, .114], dtype=np.float32))[..., None], 3, axis=2))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
