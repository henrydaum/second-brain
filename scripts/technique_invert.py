"""Invert. Metadata and implementation live together.

Invert RGB, preserving alpha. Use layer opacity to mix the effect.
"""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Invert',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Invert RGB, preserving alpha. Use layer opacity to mix the effect.',
 'controls': {},
 'example': {},
 'tags': 'negative reverse colors',
 'aliases': ['canvas_invert']}


def apply(sdk, image, controls, palette):
    return map_rgb(image, lambda rgb: 1 - rgb)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
