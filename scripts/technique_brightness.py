"""Brightness. Metadata and implementation live together.

Multiply encoded RGB; alpha is unchanged. Zero is black, one is unchanged.
"""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Brightness',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Multiply encoded RGB; alpha is unchanged. Zero is black, one is unchanged.',
 'controls': {'factor': {'type': 'number',
                         'default': 1,
                         'minimum': 0,
                         'maximum': 4,
                         'step': 0.05,
                         'description': 'Brightness multiplier.',
                         'unit': ''}},
 'example': {'factor': 1.1},
 'tags': 'light dark brighten dim',
 'aliases': ['canvas_brightness']}


def apply(sdk, image, controls, palette):
    return map_rgb(image, lambda rgb: rgb * controls["factor"])


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
