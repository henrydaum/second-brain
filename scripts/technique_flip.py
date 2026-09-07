"""Flip / mirror. Metadata and implementation live together.

Mirror horizontally and/or vertically without resampling.
"""
from .art_kit import read_image, write_png
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Flip / mirror',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Mirror horizontally and/or vertically without resampling.',
 'controls': {'horizontal': {'type': 'boolean',
                             'default': True,
                             'description': 'Mirror left to right.'},
              'vertical': {'type': 'boolean',
                           'default': False,
                           'description': 'Mirror top to bottom.'}},
 'example': {'horizontal': True},
 'tags': 'mirror reverse',
 'aliases': ['canvas_flip']}


def apply(sdk, image, controls, palette):
    from PIL import ImageOps
    result = ImageOps.mirror(image) if controls["horizontal"] else image.copy()
    return ImageOps.flip(result) if controls["vertical"] else result


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
