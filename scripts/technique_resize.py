"""Resize image. Metadata and implementation live together.

Resample the accumulated image to exact dimensions with alpha-safe Lanczos sampling. Geometry steps change subsequent layer coordinates.
"""
from .art_kit import read_image, write_png, resize_image
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Resize image',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Resample the accumulated image to exact dimensions with alpha-safe Lanczos '
                'sampling. Geometry steps change subsequent layer coordinates.',
 'controls': {'width': {'type': 'integer',
                        'step': 1,
                        'description': 'Output width.',
                        'unit': 'px',
                        'minimum': 1},
              'height': {'type': 'integer',
                         'step': 1,
                         'description': 'Output height.',
                         'unit': 'px',
                         'minimum': 1},
              'fit': {'type': 'string',
                      'default': 'stretch',
                      'enum': ['stretch', 'contain', 'cover'],
                      'description': 'Stretch changes aspect; contain pads transparent; cover '
                                     'crops centrally.'}},
 'example': {'width': 1200, 'height': 800, 'fit': 'contain'},
 'tags': 'scale dimensions thumbnail geometry',
 'aliases': ['canvas_resize']}


def apply(sdk, image, controls, palette):
    return resize_image(image, (controls["width"], controls["height"]), controls["fit"])


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
