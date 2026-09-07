"""Crop. Metadata and implementation live together.

Extract a pixel rectangle. Right/bottom are exclusive. Output dimensions become right-left by bottom-top; outside-source pixels are transparent.
"""
from .art_kit import read_image, write_png, crop_image
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Crop',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Extract a pixel rectangle. Right/bottom are exclusive. Output dimensions become '
                'right-left by bottom-top; outside-source pixels are transparent.',
 'controls': {'left': {'type': 'integer',
                       'step': 1,
                       'description': 'Left edge.',
                       'unit': 'px',
                       'default': 0},
              'top': {'type': 'integer',
                      'step': 1,
                      'description': 'Top edge.',
                      'unit': 'px',
                      'default': 0},
              'right': {'type': 'integer',
                        'step': 1,
                        'description': 'Exclusive right edge.',
                        'unit': 'px'},
              'bottom': {'type': 'integer',
                         'step': 1,
                         'description': 'Exclusive bottom edge.',
                         'unit': 'px'}},
 'example': {'left': 100, 'top': 50, 'right': 900, 'bottom': 650},
 'tags': 'trim cut reframe geometry',
 'aliases': ['canvas_crop'],
 'constraints': [{'greater': 'right', 'than': 'left'}, {'greater': 'bottom', 'than': 'top'}]}


def apply(sdk, image, controls, palette):
    return crop_image(image, (controls["left"], controls["top"], controls["right"], controls["bottom"]))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
