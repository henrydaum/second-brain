"""Rotate / straighten. Metadata and implementation live together.

Rotate counterclockwise around the centre. Expand retains the whole image and changes dimensions; corners are transparent. Quarter turns preserve pixels exactly.
"""
from .art_kit import read_image, write_png
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Rotate / straighten',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Rotate counterclockwise around the centre. Expand retains the whole image and '
                'changes dimensions; corners are transparent. Quarter turns preserve pixels '
                'exactly.',
 'controls': {'angle': {'type': 'number',
                        'default': 0,
                        'minimum': -360,
                        'maximum': 360,
                        'step': 0.1,
                        'description': 'Positive is counterclockwise.',
                        'unit': 'degrees'},
              'expand': {'type': 'boolean',
                         'default': True,
                         'description': 'Expand bounds; false keeps and clips to current '
                                        'dimensions.'}},
 'example': {'angle': -2.5, 'expand': True},
 'tags': 'rotation straighten orientation geometry',
 'aliases': ['canvas_rotate']}


def apply(sdk, image, controls, palette):
    from PIL import Image
    angle = controls["angle"] % 360
    if angle == 0:
        return image.copy()
    if angle in (90, 180, 270) and (controls["expand"] or angle == 180 or image.width == image.height):
        method = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
                  270: Image.Transpose.ROTATE_270}[angle]
        return image.transpose(method)
    return image.convert("RGBa").rotate(angle, Image.Resampling.BICUBIC,
                                        expand=controls["expand"], fillcolor=(0, 0, 0, 0)).convert("RGBA")


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
