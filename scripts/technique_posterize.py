"""Quantize each RGB channel to a chosen number of evenly spaced levels; preserve alpha."""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Posterize',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Quantize each RGB channel to a chosen number of evenly spaced levels; preserve '
                'alpha.',
 'controls': {'levels': {'type': 'integer',
                         'default': 4,
                         'minimum': 2,
                         'maximum': 256,
                         'step': 1,
                         'description': 'Number of levels per RGB channel.'}},
 'example': {'levels': 4},
 'tags': 'limited colours reduce quantize graphic'}


def apply(sdk, image, controls, palette):
    import numpy as np
    steps = controls["levels"] - 1
    return map_rgb(image, lambda rgb: np.floor(rgb * steps + .5) / steps)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
