"""Add signed RGB channel offsets to correct a colour cast. Alpha is unchanged."""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Color balance',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Add signed RGB channel offsets to correct a colour cast. Alpha is unchanged.',
 'controls': {'red': {'type': 'number',
                      'default': 0,
                      'minimum': -1,
                      'maximum': 1,
                      'step': 0.05,
                      'description': 'Signed red offset in normalized encoded RGB.'},
              'green': {'type': 'number',
                        'default': 0,
                        'minimum': -1,
                        'maximum': 1,
                        'step': 0.05,
                        'description': 'Signed green offset in normalized encoded RGB.'},
              'blue': {'type': 'number',
                       'default': 0,
                       'minimum': -1,
                       'maximum': 1,
                       'step': 0.05,
                       'description': 'Signed blue offset in normalized encoded RGB.'}},
 'example': {'red': 0.05, 'blue': -0.03},
 'tags': 'warm cool cast correction channels'}


def apply(sdk, image, controls, palette):
    import numpy as np
    offsets = np.array([controls[c] for c in ("red", "green", "blue")], dtype=np.float32)
    return map_rgb(image, lambda rgb: rgb + offsets)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
