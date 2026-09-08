"""Warp using a matching-size image map: red controls horizontal displacement and green vertical. Value 128 is neutral; map alpha scales displacement. Positive amounts sample to the right/down."""
from .art_kit import read_image, write_png, sample_rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Displacement map',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Warp using a matching-size image map: red controls horizontal displacement and '
                'green vertical. Value 128 is neutral; map alpha scales displacement. Positive '
                'amounts sample to the right/down.',
 'controls': {'path': {'type': 'string',
                       'format': 'file',
                       'description': 'Existing displacement PNG; dimensions must match the '
                                      'current image.'},
              'x': {'type': 'number',
                    'default': 10,
                    'description': 'Horizontal displacement scale in pixels.',
                    'step': 1,
                    'minimum': -4096,
                    'maximum': 4096},
              'y': {'type': 'number',
                    'default': 10,
                    'description': 'Vertical displacement scale in pixels.',
                    'step': 1,
                    'minimum': -4096,
                    'maximum': 4096},
              'edge': {'type': 'string',
                       'default': 'transparent',
                       'enum': ['transparent', 'clamp'],
                       'description': 'Sampling outside the image.'}},
 'example': {'path': '<displacement PNG>', 'x': 10, 'y': 10},
 'tags': 'warp distort texture displacement map'}


def apply(sdk, image, controls, palette):
    import numpy as np
    if controls["x"] == 0 and controls["y"] == 0:
        return image.copy()
    mapping = read_image(sdk, controls["path"])
    if mapping.size != image.size:
        raise ValueError("Displacement map must match the current image size; resize it explicitly first")
    data = np.asarray(mapping, dtype=np.float32)
    amount = (data[..., :2] - 128) / 128 * (data[..., 3:] / 255)
    yy, xx = np.indices((image.height, image.width), dtype=np.float32)
    if not np.any(amount):
        return image.copy()
    return sample_rgba(image, xx + amount[..., 0] * controls["x"], yy + amount[..., 1] * controls["y"], controls["edge"])


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
