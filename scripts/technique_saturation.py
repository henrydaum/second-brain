"""Saturation. Metadata and implementation live together.

Zero is grayscale, one preserves colour, above one boosts colour. Alpha is unchanged.
"""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Saturation',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Zero is grayscale, one preserves colour, above one boosts colour. Alpha is '
                'unchanged.',
 'controls': {'factor': {'type': 'number',
                         'default': 1,
                         'minimum': 0,
                         'maximum': 4,
                         'step': 0.05,
                         'description': 'Colour intensity multiplier.',
                         'unit': ''}},
 'example': {'factor': 1.2},
 'tags': 'color colour vivid muted desaturate',
 'aliases': ['canvas_saturation']}


def apply(sdk, image, controls, palette):
    import numpy as np
    def adjust(rgb):
        gray = (rgb @ np.array([.299, .587, .114], dtype=np.float32))[..., None]
        return gray + (rgb - gray) * controls["factor"]
    return map_rgb(image, adjust)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
