"""Contrast. Metadata and implementation live together.

Adjust around alpha-weighted mean luminance; transparent hidden RGB does not bias the pivot. One is unchanged.
"""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Contrast',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Adjust around alpha-weighted mean luminance; transparent hidden RGB does not bias '
                'the pivot. One is unchanged.',
 'controls': {'factor': {'type': 'number',
                         'default': 1,
                         'minimum': 0,
                         'maximum': 4,
                         'step': 0.05,
                         'description': 'Contrast multiplier.',
                         'unit': ''}},
 'example': {'factor': 1.15},
 'tags': 'punch flat tonal',
 'aliases': ['canvas_contrast']}


def apply(sdk, image, controls, palette):
    import numpy as np
    weights = np.asarray(image.getchannel("A"), dtype=np.float32) / 255
    def adjust(rgb):
        lum = rgb @ np.array([.299, .587, .114], dtype=np.float32)
        total = float(weights.sum())
        pivot = float((lum * weights).sum()) / total if total else 0
        return (rgb - pivot) * controls["factor"] + pivot
    return map_rgb(image, adjust)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
