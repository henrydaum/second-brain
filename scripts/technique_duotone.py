"""Duotone / palette map. Metadata and implementation live together.

Explicitly map luminance between two palette or literal colours, then mix with the original. Preserves source alpha.
"""
from .art_kit import read_image, write_png, rgba, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Duotone / palette map',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Explicitly map luminance between two palette or literal colours, then mix with '
                'the original. Preserves source alpha.',
 'controls': {'shadows': {'type': 'string',
                          'format': 'color',
                          'default': '@secondary',
                          'description': 'Dark tone.'},
              'highlights': {'type': 'string',
                             'format': 'color',
                             'default': '@accent',
                             'description': 'Light tone.'},
              'amount': {'type': 'number',
                         'default': 0.5,
                         'minimum': 0,
                         'maximum': 1,
                         'step': 0.05,
                         'description': 'Strength; zero is unchanged, one fully mapped.',
                         'unit': ''}},
 'example': {'shadows': '#182844', 'highlights': '#ffd9a0', 'amount': 0.35},
 'tags': 'grade tint palette color map',
 'aliases': ['canvas_duotone']}


def apply(sdk, image, controls, palette):
    import numpy as np
    shadows = np.array(rgba(controls["shadows"], palette)[:3], dtype=np.float32) / 255
    highlights = np.array(rgba(controls["highlights"], palette)[:3], dtype=np.float32) / 255
    def adjust(rgb):
        lum = (rgb @ np.array([.299, .587, .114], dtype=np.float32))[..., None]
        mapped = shadows * (1 - lum) + highlights * lum
        return rgb * (1 - controls["amount"]) + mapped * controls["amount"]
    return map_rgb(image, adjust)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
