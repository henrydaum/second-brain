"""Exposure. Metadata and implementation live together.

Multiply linear-light RGB by 2**stops, then encode sRGB. Alpha is unchanged; highlights may clip.
"""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Exposure',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Multiply linear-light RGB by 2**stops, then encode sRGB. Alpha is unchanged; '
                'highlights may clip.',
 'controls': {'stops': {'type': 'number',
                        'default': 0,
                        'minimum': -8,
                        'maximum': 8,
                        'step': 0.1,
                        'description': 'Positive brightens; +1 doubles linear light.',
                        'unit': 'EV'}},
 'example': {'stops': 0.4},
 'tags': 'light photographic ev',
 'aliases': ['canvas_exposure']}


def apply(sdk, image, controls, palette):
    import numpy as np
    def adjust(rgb):
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        linear = np.clip(linear * 2 ** controls["stops"], 0, 1)
        return np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
    return map_rgb(image, adjust)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
