"""Vignette. Metadata and implementation live together.

Darken edges with an elliptical falloff centred in the image. Alpha is unchanged.
"""
import math
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Vignette',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Darken edges with an elliptical falloff centred in the image. Alpha is unchanged.',
 'controls': {'amount': {'type': 'number',
                         'default': 0.35,
                         'minimum': 0,
                         'maximum': 1,
                         'step': 0.05,
                         'description': 'Edge darkening.',
                         'unit': ''},
              'radius': {'type': 'number',
                         'default': 0.5,
                         'minimum': 0,
                         'maximum': 1,
                         'step': 0.05,
                         'description': 'Start falloff at this fraction of the centre-to-corner '
                                        'distance.',
                         'unit': ''}},
 'example': {'amount': 0.25, 'radius': 0.5},
 'tags': 'edges focus darkening lens',
 'aliases': ['canvas_vignette']}


def apply(sdk, image, controls, palette):
    import numpy as np
    x = (np.arange(image.width, dtype=np.float32) + .5) / image.width * 2 - 1
    y = (np.arange(image.height, dtype=np.float32) + .5) / image.height * 2 - 1
    distance = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2) / math.sqrt(2)
    t = np.clip((distance - controls["radius"]) / max(1e-6, 1 - controls["radius"]), 0, 1)
    weight = (1 - controls["amount"] * t * t * (3 - 2 * t))[..., None]
    return map_rgb(image, lambda rgb: rgb * weight)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
