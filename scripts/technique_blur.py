"""Gaussian blur. Metadata and implementation live together.

Blur premultiplied colour and alpha together to avoid dark or coloured fringes at transparent edges.
"""
from .art_kit import read_image, write_png, gaussian_blur_array
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Gaussian blur',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Blur premultiplied colour and alpha together to avoid dark or coloured fringes at '
                'transparent edges.',
 'controls': {'radius': {'type': 'number',
                         'default': 3,
                         'minimum': 0,
                         'maximum': 200,
                         'step': 0.25,
                         'description': 'Gaussian radius; zero is unchanged.',
                         'unit': 'px'}},
 'example': {'radius': 2.5},
 'tags': 'soften defocus gaussian smooth',
 'aliases': ['canvas_blur']}


def apply(sdk, image, controls, palette):
    import numpy as np
    from PIL import Image
    if controls["radius"] == 0:
        return image.copy()
    return Image.fromarray(np.uint8(np.clip(gaussian_blur_array(image, controls["radius"]) * 255 + .5, 0, 255)))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
