"""Relative warm/cool and green/magenta correction using linear-light channel gains. These are relative adjustments, not camera Kelvin calibration."""
from .art_kit import read_image, write_png, map_rgb, srgb_to_linear, linear_to_srgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'White balance',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Relative warm/cool and green/magenta correction using linear-light channel gains. '
                'These are relative adjustments, not camera Kelvin calibration.',
 'controls': {'temperature': {'type': 'number',
                              'default': 0,
                              'description': 'Positive warms, negative cools.',
                              'step': 1,
                              'minimum': -100,
                              'maximum': 100},
              'tint': {'type': 'number',
                       'default': 0,
                       'description': 'Positive adds magenta, negative adds green.',
                       'step': 1,
                       'minimum': -100,
                       'maximum': 100}},
 'example': {'temperature': 10, 'tint': 3},
 'tags': 'warm cool temperature tint cast photo'}


def apply(sdk, image, controls, palette):
    import numpy as np
    if controls["temperature"] == 0 and controls["tint"] == 0:
        return image.copy()
    gains = np.exp(np.array([controls["temperature"], -controls["tint"], -controls["temperature"]]) * .004)
    return map_rgb(image, lambda rgb: linear_to_srgb(srgb_to_linear(rgb) * gains))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
