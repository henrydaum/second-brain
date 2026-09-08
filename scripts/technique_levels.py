"""Set encoded RGB black and white points, then adjust midtones; alpha is unchanged."""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Levels',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Set encoded RGB black and white points, then adjust midtones; alpha is unchanged.',
 'controls': {'black': {'type': 'integer',
                        'default': 0,
                        'minimum': 0,
                        'maximum': 254,
                        'step': 1,
                        'description': 'Input black point, 0?255.'},
              'white': {'type': 'integer',
                        'default': 255,
                        'minimum': 1,
                        'maximum': 255,
                        'step': 1,
                        'description': 'Input white point, 0?255.'},
              'gamma': {'type': 'number',
                        'default': 1,
                        'minimum': 0.1,
                        'maximum': 10,
                        'step': 0.05,
                        'description': 'Midtone gamma; above one brightens.'}},
 'example': {'black': 10, 'white': 245},
 'tags': 'tones histogram black white midtones',
 'constraints': [{'greater': 'white', 'than': 'black'}]}


def apply(sdk, image, controls, palette):
    import numpy as np
    def adjust(rgb):
        normalized = np.clip((rgb * 255 - controls["black"]) / (controls["white"] - controls["black"]), 0, 1)
        return normalized ** (1 / controls["gamma"])
    return map_rgb(image, adjust)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
