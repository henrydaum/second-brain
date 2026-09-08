"""Sort complete RGBA pixels inside contiguous luminance-selected runs, horizontally or vertically. Fully transparent pixels break runs. No feature placement or coordinate estimates required."""
from .art_kit import read_image, write_png
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Pixel sort',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Sort complete RGBA pixels inside contiguous luminance-selected runs, horizontally '
                'or vertically. Fully transparent pixels break runs. No feature placement or '
                'coordinate estimates required.',
 'controls': {'direction': {'type': 'string',
                            'default': 'horizontal',
                            'enum': ['horizontal', 'vertical']},
              'low': {'type': 'number',
                      'default': 0.2,
                      'description': 'Minimum normalized luminance to include.',
                      'step': 0.01,
                      'minimum': 0,
                      'maximum': 1},
              'high': {'type': 'number',
                       'default': 0.8,
                       'description': 'Maximum normalized luminance to include.',
                       'step': 0.01,
                       'minimum': 0,
                       'maximum': 1},
              'reverse': {'type': 'boolean', 'default': False}},
 'example': {'low': 0.2, 'high': 0.8},
 'tags': 'pixel sort streak melt glitch luminance'}


def apply(sdk, image, controls, palette):
    import numpy as np
    from PIL import Image
    if controls["low"] > controls["high"]:
        raise ValueError("low must not exceed high")
    data = np.array(image)
    luminance = np.asarray(image.convert("L"), dtype=np.float32) / 255
    if controls["direction"] == "vertical":
        data = data.transpose(1,0,2).copy()
        luminance = luminance.T
    for row, values in zip(data, luminance):
        eligible = (values >= controls["low"]) & (values <= controls["high"]) & (row[:,3] > 0)
        boundaries = np.flatnonzero(np.diff(np.r_[False, eligible, False]))
        for start, end in boundaries.reshape(-1,2):
            key = -values[start:end] if controls["reverse"] else values[start:end]
            order = np.argsort(key, kind="stable")
            row[start:end] = row[start:end][order]
    if controls["direction"] == "vertical":
        data = data.transpose(1,0,2)
    return Image.fromarray(data)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
