"""Select image pixels by luminance interval or RGB colour distance. Input alpha limits selection; output is an opaque grayscale mask."""
from .art_kit import read_image, write_png, selection_image, rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Range selection',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Select image pixels by luminance interval or RGB colour distance. Input alpha '
                'limits selection; output is an opaque grayscale mask.',
 'controls': {'mode': {'type': 'string', 'default': 'luminance', 'enum': ['luminance', 'color']},
              'low': {'type': 'number',
                      'default': 0,
                      'step': 1,
                      'description': 'Minimum luminance, 0?255.',
                      'minimum': 0,
                      'maximum': 255},
              'high': {'type': 'number',
                       'default': 255,
                       'step': 1,
                       'description': 'Maximum luminance, 0?255.',
                       'minimum': 0,
                       'maximum': 255},
              'color': {'type': 'string', 'format': 'color', 'default': '@accent'},
              'tolerance': {'type': 'number',
                            'default': 30,
                            'step': 1,
                            'description': 'Maximum Euclidean RGB distance to target colour.',
                            'minimum': 0,
                            'maximum': 442},
              'softness': {'type': 'number',
                           'default': 0,
                           'step': 1,
                           'description': 'Soft transition outside interval or tolerance, in '
                                          'channel units.',
                           'minimum': 0,
                           'maximum': 255},
              'feather': {'type': 'number',
                          'default': 0,
                          'step': 1,
                          'description': 'Feather radius in pixels.',
                          'minimum': 0,
                          'maximum': 200},
              'invert': {'type': 'boolean', 'default': False}},
 'example': {'mode': 'luminance', 'low': 128},
 'tags': 'selection mask luminosity colour range key feather invert'}


def apply(sdk, image, controls, palette):
    import numpy as np
    source = np.asarray(image, dtype=np.float32)
    if controls["mode"] == "luminance":
        if controls["low"] > controls["high"]:
            raise ValueError("low must not exceed high")
        values = np.asarray(image.convert("L"), dtype=np.float32)
        distance = np.maximum(controls["low"] - values, values - controls["high"])
    else:
        target = np.array(rgba(controls["color"], palette)[:3], dtype=np.float32)
        distance = np.sqrt(np.sum((source[..., :3] - target) ** 2, axis=2)) - controls["tolerance"]
    coverage = (distance <= 0).astype(np.float32) if controls["softness"] == 0 else np.clip(1 - np.maximum(distance, 0) / controls["softness"], 0, 1)
    coverage *= source[..., 3] / 255
    return selection_image(coverage, controls["feather"], controls["invert"])


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
