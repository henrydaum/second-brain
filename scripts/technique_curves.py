"""Piecewise-linear input/output curve in normalized encoded RGB. Points must span x=0 to x=1 in strictly increasing order; alpha is preserved."""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Tone curves',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Piecewise-linear input/output curve in normalized encoded RGB. Points must span '
                'x=0 to x=1 in strictly increasing order; alpha is preserved.',
 'controls': {'points': {'type': 'array',
                         'default': [[0, 0], [1, 1]],
                         'minItems': 2,
                         'items': {'type': 'array',
                                   'minItems': 2,
                                   'maxItems': 2,
                                   'items': {'type': 'number'}}},
              'channel': {'type': 'string',
                          'default': 'rgb',
                          'enum': ['rgb', 'red', 'green', 'blue'],
                          'description': 'Apply to all channels or one channel.'}},
 'example': {'points': [[0, 0], [0.25, 0.18], [0.75, 0.82], [1, 1]]},
 'tags': 'tone curve contrast midtones channels'}


def apply(sdk, image, controls, palette):
    import numpy as np
    curve = np.asarray(controls["points"], dtype=np.float64)
    if np.any(curve < 0) or np.any(curve > 1) or curve[0, 0] != 0 or curve[-1, 0] != 1 or np.any(np.diff(curve[:, 0]) <= 0):
        raise ValueError("Curve points must be in [0,1], span x=0 to x=1, and have strictly increasing x")
    def adjust(rgb):
        if controls["channel"] == "rgb":
            return np.interp(rgb, curve[:, 0], curve[:, 1])
        result = rgb.copy()
        index = ["red", "green", "blue"].index(controls["channel"])
        result[..., index] = np.interp(rgb[..., index], curve[:, 0], curve[:, 1])
        return result
    return map_rgb(image, adjust)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
