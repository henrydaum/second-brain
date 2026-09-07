"""Linear gradient. Metadata and implementation live together.

Two-colour alpha-safe gradient. Zero degrees goes left to right; 90 goes top to bottom.
"""
import math
from .art_kit import read_image, write_png, rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Linear gradient',
 'kind': 'background',
 'kinds': ['background', 'object'],
 'description': 'Two-colour alpha-safe gradient. Zero degrees goes left to right; 90 goes top to '
                'bottom.',
 'controls': {'start': {'type': 'string',
                        'format': 'color',
                        'default': '@primary',
                        'description': 'Starting colour.'},
              'end': {'type': 'string',
                      'format': 'color',
                      'default': '@accent',
                      'description': 'Ending colour.'},
              'angle': {'type': 'number',
                        'default': 0,
                        'minimum': -360,
                        'maximum': 360,
                        'step': 1,
                        'description': 'Gradient direction.',
                        'unit': 'degrees'}},
 'example': {'start': '@primary', 'end': '@accent', 'angle': 90},
 'tags': 'ramp background fade color',
 'aliases': ['canvas_gradient']}


def apply(sdk, image, controls, palette):
    import numpy as np
    from PIL import Image
    angle = math.radians(controls["angle"])
    dx, dy = math.cos(angle), math.sin(angle)
    x = np.arange(image.width, dtype=np.float32) * dx
    y = np.arange(image.height, dtype=np.float32) * dy
    values = y[:, None] + x[None, :]
    lo, hi = float(values.min()), float(values.max())
    t = ((values - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(values))[..., None]
    a, b = (np.array(rgba(controls[key], palette), dtype=np.float32) / 255 for key in ("start", "end"))
    a[:3] *= a[3]
    b[:3] *= b[3]
    result = a * (1 - t) + b * t
    result[..., :3] = np.divide(result[..., :3], result[..., 3:],
                                out=np.zeros_like(result[..., :3]), where=result[..., 3:] > 0)
    return Image.fromarray(np.uint8(np.clip(result * 255 + .5, 0, 255)))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
