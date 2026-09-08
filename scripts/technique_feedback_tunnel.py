"""Recursively composite a smaller, rotated copy inside the original image. Creates a finite tunnel, not an animation. Scale, twist, depth and opacity need no pixel coordinates."""
from .art_kit import read_image, write_png, transform_image, composite
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Feedback tunnel',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Recursively composite a smaller, rotated copy inside the original image. Creates '
                'a finite tunnel, not an animation. Scale, twist, depth and opacity need no pixel '
                'coordinates.',
 'controls': {'depth': {'type': 'integer',
                        'default': 8,
                        'description': 'Number of recursive copies; zero unchanged.',
                        'step': 1,
                        'minimum': 0,
                        'maximum': 32},
              'scale': {'type': 'number',
                        'default': 0.8,
                        'description': 'Size of each nested copy relative to the previous.',
                        'step': 0.01,
                        'minimum': 0.2,
                        'maximum': 0.98},
              'twist': {'type': 'number',
                        'default': 5,
                        'description': 'Rotation per copy in degrees.',
                        'step': 0.01,
                        'minimum': -90,
                        'maximum': 90},
              'opacity': {'type': 'number',
                          'default': 0.8,
                          'description': 'Opacity of nested copies.',
                          'step': 0.01,
                          'minimum': 0,
                          'maximum': 1},
              'center_x': {'type': 'number',
                           'default': 0.5,
                           'description': 'Horizontal centre: 0 left, 0.5 middle, 1 right.',
                           'step': 0.01,
                           'minimum': 0,
                           'maximum': 1},
              'center_y': {'type': 'number',
                           'default': 0.5,
                           'description': 'Vertical centre: 0 top, 0.5 middle, 1 bottom.',
                           'step': 0.01,
                           'minimum': 0,
                           'maximum': 1}},
 'example': {'depth': 8, 'scale': 0.8, 'twist': 5},
 'tags': 'feedback tunnel recursion nested echo infinite trippy'}


def apply(sdk, image, controls, palette):
    import math
    if controls["depth"] == 0 or controls["opacity"] == 0:
        return image.copy()
    cx, cy = controls["center_x"] * image.width, controls["center_y"] * image.height
    angle = math.radians(controls["twist"])
    a, b = math.cos(angle) / controls["scale"], -math.sin(angle) / controls["scale"]
    d, e = -b, a
    matrix = (a, b, cx - a * cx - b * cy, d, e, cy - d * cx - e * cy)
    result = image.copy()
    for _ in range(controls["depth"]):
        nested = transform_image(result, image.size, matrix)
        result = composite(image, nested, opacity=controls["opacity"])
    return result


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
