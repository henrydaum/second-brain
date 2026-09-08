"""Twist a circular region with smooth falloff toward its edge. Positive turns rotate source sampling clockwise in screen coordinates. Outside the radius is unchanged."""
from .art_kit import read_image, write_png, sample_rgba, radial_coordinates
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Swirl',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Twist a circular region with smooth falloff toward its edge. Positive turns '
                'rotate source sampling clockwise in screen coordinates. Outside the radius is '
                'unchanged.',
 'controls': {'turns': {'type': 'number',
                        'default': 0.5,
                        'description': 'Twist in revolutions; zero unchanged.',
                        'step': 0.01,
                        'minimum': -4,
                        'maximum': 4},
              'radius': {'type': 'number',
                         'default': 1,
                         'description': 'Radius in half-short-side units.',
                         'step': 0.01,
                         'minimum': 0.05,
                         'maximum': 4},
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
 'example': {'turns': 0.5},
 'tags': 'vortex spiral swirl twist trippy'}


def apply(sdk, image, controls, palette):
    import numpy as np
    if controls["turns"] == 0:
        return image.copy()
    xx, yy, radius, angle, cx, cy, unit = radial_coordinates(image.size, controls["center_x"], controls["center_y"])
    twist = controls["turns"] * 2 * np.pi * np.maximum(1 - radius / controls["radius"], 0) ** 2
    return sample_rgba(image, cx + radius * unit * np.cos(angle + twist), cy + radius * unit * np.sin(angle + twist))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
