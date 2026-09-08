"""Bulge or pinch a circular region about an image-relative centre. Positive strength magnifies the centre; negative pinches. Outside radius stays unchanged."""
from .art_kit import read_image, write_png, sample_rgba, radial_coordinates
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Fisheye lens',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Bulge or pinch a circular region about an image-relative centre. Positive '
                'strength magnifies the centre; negative pinches. Outside radius stays unchanged.',
 'controls': {'strength': {'type': 'number',
                           'default': 0.6,
                           'description': 'Bulge/pinch strength; zero unchanged.',
                           'step': 0.01,
                           'minimum': -0.95,
                           'maximum': 2},
              'radius': {'type': 'number',
                         'default': 1,
                         'description': 'Lens radius in half-short-side units.',
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
 'example': {'strength': 0.6},
 'tags': 'lens fisheye bulge pinch distortion trippy'}


def apply(sdk, image, controls, palette):
    import numpy as np
    if controls["strength"] == 0:
        return image.copy()
    xx, yy, radius, angle, cx, cy, unit = radial_coordinates(image.size, controls["center_x"], controls["center_y"])
    t = np.minimum(radius / controls["radius"], 1)
    factor = np.exp(-controls["strength"] * (1 - t) ** 2)
    return sample_rgba(image, cx + (xx - cx) * factor, cy + (yy - cy) * factor)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
