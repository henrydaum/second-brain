"""Split red and blue channels in opposite directions, radially or horizontally. Amount is a fraction of the shorter image side; original alpha is preserved and hidden colours are not sampled."""
from .art_kit import read_image, write_png, sample_rgba, radial_coordinates
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Chromatic aberration',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Split red and blue channels in opposite directions, radially or horizontally. '
                'Amount is a fraction of the shorter image side; original alpha is preserved and '
                'hidden colours are not sampled.',
 'controls': {'amount': {'type': 'number',
                         'default': 0.01,
                         'description': 'Channel separation relative to the shorter image side.',
                         'step': 0.01,
                         'minimum': -0.2,
                         'maximum': 0.2},
              'mode': {'type': 'string', 'default': 'radial', 'enum': ['radial', 'horizontal']},
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
 'example': {'amount': 0.015},
 'tags': 'rgb split chromatic aberration abberation lens colour glitch'}


def apply(sdk, image, controls, palette):
    import numpy as np
    from PIL import Image
    if controls["amount"] == 0:
        return image.copy()
    xx, yy, radius, angle, cx, cy, unit = radial_coordinates(image.size, controls["center_x"], controls["center_y"])
    distance = controls["amount"] * min(image.size)
    if controls["mode"] == "radial":
        dx, dy = (xx - cx) / unit * distance, (yy - cy) / unit * distance
    else:
        dx, dy = distance, 0
    red = np.asarray(sample_rgba(image, xx + dx, yy + dy))
    blue = np.asarray(sample_rgba(image, xx - dx, yy - dy))
    result = np.array(image)
    result[..., 0] = np.where(red[..., 3] > 0, red[..., 0], result[..., 0])
    result[..., 2] = np.where(blue[..., 3] > 0, blue[..., 2], result[..., 2])
    return Image.fromarray(result)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
