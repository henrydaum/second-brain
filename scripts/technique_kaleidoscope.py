"""Mirror one angular wedge into repeated radial segments. Zoom controls source sampling scale. Image-relative centre defaults to the middle; transparent fill outside the source."""
from .art_kit import read_image, write_png, sample_rgba, radial_coordinates
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Kaleidoscope',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Mirror one angular wedge into repeated radial segments. Zoom controls source '
                'sampling scale. Image-relative centre defaults to the middle; transparent fill '
                'outside the source.',
 'controls': {'segments': {'type': 'integer',
                           'default': 6,
                           'description': 'Number of repeated wedges.',
                           'step': 1,
                           'minimum': 2,
                           'maximum': 32},
              'angle': {'type': 'number',
                        'default': 0,
                        'description': 'Rotate the sampled wedge in degrees.',
                        'step': 0.01,
                        'minimum': -360,
                        'maximum': 360},
              'zoom': {'type': 'number',
                       'default': 1,
                       'description': 'Magnification of sampled source.',
                       'step': 0.01,
                       'minimum': 0.1,
                       'maximum': 10},
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
 'example': {'segments': 6},
 'tags': 'mirror kaleidoscope radial symmetry mandala trippy'}


def apply(sdk, image, controls, palette):
    import numpy as np
    xx, yy, radius, angle, cx, cy, unit = radial_coordinates(image.size, controls["center_x"], controls["center_y"])
    wedge = 2 * np.pi / controls["segments"]
    folded = np.abs((angle + wedge / 2) % wedge - wedge / 2) + np.deg2rad(controls["angle"])
    distance = radius * unit / controls["zoom"]
    return sample_rgba(image, cx + distance * np.cos(folded), cy + distance * np.sin(folded))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
