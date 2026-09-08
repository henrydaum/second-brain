"""Generate a seeded, multiscale value-noise texture between two palette-aware colours. Scale is in pixels; use as a background, or import its cached PNG as an overlay or displacement map."""
from .art_kit import read_image, write_png, fbm_grid, rgba, colorize_field
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Fractal noise texture',
 'kind': 'background',
 'kinds': ['background'],
 'description': 'Generate a seeded, multiscale value-noise texture between two palette-aware '
                'colours. Scale is in pixels; use as a background, or import its cached PNG as an '
                'overlay or displacement map.',
 'controls': {'scale': {'type': 'number',
                        'default': 64,
                        'description': 'Base feature size in pixels.',
                        'step': 1,
                        'minimum': 1,
                        'maximum': 4096},
              'octaves': {'type': 'integer',
                          'default': 4,
                          'description': 'Number of noise scales.',
                          'step': 1,
                          'minimum': 1,
                          'maximum': 8},
              'contrast': {'type': 'number',
                           'default': 1,
                           'description': 'Contrast about the midpoint.',
                           'step': 0.05,
                           'minimum': 0,
                           'maximum': 4},
              'low': {'type': 'string',
                      'format': 'color',
                      'default': '@primary',
                      'description': 'CSS/hex colour or live @palette role.'},
              'high': {'type': 'string',
                       'format': 'color',
                       'default': '@accent',
                       'description': 'CSS/hex colour or live @palette role.'}},
 'example': {'scale': 64, 'octaves': 4},
 'tags': 'procedural texture noise clouds paper displacement'}


def apply(sdk, image, controls, palette, seed=0):
    import numpy as np
    yy, xx = np.indices((image.height, image.width), dtype=np.float64)
    field = fbm_grid(seed, xx / controls["scale"], yy / controls["scale"], octaves=controls["octaves"])
    field = np.clip(.5 + (field - .5) * controls["contrast"], 0, 1)
    return colorize_field(field, rgba(controls["low"], palette), rgba(controls["high"], palette))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette, seed=seed)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
