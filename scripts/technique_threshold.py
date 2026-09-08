"""Map encoded RGB luminance to two colours while retaining input coverage. Colour alpha multiplies input alpha."""
from .art_kit import read_image, write_png, rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Threshold',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Map encoded RGB luminance to two colours while retaining input coverage. Colour '
                'alpha multiplies input alpha.',
 'controls': {'threshold': {'type': 'integer',
                            'default': 128,
                            'minimum': 0,
                            'maximum': 255,
                            'step': 1,
                            'description': 'Luminance cutoff; values at or above it use '
                                           'highlights.'},
              'shadows': {'type': 'string',
                          'format': 'color',
                          'default': '#000000',
                          'description': 'Literal CSS/hex colour or live @palette role.'},
              'highlights': {'type': 'string',
                             'format': 'color',
                             'default': '#ffffff',
                             'description': 'Literal CSS/hex colour or live @palette role.'}},
 'example': {'threshold': 128},
 'tags': 'binary black white silhouette two tone'}


def apply(sdk, image, controls, palette):
    import numpy as np
    from PIL import Image
    source = np.asarray(image, dtype=np.float32)
    luminance = source[..., :3] @ np.array([.2126, .7152, .0722], dtype=np.float32)
    low = np.array(rgba(controls["shadows"], palette), dtype=np.float32)
    high = np.array(rgba(controls["highlights"], palette), dtype=np.float32)
    result = np.where((luminance >= controls["threshold"])[..., None], high, low)
    result[..., 3] *= source[..., 3] / 255
    return Image.fromarray(np.uint8(np.clip(result + .5, 0, 255)))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
