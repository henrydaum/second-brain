"""Reduce luminance to two colours using a deterministic 4x4 Bayer pattern. Preserves source alpha, multiplied by chosen colour alpha."""
from .art_kit import read_image, write_png, rgba, colorize_field
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Ordered dithering',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Reduce luminance to two colours using a deterministic 4x4 Bayer pattern. '
                'Preserves source alpha, multiplied by chosen colour alpha.',
 'controls': {'shadows': {'type': 'string',
                          'format': 'color',
                          'default': '#000000',
                          'description': 'CSS/hex colour or live @palette role.'},
              'highlights': {'type': 'string',
                             'format': 'color',
                             'default': '#ffffff',
                             'description': 'CSS/hex colour or live @palette role.'},
              'strength': {'type': 'number',
                           'default': 1,
                           'description': 'Pattern strength; zero is a plain midpoint threshold.',
                           'step': 0.05,
                           'minimum': 0,
                           'maximum': 1}},
 'example': {'strength': 1},
 'tags': 'bayer dither monochrome stipple print retro'}


def apply(sdk, image, controls, palette):
    import numpy as np
    matrix = np.array([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]], dtype=np.float32)
    yy, xx = np.indices((image.height, image.width))
    threshold = .5 + controls["strength"] * ((matrix[yy % 4, xx % 4] + .5) / 16 - .5)
    luminance = np.asarray(image.convert("L"), dtype=np.float32) / 255
    alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255
    return colorize_field((luminance >= threshold).astype(np.float32), rgba(controls["shadows"], palette), rgba(controls["highlights"], palette), alpha)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
