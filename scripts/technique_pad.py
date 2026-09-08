"""Create a new canvas without resampling. Place the input at x/y; negative offsets or smaller dimensions crop it."""
from .art_kit import read_image, write_png, rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Canvas padding',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Create a new canvas without resampling. Place the input at x/y; negative offsets '
                'or smaller dimensions crop it.',
 'controls': {'width': {'type': 'integer',
                        'default': 800,
                        'minimum': 1,
                        'maximum': 2147483647,
                        'step': 1,
                        'description': 'Output width in pixels.'},
              'height': {'type': 'integer',
                         'default': 600,
                         'minimum': 1,
                         'maximum': 2147483647,
                         'step': 1,
                         'description': 'Output height in pixels.'},
              'x': {'type': 'integer',
                    'default': 0,
                    'minimum': -2147483647,
                    'maximum': 2147483647,
                    'step': 1,
                    'description': 'Input left offset.'},
              'y': {'type': 'integer',
                    'default': 0,
                    'minimum': -2147483647,
                    'maximum': 2147483647,
                    'step': 1,
                    'description': 'Input top offset.'},
              'color': {'type': 'string',
                        'format': 'color',
                        'default': 'transparent',
                        'description': 'Literal CSS/hex colour or live @palette role.'}},
 'example': {'width': 800, 'height': 600, 'x': 20, 'y': 20},
 'tags': 'extend canvas border margins transparent frame'}


def apply(sdk, image, controls, palette):
    from PIL import Image
    result = Image.new("RGBA", (controls["width"], controls["height"]), rgba(controls["color"], palette))
    result.alpha_composite(image, dest=(controls["x"], controls["y"]))
    return result


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
