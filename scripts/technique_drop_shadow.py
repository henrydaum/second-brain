"""Place a blurred alpha silhouette behind the current image. Needs transparency to be visible; pad first for room because shadows clip to the current canvas."""
from .art_kit import read_image, write_png, rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Drop shadow',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Place a blurred alpha silhouette behind the current image. Needs transparency to '
                'be visible; pad first for room because shadows clip to the current canvas.',
 'controls': {'x': {'type': 'integer',
                    'default': 8,
                    'description': 'Shadow horizontal offset in pixels.',
                    'step': 1},
              'y': {'type': 'integer',
                    'default': 8,
                    'description': 'Shadow vertical offset in pixels.',
                    'step': 1},
              'radius': {'type': 'number',
                         'default': 5,
                         'description': 'Blur radius in pixels.',
                         'step': 0.05,
                         'minimum': 0,
                         'maximum': 200},
              'opacity': {'type': 'number',
                          'default': 0.5,
                          'description': 'Shadow opacity.',
                          'step': 0.05,
                          'minimum': 0,
                          'maximum': 1},
              'color': {'type': 'string',
                        'format': 'color',
                        'default': '#000000',
                        'description': 'CSS/hex colour or live @palette role.'}},
 'example': {'x': 4, 'y': 4, 'radius': 3},
 'tags': 'shadow silhouette depth alpha compositing'}


def apply(sdk, image, controls, palette):
    from PIL import Image, ImageFilter
    if controls["opacity"] == 0:
        return image.copy()
    color = rgba(controls["color"], palette)
    alpha = image.getchannel("A").filter(ImageFilter.GaussianBlur(controls["radius"]))
    alpha = alpha.point([round(i * controls["opacity"] * color[3] / 255) for i in range(256)])
    shadow = Image.new("RGBA", image.size, color)
    shadow.putalpha(alpha)
    placed = Image.new("RGBA", image.size)
    placed.paste(shadow, (controls["x"], controls["y"]))
    return Image.alpha_composite(placed, image)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
