"""Draw an outer outline around nontransparent content using a square neighbourhood. Pad first for room; opaque photos have no internal alpha silhouette."""
from .art_kit import read_image, write_png, rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Alpha outline',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Draw an outer outline around nontransparent content using a square neighbourhood. '
                'Pad first for room; opaque photos have no internal alpha silhouette.',
 'controls': {'radius': {'type': 'integer',
                         'default': 2,
                         'description': 'Outline radius in pixels; square neighbourhood.',
                         'step': 1,
                         'minimum': 0,
                         'maximum': 32},
              'color': {'type': 'string',
                        'format': 'color',
                        'default': '@accent',
                        'description': 'CSS/hex colour or live @palette role.'},
              'opacity': {'type': 'number',
                          'default': 1,
                          'description': 'Outline opacity.',
                          'step': 0.05,
                          'minimum': 0,
                          'maximum': 1}},
 'example': {'radius': 2},
 'tags': 'stroke border silhouette alpha compositing'}


def apply(sdk, image, controls, palette):
    from PIL import Image, ImageFilter, ImageChops
    if controls["radius"] == 0 or controls["opacity"] == 0:
        return image.copy()
    alpha = image.getchannel("A")
    grown = alpha.filter(ImageFilter.MaxFilter(2 * controls["radius"] + 1))
    ring = ImageChops.subtract(grown, alpha)
    color = rgba(controls["color"], palette)
    ring = ring.point([round(i * controls["opacity"] * color[3] / 255) for i in range(256)])
    outline = Image.new("RGBA", image.size, color)
    outline.putalpha(ring)
    return Image.alpha_composite(outline, image)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
