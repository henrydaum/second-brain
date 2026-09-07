"""Solid fill. Metadata and implementation live together.

Fill with a literal colour or a live @palette-role. Transparent is allowed.
"""
from .art_kit import read_image, write_png, rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Solid fill',
 'kind': 'background',
 'kinds': ['background', 'object'],
 'description': 'Fill with a literal colour or a live @palette-role. Transparent is allowed.',
 'controls': {'color': {'type': 'string',
                        'format': 'color',
                        'default': '@background',
                        'description': 'Fill colour: CSS name, #RGB, #RRGGBB, #RRGGBBAA, '
                                       'transparent or @role.'}},
 'example': {'color': '#f5f2e8'},
 'tags': 'background color colour fill transparent',
 'aliases': ['canvas_solid']}


def apply(sdk, image, controls, palette):
    from PIL import Image
    return Image.new("RGBA", image.size, rgba(controls["color"], palette))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
