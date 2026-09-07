"""Gamma / midtones. Metadata and implementation live together.

Apply RGB ** (1/gamma). Above one brightens midtones; endpoints and alpha stay fixed.
"""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Gamma / midtones',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Apply RGB ** (1/gamma). Above one brightens midtones; endpoints and alpha stay '
                'fixed.',
 'controls': {'gamma': {'type': 'number',
                        'default': 1,
                        'minimum': 0.1,
                        'maximum': 5,
                        'step': 0.05,
                        'description': 'Midtone gamma.',
                        'unit': ''}},
 'example': {'gamma': 1.15},
 'tags': 'midtone tonal light',
 'aliases': ['canvas_gamma']}


def apply(sdk, image, controls, palette):
    return map_rgb(image, lambda rgb: rgb ** (1 / controls["gamma"]))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
