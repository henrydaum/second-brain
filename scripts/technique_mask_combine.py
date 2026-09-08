"""Combine current mask coverage with a saved mask PNG of matching dimensions. Inputs are luminance times alpha; output is opaque grayscale."""
from .art_kit import read_image, write_png, selection_image, mask_coverage
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Combine selections',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Combine current mask coverage with a saved mask PNG of matching dimensions. '
                'Inputs are luminance times alpha; output is opaque grayscale.',
 'controls': {'path': {'type': 'string',
                       'format': 'file',
                       'description': 'Existing cached mask PNG path.'},
              'operation': {'type': 'string',
                            'default': 'union',
                            'enum': ['union', 'intersect', 'subtract', 'replace']},
              'feather': {'type': 'number',
                          'default': 0,
                          'step': 1,
                          'description': 'Feather radius in pixels.',
                          'minimum': 0,
                          'maximum': 200},
              'invert': {'type': 'boolean', 'default': False}},
 'example': {'path': '<cached mask PNG>', 'operation': 'intersect'},
 'tags': 'selection mask boolean union intersect subtract combine'}


def apply(sdk, image, controls, palette):
    import numpy as np
    other = read_image(sdk, controls["path"])
    if other.size != image.size:
        raise ValueError("Selection masks must have matching dimensions; resize the mask explicitly first")
    a, b = mask_coverage(image), mask_coverage(other)
    operation = controls["operation"]
    if operation == "union": result = np.maximum(a, b)
    elif operation == "intersect": result = np.minimum(a, b)
    elif operation == "subtract": result = np.clip(a - b, 0, 1)
    else: result = b
    return selection_image(result, controls["feather"], controls["invert"])


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
