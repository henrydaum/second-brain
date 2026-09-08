"""Build an opaque black/white rectangle, ellipse or polygon selection. White selects; coordinates use the current canvas size."""
from .art_kit import read_image, write_png, selection_image
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Shape selection',
 'kind': 'background',
 'kinds': ['background', 'filter'],
 'description': 'Build an opaque black/white rectangle, ellipse or polygon selection. White '
                'selects; coordinates use the current canvas size.',
 'controls': {'shape': {'type': 'string',
                        'default': 'rectangle',
                        'enum': ['rectangle', 'ellipse', 'polygon']},
              'left': {'type': 'number', 'default': 0, 'step': 1, 'description': 'Left edge.'},
              'top': {'type': 'number', 'default': 0, 'step': 1, 'description': 'Top edge.'},
              'right': {'type': 'number',
                        'default': 100,
                        'step': 1,
                        'description': 'Exclusive right edge.'},
              'bottom': {'type': 'number',
                         'default': 100,
                         'step': 1,
                         'description': 'Exclusive bottom edge.'},
              'points': {'type': 'array',
                         'default': [[0, 0], [100, 0], [50, 100]],
                         'minItems': 3,
                         'items': {'type': 'array',
                                   'minItems': 2,
                                   'maxItems': 2,
                                   'items': {'type': 'number'}}},
              'feather': {'type': 'number',
                          'default': 0,
                          'step': 1,
                          'description': 'Feather radius in pixels.',
                          'minimum': 0,
                          'maximum': 200},
              'invert': {'type': 'boolean', 'default': False}},
 'example': {'right': 100, 'bottom': 100, 'feather': 5},
 'tags': 'selection mask rectangle ellipse polygon feather invert'}


def apply(sdk, image, controls, palette):
    from PIL import Image, ImageDraw
    import numpy as np
    scale = 2
    mask = Image.new("L", (image.width * scale, image.height * scale))
    draw = ImageDraw.Draw(mask)
    shape = controls["shape"]
    if shape == "polygon":
        draw.polygon([(round(x * scale), round(y * scale)) for x, y in controls["points"]], fill=255)
    else:
        if controls["right"] <= controls["left"] or controls["bottom"] <= controls["top"]:
            raise ValueError("right/bottom must exceed left/top for a shape selection")
        bounds = (controls["left"] * scale, controls["top"] * scale,
                  controls["right"] * scale - 1, controls["bottom"] * scale - 1)
        if shape == "rectangle": draw.rectangle(bounds, fill=255)
        else: draw.ellipse(bounds, fill=255)
    mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return selection_image(np.asarray(mask, dtype=np.float32) / 255, controls["feather"], controls["invert"])


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
