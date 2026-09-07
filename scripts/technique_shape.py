"""Rectangle / ellipse. Metadata and implementation live together.

Draw an antialiased rectangle or ellipse within an explicit pixel box. Transparent fill makes an outline.
"""
from .art_kit import read_image, write_png, rgba, antialiased_overlay
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Rectangle / ellipse',
 'kind': 'object',
 'kinds': ['object'],
 'description': 'Draw an antialiased rectangle or ellipse within an explicit pixel box. '
                'Transparent fill makes an outline.',
 'controls': {'shape': {'type': 'string',
                        'default': 'rectangle',
                        'enum': ['rectangle', 'ellipse'],
                        'description': 'Shape.'},
              'left': {'type': 'integer',
                       'step': 1,
                       'description': 'Left edge.',
                       'unit': 'px',
                       'default': 0},
              'top': {'type': 'integer',
                      'step': 1,
                      'description': 'Top edge.',
                      'unit': 'px',
                      'default': 0},
              'right': {'type': 'integer',
                        'step': 1,
                        'description': 'Exclusive right edge.',
                        'unit': 'px'},
              'bottom': {'type': 'integer',
                         'step': 1,
                         'description': 'Exclusive bottom edge.',
                         'unit': 'px'},
              'fill': {'type': 'string',
                       'format': 'color',
                       'default': '@primary',
                       'description': 'Fill colour or transparent.'},
              'stroke': {'type': 'string',
                         'format': 'color',
                         'default': 'transparent',
                         'description': 'Outline colour.'},
              'stroke_width': {'type': 'number',
                               'default': 1,
                               'minimum': 0,
                               'maximum': 1000,
                               'step': 0.5,
                               'description': 'Outline width; zero disables it.',
                               'unit': 'px'}},
 'example': {'left': 20,
             'top': 20,
             'right': 220,
             'bottom': 120,
             'fill': 'transparent',
             'stroke': '@accent',
             'stroke_width': 3},
 'tags': 'draw box circle oval outline annotation',
 'aliases': ['canvas_shape'],
 'constraints': [{'greater': 'right', 'than': 'left'}, {'greater': 'bottom', 'than': 'top'}]}


def apply(sdk, image, controls, palette):
    def paint(draw, scale):
        bounds = (controls["left"] * scale, controls["top"] * scale,
                  controls["right"] * scale - 1, controls["bottom"] * scale - 1)
        kwargs = {"fill": rgba(controls["fill"], palette)}
        if controls["stroke_width"] > 0 and rgba(controls["stroke"], palette)[3] > 0:
            kwargs.update(outline=rgba(controls["stroke"], palette), width=max(1, round(controls["stroke_width"] * scale)))
        if controls["shape"] == "rectangle":
            draw.rectangle(bounds, **kwargs)
        else:
            draw.ellipse(bounds, **kwargs)
    return antialiased_overlay(image.size, paint)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
