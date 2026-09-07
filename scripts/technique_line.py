"""Line / polyline. Metadata and implementation live together.

Draw an antialiased line through pixel coordinates. Produces an overlay, so layer opacity and blend mode work normally.
"""
from .art_kit import read_image, write_png, rgba, antialiased_overlay
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Line / polyline',
 'kind': 'object',
 'kinds': ['object'],
 'description': 'Draw an antialiased line through pixel coordinates. Produces an overlay, so layer '
                'opacity and blend mode work normally.',
 'controls': {'points': {'type': 'array',
                         'minItems': 2,
                         'description': '[[x,y], ...] in current image pixels.',
                         'items': {'type': 'array',
                                   'minItems': 2,
                                   'maxItems': 2,
                                   'items': {'type': 'number'}}},
              'width': {'type': 'number',
                        'default': 3,
                        'minimum': 0.1,
                        'maximum': 1000,
                        'step': 0.5,
                        'description': 'Stroke width.',
                        'unit': 'px'},
              'color': {'type': 'string',
                        'format': 'color',
                        'default': '@primary',
                        'description': 'Stroke colour.'}},
 'example': {'points': [[40, 40], [240, 140]], 'width': 3, 'color': '@accent'},
 'tags': 'draw stroke path annotation',
 'aliases': ['canvas_line']}


def apply(sdk, image, controls, palette):
    def paint(draw, scale):
        points = [(x * scale, y * scale) for x, y in controls["points"]]
        draw.line(points, fill=rgba(controls["color"], palette), width=max(1, round(controls["width"] * scale)), joint="curve")
    return antialiased_overlay(image.size, paint)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
