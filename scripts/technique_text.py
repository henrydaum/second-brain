"""Text. Metadata and implementation live together.

Draw text on a transparent overlay. Explicit font_path supports custom faces/Unicode coverage; the portable default is regular Latin text.
"""
from .art_kit import read_image, write_png, load_font, text, rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Text',
 'kind': 'object',
 'kinds': ['object'],
 'description': 'Draw text on a transparent overlay. Explicit font_path supports custom '
                'faces/Unicode coverage; the portable default is regular Latin text.',
 'controls': {'content': {'type': 'string', 'description': 'Text; newline creates a line break.'},
              'x': {'type': 'integer',
                    'step': 1,
                    'description': 'Left position.',
                    'unit': 'px',
                    'default': 0},
              'y': {'type': 'integer',
                    'step': 1,
                    'description': 'Top position.',
                    'unit': 'px',
                    'default': 0},
              'size': {'type': 'integer',
                       'step': 1,
                       'description': 'Font size.',
                       'unit': 'px',
                       'minimum': 1,
                       'default': 48},
              'color': {'type': 'string',
                        'format': 'color',
                        'default': '@primary',
                        'description': 'Text colour.'},
              'font_path': {'type': 'string',
                            'format': 'file',
                            'default': '',
                            'description': 'Optional TTF/OTF path; tracked automatically.'},
              'max_width': {'type': 'integer',
                            'step': 1,
                            'description': 'Wrap width; zero disables wrapping.',
                            'unit': 'px',
                            'minimum': 0,
                            'default': 0},
              'align': {'type': 'string',
                        'default': 'left',
                        'enum': ['left', 'center', 'right'],
                        'description': 'Alignment of lines within the text block.'}},
 'example': {'content': 'Summer 2026', 'x': 40, 'y': 40, 'size': 36, 'color': '#ffffff'},
 'tags': 'label caption typography annotation',
 'aliases': ['canvas_text']}


def apply(sdk, image, controls, palette):
    from PIL import Image
    result = Image.new("RGBA", image.size)
    font = load_font(sdk, controls["font_path"], controls["size"]) if controls["font_path"] else None
    text(result, (controls["x"], controls["y"]), controls["content"], size=controls["size"], color=rgba(controls["color"], palette),
         max_width=controls["max_width"] or None, align=controls["align"], font=font)
    return result


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
