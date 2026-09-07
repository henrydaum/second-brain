"""Copy to workspace/scripts/technique_your_name.py and edit metadata + apply.

The technique_ prefix makes a script discoverable. TECHNIQUE must be a literal
dictionary: discovery reads it without executing the script. Keep box and helper
dependencies. main receives the current image size and the resolved live palette.
Filters return replacement RGBA images; objects return transparent overlays;
backgrounds receive no input. Use SDK file IO and art_kit shared utilities.
File controls use format="file" so rendering fingerprints their contents.
Validate with sdk.plugins.validate(path), then search, add_layer and render_canvas.
This starter adjusts brightness while preserving alpha; replace apply as needed.
"""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Brightness',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Multiply encoded RGB; alpha is unchanged. Zero is black, one is unchanged.',
 'controls': {'factor': {'type': 'number',
                         'default': 1,
                         'minimum': 0,
                         'maximum': 4,
                         'step': 0.05,
                         'description': 'Brightness multiplier.',
                         'unit': ''}},
 'example': {'factor': 1.1},
 'tags': 'light dark brighten dim'}


def apply(sdk, image, controls, palette):
    return map_rgb(image, lambda rgb: rgb * controls["factor"])


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
