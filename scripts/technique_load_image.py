"""Load photo or image. Metadata and implementation live together.

Read an attachment/local path with EXIF orientation and alpha. Native preserves original pixels and dimensions. Use object to overlay a second image.
"""
from .art_kit import read_image, write_png, resize_image
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Load photo or image',
 'kind': 'background',
 'kinds': ['background', 'object'],
 'description': 'Read an attachment/local path with EXIF orientation and alpha. Native preserves '
                'original pixels and dimensions. Use object to overlay a second image.',
 'controls': {'path': {'type': 'string',
                       'format': 'file',
                       'description': 'Existing attachment/local image path, not a URL.'},
              'fit': {'type': 'string',
                      'default': 'native',
                      'enum': ['native', 'contain', 'cover', 'stretch'],
                      'description': 'Native keeps source size; others fit current canvas '
                                     'dimensions.'}},
 'example': {'path': '<attachment-path>', 'fit': 'native'},
 'tags': 'upload import open picture photograph overlay',
 'aliases': ['canvas_load_image']}


def apply(sdk, image, controls, palette):
    loaded = read_image(sdk, controls["path"])
    return loaded if controls["fit"] == "native" else resize_image(loaded, image.size, controls["fit"])


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
