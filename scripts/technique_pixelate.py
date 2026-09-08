"""Average premultiplied RGBA into coarse cells, then enlarge with nearest-neighbour sampling."""
from .art_kit import read_image, write_png
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Pixelate',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Average premultiplied RGBA into coarse cells, then enlarge with nearest-neighbour '
                'sampling.',
 'controls': {'block_size': {'type': 'integer',
                             'default': 8,
                             'minimum': 1,
                             'maximum': 4096,
                             'step': 1,
                             'description': 'Approximate cell size in pixels; one is unchanged.'}},
 'example': {'block_size': 8},
 'tags': 'mosaic pixel blocks abstract'}


def apply(sdk, image, controls, palette):
    from PIL import Image
    block = controls["block_size"]
    if block == 1:
        return image.copy()
    size = (max(1, (image.width + block - 1) // block), max(1, (image.height + block - 1) // block))
    return image.convert("RGBa").resize(size, Image.Resampling.BOX).resize(image.size, Image.Resampling.NEAREST).convert("RGBA")


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
