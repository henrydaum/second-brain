"""Remove isolated speckles with a median filter. Premultiplied colour prevents hidden RGB bleeding; alpha is filtered too."""
from .art_kit import read_image, write_png
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Median denoise',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Remove isolated speckles with a median filter. Premultiplied colour prevents '
                'hidden RGB bleeding; alpha is filtered too.',
 'controls': {'radius': {'type': 'integer',
                         'default': 1,
                         'minimum': 0,
                         'maximum': 5,
                         'step': 1,
                         'description': 'Neighbourhood radius in pixels; zero is unchanged.'}},
 'example': {'radius': 1},
 'tags': 'noise speckle dust cleanup smooth'}


def apply(sdk, image, controls, palette):
    from PIL import ImageFilter
    radius = controls["radius"]
    if radius == 0:
        return image.copy()
    # Filter the four premultiplied channels independently; Pillow rank filters
    # do not accept RGBa directly on every supported version.
    from PIL import Image
    channels = [c.filter(ImageFilter.MedianFilter(radius * 2 + 1)) for c in image.convert("RGBa").split()]
    return Image.merge("RGBa", channels).convert("RGBA")


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
