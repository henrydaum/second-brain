"""Apply smooth luminance-weighted exposure corrections in linear light. Positive values brighten the named tonal region; alpha is unchanged."""
from .art_kit import read_image, write_png, map_rgb, srgb_to_linear, linear_to_srgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Shadows and highlights',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Apply smooth luminance-weighted exposure corrections in linear light. Positive '
                'values brighten the named tonal region; alpha is unchanged.',
 'controls': {'shadows': {'type': 'number',
                          'default': 0,
                          'description': 'Shadow correction strength.',
                          'step': 0.05,
                          'minimum': -1,
                          'maximum': 1},
              'highlights': {'type': 'number',
                             'default': 0,
                             'description': 'Highlight correction strength.',
                             'step': 0.05,
                             'minimum': -1,
                             'maximum': 1}},
 'example': {'shadows': 0.3, 'highlights': -0.2},
 'tags': 'recover dark bright tone exposure photo'}


def apply(sdk, image, controls, palette):
    import numpy as np
    if controls["shadows"] == 0 and controls["highlights"] == 0:
        return image.copy()
    def adjust(rgb):
        luminance = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
        stops = 2 * (controls["shadows"] * (1 - luminance) ** 2 + controls["highlights"] * luminance ** 2)
        return linear_to_srgb(srgb_to_linear(rgb) * (2 ** stops)[..., None])
    return map_rgb(image, adjust)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
