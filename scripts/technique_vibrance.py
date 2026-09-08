"""Adjust saturation with less effect on already saturated colours. This is saturation-adaptive, not face or skin detection."""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Vibrance',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Adjust saturation with less effect on already saturated colours. This is '
                'saturation-adaptive, not face or skin detection.',
 'controls': {'amount': {'type': 'number',
                         'default': 0,
                         'description': 'Positive boosts muted colours; negative reduces '
                                        'saturation.',
                         'step': 0.05,
                         'minimum': -1,
                         'maximum': 1}},
 'example': {'amount': 0.35},
 'tags': 'colour saturation muted vivid photo'}


def apply(sdk, image, controls, palette):
    import numpy as np
    if controls["amount"] == 0:
        return image.copy()
    def adjust(rgb):
        maximum = rgb.max(axis=2)
        saturation = (maximum - rgb.min(axis=2)) / np.maximum(maximum, 1e-8)
        factor = 1 + controls["amount"] * (1 - saturation)
        luminance = (rgb @ np.array([.2126, .7152, .0722], dtype=np.float32))[..., None]
        return luminance + (rgb - luminance) * factor[..., None]
    return map_rgb(image, adjust)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
