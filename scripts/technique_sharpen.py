"""Sharpen. Metadata and implementation live together.

Unsharp mask using an alpha-weighted blur. Preserves alpha and avoids hidden-colour fringes.
"""
from .art_kit import read_image, write_png, map_rgb, gaussian_blur_array
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Sharpen',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Unsharp mask using an alpha-weighted blur. Preserves alpha and avoids '
                'hidden-colour fringes.',
 'controls': {'radius': {'type': 'number',
                         'default': 2,
                         'minimum': 0,
                         'maximum': 100,
                         'step': 0.25,
                         'description': 'Detail radius.',
                         'unit': 'px'},
              'amount': {'type': 'number',
                         'default': 100,
                         'minimum': 0,
                         'maximum': 500,
                         'step': 5,
                         'description': 'Detail gain; zero is unchanged.',
                         'unit': '%'},
              'threshold': {'type': 'integer',
                            'step': 1,
                            'description': 'Ignore RGB differences below this threshold.',
                            'unit': 'levels',
                            'minimum': 0,
                            'default': 3,
                            'maximum': 255}},
 'example': {'radius': 1.5, 'amount': 80, 'threshold': 3},
 'tags': 'sharpness crisp detail unsharp',
 'aliases': ['canvas_sharpen']}


def apply(sdk, image, controls, palette):
    import numpy as np
    if controls["amount"] == 0 or controls["radius"] == 0:
        return image.copy()
    soft = gaussian_blur_array(image, controls["radius"])[..., :3]
    def adjust(rgb):
        detail = rgb - soft
        detail = np.where(np.abs(detail) * 255 >= controls["threshold"], detail, 0)
        return rgb + detail * controls["amount"] / 100
    return map_rgb(image, adjust)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
