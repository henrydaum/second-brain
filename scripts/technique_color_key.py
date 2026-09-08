"""Make pixels near a target RGB colour transparent, with an optional soft distance transition. Classic chroma key; does not infer subjects or remove spill."""
from .art_kit import read_image, write_png, rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Colour key removal',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Make pixels near a target RGB colour transparent, with an optional soft distance '
                'transition. Classic chroma key; does not infer subjects or remove spill.',
 'controls': {'color': {'type': 'string',
                        'format': 'color',
                        'default': '#00ff00',
                        'description': 'CSS/hex colour or live @palette role.'},
              'tolerance': {'type': 'number',
                            'default': 30,
                            'description': 'RGB Euclidean distance fully removed.',
                            'step': 0.05,
                            'minimum': 0,
                            'maximum': 442},
              'softness': {'type': 'number',
                           'default': 20,
                           'description': 'Additional distance for transition to opaque.',
                           'step': 0.05,
                           'minimum': 0,
                           'maximum': 442},
              'amount': {'type': 'number',
                         'default': 1,
                         'description': 'Removal strength; zero unchanged.',
                         'step': 0.05,
                         'minimum': 0,
                         'maximum': 1}},
 'example': {'color': '#00ff00', 'tolerance': 30, 'softness': 20},
 'tags': 'chroma green screen background remove transparency'}


def apply(sdk, image, controls, palette):
    import numpy as np
    from PIL import Image
    if controls["amount"] == 0:
        return image.copy()
    data = np.asarray(image, dtype=np.float32)
    target = np.array(rgba(controls["color"], palette)[:3], dtype=np.float32)
    distance = np.sqrt(np.sum((data[..., :3] - target) ** 2, axis=2))
    keep = (distance > controls["tolerance"]).astype(np.float32) if controls["softness"] == 0 else np.clip((distance - controls["tolerance"]) / controls["softness"], 0, 1)
    data[..., 3] *= (1 - controls["amount"]) + controls["amount"] * keep
    return Image.fromarray(np.uint8(np.clip(data + .5, 0, 255)))


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
