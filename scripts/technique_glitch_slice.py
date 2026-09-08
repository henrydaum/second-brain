"""Shift randomly chosen horizontal bands sideways. Band height and shift are image-relative; the canvas seed fixes placement for repeatable fine-tuning. Transparent fill clips displaced bands."""
from .art_kit import read_image, write_png, sample_rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Glitch slices',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Shift randomly chosen horizontal bands sideways. Band height and shift are '
                'image-relative; the canvas seed fixes placement for repeatable fine-tuning. '
                'Transparent fill clips displaced bands.',
 'controls': {'slices': {'type': 'integer',
                         'default': 12,
                         'description': 'Number of bands.',
                         'step': 1,
                         'minimum': 0,
                         'maximum': 100},
              'height': {'type': 'number',
                         'default': 0.03,
                         'description': 'Band height as a fraction of image height.',
                         'step': 0.01,
                         'minimum': 0.001,
                         'maximum': 0.5},
              'shift': {'type': 'number',
                        'default': 0.08,
                        'description': 'Maximum shift as a fraction of image width.',
                        'step': 0.01,
                        'minimum': 0,
                        'maximum': 0.5}},
 'example': {'slices': 12, 'height': 0.03, 'shift': 0.08},
 'tags': 'glitch slice vhs datamosh bands seeded'}


def apply(sdk, image, controls, palette, seed=0):
    import numpy as np
    if controls["slices"] == 0 or controls["shift"] == 0:
        return image.copy()
    rng = np.random.default_rng(seed % (2 ** 64))
    xx, yy = np.meshgrid(np.arange(image.width, dtype=np.float32), np.arange(image.height, dtype=np.float32))
    offsets = np.zeros(image.height, dtype=np.float32)
    height = max(1, round(controls["height"] * image.height))
    # Sample the same random pairs for a fixed seed and slice count, even as
    # height/shift are tuned, so comparisons do not reshuffle the bands.
    for position, distance in rng.random((controls["slices"], 2)):
        top = min(image.height - 1, int(position * image.height))
        offsets[top:top + height] = (distance * 2 - 1) * controls["shift"] * image.width
    return sample_rgba(image, xx + offsets[:, None], yy)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette, seed=seed)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
