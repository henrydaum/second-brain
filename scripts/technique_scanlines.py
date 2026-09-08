"""Apply horizontal CRT-style dark bands. Density is a count across the image, not pixel positions. Preserves alpha."""
from .art_kit import read_image, write_png, map_rgb
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Scanlines',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Apply horizontal CRT-style dark bands. Density is a count across the image, not '
                'pixel positions. Preserves alpha.',
 'controls': {'lines': {'type': 'integer',
                        'default': 100,
                        'description': 'Number of horizontal scanline periods.',
                        'step': 1,
                        'minimum': 1,
                        'maximum': 1000},
              'strength': {'type': 'number',
                           'default': 0.35,
                           'description': 'Darkening strength; zero unchanged.',
                           'step': 0.01,
                           'minimum': 0,
                           'maximum': 1},
              'width': {'type': 'number',
                        'default': 0.35,
                        'description': 'Dark-band fraction of each period.',
                        'step': 0.01,
                        'minimum': 0,
                        'maximum': 1},
              'phase': {'type': 'number',
                        'default': 0,
                        'description': 'Vertical phase as a fraction of a period.',
                        'step': 0.01,
                        'minimum': 0,
                        'maximum': 1}},
 'example': {'lines': 100, 'strength': 0.35},
 'tags': 'crt vhs television scanlines retro'}


def apply(sdk, image, controls, palette):
    import numpy as np
    if controls["strength"] == 0 or controls["width"] == 0:
        return image.copy()
    # Integrate the periodic band over each pixel's vertical extent. This
    # avoids aliasing when the requested line count exceeds image height.
    y = np.arange(image.height, dtype=np.float64)
    start = y * controls["lines"] / image.height + controls["phase"]
    end = (y + 1) * controls["lines"] / image.height + controls["phase"]
    def integral(t):
        return np.floor(t) * controls["width"] + np.minimum(t % 1, controls["width"])
    coverage = (integral(end) - integral(start)) / (end - start)
    gain = 1 - controls["strength"] * coverage[:, None, None]
    return map_rgb(image, lambda rgb: rgb * gain)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
