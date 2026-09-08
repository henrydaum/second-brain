"""Render luminance as antialiased dark dots on a light field. Cell-averaged luminance sets dot radius; source transparency is preserved."""
from .art_kit import read_image, write_png, rgba, colorize_field
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Halftone dots',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Render luminance as antialiased dark dots on a light field. Cell-averaged '
                'luminance sets dot radius; source transparency is preserved. Reads only earlier layers. '
                'Paper is the light output colour between dots, not a selection mask; choose '
                '@background to match the canvas palette. Use the layer mask to restrict the effect.',
 'controls': {'cell_size': {'type': 'integer',
                            'default': 8,
                            'description': 'Approximate dot-cell size in pixels.',
                            'step': 1,
                            'minimum': 2,
                            'maximum': 256},
              'ink': {'type': 'string',
                      'format': 'color',
                      'default': '@primary',
                      'description': 'CSS/hex colour or live @palette role.'},
              'paper': {'type': 'string',
                        'format': 'color',
                        'default': '#ffffff',
                        'description': 'CSS/hex colour or live @palette role.'}},
 'example': {'cell_size': 8},
 'tags': 'print comic newspaper halftone dots'}


def apply(sdk, image, controls, palette):
    import numpy as np
    from PIL import Image
    cell = controls["cell_size"]
    columns = max(1, (image.width + cell - 1) // cell)
    rows = max(1, (image.height + cell - 1) // cell)
    small = image.convert("RGBa").resize((columns, rows), Image.Resampling.BOX).convert("RGBA").convert("L")
    lum = np.asarray(small, dtype=np.float32) / 255
    yy, xx = np.indices((image.height, image.width), dtype=np.float32)
    gx, gy = (xx + .5) * columns / image.width, (yy + .5) * rows / image.height
    local = lum[np.minimum(gy.astype(int), rows - 1), np.minimum(gx.astype(int), columns - 1)]
    dx = (gx % 1 - .5) * image.width / columns
    dy = (gy % 1 - .5) * image.height / rows
    radius = .5 * np.hypot(image.width / columns, image.height / rows) * np.sqrt(1 - local)
    ink = np.clip(radius - np.hypot(dx, dy) + .5, 0, 1)
    ink = np.where(local >= 1, 0, np.where(local <= 0, 1, ink))
    alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255
    return colorize_field(ink, rgba(controls["paper"], palette), rgba(controls["ink"], palette), alpha)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
