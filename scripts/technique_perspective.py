"""Rectify a convex source quadrilateral into a rectangle. Source corners are normalized image-edge coordinates, ordered top-left, top-right, bottom-right, bottom-left. Width/height zero keeps the current size."""
from .art_kit import read_image, write_png
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Perspective correction',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Rectify a convex source quadrilateral into a rectangle. Source corners are '
                'normalized image-edge coordinates, ordered top-left, top-right, bottom-right, '
                'bottom-left. Width/height zero keeps the current size.',
 'controls': {'corners': {'type': 'array',
                          'default': [[0, 0], [1, 0], [1, 1], [0, 1]],
                          'minItems': 4,
                          'items': {'type': 'array',
                                    'minItems': 2,
                                    'maxItems': 2,
                                    'items': {'type': 'number'}},
                          'maxItems': 4},
              'width': {'type': 'integer',
                        'default': 0,
                        'description': 'Output width, or zero for current.',
                        'step': 1,
                        'minimum': 0},
              'height': {'type': 'integer',
                         'default': 0,
                         'description': 'Output height, or zero for current.',
                         'step': 1,
                         'minimum': 0}},
 'example': {'corners': [[0.1, 0], [0.9, 0.1], [1, 1], [0, 0.9]]},
 'tags': 'keystone rectify document geometry perspective'}


def apply(sdk, image, controls, palette):
    import numpy as np
    from PIL import Image
    corners = np.asarray(controls["corners"], dtype=np.float64)
    if np.any(corners < 0) or np.any(corners > 1):
        raise ValueError("Source corners must use normalized coordinates in [0,1]")
    edges = np.roll(corners, -1, axis=0) - corners
    next_edges = np.roll(edges, -1, axis=0)
    cross = edges[:,0] * next_edges[:,1] - edges[:,1] * next_edges[:,0]
    if not (np.all(cross > 1e-8) or np.all(cross < -1e-8)):
        raise ValueError("Source corners must form a nondegenerate convex quadrilateral in perimeter order")
    width, height = controls["width"] or image.width, controls["height"] or image.height
    source = corners * [image.width, image.height]
    destination = [(0,0),(width,0),(width,height),(0,height)]
    rows, values = [], []
    for (x,y), (u,v) in zip(destination, source):
        rows.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
        values.extend([u,v])
    try:
        matrix = np.linalg.solve(np.asarray(rows), np.asarray(values))
    except np.linalg.LinAlgError:
        raise ValueError("Source corners cannot define a stable perspective transform") from None
    if (width,height) == image.size and np.allclose(matrix, (1,0,0,0,1,0,0,0), rtol=0, atol=1e-12):
        return image.copy()
    return image.convert("RGBa").transform((width,height), Image.Transform.PERSPECTIVE, tuple(matrix), resample=Image.Resampling.BICUBIC, fillcolor=(0,0,0,0)).convert("RGBA")


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
