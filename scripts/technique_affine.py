"""Translate, scale, rotate and shear about the image centre. Keeps the current canvas dimensions; pad first to retain content outside its edges. Positive rotation is counterclockwise."""
from .art_kit import read_image, write_png, transform_image
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'Affine transform',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': 'Translate, scale, rotate and shear about the image centre. Keeps the current '
                'canvas dimensions; pad first to retain content outside its edges. Positive '
                'rotation is counterclockwise.',
 'controls': {'x': {'type': 'number',
                    'default': 0,
                    'description': 'Horizontal translation in pixels.',
                    'step': 0.05},
              'y': {'type': 'number',
                    'default': 0,
                    'description': 'Vertical translation in pixels.',
                    'step': 0.05},
              'angle': {'type': 'number',
                        'default': 0,
                        'description': 'Counterclockwise degrees.',
                        'step': 1,
                        'minimum': -360,
                        'maximum': 360},
              'scale_x': {'type': 'number',
                          'default': 1,
                          'description': 'Horizontal scale.',
                          'step': 0.05,
                          'minimum': 0.01,
                          'maximum': 100},
              'scale_y': {'type': 'number',
                          'default': 1,
                          'description': 'Vertical scale.',
                          'step': 0.05,
                          'minimum': 0.01,
                          'maximum': 100},
              'shear_x': {'type': 'number',
                          'default': 0,
                          'description': 'Horizontal shear coefficient.',
                          'step': 0.05,
                          'minimum': -5,
                          'maximum': 5},
              'shear_y': {'type': 'number',
                          'default': 0,
                          'description': 'Vertical shear coefficient.',
                          'step': 0.05,
                          'minimum': -5,
                          'maximum': 5}},
 'example': {'angle': 5, 'scale_x': 0.9, 'scale_y': 0.9},
 'tags': 'transform move skew scale rotate geometry'}


def apply(sdk, image, controls, palette):
    import numpy as np
    angle = np.deg2rad(controls["angle"])
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    shear = np.array([[1, controls["shear_x"]], [controls["shear_y"], 1]])
    forward = rotation @ shear @ np.diag([controls["scale_x"], controls["scale_y"]])
    if abs(np.linalg.det(forward)) < 1e-8:
        raise ValueError("Affine transform is singular; change shear or scale controls")
    inverse = np.linalg.inv(forward)
    center = np.array([image.width / 2, image.height / 2])
    translation = center - inverse @ (center + [controls["x"], controls["y"]])
    matrix = (inverse[0,0], inverse[0,1], translation[0], inverse[1,0], inverse[1,1], translation[1])
    if np.allclose(matrix, (1,0,0,0,1,0), rtol=0, atol=1e-12):
        return image.copy()
    return transform_image(image, image.size, matrix)


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
