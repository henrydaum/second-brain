"""Rotate technique. Use search_techniques(script="canvas_rotate") for controls and examples."""
from .canvas_pixels import execute, rotate

box = "image_editing"
dependencies_files = ["scripts/canvas_pixels.py", "scripts/canvas_catalog.py", "scripts/art_kit.py"]


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    return execute(sdk, "canvas_rotate", rotate, kind, input_path, output_path,
                   width, height, seed, palette, controls)
