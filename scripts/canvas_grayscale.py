"""Grayscale technique. Use search_techniques(script="canvas_grayscale") for controls and examples."""
from .canvas_pixels import execute, grayscale

box = "image_editing"
dependencies_files = ["scripts/canvas_pixels.py", "scripts/canvas_catalog.py", "scripts/art_kit.py"]


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    return execute(sdk, "canvas_grayscale", grayscale, kind, input_path, output_path,
                   width, height, seed, palette, controls)
