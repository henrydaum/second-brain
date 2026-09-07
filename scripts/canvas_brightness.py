"""Brightness technique. Use search_techniques(script="canvas_brightness") for controls and examples."""
from .canvas_pixels import execute, brightness

box = "image_editing"
dependencies_files = ["scripts/canvas_pixels.py", "scripts/canvas_catalog.py", "scripts/art_kit.py"]


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    return execute(sdk, "canvas_brightness", brightness, kind, input_path, output_path,
                   width, height, seed, palette, controls)
