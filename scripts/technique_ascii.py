"""Render luminance as a grid of ASCII glyphs, optionally using each cell's sampled colour. Columns controls detail independent of image size. Keeps output size and multiplies output alpha by original coverage."""
from .art_kit import read_image, write_png, rgba
from .canvas_catalog import validate_controls

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]

TECHNIQUE = {'title': 'ASCII image',
 'kind': 'filter',
 'kinds': ['filter'],
 'description': "Render luminance as a grid of ASCII glyphs, optionally using each cell's sampled "
                'colour. Columns controls detail independent of image size. Keeps output size and '
                'multiplies output alpha by original coverage.',
 'controls': {'columns': {'type': 'integer',
                          'default': 80,
                          'description': 'Number of character columns; capped at image width.',
                          'step': 1,
                          'minimum': 1,
                          'maximum': 240},
              'characters': {'type': 'string',
                             'default': ' .:-=+*#%@',
                             'description': 'Glyph ramp ordered sparse to dense; 2 to 32 printable '
                                            'ASCII characters.'},
              'colorize': {'type': 'boolean', 'default': True},
              'foreground': {'type': 'string', 'default': '@accent', 'format': 'color'},
              'background': {'type': 'string', 'default': '#000000', 'format': 'color'}},
 'example': {'columns': 80},
 'tags': 'ascii text terminal glyph typewriter retro'}


def apply(sdk, image, controls, palette):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    ramp = controls["characters"]
    if not 2 <= len(ramp) <= 32 or any(ord(c) < 32 or ord(c) > 126 for c in ramp):
        raise ValueError("characters needs 2 to 32 printable ASCII characters, ordered sparse to dense")
    columns = min(controls["columns"], image.width)
    cell_width = image.width / columns
    rows = max(1, round(image.height / (cell_width * 2)))
    cell_height = image.height / rows
    samples = image.convert("RGBa").resize((columns,rows), Image.Resampling.BOX).convert("RGBA")
    lum = np.asarray(samples.convert("L"), dtype=np.float32) / 255
    foreground, background = rgba(controls["foreground"], palette), rgba(controls["background"], palette)
    result = Image.new("RGBA", image.size, background)
    font = ImageFont.load_default(size=max(1, round(cell_height * .85)))
    # Render each glyph once and fit it into a fixed cell. The default font
    # need not be monospace; fixed tile bounds prevent overlap.
    tiles = {}
    for char in set(ramp):
        box = font.getbbox(char)
        glyph = Image.new("L", (max(1, box[2]-box[0]), max(1, box[3]-box[1])))
        ImageDraw.Draw(glyph).text((-box[0], -box[1]), char, font=font, fill=255)
        tiles[char] = glyph
    for y in range(rows):
        for x in range(columns):
            left, top = round(x*cell_width), round(y*cell_height)
            right, bottom = round((x+1)*cell_width), round((y+1)*cell_height)
            tile_size = (max(1,right-left), max(1,bottom-top))
            char = ramp[min(len(ramp)-1, int(lum[y,x] * (len(ramp)-1) + .5))]
            coverage = tiles[char].resize(tile_size, Image.Resampling.LANCZOS)
            ink = (*samples.getpixel((x,y))[:3], foreground[3]) if controls["colorize"] else foreground
            tile = Image.new("RGBA", tile_size, ink)
            tile.putalpha(coverage.point([round(i*ink[3]/255) for i in range(256)]))
            result.alpha_composite(tile, dest=(left,top))
    alpha = np.asarray(result.getchannel("A"), dtype=np.float32) * np.asarray(image.getchannel("A"), dtype=np.float32) / 255
    result.putalpha(Image.fromarray(np.uint8(alpha + .5)))
    return result


def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    controls = validate_controls(TECHNIQUE, controls, kind)
    image = read_image(sdk, input_path) if input_path else Image.new("RGBA", (width, height))
    result = apply(sdk, image, controls, palette)
    return {"path": write_png(sdk, output_path, result),
            "width": result.width, "height": result.height}
