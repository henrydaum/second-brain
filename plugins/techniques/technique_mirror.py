from plugins.BaseTechnique import BaseTechnique, Enum

from PIL import Image, ImageOps

try:
    art_kit
except NameError:
    art_kit = None


class MirrorTechnique(BaseTechnique):
    name = 'Mirror'
    description = 'Symmetry transform. Reflects halves or makes a 4-way kaleidoscopic symmetry from the top-left quadrant. Adds instant order.'
    kind = "filter"

    mode = Enum([
        ('horizontal_lr', 'Left → Right'),
        ('horizontal_rl', 'Right → Left'),
        ('vertical_tb',   'Top → Bottom'),
        ('vertical_bt',   'Bottom → Top'),
        ('quad',          '4-way Mirror'),
        ('flip_h',        'Flip Horizontal'),
        ('flip_v',        'Flip Vertical'),
    ], default='quad')

    def run(self, canvas):
        img = canvas.image.convert("RGBA")
        s = canvas.size
        mode = self.mode
        if mode == 'flip_h':
            out = ImageOps.mirror(img)
        elif mode == 'flip_v':
            out = ImageOps.flip(img)
        elif mode == 'horizontal_lr':
            half = img.crop((0, 0, s // 2, s))
            out = Image.new("RGBA", (s, s))
            out.paste(half, (0, 0))
            out.paste(ImageOps.mirror(half), (s // 2, 0))
        elif mode == 'horizontal_rl':
            half = img.crop((s - s // 2, 0, s, s))
            out = Image.new("RGBA", (s, s))
            out.paste(ImageOps.mirror(half), (0, 0))
            out.paste(half, (s - s // 2, 0))
        elif mode == 'vertical_tb':
            half = img.crop((0, 0, s, s // 2))
            out = Image.new("RGBA", (s, s))
            out.paste(half, (0, 0))
            out.paste(ImageOps.flip(half), (0, s // 2))
        elif mode == 'vertical_bt':
            half = img.crop((0, s - s // 2, s, s))
            out = Image.new("RGBA", (s, s))
            out.paste(ImageOps.flip(half), (0, 0))
            out.paste(half, (0, s - s // 2))
        else:
            quad = img.crop((0, 0, s // 2, s // 2))
            qh = ImageOps.mirror(quad)
            qv = ImageOps.flip(quad)
            out = Image.new("RGBA", (s, s))
            out.paste(quad, (0, 0))
            out.paste(qh, (s // 2, 0))
            out.paste(qv, (0, s // 2))
            out.paste(ImageOps.flip(qh), (s // 2, s // 2))
        canvas.commit(out)
