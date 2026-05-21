from plugins.BaseSkill import BaseSkill

from PIL import Image, ImageOps

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class MirrorSkill(BaseSkill):
    name = 'Mirror'
    description = 'Symmetry transform. "horizontal" reflects the left half onto the right (or vice versa), "vertical" mirrors top to bottom, "quad" makes a 4-way kaleidoscopic symmetry from the top-left quadrant. Adds instant order. Param: mode enum.'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'enum', 'name': 'mode', 'label': 'Mode', 'options': [
            {'value': 'horizontal_lr', 'label': 'Left → Right'},
            {'value': 'horizontal_rl', 'label': 'Right → Left'},
            {'value': 'vertical_tb', 'label': 'Top → Bottom'},
            {'value': 'vertical_bt', 'label': 'Bottom → Top'},
            {'value': 'quad', 'label': '4-way Kaleidoscope'},
            {'value': 'flip_h', 'label': 'Flip Horizontal'},
            {'value': 'flip_v', 'label': 'Flip Vertical'},
        ], 'default': 'quad'},
    ]

    def run(self, canvas, mode='quad'):
        img = canvas.image.convert("RGBA")
        s = canvas.size
        mode = str(mode)
        if mode == 'flip_h':
            out = ImageOps.mirror(img)
        elif mode == 'flip_v':
            out = ImageOps.flip(img)
        elif mode == 'horizontal_lr':
            half = img.crop((0, 0, s // 2, s))
            mirrored = ImageOps.mirror(half)
            out = Image.new("RGBA", (s, s))
            out.paste(half, (0, 0))
            out.paste(mirrored, (s // 2, 0))
        elif mode == 'horizontal_rl':
            half = img.crop((s - s // 2, 0, s, s))
            mirrored = ImageOps.mirror(half)
            out = Image.new("RGBA", (s, s))
            out.paste(mirrored, (0, 0))
            out.paste(half, (s - s // 2, 0))
        elif mode == 'vertical_tb':
            half = img.crop((0, 0, s, s // 2))
            mirrored = ImageOps.flip(half)
            out = Image.new("RGBA", (s, s))
            out.paste(half, (0, 0))
            out.paste(mirrored, (0, s // 2))
        elif mode == 'vertical_bt':
            half = img.crop((0, s - s // 2, s, s))
            mirrored = ImageOps.flip(half)
            out = Image.new("RGBA", (s, s))
            out.paste(mirrored, (0, 0))
            out.paste(half, (0, s - s // 2))
        else:  # quad
            quad = img.crop((0, 0, s // 2, s // 2))
            qh = ImageOps.mirror(quad)
            qv = ImageOps.flip(quad)
            qhv = ImageOps.flip(qh)
            out = Image.new("RGBA", (s, s))
            out.paste(quad, (0, 0))
            out.paste(qh, (s // 2, 0))
            out.paste(qv, (0, s // 2))
            out.paste(qhv, (s // 2, s // 2))
        canvas.commit(out)
