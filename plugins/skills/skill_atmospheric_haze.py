from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

def _desaturate(rgb, amount):
    # rgb: float32 (H, W, 3). amount in [0, 1].
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    gray = np.stack([lum, lum, lum], axis=-1)
    return rgb * (1.0 - amount[..., None]) + gray * amount[..., None]


class AtmosphericHazeSkill(BaseSkill):
    name = 'Atmospheric Haze'
    description = 'Shift hue and reduce saturation toward the palette background as a function of vertical position, mimicking the way distant landscapes lose contrast and cool toward sky. Sky direction enum picks where the haze concentrates: top (default atmospheric perspective), bottom (low-fog inversion), or both ends. The blend uses a smoothstep so the transition reads natural, not a flat overlay. Strength dials the effect from a whisper to thick haze.'
    kind = 'transform'
    owner = 'library'
    created_at = 1779667200.0
    hidden = False
    controls = [{'type': 'palette', 'name': 'palette', 'label': 'Palette'}, {'type': 'enum', 'name': 'direction', 'label': 'Direction', 'options': [{'value': 'top', 'label': 'Top (Atmospheric)'}, {'value': 'bottom', 'label': 'Bottom (Low Fog)'}, {'value': 'both', 'label': 'Both Ends'}], 'default': 'top'}, {'type': 'slider', 'name': 'strength', 'label': 'Strength', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'default': 0.45}]

    def run(self, canvas, direction="top", strength=0.45, **_):
        img = canvas.image.convert("RGB")
        s = img.size[0]
        strength = float(art_kit.clamp(strength, 0.0, 1.0))
        direction = str(direction)

        arr = np.asarray(img, dtype=np.float32)
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32)

        ys = np.linspace(0.0, 1.0, s, dtype=np.float32)
        if direction == "top":
            mask = 1.0 - ys
        elif direction == "bottom":
            mask = ys
        else:  # both
            mask = 1.0 - 2.0 * np.abs(ys - 0.5)
            mask = np.clip(mask, 0.0, 1.0)
            mask = 1.0 - mask  # invert: 1 at ends, 0 in middle

        mask = mask * mask * (3.0 - 2.0 * mask)
        mask = mask * strength
        mask2d = np.broadcast_to(mask[:, None], (s, s)).astype(np.float32)

        # Desaturate the hazy region a touch -- distance washes out chroma.
        arr = _desaturate(arr, mask2d * 0.6)
        # Blend toward palette background.
        mask3 = mask2d[..., None]
        out = arr * (1.0 - mask3) + bg[None, None, :] * mask3
        out = np.clip(out, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
