from plugins.BaseTechnique import BaseTechnique, Pan, Slider

from PIL import Image


class CropTechnique(BaseTechnique):
    name = "Crop"
    description = "Filter: crop into the current image and resize back to canvas size; pan chooses the crop center."
    kind = "filter"
    zoom = Slider(1.0, 4.0, default=1.4, step=0.05)
    cx = Slider(0.0, 1.0, default=0.5, step=0.04)
    cy = Slider(0.0, 1.0, default=0.5, step=0.04)
    center = Pan(x="cx", y="cy")

    def run(self, canvas):
        s = canvas.size
        side = max(1, int(round(s / max(float(self.zoom), 1.0))))
        half = side / 2
        x = min(max(self.cx * s, half), s - half)
        y = min(max(self.cy * s, half), s - half)
        box = (int(round(x - half)), int(round(y - half)), int(round(x + half)), int(round(y + half)))
        canvas.commit(canvas.image.crop(box).resize((s, s), Image.Resampling.LANCZOS))
