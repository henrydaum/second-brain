from plugins.BaseTechnique import BaseTechnique, Slider

import numpy as np

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class GammaTechnique(BaseTechnique):
    name = 'Gamma'
    description = 'Power-curve tone shift. gamma<1 lifts shadows (brighter midtones), gamma>1 crushes them (darker midtones). More musical than linear brightness.'
    kind = "filter"

    gamma = Slider(0.2, 3.0, default=1.0)

    def run(self, canvas):
        arr = canvas.image_array(mode="RGB", dtype="float")
        canvas.commit_array(np.power(arr, 1.0 / self.gamma))
