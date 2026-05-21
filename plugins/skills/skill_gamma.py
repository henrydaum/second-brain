from plugins.BaseSkill import BaseSkill

import numpy as np

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class GammaSkill(BaseSkill):
    name = 'Gamma'
    description = 'Power-curve tone shift. gamma<1 lifts shadows (brighter midtones), gamma>1 crushes them (darker midtones). More musical than linear brightness. Param: gamma (0.2-3.0, default 1.0).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'gamma', 'label': 'Gamma', 'min': 0.2, 'max': 3.0, 'step': 0.05, 'default': 1.0},
    ]

    def run(self, canvas, gamma=1.0):
        g = float(art_kit.clamp(gamma, 0.1, 5.0))
        arr = canvas.image_array(mode="RGB", dtype="float")
        canvas.commit_array(np.power(arr, 1.0 / g))
