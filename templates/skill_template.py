"""
SKILL TEMPLATE
==============
This file is a self-contained reference for creating new canvas skills.
It is NOT imported by the running system — it exists for LLM consumption only.

Write skills in the same voice the system expects elsewhere:
- composed, intentional, algorithmic — not literal illustration
- every color from the palette (canvas.palette slots or art_kit.palette_color)
- every random source seeded from canvas.seed
- every code path ends with canvas.commit(image)

Skill authoring flow:
  1. Read this template, then call read_skill_guide ONCE for taste guidance.
  2. Call search_skills with the subject — if a strong match exists, use it.
  3. If you must author, prefer cloning via read_skill(slug) + create_skill
     rather than writing from scratch.
  4. Provide a module-level body to create_skill:
       - any needed imports (math, random, colorsys, numpy, PIL.*)
       - one `def run(canvas, **params):` function
     create_skill wraps that body in a BaseSkill class automatically and
     writes the file into the sandbox skills folder.
  5. Call execute_skill(slug=<returned-slug>) to render it.
  6. If execution returns an error with a hint line, read the hint and
     iterate via update_skill on the same slug.


AUTO-DISCOVERY RULES
--------------------
- File must be in plugins/skills/ (baked-in) or the sandbox skills dir
- File name must start with "skill_"
- Class must inherit from BaseSkill
- Class must define `def run(self, canvas)` for descriptor-style built-ins
  or `def run(self, canvas, **params)` for wrapped sandbox skills
- One skill class per file


CONTEXT THE SANDBOX INJECTS
---------------------------
Skills run in an isolated subprocess. Inside run(), you get:

  canvas.palette    SimpleNamespace with slots: primary, secondary, tertiary,
                    accent, background. Each slot is a string ("#RRGGBB") that
                    also behaves as an (r, g, b) sequence.
  canvas.size       Square dimension in pixels.
  canvas.width      Alias of canvas.size.
  canvas.height     Alias of canvas.size.
  canvas.seed       Integer for seeding RNGs (random.Random(canvas.seed) etc).
  canvas.image      The current image (transform skills only); raises for
                    creation skills.
  canvas.new(w, h, color=...)         Fresh RGBA image.
  canvas.create_image(color=...)      Fresh RGBA image filled with the
                                       palette background by default.
  canvas.commit(image)                **REQUIRED** — call once at the end.

  art_kit           Namespace of pre-bound helpers (no import needed):
                      palette_color(t, value=1.0)   sample palette ramp
                      vogel_spiral(n, scale=1.0)
                      fbm(seed, x, y, octaves=4, lacunarity=2.0, gain=0.5)
                      value_noise(seed, x, y)
                      radial_falloff(w, h, cx, cy)
                      flow_field(seed, scale=0.005, octaves=4)
                      voronoi_nearest(points)
                      lindenmayer(axiom, rules, iterations)
                      turtle_segments(sentence, start, heading, step, turn)
                      wave_field(sources)
                      attractor_points(name, n, seed, params=None)
                      rule_of_thirds(size)
                      hex_to_rgb / rgb_to_hex / mix_hex / oklch_to_rgb
                      lerp / clamp / smoothstep / remap
                      pi / tau


ALLOWED IMPORTS
---------------
Only these modules may be imported inside a skill:
  math, random, colorsys
  numpy (and numpy.random)
  PIL.Image, PIL.ImageDraw, PIL.ImageFilter, PIL.ImageOps,
  PIL.ImageEnhance, PIL.ImageChops, PIL.ImageColor
  plugins.BaseSkill  (literal — for `BaseSkill`, `Slider`, `Bool`, `Enum`, `Pan`, `Palette`)

Everything else (os, sys, subprocess, requests, plugins.helpers.*, etc.) is
blocked at AST validation time. There is no escape hatch.


CONTROLS
--------
Built-in skills declare up to 3 non-palette controls as BaseSkill descriptors:

  intensity = Slider(0.0, 1.0, default=0.5, step=0.01)
  mode      = Enum([("soft", "Soft"), ("crisp", "Crisp")], default="soft")
  wrap      = Bool(default=False)
  cx        = Slider(0.0, 1.0, default=0.5)
  cy        = Slider(0.0, 1.0, default=0.5)
  center    = Pan(x="cx", y="cy")
  palette   = Palette()

Read values as self.intensity/self.mode/etc. Slider values are clamped before
run() starts. Pan is only a UI grouping over two sliders; read self.cx/self.cy.
Sandbox skills created through create_skill still use the dict-form controls
schema because the tool wraps a module-level run(canvas, **params) function.
"""

# =====================================================================
# BASE CLASS (copied from plugins/BaseSkill.py for self-containment)
# =====================================================================


class BaseSkill:
    """The contract every skill implements. See plugins/BaseSkill.py."""
    name: str = ""
    description: str = ""
    kind: str = "creation"          # "creation" | "transform"
    owner: str = "library"
    created_at: float = 0.0
    controls: list = []
    hidden: bool = False
    auto_register: bool = True
    requires_services: list[str] = []
    config_settings: list = []

    def run(self, canvas):
        raise NotImplementedError


# =====================================================================
# EXAMPLE — a creation skill (full file shape; create_skill emits this)
# =====================================================================
#
# from plugins.BaseSkill import BaseSkill, Slider, Palette
# import math
# import random
# from PIL import Image, ImageDraw
#
#
# class VogelBloomSkill(BaseSkill):
#     name = "Vogel Bloom"
#     description = "A sunflower-style bloom of palette-blended cells."
#     kind = "creation"
#     palette = Palette()
#     density = Slider(200, 1500, default=600, step=50)
#
#     def run(self, canvas):
#         rng = random.Random(canvas.seed)
#         img = canvas.create_image()
#         draw = ImageDraw.Draw(img, "RGBA")
#         s = canvas.size
#         scale = s * 0.42
#         density = int(self.density)
#         for i, (x, y) in enumerate(art_kit.vogel_spiral(density, scale=scale)):
#             t = i / max(1, density - 1)
#             color = art_kit.palette_color(t)
#             r = 4 + 8 * (1.0 - t)
#             cx, cy = s / 2 + x, s / 2 + y
#             draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
#         canvas.commit(img)


# =====================================================================
# EXAMPLE — a transform skill (reads canvas.image, returns reshaped)
# =====================================================================
#
# from plugins.BaseSkill import BaseSkill, Slider
# from PIL import ImageFilter
#
#
# class BloomGlowSkill(BaseSkill):
#     name = "Bloom Glow"
#     description = "A soft luminous bloom over the current canvas."
#     kind = "transform"
#     radius = Slider(1.0, 24.0, default=8.0, step=0.5)
#
#     def run(self, canvas):
#         base = canvas.image
#         glow = base.filter(ImageFilter.GaussianBlur(radius=float(self.radius)))
#         canvas.commit(canvas.image.alpha_composite(glow) or glow)
