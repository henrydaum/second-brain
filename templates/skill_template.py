"""
SKILL TEMPLATE — the canvas-skill authoring contract.
=====================================================
This file is the single source of truth for *how* to write a canvas skill.
It is NOT imported by the running system; it exists for LLM consumption
and is served as-is by the `read_skill_guide` tool, which appends a live
index of every art_kit helper after the template.

You are building one skill: a small, deterministic Python function that
either paints a **background**, applies a **filter** to the current canvas,
or overlays an **object** on top of it.

Voice the system expects:
  - composed, intentional, algorithmic — not literal illustration
  - every color from the palette (canvas.palette slots or art_kit.palette_color)
  - every random source seeded from canvas.seed
  - every code path ends with canvas.commit(image)


SKILL AUTHORING FLOW
--------------------
1. Read this template, then call read_skill_guide ONCE per session.
2. Call search_skills with the subject — if a strong match exists, clone it.
3. If you must author, prefer cloning via read_skill(slug) + create_skill
   rather than writing from scratch.
4. While drafting, call read_skill_guide(methods=["fbm_grid", ...]) to
   inspect specific art_kit helpers — full signature, docstring, source.
5. Provide a complete BaseSkill class file to create_skill:
     - imports (math, random, colorsys, numpy, PIL.*, plugins.BaseSkill)
     - metadata as class attributes
     - descriptor controls as class attributes
     - one `def run(self, canvas):` method
6. Call execute_skill(slug=<returned-slug>) to render it.
7. If execution returns an error with a hint line, read the hint and
   iterate via update_skill on the same slug.


THE THREE KINDS
---------------
- `background` — produces a fresh image from scratch using
  `canvas.create_image()`. Always layer 0. Reading `canvas.image` from a
  background skill raises ValueError; use `canvas.create_image()` or
  `canvas.new(color=...)` instead.
- `filter` — reads the current canvas via `canvas.image`, returns a
  same-shape opaque image that replaces it. Requires a background first.
- `object` — paints onto a transparent base from `canvas.new_layer()`
  (or reads `canvas.image` for contrast decisions), commits RGBA, and
  the framework alpha-composites the result onto the prior canvas. Use
  for overlays — typography, badges, icons. Paint only what you want
  visible; leave the rest transparent. Requires a background first.


PALETTE DISCIPLINE — NON-NEGOTIABLE
-----------------------------------
Every color MUST come from `canvas.palette` slots
(`primary`, `secondary`, `tertiary`, `accent`, `background`) or
`art_kit.palette_color(t)`. Hardcoded hex strings or RGB tuples
(e.g. `(255, 80, 80)`, `"#ff5050"`) break palette swapping and are the
single most common bug.

- Wrong:  `draw.ellipse(box, fill=(255, 80, 80, 255))`
- Right:  `draw.ellipse(box, fill=canvas.palette.primary)`
- Right:  `draw.ellipse(box, fill=art_kit.palette_color(t))`

Map luminance to palette for tonal range — `palette_color(t)` interpolates
along the canvas's luminance-sorted ramp, turning any 0..1 gradient into
a palette-aware gradient.

- **Background controls mood.** Let `palette.background` dominate; reserve
  the brighter slots for the subject.
- **Accent sparingly.** `palette.accent` only on focal highlights, edges,
  or a ≤10% sprinkle. Big fields of accent look gaudy.

The post-render validator detects palette drift > 15% of pixels and
surfaces a warning. If you need a color genuinely outside the palette
(rare), ask the user first.


AUTO-DISCOVERY RULES
--------------------
- File must live in `plugins/skills/` (baked-in) or the sandbox skills dir.
- File name must start with `skill_`.
- Class must inherit from `BaseSkill`.
- Class must define `def run(self, canvas)`.
- One skill class per file.


CONTEXT THE SANDBOX INJECTS
---------------------------
Skills run in an isolated subprocess. Inside run(), you get:

  canvas.palette     SimpleNamespace with slots: primary, secondary,
                     tertiary, accent, background. Each slot is a hex
                     string ("#RRGGBB") that also behaves as an RGB tuple.
  canvas.palette.colors                       dict of all five slots.
  canvas.size / .width / .height              square dimension in pixels.
  canvas.seed                                 int — seed every RNG.
  canvas.image                                (filter/object only) copy of
                                              the current canvas image.
  canvas.image_array(mode="RGB", dtype="float")
                                              (filter/object only) current
                                              image as a numpy array.
                                              dtype="float" → float32 in
                                              [0,1]; dtype="uint8" → raw.
  canvas.new(color=...)                       fresh RGBA image at canvas
                                              size.
  canvas.create_image(color=...)              shorthand for
                                              new(color=palette.background).
                                              Backgrounds.
  canvas.new_layer()                          fully-transparent RGBA at
                                              canvas size. Objects.
  canvas.commit(image)                        **REQUIRED** — call once at
                                              the end with an RGBA image.
  canvas.commit_array(arr)                    same as commit, for numpy
                                              HxWxC arrays (float in [0,1]
                                              or uint8; C=3 or 4). Handles
                                              clip + dtype + RGBA convert.
                                              Prefer this in numpy-heavy
                                              filters.

  art_kit            Namespace of pre-bound helpers (no import needed).
                     Call read_skill_guide() to see the full index, or
                     read_skill_guide(methods=[...]) for specific helpers.


ALLOWED IMPORTS
---------------
Only these modules may be imported inside a skill:
  math, random, colorsys
  numpy (and numpy.random)
  PIL.Image, PIL.ImageDraw, PIL.ImageFilter, PIL.ImageOps,
  PIL.ImageEnhance, PIL.ImageChops, PIL.ImageColor
  plugins.BaseSkill  (literal — for `BaseSkill`, `Slider`, `Bool`, `Enum`,
                       `Pan`, `Palette`, `Text`)

Everything else (os, sys, subprocess, requests, plugins.helpers.*, scipy,
matplotlib, cv2, torch) is blocked at AST validation time. There is no
escape hatch. Reach for numpy if a math function is missing.


DETERMINISM RULES
-----------------
The palette swatch buttons re-render every image by replaying the skill
chain with a new palette. If your skill is non-deterministic, the user
sees a different image — bad.

- Seed every random source from canvas.seed:
    rng = random.Random(canvas.seed)
- For numpy:
    rng_np = numpy.random.default_rng(canvas.seed)
- Never read the wall clock. Never read environment variables. Never
  touch the filesystem. The sandbox blocks all of them anyway.
- All art_kit helpers are deterministic given the same seed.


CONTROLS
--------
A skill may expose up to **3 non-palette controls** plus an optional
`Palette()` for palette-aware skills. Controls render as widgets on the
canvas — sliders, toggles, arrow pads — and re-run the chain whenever the
user adjusts one. Used well, they turn near-miss completions into
successes without another agent turn.

Declare controls as class attributes:

  intensity = Slider(0.0, 1.0, default=0.5, step=0.01)
  mode      = Enum([("soft", "Soft"), ("crisp", "Crisp")], default="soft")
  wrap      = Bool(default=False)
  cx        = Slider(0.0, 1.0, default=0.5)
  cy        = Slider(0.0, 1.0, default=0.5)
  center    = Pan(x="cx", y="cy")
  palette   = Palette()

Read values as `self.intensity` / `self.mode` / etc. Slider values are
clamped before run() starts. `Pan` is only a UI grouping over two
sliders; read self.cx/self.cy, not self.center. `Palette()` declares a
layer-specific palette override and does not count toward the cap; only
use it on skills that actually read palette colors.

Do not define get_controls(), controls = [...], or plain defaults like
slot = "primary" for user controls; only descriptor instances assigned
directly on the class become controls.

Descriptor reference:

  Slider   numeric value;  Slider(0.1, 20.0, default=1.0, step=0.1)
  Enum     discrete value; Enum([("soft", "Soft")], default="soft")
  Bool     toggle;         Bool(default=False)
  Pan      UI arrow pad over two existing Sliders (x="cx", y="cy")
  Palette  per-layer palette override; Palette()
  Text     short string;   Text(default="NEW", max_length=12)

**Pick controls that generalize.** A `zoom` slider works for fractals,
tilings, and spirals — anything where a scale matters. A `density` slider
works for fields of dots, lines, or strokes. Reach for these abstract
dials before skill-specific ones. Prefer to include `Palette()` on
palette-aware background skills so users can explore color choices
freely.


COMPOSITION RULES
-----------------
- **Rule of thirds.** Place the focal point on one of the four
  `art_kit.rule_of_thirds(size).points`. Place horizons on a `horizons`
  line, not center.
- **Negative space.** Leave 30–50% of the canvas as `palette.background`
  for any subject-focused image. Density kills focus.
- **Focal contrast.** The brightest or most-saturated patch should be
  the subject.
- **Leading lines.** Curves or grids should aim toward the focal point.


ESTABLISHED METHODS — REACH FOR THESE FIRST
-------------------------------------------
When the user asks for a natural subject, pick a known method before
freehanding. The encyclopedia (loaded in the system prompt) has the
math; this is the short list of which method fits which subject:

- **Vogel / golden-angle spiral** → sunflowers, dandelion seeds, star
  fields. `art_kit.vogel_spiral(n)`. Vary point size with index for depth.
- **Voronoi tiling** → cellular tissues, basalt columns, cracked earth,
  stained glass. Seed with `jittered_grid`, then argmin over a numpy
  distance broadcast (the fast path).
- **Flow fields / fbm streamlines** → wind, water, hair, organic curves.
  Sample `art_kit.flow_field` for an angle at each point; step short
  particles; draw each path with a palette color.
- **L-systems / recursive branching** → trees, lightning, river deltas,
  coral. Use `lindenmayer` + `turtle_segments`. Depth 6–9; narrow stroke
  near the tips and shift the palette ramp toward `accent`.
- **Recursive subdivision** → Mondrian, low-poly mountains, fractured
  glass. Subdivide a rectangle along its long axis; stop at random depth.
- **Strange attractors (Clifford, de Jong)** → wispy ribbon abstracts.
  Iterate `attractor_points` for ~200k steps; accumulate into a density
  buffer; color-map by density.
- **Sediment bands / stratification** → landscapes, sunsets, geological
  sections. Stack horizontal bands with sine-modulated edges; use the
  palette luminance ramp top-to-bottom.
- **fbm fields** → clouds, terrain, nebulas, magma. `art_kit.fbm_grid`
  on a numpy lattice; map values through `palette_color`.

For suns / dawns / sunsets specifically: anchor the sun on a rule-of-thirds
intersection, gradient the sky between `palette.background` and
`palette.primary`, add a radial glow with `radial_falloff`, then
post-process with `palette_grade` + `bloom_glow`.


CHAINING STRATEGY
-----------------
A single background rarely produces a finished image. High-quality pattern:

  1. **Background skill** — establishes geometry and base palette.
  2. **palette_grade** — re-maps luminance to the palette for cohesion.
  3. One of **bloom_glow**, **vignette**, **film_grain**, **sharpen**
     — adds atmosphere or sharpens detail.

Keep filter chains ≤ 3 deep. The runtime enforces a hard cap of 4 total
chain entries (1 background + 3 filters or objects); past that,
execute_skill errors and the user must delete a layer first.


PERFORMANCE — STAY UNDER THE 30s TIMEOUT
----------------------------------------
Skills run in a subprocess with a hard 30-second wall-clock timeout and
a 768 MB memory cap. Per-pixel Python loops at 1024² always time out.

- Vectorize with numpy. Use `art_kit.centered_grid(size)` to build
  coordinate arrays once; `bilinear_sample` to warp.
- For 2D noise fields, prefer `art_kit.fbm_grid` (numpy-vectorized) over
  scalar `art_kit.fbm` in nested loops — ~30-80× faster on a 160² grid.
- For Voronoi, do the per-pixel argmin inline with numpy broadcasting,
  not via `voronoi_nearest` in a Python loop.
- For very dense work, sample on a coarse grid (e.g. 160² or 256²) and
  upscale with PIL `BICUBIC` — the eye can't tell, and it's an order of
  magnitude cheaper.


COMMON PITFALLS
---------------
- Forgetting `canvas.commit(image)` → the runtime errors. Always commit.
- Calling `canvas.image` in a background skill → ValueError. Use
  `canvas.create_image()` or `canvas.new(...)`.
- Using `random.random()` without seeding → palette replay produces a
  different image. Always go through `random.Random(canvas.seed)`.
- Drawing everything edge-to-edge → no focal point. Leave breathing room.
- Hex literals like `"#ff00aa"` → ignores the palette. Use slots or
  `palette_color`.
- Nested Python per-pixel loops at full resolution → 30s timeout.
"""

# =====================================================================
# BASE CLASS (copied from plugins/BaseSkill.py for self-containment)
# =====================================================================


class BaseSkill:
    """The contract every skill implements. See plugins/BaseSkill.py."""
    name: str = ""
    description: str = ""
    kind: str = "background"        # "background" | "filter" | "object"
    owner: str = "library"
    created_at: float = 0.0
    hidden: bool = False
    auto_register: bool = True
    requires_services: list[str] = []
    config_settings: list = []

    def run(self, canvas):
        raise NotImplementedError


# =====================================================================
# EXAMPLE — background skill (full file shape)
# =====================================================================
#
# from plugins.BaseSkill import BaseSkill, Slider, Palette
# import random
# from PIL import ImageDraw
#
#
# class SunflowerFieldSkill(BaseSkill):
#     name = "Sunflower Field"
#     description = "Vogel-spiral sunflower with palette-graded petals."
#     kind = "background"
#
#     palette = Palette()
#     count = Slider(200, 1800, default=900, step=50)
#     petal_size = Slider(0.006, 0.04, default=0.018, step=0.002)
#
#     def run(self, canvas):
#         rng = random.Random(canvas.seed)                  # seed
#         img = canvas.create_image()                       # palette.background fill
#         draw = ImageDraw.Draw(img, "RGBA")
#         s = canvas.size
#         count = int(self.count)
#         for i, (nx, ny) in enumerate(art_kit.vogel_spiral(count)):
#             x = s * (0.5 + nx * 0.45)
#             y = s * (0.5 + ny * 0.45)
#             t = i / max(1, count - 1)
#             color = art_kit.palette_color(t)
#             r = s * self.petal_size * (1 - 0.4 * t)
#             draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
#         canvas.commit(img)


# =====================================================================
# EXAMPLE — filter skill (numpy + warp)
# =====================================================================
#
# from plugins.BaseSkill import BaseSkill, Slider
#
#
# class BarrelWarpSkill(BaseSkill):
#     name = "Barrel Warp"
#     description = "Radial barrel distortion of the current canvas."
#     kind = "filter"
#     strength = Slider(-1.0, 1.0, default=0.6, step=0.05)
#
#     def run(self, canvas):
#         arr = canvas.image_array(mode="RGB", dtype="float")
#         xx, yy, nx, ny = art_kit.centered_grid(canvas.size)
#         r2 = nx * nx + ny * ny
#         scale = 1.0 + float(self.strength) * r2
#         cx = (canvas.size - 1) / 2.0
#         sx = cx + nx * scale * cx
#         sy = cx + ny * scale * cx
#         canvas.commit_array(art_kit.bilinear_sample(arr, sx, sy))


# =====================================================================
# EXAMPLE — object skill (overlay)
# =====================================================================
#
# from plugins.BaseSkill import BaseSkill, Text, Slider, Palette
#
#
# class CornerBadgeSkill(BaseSkill):
#     name = "Corner Badge"
#     description = "A small accent-colored label in the top-left."
#     kind = "object"
#
#     palette = Palette()
#     label = Text(default="NEW", max_length=12)
#     size_pct = Slider(4, 20, default=8, step=0.5)
#
#     def run(self, canvas):
#         img = canvas.new_layer()                          # transparent RGBA
#         s = canvas.size
#         pad = int(s * 0.04)
#         h = int(s * float(self.size_pct) / 100.0)
#         art_kit.text(
#             img, (pad, pad), str(self.label),
#             size=h, color=canvas.palette.accent, anchor="lt",
#         )
#         canvas.commit(img)                                # framework composites
