# Canvas skill authoring guide

You are building one skill: a small, deterministic Python function that either
**creates** a new image on the canvas or **transforms** the current one. This
guide is the single source of truth for how to do that well.

Before authoring a new skill from scratch, **search_skills first** — the
built-in library already has high-quality references for common subjects.
Clone-and-adjust beats freehand every time.

---

## 1. Skill file template

Every skill is one Python file with five metadata constants followed by a
`run(canvas, **params)` function:

```python
SKILL_NAME = "Sunflower Field"
SKILL_DESCRIPTION = "Vogel-spiral sunflower seed pattern with palette-graded petals."
SKILL_KIND = "creation"   # or "transform"
SKILL_OWNER = "library"   # set automatically when the agent creates a skill
SKILL_CREATED_AT = 0.0    # set automatically

import math, random
from PIL import Image, ImageDraw

def run(canvas, count=900, petal_size=0.018):
    rng = random.Random(canvas.seed)              # seed every random source
    img = canvas.create_image()                   # palette.background fill
    draw = ImageDraw.Draw(img, "RGBA")
    s = canvas.size
    for i, (nx, ny) in enumerate(art_kit.vogel_spiral(int(count))):
        x = s * (0.5 + nx * 0.45)
        y = s * (0.5 + ny * 0.45)
        t = i / max(1, int(count) - 1)
        color = art_kit.palette_color(t)
        r = s * float(petal_size) * (1 - 0.4 * t)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    canvas.commit(img)
```

The agent fills `SKILL_OWNER` and `SKILL_CREATED_AT` via `create_skill`; you
don't write them yourself.

---

## 2. Canvas + art_kit reference

`canvas` is injected by the runtime:

| Attribute / method        | Purpose                                                 |
|---------------------------|---------------------------------------------------------|
| `canvas.palette.primary` (also `secondary`, `tertiary`, `accent`, `background`) | Hex strings; also unpack as RGB tuples. |
| `canvas.palette.colors`   | Dict of all five slots.                                 |
| `canvas.size` / `.width` / `.height` | Square dimension in pixels.                 |
| `canvas.seed`             | Integer; seed every RNG with this.                      |
| `canvas.image`            | (transform only) A copy of the current canvas image.    |
| `canvas.new(color=...)`   | Returns a fresh RGBA image at canvas size.              |
| `canvas.create_image()`   | Shorthand for `new(color=palette.background)`.          |
| `canvas.commit(image)`    | **Required.** Hands the finished image to the runtime.  |

`art_kit` is injected too (no import needed):

| Helper                              | Returns                                              |
|-------------------------------------|------------------------------------------------------|
| `art_kit.lerp(a, b, t)`             | Linear interpolation.                                |
| `art_kit.smoothstep(t)`             | Smooth Hermite ramp in [0,1].                        |
| `art_kit.clamp(x, lo, hi)`          |                                                      |
| `art_kit.remap(x, a, b, c, d)`      | Re-range x.                                          |
| `art_kit.hex_to_rgb(h)` / `rgb_to_hex(rgb)` / `mix_hex(a, b, t)` | Color math. |
| `art_kit.palette_color(t)`          | Sample the palette luminance ramp at t∈[0,1]. Pre-bound to this canvas. |
| `art_kit.oklch_to_rgb(l, c, h)`     | OKLch → sRGB. Hue in turns (0..1).                   |
| `art_kit.rule_of_thirds(size)`      | `.points`, `.verticals`, `.horizons` for composition.|
| `art_kit.vogel_spiral(n)`           | n points distributed on a disc via the golden angle. |
| `art_kit.jittered_grid(rng, c, r)`  | Cell-centered grid in [0,1]² with jitter.            |
| `art_kit.value_noise(seed, x, y)`   | Smooth 2D value noise in [0,1].                      |
| `art_kit.fbm(seed, x, y, octaves)`  | Fractal Brownian motion over value_noise.            |
| `art_kit.radial_falloff(w, h)`      | Closure: 1 at center → 0 at corner.                  |

Allowed imports (the sandbox blocks everything else): `math`, `random`,
`colorsys`, `numpy`, `numpy.random`, `PIL.Image`, `PIL.ImageDraw`,
`PIL.ImageFilter`, `PIL.ImageOps`, `PIL.ImageEnhance`, `PIL.ImageChops`,
`PIL.ImageColor`.

---

## 3. Established generative methods

When the user asks for a natural subject, reach for a known method before
freehanding:

- **Vogel / golden-angle spiral** — sunflowers, daisies, dandelion seeds, star
  fields. Use `art_kit.vogel_spiral(n)`. Vary petal size with index for depth.
- **Voronoi tiling** — cellular tissues, basalt columns, cracked-earth,
  stained glass. Generate seed points on a `jittered_grid`, assign each pixel
  to the nearest seed, fill with palette ramp by cell index.
- **Flow fields / fbm streamlines** — wind, water, hair, organic curves.
  Sample `art_kit.fbm` for a vector angle at each point, step short particles
  along the field, draw each path with a palette color.
- **L-systems / recursive branching** — trees, lightning, river deltas, coral.
  Recurse depth ~6–9, narrow stroke width and shift palette ramp toward
  `accent` near the tips.
- **Recursive subdivision** — mondrian, low-poly mountains, fractured glass.
  Subdivide a rectangle along its long axis, stop at a random depth, fill.
- **Strange attractors (Clifford, de Jong)** — wispy, ribbon-like abstracts.
  Iterate a 2-param map for ~200k steps with a deterministic RNG, accumulate
  into a density buffer, color-map by density.
- **Sediment bands / stratification** — landscapes, sunsets, geological
  cross-sections. Stack horizontal bands with sine-modulated edges; use the
  palette luminance ramp top-to-bottom.

For suns/dawns/sunsets specifically: anchor the sun on a rule-of-thirds
intersection, gradient the sky between `palette.background` and `palette.primary`,
add a radial glow with `radial_falloff`, then post-process with `palette_grade`
+ `bloom_glow`.

---

## 4. Composition rules

- **Rule of thirds.** Place the focal point on one of the four
  `art_kit.rule_of_thirds(size).points`. Place horizons on a `horizons`
  line, not center.
- **Negative space.** Leave 30–50% of the canvas as `palette.background` for
  any subject-focused image. Density kills focus.
- **Focal contrast.** The brightest or most-saturated patch should be the
  subject. Reserve `palette.accent` for ≤10% of pixels — it's seasoning, not
  a base color.
- **Leading lines.** Curves or grids should aim toward the focal point.

---

## 5. Palette discipline

- **Never hardcode hex** unless the user explicitly asks. Pull every fill and
  stroke from `canvas.palette` slots or `art_kit.palette_color(t)`.
- **Map luminance to palette** for tonal range. `palette_color(t)` interpolates
  along the palette's luminance-sorted ramp; a normal gradient becomes a
  palette-aware gradient.
- **Background controls mood.** Don't fight `palette.background` — let it
  dominate, and reserve the brighter slots for the subject.
- **Accent sparingly.** `palette.accent` only on focal highlights, edges, or a
  ≤10% sprinkle. Big fields of accent look gaudy.

---

## 6. Determinism rules

The palette swatch buttons re-render every image by replaying the skill chain
with a new palette. If your skill is non-deterministic, the user sees a
different image — bad.

- Seed every random source from `canvas.seed`: `rng = random.Random(canvas.seed)`.
- For numpy: `rng_np = numpy.random.default_rng(canvas.seed)`.
- Never read the wall clock, never read environment variables, never touch the
  filesystem.
- All `art_kit` helpers are deterministic given the same seed.

---

## 7. When to chain skills

A single creation rarely produces a finished image. The high-quality pattern is:

1. **Creation skill** — establishes geometry and base palette.
2. **`palette_grade`** — re-maps luminance to the palette for a cohesive feel.
3. **One of `bloom_glow`, `vignette`, `film_grain`, or `sharpen`** — adds
   atmosphere or finishes detail.

Keep transform chains ≤3 deep so palette re-render stays snappy.

---

## 8. Declaring user-facing controls (optional but encouraged)

A skill may expose up to **3 non-palette controls** plus an optional palette
control to the user. These render as widgets on the canvas — sliders, toggles,
arrow pads, etc. — and re-run the skill chain whenever the user adjusts one.
Used well, controls turn near-miss completions into successes without another
agent turn.

Pass `controls` to `create_skill`. Every control name (except `palette`) must
correspond to a keyword parameter of your `run(canvas, ...)` function — or be
`seed` (which lives on the chain entry).

```python
controls = [
    {"type": "slider", "name": "zoom", "label": "Zoom",
     "min": 0.1, "max": 20.0, "step": 0.1, "default": 1.0},
    {"type": "pan", "name": "center", "label": "Pan",
     "x_param": "cx", "y_param": "cy", "step": 0.1,
     "x_default": -0.5, "y_default": 0.0},
    {"type": "palette"},
]
```

Control types — keep things general-purpose; one widget should map cleanly to
one knob the user wants to turn:

| type     | Schema                                                          | Notes |
|----------|-----------------------------------------------------------------|-------|
| `slider` | `name, label, min, max, step, default` (numeric)                | Sets one numeric run param. |
| `enum`   | `name, label, options:[{value,label}], default`                 | Sets one param to a discrete value. |
| `bool`   | `name, label, default`                                          | Sets one boolean run param. |
| `pan`    | `name, label, x_param, y_param, step, x_default, y_default`     | Two-axis arrow pad; updates both params together. |
| `button` | `name, label, param, action:"randomize"`                        | One-shot action; the only action today is `randomize` (rolls a new value for `param`, usually `seed`). |
| `palette`| no extras                                                       | Lets the user swap the canvas palette for this entry. Doesn't count toward the cap. |

**Pick controls that generalize.** A `zoom` slider works for fractals, tilings,
spirals — anything where a scale matters. A `density` slider works for fields
of dots, lines, or strokes. Reach for these abstract dials before you reach
for skill-specific ones.

Prefer to include a `palette` control on creation skills — it lets users
explore color choices freely instead of being locked into whatever palette the
agent picked.
## 9. Common pitfalls

- Forgetting `canvas.commit(image)` → the runtime errors. Always commit.
- Calling `canvas.image` in a creation skill → raises ValueError. Use
  `canvas.create_image()` or `canvas.new(...)` instead.
- Using `random.random()` without seeding → palette replay produces a
  different image. Always go through a seeded `random.Random(canvas.seed)`.
- Drawing everything edge-to-edge → no focal point. Leave breathing room.
- Hex literals like `"#ff00aa"` → ignores the palette. Use slots.
