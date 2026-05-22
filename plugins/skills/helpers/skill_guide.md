# Canvas skill authoring guide

You are building one skill: a small, deterministic Python function that either
paints a **background**, applies an **filter** to the current canvas, or
overlays an **object** on top of it. This guide is the single source of
truth for how to do that well.

The three kinds:

- `background` — produces a fresh image from scratch using
  `canvas.create_image()`. Always layer 0.
- `filter` — reads the current canvas via `canvas.image`, returns a
  same-shape opaque image that replaces it. Requires a background first.
- `object` — paints onto a transparent base from `canvas.new_layer()`
  (or onto `canvas.image` if you want to read the prior pixels), commits
  RGBA, and the framework alpha-composites the result onto the prior
  canvas. Use for overlays — typography, badges, icons. Paint only what
  you want visible; leave the rest fully transparent. Requires a
  background first.

## Palette: non-negotiable

Every color in your skill MUST come from `canvas.palette` slots
(`primary`, `secondary`, `tertiary`, `accent`, `background`) or
`art_kit.palette_color(t)`. **Never hardcode hex strings or RGB tuples**
(e.g. `(255, 80, 80)`, `"#ff5050"`) unless the user explicitly asks for a
named color. Hardcoded colors break palette swapping and ignore the user's
chosen palette — they are the single most common bug.

- Wrong: `draw.ellipse(box, fill=(255, 80, 80, 255))`
- Right: `draw.ellipse(box, fill=canvas.palette.primary)`
- Right: `draw.ellipse(box, fill=art_kit.palette_color(t))`

Reserve `palette.accent` for ≤10% of pixels. Let `palette.background` set
the mood.

Before authoring a new skill from scratch, **search_skills first** — the
built-in library already has high-quality references for common subjects.
Clone-and-adjust beats freehand every time.

---

## 1. Skill file template

Skills use descriptor controls on the `BaseSkill` class.
Each parameter is declared once, then read as `self.<name>` inside `run()`.
Slider values are defaulted and clamped by the runtime before `run()` starts.

```python
import random
from PIL import ImageDraw
from plugins.BaseSkill import BaseSkill, Slider, Palette


class SunflowerFieldSkill(BaseSkill):
    name = "Sunflower Field"
    description = "Vogel-spiral sunflower seed pattern with palette-graded petals."
    kind = "background"

    palette = Palette()
    count = Slider(200, 1800, default=900, step=50)
    petal_size = Slider(0.006, 0.04, default=0.018, step=0.002)

    def run(self, canvas):
        rng = random.Random(canvas.seed)              # seed every random source
        img = canvas.create_image()                   # palette.background fill
        draw = ImageDraw.Draw(img, "RGBA")
        s = canvas.size
        count = int(self.count)
        for i, (nx, ny) in enumerate(art_kit.vogel_spiral(count)):
            x = s * (0.5 + nx * 0.45)
            y = s * (0.5 + ny * 0.45)
            t = i / max(1, count - 1)
            color = art_kit.palette_color(t)
            r = s * self.petal_size * (1 - 0.4 * t)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        canvas.commit(img)
```

### filter skill template (numpy + warp)

For lens / warp / glitch filters, build a coordinate map and resample. The
`centered_grid`, `bilinear_sample`, `image_array`, and `commit_array` helpers
remove almost all of the boilerplate:

```python
from plugins.BaseSkill import BaseSkill, Slider


class BarrelWarpSkill(BaseSkill):
    name = "Barrel Warp"
    description = "A radial barrel distortion."
    kind = "filter"
    strength = Slider(-1.0, 1.0, default=0.6, step=0.05)

    def run(self, canvas):
        strength = self.strength
        arr = canvas.image_array(mode="RGB", dtype="float")   # float32 in [0,1]
        xx, yy, nx, ny = art_kit.centered_grid(canvas.size)   # pixel + normalized coords
        r2 = nx * nx + ny * ny
        scale = 1.0 + strength * r2
        cx = (canvas.size - 1) / 2.0
        sx = cx + nx * scale * cx
        sy = cx + ny * scale * cx
        canvas.commit_array(art_kit.bilinear_sample(arr, sx, sy))
```

For PIL-only filters (blur, solarize, enhance), use `canvas.image` →
filter → `canvas.commit(...)` as usual.

### Object skill template (overlay)

Object skills paint onto a transparent base; the framework composites
your output onto the prior canvas. Only the pixels you paint show up,
so don't bother filling the rest.

```python
from plugins.BaseSkill import BaseSkill, Text, Slider, Palette


class CornerBadgeSkill(BaseSkill):
    name = "Corner Badge"
    description = "A small accent-colored chip in the top-left corner."
    kind = "object"

    palette = Palette()
    label = Text(default="NEW", max_length=12)
    size_pct = Slider(4, 20, default=8, step=0.5)

    def run(self, canvas):
        img = canvas.new_layer()                 # fully transparent RGBA
        s = canvas.size
        pad = int(s * 0.04)
        h = int(s * float(self.size_pct) / 100.0)
        art_kit.text(
            img, (pad, pad), str(self.label),
            size=h, color=canvas.palette.accent, anchor="lt",
        )
        canvas.commit(img)                       # framework composites
```

If you need to read the underlying canvas (e.g. to pick contrast for the
overlay), `canvas.image` is also available to object skills.

---

## 2. Canvas + art_kit reference

`canvas` is injected by the runtime:

| Attribute / method        | Purpose                                                 |
|---------------------------|---------------------------------------------------------|
| `canvas.palette.primary` (also `secondary`, `tertiary`, `accent`, `background`) | Hex strings; also unpack as RGB tuples. |
| `canvas.palette.colors`   | Dict of all five slots.                                 |
| `canvas.size` / `.width` / `.height` | Square dimension in pixels.                 |
| `canvas.seed`             | Integer; seed every RNG with this.                      |
| `canvas.image`            | (filter/object only) A copy of the current canvas image. |
| `canvas.image_array(mode="RGB", dtype="float")` | (filter/object only) The current image as a numpy array. `dtype="float"` → float32 in [0,1]; `dtype="uint8"` → raw bytes. Saves the asarray/divide step. |
| `canvas.new(color=...)`   | Returns a fresh RGBA image at canvas size.              |
| `canvas.create_image()`   | Shorthand for `new(color=palette.background)`. Creations. |
| `canvas.new_layer()`      | Fully-transparent RGBA at canvas size. Objects.         |
| `canvas.commit(image)`    | **Required.** Hands the finished PIL image to the runtime. |
| `canvas.commit_array(arr)`| Same as `commit`, but accepts a numpy HxWxC array (float in [0,1] or uint8; C=3 or 4). Handles clip + dtype + Image.fromarray + RGBA convert for you. Prefer this in numpy-heavy filters. |

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
| `art_kit.value_noise(seed, x, y)`   | Smooth 2D value noise in [0,1]. Scalar — fine for sparse sampling. |
| `art_kit.fbm(seed, x, y, octaves)`  | Fractal Brownian motion over value_noise. Scalar — fine for sparse sampling. |
| `art_kit.value_noise_grid(seed, xx, yy)` | **Vectorized** value noise on numpy arrays. Use when filling a whole lattice (returns same-shape float64). Different hash than scalar, so outputs at the same seed differ. |
| `art_kit.fbm_grid(seed, xx, yy, octaves)` | **Vectorized** fbm on numpy arrays. Drop-in replacement for nested Python loops over `fbm` — ~30-80× faster on a 160² grid. Different hash than scalar `fbm`. |
| `art_kit.radial_falloff(w, h)`      | Closure: 1 at center → 0 at corner.                  |
| `art_kit.centered_grid(size)`       | `(xx, yy, nx, ny)` — pixel coords + normalized [-1,+1] coords. The standard opener for any radial / warp filter. |
| `art_kit.bilinear_sample(arr, fx, fy)` | Bilinear resample at fractional coords. `arr` is 2D (H,W) or 3D (H,W,C); `fx/fy` are float arrays. Coords outside the array clamp to the edge. |

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

A single background rarely produces a finished image. The high-quality pattern is:

1. **Background skill** — establishes geometry and base palette.
2. **`palette_grade`** — re-maps luminance to the palette for a cohesive feel.
3. **One of `bloom_glow`, `vignette`, `film_grain`, or `sharpen`** — adds
   atmosphere or finishes detail.

Keep filter chains ≤3 deep so palette re-render stays snappy. The runtime
enforces a hard cap of 4 total chain entries (1 background + 3 filters or
objects); past that, `execute_skill` errors and the user must delete a layer
first.

---

## 8. Declaring user-facing controls (optional but encouraged)

A skill may expose up to **3 non-palette controls** plus an optional palette
control when the skill actually uses `canvas.palette` or `art_kit.palette_color`.
These render as widgets on the canvas — sliders, toggles, arrow pads, etc. —
and re-run the skill chain whenever the user adjusts one.
Used well, controls turn near-miss completions into successes without another
agent turn.

Declare controls as class attributes:

```python
from plugins.BaseSkill import BaseSkill, Slider, Bool, Enum, Pan, Palette


class ExampleSkill(BaseSkill):
    name = "Example"
    description = "A controllable example."
    kind = "filter"

    palette = Palette()
    zoom = Slider(0.1, 20.0, default=1.0, step=0.1)
    wrap = Bool(default=False)
    mode = Enum([("soft", "Soft"), ("crisp", "Crisp")], default="soft")
    cx = Slider(0.0, 1.0, default=0.5, step=0.05)
    cy = Slider(0.0, 1.0, default=0.5, step=0.05)
    center = Pan(x="cx", y="cy")

    def run(self, canvas):
        # self.zoom/self.wrap/self.mode/self.cx/self.cy are ready to use.
        ...
```

`Pan` is only a UI grouping over two slider values; read `self.cx` and
`self.cy`, not `self.center`. `Palette` declares a layer-specific palette
override and should only be used by skills that actually read palette colors.
Do not define `get_controls()`, `controls = [...]`, or plain defaults like
`slot = "primary"` for user controls; the runtime only scans descriptor
instances assigned directly on the class.

Descriptor controls — keep things general-purpose; one widget should map
cleanly to one knob the user wants to turn:

| descriptor | Example | Notes |
|------------|---------|-------|
| `Slider` | `zoom = Slider(0.1, 20.0, default=1.0, step=0.1)` | Numeric value; auto-clamped. |
| `Enum` | `mode = Enum([("soft", "Soft")], default="soft")` | Discrete string/value choice. |
| `Bool` | `wrap = Bool(default=False)` | Boolean toggle. |
| `Pan` | `center = Pan(x="cx", y="cy")` | UI arrow pad over two Slider values. |
| `Palette` | `palette = Palette()` | Layer-specific palette override; does not count toward the cap. |

**Pick controls that generalize.** A `zoom` slider works for fractals, tilings,
spirals — anything where a scale matters. A `density` slider works for fields
of dots, lines, or strokes. Reach for these abstract dials before you reach
for skill-specific ones.

Prefer to include `Palette()` on palette-aware background skills — it lets users
explore color choices freely instead of being locked into whatever palette the
agent picked.

## 9. Common pitfalls

- Forgetting `canvas.commit(image)` → the runtime errors. Always commit.
- Calling `canvas.image` in a background skill → raises ValueError. Use
  `canvas.create_image()` or `canvas.new(...)` instead.
- Using `random.random()` without seeding → palette replay produces a
  different image. Always go through a seeded `random.Random(canvas.seed)`.
- Drawing everything edge-to-edge → no focal point. Leave breathing room.
- Hex literals like `"#ff00aa"` → ignores the palette. Use slots.
