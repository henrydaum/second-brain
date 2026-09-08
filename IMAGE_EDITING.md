# Image editing suite

The bundle keeps the Art version's ordered recipe and prefix cache. The canvas
service owns state and undo/redo; scripts own pixels, and the existing sandbox
owns their execution. The bundle includes 53 classic editing techniques, their
shared helpers, and a searchable catalogue. No kernel changes are required.

## What changed from Art

| Area | Current contract |
| --- | --- |
| Dimensions | Independent positive integer width and height; no 16–8192 clamp or aspect presets. |
| Starting image | Transparent RGBA; an empty recipe renders a valid PNG. |
| Background | Optional step at index zero. Adding one replaces an existing background or inserts before other steps. |
| Objects | Produce overlays; the renderer composites them over the accumulated image. |
| Filters | Produce full replacements; opacity and masks interpolate premultiplied RGBA, including alpha. |
| Layers | Stable IDs, editable script/controls, name, visibility, opacity, blend mode, offset, mask and dependencies. |
| History | Deep snapshots; invalid edits and failed saves leave the published canvas and history unchanged. |
| Deletion | Removing index zero preserves subsequent steps. Clear is a separate action. |
| Session selection | Uses the SDK's actual `key`, user and conversation; selection persists across restarts. |
| Cache | Includes source/helper contents, dimensions, palette values, seed, compositing properties and declared file inputs. |
| Fonts | Portable regular default; custom fonts are explicitly read through the SDK. No dependency on the old application's fonts folder. |

## Start here: discover, edit, render

`search_techniques()` lists the suite. Search ordinary words with
`search_techniques(query="sharpness")`, or get the exact controls and a usable
example with `search_techniques(script="technique_sharpen")`. The catalogue returns
control types, defaults, bounds, units and suggested increments. Increments are
not quantization: saturation 1.125 is valid even though its suggested step is 0.05.
Numeric controls accept numeric strings such as `"25"` and `"3.5"`; integer
controls accept integer strings such as `"25"` without truncating fractions.
Boolean controls accept `true`/`false` strings, while booleans are never accepted
as numbers. Normalization also applies to array items, and saved controls use
the resulting typed values. Invalid values report the control and received input.
`search_techniques(recipe="photo")` and `recipe="composition"` return complete
worked sequences of tool calls without changing any canvas.

For an uploaded photo, use the actual local attachment path:

```python
add_layer(script="technique_load_image", controls={"path": "<actual attachment path>"})
render_canvas()
add_layer(script="technique_crop", controls={"left": 100, "top": 50, "right": 900, "bottom": 650})
add_layer(script="technique_saturation", controls={"factor": 1.1})
add_layer(script="technique_sharpen", controls={"radius": 1.5, "amount": 60, "threshold": 3})
render_canvas()
```

The crop coordinates above are an example for a sufficiently large image: read
the first render's dimensions and choose a useful rectangle. Import applies EXIF
orientation and preserves native pixels by default. It does not upload the photo
to any external service. `kind="object"` imports an additional image as an overlay.
Use `fit="contain"`, `"cover"` or `"stretch"` to fit that import to the current size.

A blank design starts with `manage_layers(action="create", width=800, height=500)`.
Then add `technique_solid` or `technique_gradient` and object steps such as `technique_text`.
`create` makes a new canvas; `inspect`, `list` and `select` let you resume existing work.

### The 53 techniques

| Script suffix (all use `technique_`) | Main adjustable controls |
| --- | --- |
| load_image | path, fit |
| crop | left, top, right, bottom (exclusive edges) |
| resize | width, height, fit |
| rotate | angle (positive counterclockwise), expand |
| flip | horizontal, vertical |
| brightness | factor (1 unchanged) |
| contrast | factor (1 unchanged) |
| saturation | factor (0 grayscale, 1 unchanged) |
| exposure | stops (0 unchanged; +1 doubles linear light) |
| gamma | gamma (1 unchanged; above 1 brightens midtones) |
| blur | radius in pixels |
| sharpen | radius, amount in percent, threshold |
| grayscale | no controls; mix using layer opacity |
| invert | no controls; mix using layer opacity |
| solid | color |
| gradient | start, end, angle |
| line | points, width, color |
| shape | shape (rectangle/ellipse), box, fill, stroke, stroke_width |
| text | content, x, y, size, color, font_path, max_width, align |
| duotone | shadows, highlights, amount |
| vignette | amount, radius |
| levels | black, white, gamma |
| color_balance | red, green, blue signed offsets |
| threshold | threshold, shadows, highlights |
| posterize | levels per channel |
| pixelate | block_size |
| median | radius |
| pad | width, height, x, y, color |

### Fine adjustments and history

Inspect first: `manage_layers(action="inspect")` exposes indices and stable IDs.
`manage_layers(action="controls", layer_id="...")` returns current values and their
specifications. Change one control with `set_control` and `name`/`value`, or patch
several with `set_controls` and `controls={...}`. These validate shipped controls
before editing and use one undo step. Use the existing layer rather than stacking
a second filter just to change its strength.

`update` changes compositing properties; its `controls` field replaces the entire
control dictionary (defaults fill omitted values). `duplicate`, `move`, `delete`,
`undo` and `redo` support iteration. `layer_id` targets a stable identity; otherwise
reinspect indices after reordering. `controls` lookup describes shipped techniques;
custom scripts still expose their current values through `inspect`.

### Geometry and order

Crop and resize change actual output dimensions. Expanded rotation and native
photo import can also change them. Each following step receives that current
image's dimensions, including when rendering resumes from a cached prefix.
Coordinates are pixels measured from its top-left corner. Masks must match the
input dimensions of the step where they are used. Dimension-changing steps need
full opacity and no mask, since differently sized images cannot be interpolated.

Canvas state width/height describe the **starting** empty surface; they do not
change when a crop is added. `render_canvas` reports final dimensions.
`set_dimensions` replays the recipe from a new starting surface; it does not
resample a native source photo. Use `technique_resize` to resize pixels.

Usually do geometry early, tonal edits next, sharpen near the end, and draw text
or annotations last so they remain crisp. Render and inspect before placing
objects after a crop or rotation. This is a recipe: a later filter affects every
visible step beneath it, including earlier text and objects.

### Palette behaviour

Photos are never forced into a palette. Brightness, saturation, exposure and
other ordinary adjustments preserve alpha and use the photo's own colours.
Literal colours (CSS names, hex RGB/RGBA, or `transparent`) stay fixed.
`@primary`, `@secondary`, `@tertiary`, `@accent`, `@background` are live references
for fills, gradients, lines, shapes, text and explicit duotone mapping.

`manage_layers(action="palettes")` lists presets. Set canvas-specific overrides:

```python
manage_layers(action="set_palette", colors={"primary": "#182844", "accent": "#ffd9a0"})
```

The supplied dictionary replaces prior overrides; `{}` restores preset values.
Changing preset ID also clears overrides. Palette changes are undoable and survive
restart. Duotone interpolates RGB between its two colours but preserves photo alpha;
alpha components of its shadow/highlight colours are ignored.

### Rendering and export

`render_canvas(seed=0)` is deterministic, including zero. Ordinary classic edits do
not use randomness. `force=True` bypasses the cache while keeping the seed; `out`
copies the result to an explicit PNG export path. Always render, inspect and refine
before considering an edit finished.

## Layer authoring contract

Scripts declare `box = "image_editing"` and
`dependencies_files = ["scripts/art_kit.py"]` when using that library, with
relative imports such as `from .art_kit import read_image, write_png`.
Use the SDK script directory and the existing script runner; do not create a
second worker pool or import kernel code.

The entry signature is:

```python
def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    ...
```

The script writes a PNG to `output_path` through `write_png(sdk, path, image)`.
Use `read_image(sdk, path)` to import images with alpha and EXIF orientation.
Treat `input_path` as read-only. Backgrounds receive `None`; filters and objects
receive the preceding flattened RGBA image, even when the recipe starts empty.

An object must output only its overlay. Returning the input with the object
already drawn would composite the input twice. Object images can be smaller
than the canvas; `offset: [x, y]` places them and clips outside the canvas.
Filters and backgrounds use normal blend mode and zero offset. They may change
dimensions only with full opacity and no mask. Later steps receive the new dimensions.

Layer properties:

- `visible`: boolean; hidden steps pass the accumulated image through.
- `opacity`: number from 0 to 1.
- `blend_mode`: `normal`, `multiply`, `screen`, `overlay`, `darken`, `lighten`, or `difference` (objects).
- `mask`: optional image path at canvas size. Luminance multiplied by alpha controls coverage; white reveals and black hides.
- `offset`: integer `[x, y]` (objects).
- `dependencies`: paths of every external file read by a **custom** script. Shipped techniques automatically derive image/font dependencies from their current controls; mask paths are always included. Contents are hashed, so replacing an image at the same path invalidates its cached import.

Scripts must be deterministic for their inputs. Relative Python helpers are
hashed recursively. Dynamic imports, network responses, environment settings
and third-party library versions are not automatically fingerprinted: pass their
relevant values as controls or force a render after they change. Cache files are
validated before reuse, and failed output is never published as a completed step.

## Reusable pixel operations

`art_kit` retains the old math/noise/composition utilities and adds:

- `read_image`, `write_png`, and `load_font`: SDK-mediated image/font I/O.
- `composite`: alpha-correct object blending and masked filter interpolation.
- `resize_image`: stretch, transparent contain, or cropped cover; premultiplied-alpha sampling.
- `crop_image`: arbitrary rectangular crop with transparent out-of-bounds pixels.
- `transform_image`: affine translation, scaling, rotation and skew with transparent fill.

`text` and `text_bbox` accept `font=load_font(sdk, path, size)` for custom faces
and styles. The portable fallback is regular only.

## Practical boundaries

This is an 8-bit RGBA raster recipe engine, not a vector document or a PSD
implementation. Blend calculations use encoded RGB values, not a managed linear
color workspace. Resolution has no service-level cap, but allocations, Pillow's
decoder safeguards and sandbox memory/time limits still apply. This is not a
tiled, out-of-core renderer. Render caches remain under workspace/canvas_renders;
they can be removed to reclaim disk space without deleting canvas recipes.

This is classic deterministic editing, not generative AI. Discovery is a small
local catalogue with keyword matching; no embedding service, indexing job or model
is needed. Tests cover every shipped script and composite workflows through the
actual sandbox, not just standalone pixel calls.

## Authoring and auditing techniques

Each `scripts/technique_*.py` is an audit unit: literal `TECHNIQUE` metadata,
control schemas and examples, the effect's `apply` function, and the sandbox
`main` entry point. Shared pixel, colour, compositing and IO utilities live in
`art_kit.py`. `canvas_catalog.py` discovers and validates declarations; it does
not implement effects or maintain a registration list.

Call `search_techniques(guide=True)` for the shipped template and workflow.
Copy it to `workspace/scripts/technique_your_name.py`, edit metadata and `apply`,
validate with `sdk.plugins.validate(path)`, then search for the new name, add it
as a layer and render. No central file edit or restart is required. New controls
are available through the existing inspection and fine-adjustment tools.

Discovery parses metadata with AST without importing or executing techniques.
It searches workspace, installed and bundled script directories in that order;
the first file of a given name wins. Search includes its origin and source path.
Invalid declarations are reported, including invalid workspace overrides, rather
than silently falling back. Renames, edits and deletions appear on the next lookup.
The renderer executes the resolved script through the existing sandbox runner.

The original 21 techniques declare their former `canvas_*` name as an alias so saved
recipes continue working. New recipes use `technique_*`. Ambiguous aliases are
rejected. The template has no aliases; add them only when deliberately renaming
a technique. Scripts without the prefix remain usable with an explicit layer
kind, but do not participate in technique discovery or control schemas.

## Render folders and additional editing steps

Renders use `workspace/canvas_renders/<recipe-hash>/<seed>.png`, matching Art's
seed-pool organization. Each intermediate recipe prefix gets its own folder;
identical inputs share cached results across canvases. Controls, file contents,
helpers and palette values affect the recipe hash; seed selects a PNG within it.
Old flat cache files are left untouched and may be deleted to reclaim space.
They are not reused by this renderer version; recipes themselves are unaffected.

Levels adjusts black/white points and midtones. Color balance corrects casts with
signed RGB offsets. Median removes speckles; use a small radius before sharpening.
Threshold and posterize create graphic colour reductions; threshold colours can
reference the palette. Pixelate uses block averaging with transparency preserved.
Padding extends or crops the canvas without resampling; place it early if later
layers should use its new coordinates. Like resize, pad needs full layer opacity
and no mask when it changes dimensions. Render and adjust existing layer controls.

## Live canvas context and selections

The canvas service uses `agent_prompt_refresh = "call"`: its prompt is rebuilt
before every LLM request, including repeated calls within the same turn. It
includes the selected canvas ID, starting size, seed, effective palette, ordered
layers with IDs and stored controls, mask/compositing settings, and undo/redo
availability. Reading it never creates or selects a canvas. Rendered dimensions
still come from the render result, since geometry steps can change the size.

`render_canvas` now returns `pool_hash` with `seed`. A successful render records
its editable recipe in the service database. `manage_layers(action="cached",
pool_hash=..., seed=...)` resolves the PNG, dimensions, and recipe.
`manage_layers(action="remix", pool_hash=..., seed=...)` creates and selects an
independent canvas from that snapshot. The source canvas is untouched. If cache
pixels are removed, the saved recipe remains remixable. External input files and
technique implementations are referenced, not archived: rerendering a remix uses
their current contents. Existing renders need to be rendered again once to record
a snapshot. These are local references, not publicly hosted URLs or QR links.

Selections are ordinary canvases: keep the photo's canvas ID, create a mask
canvas at the photo's *rendered* size, and use:

- `technique_mask_shape`: rectangle, ellipse, or polygon; feather and invert.
- `technique_mask_range`: import the photo, then select luminance or colour range.
- `technique_mask_combine`: union, intersection, subtraction, or replacement with
  another cached mask path. File contents participate in cache invalidation.

Render the selection and keep its PNG path (or pool hash and seed). Select the
photo canvas again, then set the target layer's `mask` property to that path via
`manage_layers(action="update", layer_id=..., properties={"mask": ...})`.
White reveals the edit, black hides it; masks have opaque grayscale output so
inversion works outside the original selection too. Sizes must match exactly.
These references are frozen PNG snapshots: editing the mask recipe produces a new
path, and the agent explicitly updates the target layer to use it. There are no
live cross-canvas dependencies, cycles, or additional tools.

## Photo corrections, geometry and effects batch

All names below use the `technique_` prefix. Use `search_techniques(script=...)`
for their full controls; each script owns its metadata and implementation.

| Technique | Controls and intended use |
| --- | --- |
| curves | Piecewise-linear normalized points; RGB or individual channel. Points span x=0 to x=1. |
| white_balance | Relative temperature and tint using linear-light gains, not absolute Kelvin. |
| shadows_highlights | Signed corrections for dark and bright regions; zero is unchanged. |
| vibrance | Saturation adjustment weighted toward muted colours; no skin detection. |
| affine | Translation, scale, counterclockwise rotation, shear; fixed output canvas. |
| perspective | Four normalized source corners in perimeter order; optional output width/height. |
| drop_shadow | Alpha silhouette, offset, blur, colour and opacity; clips to canvas bounds. |
| outline | Outer alpha outline using a square neighbourhood, colour and opacity. |
| color_key | RGB target, tolerance, soft transition and strength; removes colour to transparency. |
| dither | Deterministic two-colour Bayer pattern with adjustable strength. |
| halftone | Luminance-controlled dot grid, cell size, ink and paper colours. |
| displace | Same-size PNG map; red/green steer x/y sampling, 128 is neutral, alpha scales strength. |
| texture | Seeded fractal value noise with scale, octaves, contrast and two palette-aware colours. |

For a photo, start with geometry, then white balance, curves and tonal corrections;
sharpen near the end and put annotations last. Straightening already uses
`technique_rotate` with a small angle; expanded rotation followed by crop retains
control of the output bounds. Affine preserves the current size. Perspective can
change it and therefore requires full opacity and no mask when doing so.

For a cutout, use `color_key` only for a suitable flat-colour background. It is
classic chroma key, not subject recognition. Pad with transparency before adding
an outline or shadow, so there is room around the alpha silhouette. A fully opaque
photo has no internal alpha boundary for these effects to follow.

For texture-driven displacement, render a separate texture canvas at the target
image's rendered dimensions, using `low='#000000'` and `high='#ffffff'` for a
neutral grayscale map. Reselect the photo and supply that cached PNG as the
displacement `path`. Maps are snapshots with automatic file fingerprinting, not
live dependencies. Colour, alpha, interpolation and sampling primitives remain
in `art_kit`; no extra tools or kernel changes are needed.

## Glitch and psychedelic effects

`search_techniques(recipe="glitch")` gives a photo-based recipe;
`recipe="trippy"` starts with a procedural texture. Both batch their edits before
rendering. Keep the render seed fixed while adjusting existing layer controls.

| Technique (all use `technique_`) | Controls that avoid guessed pixel positions |
| --- | --- |
| chromatic_aberration | Separation as a fraction of the shorter image side; radial or horizontal. |
| fisheye | Bulge/pinch strength and relative radius, centred by default. |
| feedback_tunnel | Finite recursive copies: depth, scale, twist and opacity. |
| ascii | Character columns, glyph ramp, sampled colour or palette foreground. |
| scanlines | Line count across the image, strength, band fraction and phase. |
| glitch_slice | Seeded bands; count, fractional height and fractional horizontal shift. |
| pixel_sort | Horizontal/vertical runs selected by luminance range; ascending/descending. |
| kaleidoscope | Repeated angular wedges, zoom, angle and normalized centre. |
| swirl | Turns and relative radius, with smooth falloff around a normalized centre. |

For radial effects, centre coordinates range from 0 to 1; 0.5 is the image centre.
Radius uses half the shorter image side, so circular effects remain round on a
portrait or panorama. These are geometric controls, not feature detection. Do
not guess eye positions or freehand natural objects: prefer formulas, symmetry,
grids, seeded patterns, and imported source photos for natural-looking elements.

### Filter order, masks, opacity, and previews

A filter reads **only the accumulated earlier layers** and returns a full image
replacement at that step. Later objects cannot affect its luminance calculations.
Put halftone before text to retain clean text; put it after text to halftone both.
Halftone `paper` is the light output colour between dots, not a region selector.
Choose `@background` to match the palette background if desired.

Any layer can use `properties={"opacity":0.55,"mask":"<same-size mask PNG>"}`.
These are layer properties, not entries in a technique's `controls`. Change them
with `manage_layers(action="update", layer_id=..., properties=...)`. Numeric
strings for opacity are accepted; booleans, nonfinite and out-of-range values are
rejected with the received value. Dimension-changing steps still need full
opacity and no mask. Use a mask to halftone a maze while preserving an orb, or put
the orb in a later object layer.

Add/manage/inspect do not render or attach images. Only `render_canvas` attaches
the PNG. It reports `attachment_path` and `image_sha256` for the exact encoded
bytes; exports use those same bytes. Even when an export filename is reused,
the attachment points to the recipe/seed cache path, keeping revisions distinct.
The pool hash identifies recipe inputs; `image_sha256` identifies the PNG bytes.
