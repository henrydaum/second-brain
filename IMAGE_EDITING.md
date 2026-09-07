# Image editing suite

The bundle keeps the Art version's ordered recipe and prefix cache. The canvas
service owns state and undo/redo; scripts own pixels, and the existing sandbox
owns their execution. The bundle includes 21 classic editing techniques, their
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
example with `search_techniques(script="canvas_sharpen")`. The catalogue returns
control types, defaults, bounds, units and suggested increments. Increments are
not quantization: saturation 1.125 is valid even though its suggested step is 0.05.
`search_techniques(recipe="photo")` and `recipe="composition"` return complete
worked sequences of tool calls without changing any canvas.

For an uploaded photo, use the actual local attachment path:

```python
add_layer(script="canvas_load_image", controls={"path": "<actual attachment path>"})
render_canvas()
add_layer(script="canvas_crop", controls={"left": 100, "top": 50, "right": 900, "bottom": 650})
add_layer(script="canvas_saturation", controls={"factor": 1.1})
add_layer(script="canvas_sharpen", controls={"radius": 1.5, "amount": 60, "threshold": 3})
render_canvas()
```

The crop coordinates above are an example for a sufficiently large image: read
the first render's dimensions and choose a useful rectangle. Import applies EXIF
orientation and preserves native pixels by default. It does not upload the photo
to any external service. `kind="object"` imports an additional image as an overlay.
Use `fit="contain"`, `"cover"` or `"stretch"` to fit that import to the current size.

A blank design starts with `manage_layers(action="create", width=800, height=500)`.
Then add `canvas_solid` or `canvas_gradient` and object steps such as `canvas_text`.
`create` makes a new canvas; `inspect`, `list` and `select` let you resume existing work.

### The 21 techniques

| Script suffix (all use `canvas_`) | Main adjustable controls |
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
resample a native source photo. Use `canvas_resize` to resize pixels.

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
