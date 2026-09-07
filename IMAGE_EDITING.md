# Image editing foundations

The bundle keeps the Art version's ordered recipe and prefix cache. The canvas
service owns state and undo/redo; scripts own pixels, and the existing sandbox
owns their execution. No kernel changes or technique catalogue are required.

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

## Agent workflow

1. `manage_layers(action="create", width=1600, height=900)` creates and selects a canvas.
2. `add_layer(script="my_layer", kind="object", controls={...}, properties={...})` adds a step.
3. `manage_layers(action="inspect")` exposes indices, stable IDs and current controls.
4. `manage_layers(action="update", chain_index=0, properties={"opacity": 0.5})` edits one step.
5. `render_canvas()` renders or reuses the longest matching prefix.

Use `set_control` to patch one control; `update` with `controls` replaces the
whole control dictionary. `duplicate`, `move`, `delete`, `undo` and `redo` work on
indices. Reinspect after reordering. `list` and `select` support multiple canvases.
`set_dimensions` replays the recipe at the new size; pixel resampling and cropping
belong in scripts, using the helpers below.

`render_canvas(seed=0)` is deterministic, including zero. `force_new_seed=True`
chooses another seed. `force=True` bypasses all cached steps while keeping the
seed. `out` copies the result to an explicit PNG export path.

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
Filters and backgrounds must return the exact canvas dimensions and use normal
blend mode with zero offset.

Layer properties:

- `visible`: boolean; hidden steps pass the accumulated image through.
- `opacity`: number from 0 to 1.
- `blend_mode`: `normal`, `multiply`, `screen`, `overlay`, `darken`, `lighten`, or `difference` (objects).
- `mask`: optional image path at canvas size. Luminance multiplied by alpha controls coverage; white reveals and black hides.
- `offset`: integer `[x, y]` (objects).
- `dependencies`: paths of **every external file read by the script**, including source images and fonts. Contents are hashed; mask paths are included automatically.

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

No production techniques ship in this change. Tests create minimal scripts in
temporary directories, including actual nested sandbox execution.
