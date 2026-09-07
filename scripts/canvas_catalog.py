"""Classic editing catalogue: discoverable controls shared by tools and scripts.

No model, embeddings or pixel imports. Numeric steps are suggested adjustment
increments, not quantization: callers may use any finite value in the range.
"""
from copy import deepcopy
import math
import re

box = "image_editing"


def number(default, minimum, maximum, step, description, unit=""):
    return dict(type="number", default=default, minimum=minimum, maximum=maximum,
                step=step, description=description, unit=unit)


def integer(default, minimum, description, maximum=None, unit="px"):
    field = dict(type="integer", step=1, description=description, unit=unit)
    if minimum is not None:
        field["minimum"] = minimum
    if default is not None:
        field["default"] = default
    if maximum is not None:
        field["maximum"] = maximum
    return field


def choice(default, options, description):
    return dict(type="string", default=default, enum=options, description=description)


def color(default, description):
    return dict(type="string", format="color", default=default, description=description)


def entry(title, kind, description, controls, example, tags="", kinds=None):
    return dict(title=title, kind=kind, kinds=kinds or [kind], description=description,
                controls=controls, example=example, tags=tags)


CATALOG = {
    "canvas_load_image": entry("Load photo or image", "background",
        "Read an attachment/local path with EXIF orientation and alpha. Native preserves original pixels and dimensions. Use object to overlay a second image.",
        {"path": {"type": "string", "format": "file", "description": "Existing attachment/local image path, not a URL."},
         "fit": choice("native", ["native", "contain", "cover", "stretch"], "Native keeps source size; others fit current canvas dimensions.")},
        {"path": "<attachment-path>", "fit": "native"}, "upload import open picture photograph overlay", ["background", "object"]),
    "canvas_crop": entry("Crop", "filter",
        "Extract a pixel rectangle. Right/bottom are exclusive. Output dimensions become right-left by bottom-top; outside-source pixels are transparent.",
        {"left": integer(0, None, "Left edge."), "top": integer(0, None, "Top edge."),
         "right": integer(None, None, "Exclusive right edge."), "bottom": integer(None, None, "Exclusive bottom edge.")},
        {"left": 100, "top": 50, "right": 900, "bottom": 650}, "trim cut reframe geometry"),
    "canvas_resize": entry("Resize image", "filter",
        "Resample the accumulated image to exact dimensions with alpha-safe Lanczos sampling. Geometry steps change subsequent layer coordinates.",
        {"width": integer(None, 1, "Output width."), "height": integer(None, 1, "Output height."),
         "fit": choice("stretch", ["stretch", "contain", "cover"], "Stretch changes aspect; contain pads transparent; cover crops centrally.")},
        {"width": 1200, "height": 800, "fit": "contain"}, "scale dimensions thumbnail geometry"),
    "canvas_rotate": entry("Rotate / straighten", "filter",
        "Rotate counterclockwise around the centre. Expand retains the whole image and changes dimensions; corners are transparent. Quarter turns preserve pixels exactly.",
        {"angle": number(0, -360, 360, .1, "Positive is counterclockwise.", "degrees"),
         "expand": {"type": "boolean", "default": True, "description": "Expand bounds; false keeps and clips to current dimensions."}},
        {"angle": -2.5, "expand": True}, "rotation straighten orientation geometry"),
    "canvas_flip": entry("Flip / mirror", "filter", "Mirror horizontally and/or vertically without resampling.",
        {"horizontal": {"type": "boolean", "default": True, "description": "Mirror left to right."},
         "vertical": {"type": "boolean", "default": False, "description": "Mirror top to bottom."}},
        {"horizontal": True}, "mirror reverse"),
    "canvas_brightness": entry("Brightness", "filter", "Multiply encoded RGB; alpha is unchanged. Zero is black, one is unchanged.",
        {"factor": number(1, 0, 4, .05, "Brightness multiplier.")}, {"factor": 1.1}, "light dark brighten dim"),
    "canvas_contrast": entry("Contrast", "filter", "Adjust around alpha-weighted mean luminance; transparent hidden RGB does not bias the pivot. One is unchanged.",
        {"factor": number(1, 0, 4, .05, "Contrast multiplier.")}, {"factor": 1.15}, "punch flat tonal"),
    "canvas_saturation": entry("Saturation", "filter", "Zero is grayscale, one preserves colour, above one boosts colour. Alpha is unchanged.",
        {"factor": number(1, 0, 4, .05, "Colour intensity multiplier.")}, {"factor": 1.2}, "color colour vivid muted desaturate"),
    "canvas_exposure": entry("Exposure", "filter", "Multiply linear-light RGB by 2**stops, then encode sRGB. Alpha is unchanged; highlights may clip.",
        {"stops": number(0, -8, 8, .1, "Positive brightens; +1 doubles linear light.", "EV")}, {"stops": .4}, "light photographic ev"),
    "canvas_gamma": entry("Gamma / midtones", "filter", "Apply RGB ** (1/gamma). Above one brightens midtones; endpoints and alpha stay fixed.",
        {"gamma": number(1, .1, 5, .05, "Midtone gamma.")}, {"gamma": 1.15}, "midtone tonal light"),
    "canvas_blur": entry("Gaussian blur", "filter", "Blur premultiplied colour and alpha together to avoid dark or coloured fringes at transparent edges.",
        {"radius": number(3, 0, 200, .25, "Gaussian radius; zero is unchanged.", "px")}, {"radius": 2.5}, "soften defocus gaussian smooth"),
    "canvas_sharpen": entry("Sharpen", "filter", "Unsharp mask using an alpha-weighted blur. Preserves alpha and avoids hidden-colour fringes.",
        {"radius": number(2, 0, 100, .25, "Detail radius.", "px"),
         "amount": number(100, 0, 500, 5, "Detail gain; zero is unchanged.", "%"),
         "threshold": integer(3, 0, "Ignore RGB differences below this threshold.", maximum=255, unit="levels")},
        {"radius": 1.5, "amount": 80, "threshold": 3}, "sharpness crisp detail unsharp"),
    "canvas_grayscale": entry("Grayscale", "filter", "Convert RGB to luminance while preserving alpha.", {}, {}, "black white monochrome greyscale"),
    "canvas_invert": entry("Invert", "filter", "Invert RGB, preserving alpha. Use layer opacity to mix the effect.", {}, {}, "negative reverse colors"),
    "canvas_solid": entry("Solid fill", "background", "Fill with a literal colour or a live @palette-role. Transparent is allowed.",
        {"color": color("@background", "Fill colour: CSS name, #RGB, #RRGGBB, #RRGGBBAA, transparent or @role.")},
        {"color": "#f5f2e8"}, "background color colour fill transparent", ["background", "object"]),
    "canvas_gradient": entry("Linear gradient", "background", "Two-colour alpha-safe gradient. Zero degrees goes left to right; 90 goes top to bottom.",
        {"start": color("@primary", "Starting colour."), "end": color("@accent", "Ending colour."),
         "angle": number(0, -360, 360, 1, "Gradient direction.", "degrees")},
        {"start": "@primary", "end": "@accent", "angle": 90}, "ramp background fade color", ["background", "object"]),
    "canvas_line": entry("Line / polyline", "object", "Draw an antialiased line through pixel coordinates. Produces an overlay, so layer opacity and blend mode work normally.",
        {"points": {"type": "array", "minItems": 2, "description": "[[x,y], ...] in current image pixels.",
                    "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}},
         "width": number(3, .1, 1000, .5, "Stroke width.", "px"), "color": color("@primary", "Stroke colour.")},
        {"points": [[40, 40], [240, 140]], "width": 3, "color": "@accent"}, "draw stroke path annotation"),
    "canvas_shape": entry("Rectangle / ellipse", "object", "Draw an antialiased rectangle or ellipse within an explicit pixel box. Transparent fill makes an outline.",
        {"shape": choice("rectangle", ["rectangle", "ellipse"], "Shape."),
         "left": integer(0, None, "Left edge."), "top": integer(0, None, "Top edge."),
         "right": integer(None, None, "Exclusive right edge."), "bottom": integer(None, None, "Exclusive bottom edge."),
         "fill": color("@primary", "Fill colour or transparent."), "stroke": color("transparent", "Outline colour."),
         "stroke_width": number(1, 0, 1000, .5, "Outline width; zero disables it.", "px")},
        {"left": 20, "top": 20, "right": 220, "bottom": 120, "fill": "transparent", "stroke": "@accent", "stroke_width": 3}, "draw box circle oval outline annotation"),
    "canvas_text": entry("Text", "object", "Draw text on a transparent overlay. Explicit font_path supports custom faces/Unicode coverage; the portable default is regular Latin text.",
        {"content": {"type": "string", "description": "Text; newline creates a line break."},
         "x": integer(0, None, "Left position."), "y": integer(0, None, "Top position."),
         "size": integer(48, 1, "Font size."), "color": color("@primary", "Text colour."),
         "font_path": {"type": "string", "format": "file", "default": "", "description": "Optional TTF/OTF path; tracked automatically."},
         "max_width": integer(0, 0, "Wrap width; zero disables wrapping."),
         "align": choice("left", ["left", "center", "right"], "Alignment of lines within the text block.")},
        {"content": "Summer 2026", "x": 40, "y": 40, "size": 36, "color": "#ffffff"}, "label caption typography annotation"),
    "canvas_duotone": entry("Duotone / palette map", "filter", "Explicitly map luminance between two palette or literal colours, then mix with the original. Preserves source alpha.",
        {"shadows": color("@secondary", "Dark tone."), "highlights": color("@accent", "Light tone."),
         "amount": number(.5, 0, 1, .05, "Strength; zero is unchanged, one fully mapped.")},
        {"shadows": "#182844", "highlights": "#ffd9a0", "amount": .35}, "grade tint palette color map"),
    "canvas_vignette": entry("Vignette", "filter", "Darken edges with an elliptical falloff centred in the image. Alpha is unchanged.",
        {"amount": number(.35, 0, 1, .05, "Edge darkening."),
         "radius": number(.5, 0, 1, .05, "Start falloff at this fraction of the centre-to-corner distance.")},
        {"amount": .25, "radius": .5}, "edges focus darkening lens"),
}


def _check(value, schema, label):
    kind = schema["type"]
    valid = {"number": type(value) in (int, float), "integer": type(value) is int,
             "string": isinstance(value, str), "boolean": type(value) is bool,
             "array": isinstance(value, (list, tuple))}[kind]
    if not valid:
        raise ValueError(f"{label} must be {kind}")
    if kind in ("number", "integer"):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        if schema.get("minimum") is not None and value < schema["minimum"]:
            raise ValueError(f"{label} must be >= {schema['minimum']}")
        if schema.get("maximum") is not None and value > schema["maximum"]:
            raise ValueError(f"{label} must be <= {schema['maximum']}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{label} must be one of {schema['enum']}")
    if kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            raise ValueError(f"{label} has invalid length")
        for item in value:
            _check(item, schema["items"], label + "[]")
    if schema.get("format") == "file" and not value.strip() and "default" not in schema:
        raise ValueError(f"{label} needs a file path")
    if schema.get("format") == "color" and not value.strip():
        raise ValueError(f"{label} needs a colour or @palette-role")


def prepare(script, controls=None, kind=None):
    """Normalize shipped controls without quantizing; custom scripts remain usable."""
    if not isinstance(script, str) or not script:
        raise ValueError("script is required")
    script = script.removesuffix(".py")
    spec = CATALOG.get(script)
    controls = {} if controls is None else deepcopy(controls)
    if not isinstance(controls, dict):
        raise ValueError("controls must be an object")
    if spec is None:
        if kind not in ("background", "filter", "object"):
            raise ValueError("unknown technique; search_techniques lists names. Custom scripts require kind.")
        return dict(script=script, kind=kind, controls=controls, dependencies=[], custom=True)
    kind = kind or spec["kind"]
    if kind not in spec["kinds"]:
        raise ValueError(f"{script} supports kinds {spec['kinds']}")
    unknown = set(controls) - set(spec["controls"])
    if unknown:
        raise ValueError(f"{script}: unknown controls {sorted(unknown)}; expected {list(spec['controls'])}")
    for name, schema in spec["controls"].items():
        if name not in controls:
            if "default" not in schema:
                raise ValueError(f"{script}: required control {name}: {schema['description']}")
            controls[name] = deepcopy(schema["default"])
        _check(controls[name], schema, name)
    if script in ("canvas_crop", "canvas_shape"):
        if controls["right"] <= controls["left"] or controls["bottom"] <= controls["top"]:
            raise ValueError("right must exceed left and bottom must exceed top")
    files = [controls[name] for name, schema in spec["controls"].items()
             if schema.get("format") == "file" and controls[name]]
    return dict(script=script, kind=kind, controls=controls, dependencies=files, custom=False)


def main(sdk, action="search", query="", script=None, controls=None, kind=None, recipe=None):
    if recipe:
        recipes = {
            "photo": [
                {"tool": "add_layer", "args": {"script": "canvas_load_image", "controls": {"path": "<actual attachment path>"}}},
                {"tool": "render_canvas", "args": {}, "note": "Inspect the native dimensions and choose a crop if needed."},
                {"tool": "add_layer", "args": {"script": "canvas_saturation", "controls": {"factor": 1.1}}},
                {"tool": "add_layer", "args": {"script": "canvas_sharpen", "controls": {"radius": 1.5, "amount": 60, "threshold": 3}}},
                {"tool": "render_canvas", "args": {}, "note": "Inspect, then fine-tune the existing layer's controls; do not stack another adjustment."},
            ],
            "composition": [
                {"tool": "manage_layers", "args": {"action": "create", "width": 800, "height": 500}},
                {"tool": "manage_layers", "args": {"action": "set_palette", "colors": {"primary": "#182844", "accent": "#ffd9a0"}}},
                {"tool": "add_layer", "args": {"script": "canvas_gradient", "controls": {"start": "@primary", "end": "@accent", "angle": 30}}},
                {"tool": "add_layer", "args": {"script": "canvas_text", "controls": {"content": "Hello", "x": 60, "y": 60, "size": 48, "color": "#ffffff"}}},
                {"tool": "render_canvas", "args": {}},
            ],
        }
        if recipe not in recipes:
            raise ValueError("recipe must be photo or composition")
        return {"recipe": recipe, "steps": recipes[recipe],
                "note": "Examples are starting points. Use actual input paths and tailor edits to the request; no changes have been made by this lookup."}
    if action == "prepare":
        return prepare(script, controls, kind)
    if script:
        name = script.removesuffix(".py")
        if name not in CATALOG:
            raise ValueError(f"unknown shipped technique: {script}; omit script to list available techniques")
        spec = deepcopy(CATALOG[name])
        spec["script"] = name
        spec["example"] = {"script": name, "kind": spec["kind"], "controls": spec["example"]}
        return spec
    words = re.findall(r"[a-z0-9]+", query.lower())
    ranked = []
    for name, spec in CATALOG.items():
        text = " ".join([name, spec["title"], spec["description"], spec["tags"]]).lower()
        score = sum(word in text for word in words)
        if not words or score:
            ranked.append((score, name, spec))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [{"script": name, "kind": spec["kind"], "description": spec["description"]}
            for _, name, spec in ranked[:(8 if words else len(ranked))]]
