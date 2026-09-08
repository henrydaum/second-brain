"""Discover technique_*.py and read their literal TECHNIQUE declarations.

Discovery never imports or executes a technique. Workspace wins over installed,
then bundled, matching script resolution. A malformed override is reported rather
than silently falling back to a different implementation. Each lookup reads current
files; new, edited and removed techniques need no registration or restart.
"""
import ast
from copy import deepcopy
import math
import re

box = "image_editing"


def _check(value, schema, label, coerce=False):
    kind = schema["type"]
    original = value
    if coerce and isinstance(value, str):
        text = value.strip()
        if kind in ("number", "integer"):
            try:
                value = int(text)
            except ValueError:
                if kind == "number":
                    try:
                        value = float(text)
                    except ValueError:
                        pass
        elif kind == "boolean" and text.lower() in ("true", "false", "yes", "no", "1", "0"):
            value = text.lower() in ("true", "yes", "1")
    valid = {"number": type(value) in (int, float), "integer": type(value) is int,
             "string": isinstance(value, str), "boolean": type(value) is bool,
             "array": isinstance(value, (list, tuple))}[kind]
    if not valid:
        hint = {"number": "Use a number such as 3.5 (numeric strings are accepted).",
                "integer": "Use a whole integer such as 25; fractions are not rounded.",
                "boolean": "Use true or false (also accepts strings yes/no and 1/0)."}.get(kind, "")
        raise ValueError(f"{label} must be {kind}; received {original!r} "
                         f"({type(original).__name__}). {hint}".rstrip())
    if kind in ("number", "integer"):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite; received {original!r}")
        if schema.get("minimum") is not None and value < schema["minimum"]:
            raise ValueError(f"{label} must be >= {schema['minimum']}; received {original!r}")
        if schema.get("maximum") is not None and value > schema["maximum"]:
            raise ValueError(f"{label} must be <= {schema['maximum']}; received {original!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{label} must be one of {schema['enum']}")
    if kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            raise ValueError(f"{label} has invalid length")
        value = [_check(item, schema["items"], f"{label}[{index}]", coerce=coerce)
                 for index, item in enumerate(value)]
    if schema.get("format") == "file" and not value.strip() and "default" not in schema:
        raise ValueError(f"{label} needs a file path")
    if schema.get("format") == "color" and not value.strip():
        raise ValueError(f"{label} needs a colour or @palette-role")
    return value


def validate_controls(spec, controls=None, kind=None):
    """Validate one declaration's inputs, retaining arbitrary fractional values."""
    controls = {} if controls is None else deepcopy(controls)
    if not isinstance(controls, dict):
        raise ValueError("controls must be an object")
    kind = kind or spec["kind"]
    if kind not in spec.get("kinds", [spec["kind"]]):
        raise ValueError(f"supported layer kinds: {spec.get('kinds', [spec['kind']])}")
    unknown = set(controls) - set(spec["controls"])
    if unknown:
        raise ValueError(f"unknown controls {sorted(unknown)}; expected {list(spec['controls'])}")
    for name, schema in spec["controls"].items():
        if name not in controls:
            if "default" not in schema:
                raise ValueError(f"required control {name}: {schema.get('description', '')}")
            controls[name] = deepcopy(schema["default"])
        controls[name] = _check(controls[name], schema, name, coerce=True)
    for rule in spec.get("constraints", []):
        if controls[rule["greater"]] <= controls[rule["than"]]:
            raise ValueError(f"{rule['greater']} must exceed {rule['than']}")
    return controls


def _schema(schema, name):
    if not isinstance(schema, dict) or schema.get("type") not in ("number", "integer", "string", "boolean", "array"):
        raise ValueError(f"{name}: supported control type is required")
    for bound in ("minimum", "maximum", "step"):
        if bound in schema and (type(schema[bound]) not in (int, float) or not math.isfinite(schema[bound])):
            raise ValueError(f"{name}.{bound} must be finite numeric data")
    if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
        raise ValueError(f"{name}: minimum exceeds maximum")
    if schema.get("step", 1) <= 0:
        raise ValueError(f"{name}: step must be positive")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError(f"{name}: enum must be a nonempty list")
    if schema.get("format") not in (None, "file", "color"):
        raise ValueError(f"{name}: unsupported format")
    if "format" in schema and schema["type"] != "string":
        raise ValueError(f"{name}: file/color controls must be strings")
    if schema["type"] == "array":
        for key in ("minItems", "maxItems"):
            if key in schema and (type(schema[key]) is not int or schema[key] < 0):
                raise ValueError(f"{name}.{key} must be a nonnegative integer")
        if schema.get("minItems", 0) > schema.get("maxItems", math.inf):
            raise ValueError(f"{name}: minItems exceeds maxItems")
        _schema(schema.get("items"), name + "[]")
    if "default" in schema:
        _check(schema["default"], schema, name)


def _declaration(source):
    tree = ast.parse(source)
    values = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "TECHNIQUE" for t in targets):
            values.append(ast.literal_eval(node.value))
    if len(values) != 1 or not isinstance(values[0], dict):
        raise ValueError("exactly one literal TECHNIQUE dictionary is required")
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    if len(mains) != 1:
        raise ValueError("one main(sdk, ...) function is required")
    args = mains[0].args
    if not args.args or args.args[0].arg != "sdk" or args.posonlyargs:
        raise ValueError("main must start with sdk and accept named layer arguments")
    required = {"kind", "input_path", "output_path", "width", "height", "seed", "palette", "controls"}
    if args.kwarg is None and not required <= {a.arg for a in args.args + args.kwonlyargs}:
        raise ValueError("main must accept the complete layer contract")
    spec = values[0]
    for key in ("title", "description"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"TECHNIQUE.{key} must be a nonempty string")
    if spec.get("kind") not in ("background", "filter", "object"):
        raise ValueError("TECHNIQUE.kind must be background, filter or object")
    kinds = spec.get("kinds", [spec["kind"]])
    if not isinstance(kinds, list) or spec["kind"] not in kinds or any(k not in ("background", "filter", "object") for k in kinds):
        raise ValueError("TECHNIQUE.kinds must include its default kind")
    if not isinstance(spec.get("controls"), dict):
        raise ValueError("TECHNIQUE.controls must be a dictionary")
    for name, schema in spec["controls"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("control names must be nonempty strings")
        _schema(schema, name)
    if not isinstance(spec.get("tags", ""), str):
        raise ValueError("TECHNIQUE.tags must be a string of search words")
    aliases = spec.get("aliases", [])
    if not isinstance(aliases, list) or any(not isinstance(a, str) or not re.fullmatch(r"[a-zA-Z0-9_]+", a) for a in aliases):
        raise ValueError("TECHNIQUE.aliases must be a list of filename stems")
    constraints = spec.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("TECHNIQUE.constraints must be a list")
    for rule in constraints:
        if not isinstance(rule, dict) or set(rule) != {"greater", "than"} or any(
                name not in spec["controls"] or spec["controls"][name]["type"] not in ("number", "integer")
                for name in rule.values()):
            raise ValueError("constraints must compare numeric controls using greater/than")
    if not isinstance(spec.get("example"), dict):
        raise ValueError("TECHNIQUE.example must contain example controls")
    validate_controls(spec, spec["example"])
    return spec


def discover(sdk):
    """Return valid declarations and diagnostics, with no technique code executed."""
    techniques, errors, seen = {}, {}, set()
    for origin in ("workspace", "installed", "bundled"):
        directory = sdk.path.join(sdk.paths.get(origin), "scripts")
        if not sdk.fs.exists(directory):
            continue
        entries = sdk.fs.list(directory, pattern="technique_*.py", details=True)
        for entry in sorted(entries, key=lambda entry: entry["path"]):
            if entry["is_dir"]:
                continue
            path = entry["path"]
            name = sdk.path.name(path).removesuffix(".py")
            if name in seen:
                continue
            seen.add(name)
            # Read failures, including permission denials, remain real failures.
            source = sdk.fs.read(path)
            try:
                if not re.fullmatch(r"technique_[a-zA-Z0-9_]+", name):
                    raise ValueError("technique filenames must use letters, digits and underscores")
                spec = _declaration(source)
                techniques[name] = dict(spec, script=name, source_path=path, origin=origin)
            except (ValueError, SyntaxError, TypeError, KeyError) as exc:
                errors[name] = {"script": name, "source_path": path, "origin": origin, "error": str(exc)}
    return {"techniques": techniques, "errors": errors}


def _lookup(inventory, name):
    if name in inventory["errors"]:
        raise ValueError(f"{name}: {inventory['errors'][name]['error']}")
    if name in inventory["techniques"]:
        return inventory["techniques"][name]
    matches = [spec for spec in inventory["techniques"].values() if name in spec.get("aliases", [])]
    if len(matches) > 1:
        raise ValueError(f"ambiguous technique alias {name}; use a technique_ filename")
    return matches[0] if matches else None


def prepare(sdk, script, controls=None, kind=None, inventory=None):
    """Resolve a technique or legacy alias, then validate and derive file inputs."""
    if not isinstance(script, str) or not script:
        raise ValueError("script is required")
    name = script.removesuffix(".py")
    if not re.fullmatch(r"[a-zA-Z0-9_]+", name):
        raise ValueError("script must be a filename without directories")
    inventory = discover(sdk) if inventory is None else inventory
    spec = _lookup(inventory, name)
    if spec is None:
        if name.startswith("technique_") or kind not in ("background", "filter", "object"):
            raise ValueError(f"unknown technique {name}; search_techniques lists discovered files and errors")
        if controls is not None and not isinstance(controls, dict):
            raise ValueError("controls must be an object")
        return dict(script=name, kind=kind, controls=deepcopy(controls or {}), dependencies=[], custom=True)
    controls = validate_controls(spec, controls, kind)
    files = [controls[name] for name, schema in spec["controls"].items()
             if schema.get("format") == "file" and controls[name]]
    return dict(script=spec["script"], source_path=spec["source_path"], kind=kind or spec["kind"],
                controls=controls, dependencies=files, custom=False)


def main(sdk, action="search", query="", script=None, controls=None, kind=None, recipe=None):
    if action == "guide":
        for root in ("workspace", "installed", "bundled"):
            path = sdk.path.join(sdk.paths.get(root), "scripts", "canvas_technique_template.py")
            if sdk.fs.exists(path):
                return {"template_path": path, "template": sdk.fs.read(path),
                        "workflow": ["Save a copy as workspace/scripts/technique_your_name.py.",
                                     "Edit its literal TECHNIQUE metadata and its apply function together.",
                                     "Use art_kit for shared image, colour and pixel utilities.",
                                     "Validate using sdk.plugins.validate(path).",
                                     "search_techniques(script='technique_your_name') discovers it immediately.",
                                     "add_layer(script='technique_your_name', controls={...}), then render_canvas."]}
        raise ValueError("technique template is missing; update the Image Editing bundle")
    if recipe:
        recipes = {
            "photo": [
                {"tool": "add_layer", "args": {"script": "technique_load_image", "controls": {"path": "<actual attachment path>"}}},
                {"tool": "render_canvas", "args": {}, "note": "Inspect the native dimensions and choose a crop if needed."},
                {"tool": "add_layer", "args": {"script": "technique_saturation", "controls": {"factor": 1.1}}},
                {"tool": "add_layer", "args": {"script": "technique_sharpen", "controls": {"radius": 1.5, "amount": 60, "threshold": 3}}},
                {"tool": "render_canvas", "args": {}, "note": "Inspect, then fine-tune the existing layer's controls; do not stack another adjustment."},
            ],
            "composition": [
                {"tool": "manage_layers", "args": {"action": "create", "width": 800, "height": 500}},
                {"tool": "manage_layers", "args": {"action": "set_palette", "colors": {"primary": "#182844", "accent": "#ffd9a0"}}},
                {"tool": "add_layer", "args": {"script": "technique_gradient", "controls": {"start": "@primary", "end": "@accent", "angle": 30}}},
                {"tool": "add_layer", "args": {"script": "technique_text", "controls": {"content": "Hello", "x": 60, "y": 60, "size": 48, "color": "#ffffff"}}},
                {"tool": "render_canvas", "args": {}},
            ],
        }
        if recipe not in recipes:
            raise ValueError("recipe must be photo or composition")
        return {"recipe": recipe, "steps": recipes[recipe],
                "note": "Examples are starting points. Use actual input paths and tailor edits to the request; no changes have been made by this lookup."}
    if action == "prepare":
        return prepare(sdk, script, controls, kind)
    inventory = discover(sdk)
    if script:
        spec = _lookup(inventory, script.removesuffix(".py"))
        if spec is None:
            raise ValueError(f"unknown technique: {script}; omit script to list discovered files")
        spec = deepcopy(spec)
        spec["example"] = {"script": spec["script"], "kind": spec["kind"], "controls": spec["example"]}
        return spec
    words = re.findall(r"[a-z0-9]+", query.lower())
    ranked = []
    for name, spec in inventory["techniques"].items():
        text = " ".join([name, spec["title"], spec["description"], spec.get("tags", ""), *spec.get("aliases", [])]).lower()
        score = sum(word in text for word in words)
        if not words or score:
            ranked.append((score, name, spec))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [{"script": name, "kind": spec["kind"], "description": spec["description"],
             "source_path": spec["source_path"], "origin": spec["origin"]}
            for _, name, spec in ranked[:(8 if words else len(ranked))]] + list(inventory["errors"].values())
