"""Technique persistence + AST validation for the class-based plugin format.

A technique is one Python file under ``plugins/techniques/technique_<slug>.py`` (baked-in)
or ``DATA_DIR/sandbox_techniques/technique_<slug>.py`` (sandbox) that defines a
``class <X>(BaseTechnique)`` with metadata/descriptor controls as class
attributes and a ``def run(self, canvas)`` method.

This module owns:
- AST validation (which imports + attribute accesses are safe, structural
  requirements for the BaseTechnique class).
- File formatting: targeted in-place rewrites for managed metadata
  (``owner``, ``created_at``, ``hidden``) and ownership transfer.
- Filesystem-level write/update/delete operations.

Runtime lookup, search, and embedding cache live in
``plugins.techniques.helpers.technique_registry`` — keep this module free of
in-memory state so it can be called from any process (parent or child
sandbox).
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from paths import SANDBOX_TECHNIQUES


# ---------------------------------------------------------------------------
# AST validation — first line of defense for the subprocess sandbox.
# ---------------------------------------------------------------------------

# Literal-only allowlist. Top-level fallback covers PIL.* etc. We explicitly
# admit "plugins.BaseTechnique" as a literal so techniques can import BaseTechnique, but
# we do NOT admit the bare "plugins" namespace — that would let a technique reach
# into plugins.helpers.web_auth etc.
_ALLOWED_IMPORTS = {
    "math", "random", "colorsys",
    "numpy", "numpy.random",
    "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFilter",
    "PIL.ImageOps", "PIL.ImageEnhance", "PIL.ImageChops", "PIL.ImageColor",
    "plugins.BaseTechnique",
}

_BANNED_NAMES = {
    "__import__", "eval", "exec", "compile", "open",
    "globals", "locals", "vars", "input", "breakpoint",
    "exit", "quit", "help", "copyright", "credits", "license",
    "memoryview", "__loader__", "__spec__", "__file__",
}

_BANNED_ATTRS = {
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__code__", "__closure__", "__dict__",
    "__builtins__", "__import__", "__getattribute__",
    "f_globals", "f_locals", "f_back", "gi_frame",
    "open", "save", "load", "loadtxt", "genfromtxt", "fromfile", "tofile",
}


class TechniqueValidationError(ValueError):
    """Raised when technique code fails AST validation."""


def _is_import_allowed(mod: str) -> bool:
    """Literal allowlist with a top-level fallback ONLY for non-`plugins` modules.

    `plugins.BaseTechnique` is admitted as a literal. Any other `plugins.*` import
    is rejected to keep techniques out of the rest of the helpers tree.
    """
    if mod in _ALLOWED_IMPORTS:
        return True
    if mod.startswith("plugins"):
        return False
    top = mod.split(".")[0]
    return top in _ALLOWED_IMPORTS


def _find_base_technique_class(tree: ast.Module) -> ast.ClassDef | None:
    """Return the first ClassDef whose bases name BaseTechnique (Name or Attribute)."""
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseTechnique":
                return node
            if isinstance(base, ast.Attribute) and base.attr == "BaseTechnique":
                return node
    return None


def _find_run_method(cls_node: ast.ClassDef) -> ast.FunctionDef | None:
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "run":
            return item
    return None


def _literal_control_attrs(cls_node: ast.ClassDef) -> list[int]:
    lines: list[int] = []
    for item in cls_node.body:
        if isinstance(item, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "controls" for t in item.targets):
                lines.append(item.lineno)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == "controls":
            lines.append(item.lineno)
    return lines


def _run_signature_error(run: ast.FunctionDef) -> str | None:
    args = run.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        return "run must be declared exactly as `def run(self, canvas)`"
    names = [a.arg for a in args.args]
    if names != ["self", "canvas"]:
        return "run must be declared exactly as `def run(self, canvas)`"
    return None


_DESCRIPTOR_KWARGS = {
    "Slider": {"default", "step", "label"},
    "Bool": {"default", "label"},
    "Enum": {"default", "label"},
    "Pan": {"x", "y", "label", "step"},
    "Text": {"default", "max_length", "placeholder", "label"},
    "Palette": {"label"},
}


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _descriptor_call_errors(cls_node: ast.ClassDef) -> list[str]:
    errors: list[str] = []
    for item in cls_node.body:
        if not isinstance(item, (ast.Assign, ast.AnnAssign)) or not isinstance(item.value, ast.Call):
            continue
        name = _call_name(item.value.func)
        if name not in _DESCRIPTOR_KWARGS:
            continue
        attr = "control"
        target = item.target if isinstance(item, ast.AnnAssign) else item.targets[0]
        if isinstance(target, ast.Name):
            attr = target.id
        allowed = _DESCRIPTOR_KWARGS[name]
        unknown = [kw.arg for kw in item.value.keywords if kw.arg and kw.arg not in allowed]
        if unknown:
            errors.append(f"{attr}: {name} got unsupported keyword(s): {', '.join(unknown)}")
        if name == "Palette" and item.value.args:
            errors.append(f"{attr}: Palette() only exposes a layer palette override; use Enum([...], label='Slot') to choose a palette slot")
    return errors


# Class-attribute names that BaseTechnique metadata reserves; literal class-level
# assignments to these are NOT misnamed control attempts.
_METADATA_NAMES = frozenset({
    "name", "description", "kind", "owner", "created_at", "hidden",
    "auto_register", "requires_services", "config_settings", "controls",
})

_PALETTE_SLOT_VALUES = frozenset({
    "primary", "secondary", "tertiary", "accent", "background",
})


def _literal_kwarg(call: ast.Call, key: str):
    """Return the literal-eval'd value for ``key`` if present, else MISSING."""
    for kw in call.keywords:
        if kw.arg == key:
            try:
                return ast.literal_eval(kw.value)
            except Exception:
                return _MISSING
    return _MISSING


def _literal_arg(call: ast.Call, index: int):
    if index >= len(call.args):
        return _MISSING
    try:
        return ast.literal_eval(call.args[index])
    except Exception:
        return _MISSING


_MISSING = object()


def _enum_option_values(options) -> list | None:
    """Mirror Enum.__init__'s option-normalization to extract value keys.

    Returns None if any option is in a form we can't normalize statically
    (skip the default-in-options check in that case rather than false-alarm).
    """
    if not isinstance(options, (list, tuple)) or not options:
        return None
    values: list = []
    for o in options:
        if isinstance(o, str):
            values.append(o)
        elif isinstance(o, dict):
            if "value" not in o:
                return None
            values.append(o["value"])
        elif isinstance(o, (list, tuple)) and o:
            values.append(o[0])
        else:
            return None
    return values


def _descriptor_semantic_errors(cls_node: ast.ClassDef) -> list[str]:
    """Catch control bugs at AST time that would otherwise only surface
    inside the subprocess: Enum default ∉ options, Slider max ≤ min,
    Pan referencing non-Slider attrs, > 4 non-palette controls, ``palette``
    as a non-Palette control, palette-slot strings assigned as bare class
    attributes (a common misattempt at a control default).
    """
    errors: list[str] = []
    sliders: dict[str, dict] = {}        # attr_name -> {min, max, default}
    pans: list[tuple[str, str, str]] = []  # (attr_name, x_param, y_param)
    descriptor_kinds: list[tuple[str, str]] = []  # (attr_name, descriptor_name)
    pan_consumed: set[str] = set()

    for item in cls_node.body:
        # Heuristic: catch `slot = "primary"` style class-attribute defaults
        # — they look like a control attempt but produce no UI and read as
        # a plain string from self.slot. Flag literal-string assignments
        # whose value matches a palette slot name.
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            attr = item.targets[0].id
            if attr not in _METADATA_NAMES and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                if item.value.value in _PALETTE_SLOT_VALUES:
                    errors.append(
                        f"class '{cls_node.name}': '{attr} = {item.value.value!r}' is not a control — "
                        f"it just sets a plain class attribute. To let the user pick a palette slot, "
                        f"declare it as Enum: `{attr} = Enum(['background','primary','secondary','tertiary','accent'], "
                        f"default='{item.value.value}', label='...')`"
                    )

        if not isinstance(item, (ast.Assign, ast.AnnAssign)) or not isinstance(item.value, ast.Call):
            continue
        descriptor_name = _call_name(item.value.func)
        if descriptor_name not in _DESCRIPTOR_KWARGS:
            continue
        target = item.target if isinstance(item, ast.AnnAssign) else item.targets[0]
        attr_name = target.id if isinstance(target, ast.Name) else "control"
        call = item.value
        descriptor_kinds.append((attr_name, descriptor_name))

        # Block accidental override of the framework's `palette` pop hook.
        if attr_name == "palette" and descriptor_name != "Palette":
            errors.append(
                f"control name 'palette' is reserved for the Palette() descriptor; "
                f"rename '{attr_name}' (e.g. 'palette_slot', 'color_role') or switch to Palette()"
            )

        if descriptor_name == "Slider":
            lo = _literal_arg(call, 0)
            hi = _literal_arg(call, 1)
            default = _literal_kwarg(call, "default")
            if lo is _MISSING:
                lo = _literal_kwarg(call, "min")
            if hi is _MISSING:
                hi = _literal_kwarg(call, "max")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                if hi <= lo:
                    errors.append(
                        f"{attr_name}: Slider needs max > min (got min={lo}, max={hi}); "
                        f"swap them or widen the range"
                    )
                if isinstance(default, (int, float)) and not (lo <= default <= hi):
                    errors.append(
                        f"{attr_name}: Slider default={default} is outside [min={lo}, max={hi}]; "
                        f"pick a default inside the declared range"
                    )
                sliders[attr_name] = {"min": lo, "max": hi, "default": default}

        elif descriptor_name == "Enum":
            options = _literal_arg(call, 0)
            if options is _MISSING:
                options = _literal_kwarg(call, "options")
            default = _literal_kwarg(call, "default")
            if options is not _MISSING:
                if not options:
                    errors.append(f"{attr_name}: Enum needs at least one option")
                else:
                    values = _enum_option_values(options)
                    if values is not None and default is not _MISSING and default not in values:
                        errors.append(
                            f"{attr_name}: Enum default={default!r} is not in options "
                            f"{values!r}; pick one of the declared option values"
                        )

        elif descriptor_name == "Pan":
            x = _literal_kwarg(call, "x")
            y = _literal_kwarg(call, "y")
            if x is _MISSING and len(call.args) >= 1:
                x = _literal_arg(call, 0)
            if y is _MISSING and len(call.args) >= 2:
                y = _literal_arg(call, 1)
            if isinstance(x, str) and isinstance(y, str):
                pans.append((attr_name, x, y))

    # Pan references resolve only after we've seen every slider in the class.
    for pan_attr, x_param, y_param in pans:
        missing = [p for p in (x_param, y_param) if p not in sliders]
        if missing:
            errors.append(
                f"{pan_attr}: Pan(x={x_param!r}, y={y_param!r}) — "
                f"{', '.join(repr(m) for m in missing)} not declared as Slider on the same class. "
                f"Add the Sliders first, then point Pan at their names."
            )
            continue
        pan_consumed.add(x_param)
        pan_consumed.add(y_param)

    # Count user-facing controls. Palette doesn't count toward the cap;
    # Sliders absorbed by a Pan share their widget with that Pan, so they
    # count once (via the Pan), not twice.
    visible = 0
    for attr_name, descriptor_name in descriptor_kinds:
        if descriptor_name == "Palette":
            continue
        if descriptor_name == "Slider" and attr_name in pan_consumed:
            continue
        visible += 1
    if visible > 4:
        errors.append(
            f"class '{cls_node.name}': {visible} non-palette controls declared, cap is 4. "
            f"Drop the least-useful ones or fold two scalars into a Pan (which counts as one widget)."
        )

    return errors


def _unsupported_control_api_errors(cls_node: ast.ClassDef) -> list[str]:
    errors: list[str] = []
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "get_controls":
            errors.append(
                f"class '{cls_node.name}' defines get_controls(); declare controls directly as class attributes, "
                "e.g. `slot = Enum([...], default='primary', label='Palette Slot')`"
            )
    return errors


def validate_technique_code(source: str) -> list[str]:
    """Return a list of violations. Empty list means the code is acceptable.

    Rules:
      * Imports limited to _ALLOWED_IMPORTS (literal or non-`plugins` top-level).
      * No references to dangerous names or dunder escape hatches.
      * No `from x import *`.
      * Must define a `class X(BaseTechnique)` with a `def run(...)` method.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    errors: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_import_allowed(alias.name):
                    errors.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not _is_import_allowed(mod):
                errors.append(f"disallowed import: from {mod}")
            for alias in node.names:
                if alias.name == "*":
                    errors.append("wildcard imports are not allowed")
        elif isinstance(node, ast.Name):
            if node.id in _BANNED_NAMES:
                errors.append(f"disallowed name: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                errors.append(f"disallowed attribute access: .{node.attr}")
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in {"__init__", "__name__"}:
                errors.append(f"disallowed dunder attribute: .{node.attr}")

    cls_node = _find_base_technique_class(tree)
    if cls_node is None:
        errors.append("no class inheriting from BaseTechnique found — every technique must define `class <Name>(BaseTechnique):`")
    else:
        for lineno in _literal_control_attrs(cls_node):
            errors.append(f"class '{cls_node.name}' declares literal controls at line {lineno}; use Slider/Enum/Bool/Pan/Text/Palette descriptors")
        errors.extend(_descriptor_call_errors(cls_node))
        errors.extend(_descriptor_semantic_errors(cls_node))
        errors.extend(_unsupported_control_api_errors(cls_node))
        run = _find_run_method(cls_node)
        if run is None:
            errors.append(f"class '{cls_node.name}' must define `def run(self, canvas)`")
        else:
            err = _run_signature_error(run)
            if err:
                errors.append(f"class '{cls_node.name}': {err}")

    return errors


def assert_valid(source: str) -> None:
    errors = validate_technique_code(source)
    if errors:
        raise TechniqueValidationError("; ".join(errors))


_KINDS = ("background", "object", "filter")


def source_uses_palette(source: str | None) -> bool:
    if not source:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        (isinstance(n, ast.Attribute)
        and ((n.attr == "palette" and isinstance(n.value, ast.Name) and n.value.id == "canvas") or n.attr == "palette_color"))
        or (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {"new", "create_image"}
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "canvas")
        for n in ast.walk(tree)
    )


# ---------------------------------------------------------------------------
# Technique dataclass — runner-facing DTO.
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("_", (name or "").strip().lower()).strip("_")
    return slug or "untitled"


@dataclass
class Technique:
    """Runner-facing DTO. Populated from a BaseTechnique instance via
    TechniqueRegistry, or directly from a source file for tools that need to
    read a technique without instantiating it."""
    slug: str
    path: str
    name: str
    description: str
    kind: str
    owner: str
    code: str            # the full file source — exec'd in the sandbox
    created_at: float
    controls: list = None
    hidden: bool = False

    def __post_init__(self):
        if self.controls is None:
            self.controls = []

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Path helpers.
# ---------------------------------------------------------------------------

def _sandbox_path_for(slug: str) -> Path:
    return SANDBOX_TECHNIQUES / f"technique_{slug}.py"


def _built_in_path_for(slug: str) -> Path:
    from paths import ROOT_DIR
    return ROOT_DIR / "plugins" / "techniques" / f"technique_{slug}.py"


def is_built_in(path: str | Path) -> bool:
    from paths import ROOT_DIR
    try:
        return (ROOT_DIR / "plugins" / "techniques").resolve() in Path(path).resolve().parents
    except Exception:
        return False


def _coerce_created_at(raw, fallback_path: Path) -> float:
    if raw is None or raw == "":
        return float(fallback_path.stat().st_mtime) if fallback_path.exists() else time.time()
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            pass
        try:
            from datetime import datetime
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            pass
    return float(fallback_path.stat().st_mtime) if fallback_path.exists() else time.time()


# ---------------------------------------------------------------------------
# DTO from a BaseTechnique instance.
# ---------------------------------------------------------------------------

def to_technique_record(instance) -> Technique:
    """Build a runner-facing Technique from a registered BaseTechnique instance.

    The class is the source of truth for metadata; ``code`` is read from
    ``_source_path`` (set by plugin_discovery at load time)."""
    source_path = Path(getattr(instance, "_source_path", "") or "")
    code = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
    controls = list(getattr(instance, "controls", []) or [])
    if not source_uses_palette(code):
        controls = [c for c in controls if not (isinstance(c, dict) and c.get("type") == "palette")]
    slug = slugify(getattr(instance, "name", "") or source_path.stem.removeprefix("technique_"))
    return Technique(
        slug=slug,
        path=str(source_path.resolve()) if source_path else "",
        name=str(getattr(instance, "name", "") or slug),
        description=str(getattr(instance, "description", "") or ""),
        kind=str(getattr(instance, "kind", "") or "background"),
        owner=str(getattr(instance, "owner", "") or ""),
        code=code,
        created_at=float(getattr(instance, "created_at", 0.0) or _coerce_created_at(None, source_path)),
        controls=controls,
        hidden=bool(getattr(instance, "hidden", False)),
    )


# ---------------------------------------------------------------------------
# In-place class-body rewrites for soft-delete + ownership transfer.
# ---------------------------------------------------------------------------

def _replace_class_attr(source: str, attr: str, new_value_repr: str) -> str:
    """Replace `<attr> = ...` inside the BaseTechnique subclass body. If the
    assignment isn't present, insert it after the last existing class-attr
    assignment (so it sits with the rest of the metadata)."""
    tree = ast.parse(source)
    cls = _find_base_technique_class(tree)
    if cls is None:
        raise TechniqueValidationError("no BaseTechnique subclass found")

    target = None
    last_assign_end = None
    for item in cls.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            last_assign_end = getattr(item, "end_lineno", item.lineno)
            if item.targets[0].id == attr:
                target = item
                break

    lines = source.splitlines(keepends=True)
    replacement = f"    {attr} = {new_value_repr}\n"

    if target is not None:
        start = target.lineno - 1
        end = getattr(target, "end_lineno", target.lineno)
        lines[start:end] = [replacement]
    elif last_assign_end is not None:
        lines.insert(last_assign_end, replacement)
    else:
        # Empty class body — insert right after the class header.
        lines.insert(cls.lineno, replacement)
    return "".join(lines)


def set_hidden_in_source(source: str, hidden: bool) -> str:
    return _replace_class_attr(source, "hidden", repr(bool(hidden)))


def set_owner_in_source(source: str, owner: str) -> str:
    return _replace_class_attr(source, "owner", repr(str(owner)))


# ---------------------------------------------------------------------------
# Filesystem ops used by the tools. The tools then refresh the registry.
# ---------------------------------------------------------------------------

def write_technique(
    *, name: str, description: str, kind: str, owner: str, code: str,
) -> tuple[Technique, Path]:
    """Create a new sandbox technique. Returns (technique_dto, file_path).

    Validates the user's full BaseTechnique class source, stamps managed
    metadata, and writes the file. The caller is responsible for asking the
    registry to load it.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    name = (name or "").strip()
    description = (description or "").strip()
    if not name:
        raise ValueError("name is required")
    slug = slugify(name)
    if not slug:
        raise ValueError("name produced an empty slug")

    sandbox_path = _sandbox_path_for(slug)
    if sandbox_path.exists() or _built_in_path_for(slug).exists():
        raise FileExistsError(f"a technique named '{slug}' already exists")

    created_at = time.time()
    assert_valid(code)
    file_source = _rewrite_metadata_only(
        code, name=name, description=description, kind=kind,
        owner=owner or "", created_at=created_at, hidden=False,
    )
    assert_valid(file_source)

    SANDBOX_TECHNIQUES.mkdir(parents=True, exist_ok=True)
    sandbox_path.write_text(file_source, encoding="utf-8")

    technique = Technique(
        slug=slug, path=str(sandbox_path.resolve()),
        name=name, description=description,
        kind=kind, owner=owner or "", code=file_source,
        created_at=created_at, controls=[],
    )
    return technique, sandbox_path


def rewrite_technique(
    path: Path, *, owner_session_key: str,
    name: str | None = None, description: str | None = None,
    code: str | None = None,
) -> Technique:
    """Update an existing sandbox technique in place. ``code`` is full technique source."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no technique file at {path}")
    if is_built_in(path):
        raise PermissionError(f"'{path.name}' is a built-in technique and is read-only. Create a new technique to fork it.")

    # Read existing metadata via a temporary module load done via AST only —
    # we never exec the file in this process.
    existing_src = path.read_text(encoding="utf-8")
    existing = _read_class_metadata(existing_src)
    if existing.get("owner") and existing["owner"] != owner_session_key:
        raise PermissionError(f"only owner '{existing['owner']}' can update this technique")

    new_name = (name if name is not None else existing.get("name") or "").strip()
    new_desc = (description if description is not None else existing.get("description") or "").strip()
    new_code = code  # user-supplied body, or None to keep current
    new_kind = existing.get("kind") or "background"
    new_owner = existing.get("owner") or owner_session_key
    new_created_at = float(existing.get("created_at") or time.time())
    new_hidden = bool(existing.get("hidden", False))

    src = existing_src if new_code is None else new_code
    assert_valid(src)
    out_source = _rewrite_metadata_only(
        src,
        name=new_name, description=new_desc, kind=new_kind,
        owner=new_owner, created_at=new_created_at, hidden=new_hidden,
    )

    assert_valid(out_source)
    path.write_text(out_source, encoding="utf-8")

    slug = path.stem.removeprefix("technique_")
    return Technique(
        slug=slug, path=str(path.resolve()),
        name=new_name, description=new_desc,
        kind=new_kind, owner=new_owner, code=out_source,
        created_at=new_created_at, controls=[], hidden=new_hidden,
    )


def soft_delete_technique(path: Path, *, owner_session_key: str) -> bool:
    """Flip ``hidden = True`` in the class body. Returns False if path missing."""
    path = Path(path)
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8")
    meta = _read_class_metadata(src)
    if (meta.get("owner")
            and meta["owner"] != owner_session_key
            and not is_built_in(path)):
        raise PermissionError(f"only owner '{meta['owner']}' can hide this technique")
    if meta.get("hidden"):
        return True
    path.write_text(set_hidden_in_source(src, True), encoding="utf-8")
    return True


def anonymize_owner_in_dir(directory: Path, owner_values) -> int:
    """Rewrite ``owner`` to 'anonymous' in every technique file under *directory*
    whose owner matches one of ``owner_values`` (case-insensitive). Returns
    the number of files rewritten."""
    targets = {str(v).strip().lower() for v in owner_values if v}
    if not targets or not directory.exists():
        return 0
    rewritten = 0
    for path in directory.glob("technique_*.py"):
        try:
            src = path.read_text(encoding="utf-8")
            meta = _read_class_metadata(src)
        except Exception:
            continue
        owner = str(meta.get("owner") or "").strip().lower()
        if not owner or owner not in targets or owner == "anonymous":
            continue
        try:
            path.write_text(set_owner_in_source(src, "anonymous"), encoding="utf-8")
            rewritten += 1
        except Exception:
            continue
    return rewritten


# ---------------------------------------------------------------------------
# Class-attribute extraction via AST literal_eval (no exec).
# ---------------------------------------------------------------------------

_META_FIELDS = {"name", "description", "kind", "owner", "created_at", "hidden"}


def _read_class_metadata(source: str) -> dict:
    """Extract BaseTechnique class-attribute metadata using literal_eval only."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    cls = _find_base_technique_class(tree)
    if cls is None:
        return {}
    out: dict = {}
    for item in cls.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            name = item.targets[0].id
            if name in _META_FIELDS:
                try:
                    out[name] = ast.literal_eval(item.value)
                except Exception:
                    pass
    return out


def _rewrite_metadata_only(source: str, *, name: str, description: str, kind: str,
                            owner: str, created_at: float, hidden: bool) -> str:
    """Replace each metadata attr in place; leaves body code untouched."""
    out = source
    out = _replace_class_attr(out, "name", repr(name))
    out = _replace_class_attr(out, "description", repr(description))
    out = _replace_class_attr(out, "kind", repr(kind))
    out = _replace_class_attr(out, "owner", repr(owner))
    out = _replace_class_attr(out, "created_at", repr(float(created_at)))
    out = _replace_class_attr(out, "hidden", repr(bool(hidden)))
    return out
