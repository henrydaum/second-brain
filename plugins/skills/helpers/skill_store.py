"""Skill persistence + AST validation for the class-based plugin format.

A skill is one Python file under ``plugins/skills/skill_<slug>.py`` (baked-in)
or ``DATA_DIR/sandbox_skills/skill_<slug>.py`` (sandbox) that defines a
``class <X>(BaseSkill)`` with metadata as class attributes and a
``def run(self, canvas, **params)`` method.

This module owns:
- AST validation (which imports + attribute accesses are safe, structural
  requirements for the BaseSkill class).
- Control schema validation.
- File formatting: wrapping a user-authored ``def run(canvas, ...)`` body
  inside a generated BaseSkill class, plus targeted in-place rewrites for
  soft-delete (``hidden``) and ownership transfer.
- Filesystem-level write/update/delete operations.

Runtime lookup, search, and embedding cache live in
``plugins.skills.helpers.skill_registry`` — keep this module free of
in-memory state so it can be called from any process (parent or child
sandbox).
"""

from __future__ import annotations

import ast
import re
import textwrap
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from paths import SANDBOX_SKILLS


# ---------------------------------------------------------------------------
# AST validation — first line of defense for the subprocess sandbox.
# ---------------------------------------------------------------------------

# Literal-only allowlist. Top-level fallback covers PIL.* etc. We explicitly
# admit "plugins.BaseSkill" as a literal so skills can import BaseSkill, but
# we do NOT admit the bare "plugins" namespace — that would let a skill reach
# into plugins.helpers.web_auth etc.
_ALLOWED_IMPORTS = {
    "math", "random", "colorsys",
    "numpy", "numpy.random",
    "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFilter",
    "PIL.ImageOps", "PIL.ImageEnhance", "PIL.ImageChops", "PIL.ImageColor",
    "plugins.BaseSkill",
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


class SkillValidationError(ValueError):
    """Raised when skill code fails AST validation."""


def _is_import_allowed(mod: str) -> bool:
    """Literal allowlist with a top-level fallback ONLY for non-`plugins` modules.

    `plugins.BaseSkill` is admitted as a literal. Any other `plugins.*` import
    is rejected to keep skills out of the rest of the helpers tree.
    """
    if mod in _ALLOWED_IMPORTS:
        return True
    if mod.startswith("plugins"):
        return False
    top = mod.split(".")[0]
    return top in _ALLOWED_IMPORTS


def _find_base_skill_class(tree: ast.Module) -> ast.ClassDef | None:
    """Return the first ClassDef whose bases name BaseSkill (Name or Attribute)."""
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseSkill":
                return node
            if isinstance(base, ast.Attribute) and base.attr == "BaseSkill":
                return node
    return None


def _find_run_method(cls_node: ast.ClassDef) -> ast.FunctionDef | None:
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "run":
            return item
    return None


def validate_skill_code(source: str) -> list[str]:
    """Return a list of violations. Empty list means the code is acceptable.

    Rules:
      * Imports limited to _ALLOWED_IMPORTS (literal or non-`plugins` top-level).
      * No references to dangerous names or dunder escape hatches.
      * No `from x import *`.
      * Must define a `class X(BaseSkill)` with a `def run(...)` method.
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

    cls_node = _find_base_skill_class(tree)
    if cls_node is None:
        errors.append("no class inheriting from BaseSkill found — every skill must define `class <Name>(BaseSkill):`")
    elif _find_run_method(cls_node) is None:
        errors.append(f"class '{cls_node.name}' must define `def run(self, canvas, **params)`")

    return errors


def assert_valid(source: str) -> None:
    errors = validate_skill_code(source)
    if errors:
        raise SkillValidationError("; ".join(errors))


# ---------------------------------------------------------------------------
# AST helpers for control-schema validation.
# ---------------------------------------------------------------------------

def _run_param_names_from_tree(tree: ast.Module) -> set[str]:
    """Extract keyword parameter names from the BaseSkill subclass's run method."""
    cls = _find_base_skill_class(tree)
    if cls is None:
        return set()
    run = _find_run_method(cls)
    if run is None:
        return set()
    args = run.args
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    names = {a.arg for a in all_args}
    names.discard("self")
    names.discard("canvas")
    return names


def extract_run_params(source: str) -> set[str]:
    """Return the keyword-param names of the BaseSkill subclass's run method."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return _run_param_names_from_tree(tree)


def _coerce_number(value, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkillValidationError(f"control field '{field}' must be a number, got {type(value).__name__}")
    return float(value)


def _code_uses_palette(code: str | None) -> bool:
    if not code:
        return False
    return ("canvas.palette" in code) or ("palette_color" in code)


MAX_NON_PALETTE_CONTROLS = 3
_CONTROL_TYPES = {"slider", "enum", "bool", "pan", "button", "palette"}
_BUTTON_ACTIONS = {"randomize"}
_KINDS = ("creation", "transform")


def validate_controls(controls, run_param_names: set[str], code: str | None = None) -> list[dict]:
    """Validate and normalize a list of control entries. Same contract as
    before the refactor — see plan / earlier docstring."""
    if controls is None:
        controls = []
    if not isinstance(controls, list):
        raise SkillValidationError("controls must be a list")
    valid_params = set(run_param_names) | {"palette", "seed"}
    seen_names: set[str] = set()
    non_palette = 0
    out: list[dict] = []
    for i, raw in enumerate(controls):
        if not isinstance(raw, dict):
            raise SkillValidationError(f"control #{i} must be a dict")
        c = dict(raw)
        ctype = c.get("type")
        if ctype not in _CONTROL_TYPES:
            raise SkillValidationError(f"control #{i} has invalid type {ctype!r}; allowed: {sorted(_CONTROL_TYPES)}")

        if ctype == "palette":
            c.setdefault("name", "palette")
            c.setdefault("label", "Palette")
        else:
            non_palette += 1
            name = c.get("name")
            if not isinstance(name, str) or not name:
                raise SkillValidationError(f"control #{i} ({ctype}) missing 'name'")
            if ctype != "pan" and name not in valid_params:
                raise SkillValidationError(
                    f"control '{name}' is not a parameter of run(); add it to the run signature"
                )
            c.setdefault("label", name.replace("_", " ").title())

        if c["name"] in seen_names:
            raise SkillValidationError(f"duplicate control name '{c['name']}'")
        seen_names.add(c["name"])

        if ctype == "slider":
            lo = _coerce_number(c.get("min"), field="min")
            hi = _coerce_number(c.get("max"), field="max")
            if hi <= lo:
                raise SkillValidationError(f"slider '{c['name']}' needs max > min")
            c["min"], c["max"] = lo, hi
            c["step"] = _coerce_number(c.get("step", (hi - lo) / 100.0), field="step")
            c["default"] = _coerce_number(c.get("default", lo), field="default")
        elif ctype == "bool":
            c["default"] = bool(c.get("default", False))
        elif ctype == "enum":
            options = c.get("options")
            if not isinstance(options, list) or not options:
                raise SkillValidationError(f"enum '{c['name']}' needs a non-empty 'options' list")
            norm_opts = []
            for j, opt in enumerate(options):
                if not isinstance(opt, dict) or "value" not in opt:
                    raise SkillValidationError(f"enum '{c['name']}' option #{j} must be a dict with 'value'")
                norm_opts.append({"value": opt["value"], "label": str(opt.get("label", opt["value"]))})
            c["options"] = norm_opts
            c.setdefault("default", norm_opts[0]["value"])
        elif ctype == "pan":
            xp, yp = c.get("x_param"), c.get("y_param")
            if not (isinstance(xp, str) and isinstance(yp, str)):
                raise SkillValidationError(f"pan '{c['name']}' needs string x_param and y_param")
            if xp not in valid_params or yp not in valid_params:
                raise SkillValidationError(f"pan '{c['name']}' references unknown run() params")
            c["step"] = _coerce_number(c.get("step", 0.1), field="step")
            xd = _coerce_number(c.get("x_default", 0.0), field="x_default")
            yd = _coerce_number(c.get("y_default", 0.0), field="y_default")
            c["x_default"], c["y_default"] = xd, yd
            c["default"] = {xp: xd, yp: yd}
        elif ctype == "button":
            action = c.get("action") or "randomize"
            if action not in _BUTTON_ACTIONS:
                raise SkillValidationError(f"button '{c['name']}' has unknown action {action!r}")
            param = c.get("param") or "seed"
            if param not in valid_params:
                raise SkillValidationError(f"button '{c['name']}' targets unknown param {param!r}")
            c["action"] = action
            c["param"] = param
        out.append(c)

    if non_palette > MAX_NON_PALETTE_CONTROLS:
        raise SkillValidationError(
            f"a skill may declare at most {MAX_NON_PALETTE_CONTROLS} non-palette controls (got {non_palette})"
        )
    if _code_uses_palette(code) and not any(c.get("type") == "palette" for c in out):
        out.insert(0, {"type": "palette", "name": "palette", "label": "Palette"})
    return out


# ---------------------------------------------------------------------------
# Skill dataclass — runner-facing DTO.
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("_", (name or "").strip().lower()).strip("_")
    return slug or "untitled"


def class_name_for_slug(slug: str) -> str:
    """PascalCase + 'Skill' suffix. Matches naming used by build_plugin tools."""
    parts = [p for p in slug.split("_") if p]
    body = "".join(p[:1].upper() + p[1:] for p in parts) or "Untitled"
    return body + "Skill"


@dataclass
class Skill:
    """Runner-facing DTO. Populated from a BaseSkill instance via
    SkillRegistry, or directly from a source file for tools that need to
    read a skill without instantiating it."""
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
    return SANDBOX_SKILLS / f"skill_{slug}.py"


def _built_in_path_for(slug: str) -> Path:
    from paths import ROOT_DIR
    return ROOT_DIR / "plugins" / "skills" / f"skill_{slug}.py"


def is_built_in(path: str | Path) -> bool:
    from paths import ROOT_DIR
    try:
        return (ROOT_DIR / "plugins" / "skills").resolve() in Path(path).resolve().parents
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
# DTO from a BaseSkill instance.
# ---------------------------------------------------------------------------

def to_skill_record(instance) -> Skill:
    """Build a runner-facing Skill from a registered BaseSkill instance.

    The class is the source of truth for metadata; ``code`` is read from
    ``_source_path`` (set by plugin_discovery at load time)."""
    source_path = Path(getattr(instance, "_source_path", "") or "")
    code = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
    slug = slugify(getattr(instance, "name", "") or source_path.stem.removeprefix("skill_"))
    return Skill(
        slug=slug,
        path=str(source_path.resolve()) if source_path else "",
        name=str(getattr(instance, "name", "") or slug),
        description=str(getattr(instance, "description", "") or ""),
        kind=str(getattr(instance, "kind", "") or "creation"),
        owner=str(getattr(instance, "owner", "") or ""),
        code=code,
        created_at=float(getattr(instance, "created_at", 0.0) or _coerce_created_at(None, source_path)),
        controls=list(getattr(instance, "controls", []) or []),
        hidden=bool(getattr(instance, "hidden", False)),
    )


# ---------------------------------------------------------------------------
# File formatting — wrap user-authored body in a BaseSkill class.
# ---------------------------------------------------------------------------

_DEF_RUN_RE = re.compile(r"\bdef\s+run\s*\(")


def _split_imports_and_body(user_code: str) -> tuple[list[str], list[str]]:
    """Walk the user's source AST. Return (import_blocks, body_blocks).

    Imports stay at module level. Everything else becomes part of the class
    body (after the metadata block). The `def run(canvas, ...)` function is
    rewritten to `def run(self, canvas, ...)` and indented one level.
    """
    tree = ast.parse(user_code)
    imports: list[str] = []
    body: list[str] = []
    for node in tree.body:
        seg = ast.get_source_segment(user_code, node)
        if seg is None:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(seg)
        elif isinstance(node, ast.FunctionDef) and node.name == "run":
            indented = textwrap.indent(seg, "    ")
            indented = _DEF_RUN_RE.sub("def run(self, ", indented, count=1)
            body.append(indented)
        else:
            body.append(textwrap.indent(seg, "    "))
    return imports, body


def wrap_user_code_in_class(*, class_name: str, name: str, description: str,
                             kind: str, owner: str, created_at: float,
                             controls: list, hidden: bool, user_code: str) -> str:
    """Generate the file source for a new skill.

    Takes the user-supplied module-level ``def run(canvas, ...)`` (plus any
    imports the user added) and wraps it inside a generated BaseSkill
    subclass with metadata as class attributes.
    """
    imports, body = _split_imports_and_body(user_code)
    if not any(_DEF_RUN_RE.search(b) for b in body):
        raise SkillValidationError("user code must contain `def run(canvas, ...)`")

    parts: list[str] = ["from plugins.BaseSkill import BaseSkill"]
    if imports:
        parts.append("\n".join(imports))
    header = "\n\n".join(parts) + "\n\n\n"

    attrs = (
        f"    name = {name!r}\n"
        f"    description = {description!r}\n"
        f"    kind = {kind!r}\n"
        f"    owner = {owner!r}\n"
        f"    created_at = {float(created_at)!r}\n"
        f"    hidden = {bool(hidden)!r}\n"
    )
    if controls:
        attrs += f"    controls = {controls!r}\n"

    return f"{header}class {class_name}(BaseSkill):\n{attrs}\n" + "\n\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# In-place class-body rewrites for soft-delete + ownership transfer.
# ---------------------------------------------------------------------------

def _replace_class_attr(source: str, attr: str, new_value_repr: str) -> str:
    """Replace `<attr> = ...` inside the BaseSkill subclass body. If the
    assignment isn't present, insert it after the last existing class-attr
    assignment (so it sits with the rest of the metadata)."""
    tree = ast.parse(source)
    cls = _find_base_skill_class(tree)
    if cls is None:
        raise SkillValidationError("no BaseSkill subclass found")

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

def write_skill(
    *, name: str, description: str, kind: str, owner: str, code: str,
    controls: list | None = None,
) -> tuple[Skill, Path]:
    """Create a new sandbox skill. Returns (skill_dto, file_path).

    Validates the user's code (which must contain ``def run(canvas, ...)``),
    wraps it in a generated BaseSkill subclass, and writes the file. The
    caller is responsible for asking the registry to load it.
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
        raise FileExistsError(f"a skill named '{slug}' already exists")

    normalized_controls = validate_controls(controls, extract_run_params(code) or _run_param_names_in_body(code), code=code)
    cls_name = class_name_for_slug(slug)
    created_at = time.time()
    file_source = wrap_user_code_in_class(
        class_name=cls_name, name=name, description=description, kind=kind,
        owner=owner or "", created_at=created_at,
        controls=normalized_controls, hidden=False, user_code=code,
    )
    # Validate the final generated file (not just the user's body).
    assert_valid(file_source)

    SANDBOX_SKILLS.mkdir(parents=True, exist_ok=True)
    sandbox_path.write_text(file_source, encoding="utf-8")

    skill = Skill(
        slug=slug, path=str(sandbox_path.resolve()),
        name=name, description=description,
        kind=kind, owner=owner or "", code=file_source,
        created_at=created_at, controls=normalized_controls,
    )
    return skill, sandbox_path


def _run_param_names_in_body(user_code: str) -> set[str]:
    """Extract `def run(canvas, ...)` params from the user's pre-wrap body."""
    try:
        tree = ast.parse(user_code)
    except SyntaxError:
        return set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            args = node.args
            all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
            names = {a.arg for a in all_args}
            names.discard("canvas")
            return names
    return set()


def rewrite_skill(
    path: Path, *, owner_session_key: str,
    name: str | None = None, description: str | None = None,
    code: str | None = None, controls: list | None = None,
) -> Skill:
    """Update an existing sandbox skill in place. Re-wraps the user's body
    when ``code`` is provided. Built-in skills are read-only; raises
    PermissionError if the target is built-in or owner-mismatched.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no skill file at {path}")
    if is_built_in(path):
        raise PermissionError(f"'{path.name}' is a built-in skill and is read-only. Create a new skill to fork it.")

    # Read existing metadata via a temporary module load done via AST only —
    # we never exec the file in this process.
    existing_src = path.read_text(encoding="utf-8")
    existing = _read_class_metadata(existing_src)
    if existing.get("owner") and existing["owner"] != owner_session_key:
        raise PermissionError(f"only owner '{existing['owner']}' can update this skill")

    new_name = (name if name is not None else existing.get("name") or "").strip()
    new_desc = (description if description is not None else existing.get("description") or "").strip()
    new_code = code  # user-supplied body, or None to keep current
    new_kind = existing.get("kind") or "creation"
    new_owner = existing.get("owner") or owner_session_key
    new_created_at = float(existing.get("created_at") or time.time())
    new_hidden = bool(existing.get("hidden", False))

    if new_code is None:
        # Keep the existing run-method body. Re-emit attrs only.
        new_controls = validate_controls(
            controls if controls is not None else existing.get("controls") or [],
            extract_run_params(existing_src), code=existing_src,
        )
        out_source = _rewrite_metadata_only(
            existing_src,
            name=new_name, description=new_desc, kind=new_kind,
            owner=new_owner, created_at=new_created_at,
            controls=new_controls, hidden=new_hidden,
        )
    else:
        new_controls = validate_controls(
            controls if controls is not None else existing.get("controls") or [],
            _run_param_names_in_body(new_code), code=new_code,
        )
        cls_name = class_name_for_slug(path.stem.removeprefix("skill_"))
        out_source = wrap_user_code_in_class(
            class_name=cls_name, name=new_name, description=new_desc, kind=new_kind,
            owner=new_owner, created_at=new_created_at,
            controls=new_controls, hidden=new_hidden, user_code=new_code,
        )

    assert_valid(out_source)
    path.write_text(out_source, encoding="utf-8")

    slug = path.stem.removeprefix("skill_")
    return Skill(
        slug=slug, path=str(path.resolve()),
        name=new_name, description=new_desc,
        kind=new_kind, owner=new_owner, code=out_source,
        created_at=new_created_at, controls=new_controls, hidden=new_hidden,
    )


def soft_delete_skill(path: Path, *, owner_session_key: str) -> bool:
    """Flip ``hidden = True`` in the class body. Returns False if path missing."""
    path = Path(path)
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8")
    meta = _read_class_metadata(src)
    if (meta.get("owner")
            and meta["owner"] != owner_session_key
            and not is_built_in(path)):
        raise PermissionError(f"only owner '{meta['owner']}' can hide this skill")
    if meta.get("hidden"):
        return True
    path.write_text(set_hidden_in_source(src, True), encoding="utf-8")
    return True


def anonymize_owner_in_dir(directory: Path, owner_values) -> int:
    """Rewrite ``owner`` to 'anonymous' in every skill file under *directory*
    whose owner matches one of ``owner_values`` (case-insensitive). Returns
    the number of files rewritten."""
    targets = {str(v).strip().lower() for v in owner_values if v}
    if not targets or not directory.exists():
        return 0
    rewritten = 0
    for path in directory.glob("skill_*.py"):
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

_META_FIELDS = {"name", "description", "kind", "owner", "created_at", "controls", "hidden"}


def _read_class_metadata(source: str) -> dict:
    """Extract BaseSkill class-attribute metadata using literal_eval only."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    cls = _find_base_skill_class(tree)
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
                            owner: str, created_at: float, controls: list, hidden: bool) -> str:
    """Replace each metadata attr in place; leaves body code untouched."""
    out = source
    out = _replace_class_attr(out, "name", repr(name))
    out = _replace_class_attr(out, "description", repr(description))
    out = _replace_class_attr(out, "kind", repr(kind))
    out = _replace_class_attr(out, "owner", repr(owner))
    out = _replace_class_attr(out, "created_at", repr(float(created_at)))
    out = _replace_class_attr(out, "hidden", repr(bool(hidden)))
    if controls:
        out = _replace_class_attr(out, "controls", repr(controls))
    return out
