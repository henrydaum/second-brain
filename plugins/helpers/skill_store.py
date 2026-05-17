"""Skill storage, AST validation, and in-memory vector search.

A skill is one Python file under DATA_DIR/skills/<slug>.skill.py with
module-level metadata constants (SKILL_NAME, SKILL_DESCRIPTION, SKILL_KIND,
SKILL_OWNER, SKILL_CREATED_AT) and a `def run(canvas, **params)` function.

Skills are embedded synchronously by the create/update tools using the text
embedder service. Cosine similarity search runs in memory — the catalog is
small (dozens) and embedding lookups are fast.
"""

from __future__ import annotations

import ast
import json
import re
import threading
import time
from array import array
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from paths import SKILLS_DIR, BUILT_IN_SKILLS_DIR


# ---------------------------------------------------------------------------
# AST validation — the first line of defense for the subprocess sandbox.
# ---------------------------------------------------------------------------

_ALLOWED_IMPORTS = {
    "math", "random", "colorsys",
    "numpy", "numpy.random",
    "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFilter",
    "PIL.ImageOps", "PIL.ImageEnhance", "PIL.ImageChops", "PIL.ImageColor",
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


def validate_skill_code(source: str) -> list[str]:
    """Return a list of violations. Empty list means the code is acceptable.

    Rules:
      * imports limited to _ALLOWED_IMPORTS
      * no references to dangerous names or dunder escape hatches
      * no `from x import *`
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    errors: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS and alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    errors.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0]
            if mod not in _ALLOWED_IMPORTS and top not in _ALLOWED_IMPORTS:
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

    return errors


def assert_valid(source: str) -> None:
    errors = validate_skill_code(source)
    if errors:
        raise SkillValidationError("; ".join(errors))


def _run_param_names(source: str) -> set[str]:
    """Extract the keyword parameter names from the skill's `def run(canvas, ...)`."""
    try:
        tree = ast.parse(source)
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


def _coerce_number(value, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkillValidationError(f"control field '{field}' must be a number, got {type(value).__name__}")
    return float(value)


def validate_controls(controls, run_param_names: set[str]) -> list[dict]:
    """Validate and normalize a list of SKILL_CONTROLS entries.

    Returns the normalized list. Raises SkillValidationError on any problem.
    The 'palette' and 'seed' names are always considered valid param targets —
    'palette' is resolved by the runner, 'seed' lives on the chain entry.
    """
    if controls in (None, []):
        return []
    if not isinstance(controls, list):
        raise SkillValidationError("SKILL_CONTROLS must be a list")
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
        # palette has no type-specific fields.
        out.append(c)

    if non_palette > MAX_NON_PALETTE_CONTROLS:
        raise SkillValidationError(
            f"a skill may declare at most {MAX_NON_PALETTE_CONTROLS} non-palette controls (got {non_palette})"
        )
    return out


def extract_run_params(source: str) -> set[str]:
    """Public alias for the AST-based run() parameter extractor."""
    return _run_param_names(source)


# ---------------------------------------------------------------------------
# Skill files: metadata extraction, naming.
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9_]+")
_KINDS = ("creation", "transform")

# Caps and types for user-facing skill controls. Palette doesn't count toward
# the cap because it's a universal, well-understood control.
MAX_NON_PALETTE_CONTROLS = 3
_CONTROL_TYPES = {"slider", "enum", "bool", "pan", "button", "palette"}
_BUTTON_ACTIONS = {"randomize"}


@dataclass
class Skill:
    slug: str
    path: str
    name: str
    description: str
    kind: str
    owner: str
    code: str
    created_at: float
    controls: list = None  # list[dict] — user-facing controls; empty when none declared

    def __post_init__(self):
        if self.controls is None:
            self.controls = []

    def to_dict(self) -> dict:
        return asdict(self)


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("_", (name or "").strip().lower()).strip("_")
    return slug or "untitled"


def _path_for(slug: str) -> Path:
    """Writable path for a slug. Writes always go to the sandbox dir; the
    built-in directory is read-only from the store's perspective."""
    return SKILLS_DIR / f"{slug}.skill.py"


def _is_built_in(path: Path) -> bool:
    try:
        return BUILT_IN_SKILLS_DIR.resolve() in Path(path).resolve().parents
    except Exception:
        return False


def _iter_skill_paths() -> Iterable[Path]:
    """Yield every skill file from both the sandbox and built-in dirs.

    Sandbox files come first so collision-dedupe (by slug) keeps the user/agent
    override when one exists.
    """
    seen: set[str] = set()
    for root in (SKILLS_DIR, BUILT_IN_SKILLS_DIR):
        for path in root.glob("*.skill.py"):
            slug = path.name.rsplit(".skill.py", 1)[0]
            if slug in seen:
                continue
            seen.add(slug)
            yield path


def _format_skill_file(skill: Skill) -> str:
    controls_line = ""
    if skill.controls:
        # repr() on lists of dicts of primitives round-trips through ast.literal_eval.
        controls_line = f'SKILL_CONTROLS = {skill.controls!r}\n'
    return (
        f'SKILL_NAME = {json.dumps(skill.name)}\n'
        f'SKILL_DESCRIPTION = {json.dumps(skill.description)}\n'
        f'SKILL_KIND = {json.dumps(skill.kind)}\n'
        f'SKILL_OWNER = {json.dumps(skill.owner)}\n'
        f'SKILL_CREATED_AT = {skill.created_at!r}\n'
        f'{controls_line}'
        f'\n'
        f'{skill.code.rstrip()}\n'
    )


def _extract_metadata(source: str) -> dict:
    """AST-only metadata extraction (no exec)."""
    tree = ast.parse(source)
    out: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("SKILL_NAME", "SKILL_DESCRIPTION", "SKILL_KIND", "SKILL_OWNER", "SKILL_CREATED_AT", "SKILL_CONTROLS"):
                try:
                    out[name] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return out


def metadata_from_source(source: str) -> dict:
    return _extract_metadata(source)


def _strip_metadata(source: str) -> str:
    """Return the source with the leading SKILL_* assignments removed (for round-tripping)."""
    tree = ast.parse(source)
    drop_lines = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id.startswith("SKILL_"):
                end_lineno = getattr(node, "end_lineno", node.lineno)
                for ln in range(node.lineno, end_lineno + 1):
                    drop_lines.add(ln)
    lines = source.splitlines()
    kept = [line for i, line in enumerate(lines, start=1) if i not in drop_lines]
    return "\n".join(kept).lstrip("\n")


def _load_skill_from_path(path: Path) -> Skill | None:
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        meta = _extract_metadata(source)
    except Exception:
        return None
    if "SKILL_NAME" not in meta or "SKILL_KIND" not in meta:
        return None
    code_body = _strip_metadata(source)
    slug = path.name.rsplit(".skill.py", 1)[0]
    raw_controls = meta.get("SKILL_CONTROLS")
    raw_list = list(raw_controls) if isinstance(raw_controls, list) else []
    # Normalize on load so hand-written skills get the same defaulting as
    # ones created via create_skill -- notably, palette controls without an
    # explicit name get name="palette" so schema lookups work.
    try:
        controls = validate_controls(raw_list, _run_param_names(code_body))
    except SkillValidationError:
        controls = raw_list
    return Skill(
        slug=slug,
        path=str(path.resolve()),
        name=str(meta.get("SKILL_NAME") or slug),
        description=str(meta.get("SKILL_DESCRIPTION") or ""),
        kind=str(meta.get("SKILL_KIND") or "creation"),
        owner=str(meta.get("SKILL_OWNER") or ""),
        code=code_body,
        created_at=float(meta.get("SKILL_CREATED_AT") or path.stat().st_mtime),
        controls=controls,
    )


# ---------------------------------------------------------------------------
# Embedding cache (in-memory, populated on demand).
# ---------------------------------------------------------------------------

_emb_lock = threading.RLock()
# slug -> {"mtime": float, "vector": list[float], "name": str, "description": str, "kind": str, "owner": str}
_emb_cache: dict[str, dict] = {}


def _embed_text(text_embedder, text: str) -> list[float] | None:
    if not text_embedder or not getattr(text_embedder, "loaded", False):
        try:
            text_embedder.load() if text_embedder else None
        except Exception:
            return None
    if not text_embedder or not getattr(text_embedder, "loaded", False):
        return None
    try:
        vec = text_embedder.encode([text])[0]
        return [float(x) for x in vec]
    except Exception:
        return None


def _ensure_embeddings(text_embedder) -> None:
    """Reconcile the in-memory embedding cache against on-disk skills."""
    with _emb_lock:
        existing_slugs = set()
        for path in _iter_skill_paths():
            skill = _load_skill_from_path(path)
            if skill is None:
                continue
            existing_slugs.add(skill.slug)
            cached = _emb_cache.get(skill.slug)
            mtime = path.stat().st_mtime
            if cached and cached.get("mtime") == mtime:
                continue
            vec = _embed_text(text_embedder, f"{skill.name}\n{skill.description}")
            if vec is None:
                continue
            _emb_cache[skill.slug] = {
                "mtime": mtime,
                "vector": vec,
                "name": skill.name,
                "description": skill.description,
                "kind": skill.kind,
                "owner": skill.owner,
            }
        # Drop cache entries for deleted skills.
        for slug in list(_emb_cache):
            if slug not in existing_slugs:
                _emb_cache.pop(slug, None)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    # Inputs from the text embedder are already L2-normalized.
    return float(sum(x * y for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Public API used by the tools.
# ---------------------------------------------------------------------------


def list_skills() -> list[Skill]:
    out: list[Skill] = []
    for path in _iter_skill_paths():
        skill = _load_skill_from_path(path)
        if skill is not None:
            out.append(skill)
    out.sort(key=lambda s: s.created_at, reverse=True)
    return out


def read_skill(slug: str) -> Skill | None:
    sandbox = _path_for(slug)
    if sandbox.is_file():
        return _load_skill_from_path(sandbox)
    built_in = BUILT_IN_SKILLS_DIR / f"{slug}.skill.py"
    if built_in.is_file():
        return _load_skill_from_path(built_in)
    return None


def write_skill(
    *, name: str, description: str, kind: str, owner: str, code: str,
    controls: list | None = None,
    text_embedder=None,
) -> Skill:
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    assert_valid(code)
    normalized_controls = validate_controls(controls, _run_param_names(code))
    slug = slugify(name)
    if not slug:
        raise ValueError("name produced an empty slug")
    path = _path_for(slug)
    if path.exists():
        raise FileExistsError(f"a skill named '{slug}' already exists")
    skill = Skill(
        slug=slug, path=str(path.resolve()),
        name=name.strip(), description=description.strip(),
        kind=kind, owner=owner or "",
        code=code, created_at=time.time(),
        controls=normalized_controls,
    )
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(_format_skill_file(skill), encoding="utf-8")
    if text_embedder is not None:
        _ensure_embeddings(text_embedder)
    return skill


def update_skill(
    slug: str, *, owner: str, name: str | None = None, description: str | None = None,
    code: str | None = None, controls: list | None = None,
    text_embedder=None,
) -> Skill:
    existing = read_skill(slug)
    if existing is None:
        raise FileNotFoundError(f"no skill named '{slug}'")
    if _is_built_in(existing.path):
        raise PermissionError(
            f"'{slug}' is a built-in skill and is read-only. Create a new skill "
            f"(e.g. '{slug}_v2') to fork it."
        )
    if existing.owner and existing.owner != owner:
        raise PermissionError(f"only owner '{existing.owner}' can update this skill")
    new_name = name if name is not None else existing.name
    new_desc = description if description is not None else existing.description
    new_code = code if code is not None else existing.code
    if code is not None:
        assert_valid(new_code)
    if controls is not None:
        new_controls = validate_controls(controls, _run_param_names(new_code))
    else:
        new_controls = list(existing.controls or [])
    updated = Skill(
        slug=existing.slug, path=existing.path,
        name=new_name.strip(), description=new_desc.strip(),
        kind=existing.kind, owner=existing.owner,
        code=new_code, created_at=existing.created_at,
        controls=new_controls,
    )
    Path(existing.path).write_text(_format_skill_file(updated), encoding="utf-8")
    if text_embedder is not None:
        _emb_cache.pop(slug, None)
        _ensure_embeddings(text_embedder)
    return updated


def delete_skill(slug: str, *, owner: str) -> bool:
    existing = read_skill(slug)
    if existing is None:
        return False
    if _is_built_in(existing.path):
        raise PermissionError(f"'{slug}' is a built-in skill and cannot be deleted.")
    if existing.owner and existing.owner != owner:
        raise PermissionError(f"only owner '{existing.owner}' can delete this skill")
    try:
        Path(existing.path).unlink()
    except FileNotFoundError:
        pass
    with _emb_lock:
        _emb_cache.pop(slug, None)
    return True


def search_skills(
    query: str, *, top_k: int = 5, kind: str | None = None, text_embedder=None,
) -> list[dict]:
    _ensure_embeddings(text_embedder)
    qvec = _embed_text(text_embedder, query) if text_embedder else None
    if qvec is None:
        # Fall back to lexical substring scoring so the tool still works without an embedder.
        q = (query or "").lower()
        out: list[dict] = []
        for skill in list_skills():
            if kind and skill.kind != kind:
                continue
            score = 0.0
            haystack = f"{skill.name}\n{skill.description}".lower()
            if q and q in haystack:
                score = 1.0
            out.append({"slug": skill.slug, "name": skill.name, "description": skill.description,
                        "kind": skill.kind, "owner": skill.owner, "score": score})
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:top_k]
    rows: list[dict] = []
    with _emb_lock:
        for slug, entry in _emb_cache.items():
            if kind and entry["kind"] != kind:
                continue
            rows.append({
                "slug": slug, "name": entry["name"], "description": entry["description"],
                "kind": entry["kind"], "owner": entry["owner"],
                "score": _cosine(qvec, entry["vector"]),
            })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_k]


def warm_embeddings(text_embedder) -> None:
    """Public entry-point used by callers that want to pre-populate the cache."""
    _ensure_embeddings(text_embedder)
