"""The validation script — a conformance linter, not a security boundary.

It reads a plugin file with :mod:`ast` and never imports it, so checking a file
cannot run it. What it catches is *carelessness*: reaching for the environment
directly instead of asking, declaring a name that already exists, breaking the
plugin contract. That is the whole threat model — an agent with good intentions
and no judgement.

**It cannot be perfect and does not need to be.** Static analysis of Python is
defeatable by anyone trying (``getattr(__builtins__, "op" + "en")`` and a dozen
cousins), which is why the subprocess exists. Against code that is merely
thoughtless, a linter with good error messages is the right tool — and the
error messages are the point. Every finding names the Request that should have
been used, because the feedback loop is what makes the constraint teachable
instead of a wall the agent keeps walking into.

Three severities:

- ``ERROR``   — will not load. A direct effect, or a broken contract.
- ``WARNING`` — loads with a disclaimer. A foreign library cannot be validated
  (it may be binary, or not Python at all), so its actions cannot be turned
  into Requests. Subprocess it and say so.
- ``NOTE``    — advisory. A declared value above the kernel's ceiling still
  works; it just gets clamped.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

ERROR = "error"
WARNING = "warning"
NOTE = "note"

FAMILIES = {"tool": "BaseTool", "task": "BaseTask", "service": "BaseService",
            "command": "BaseCommand", "frontend": "BaseFrontend"}

# The contract itself. A plugin has to import its base class to subclass it,
# and these carry declarations rather than behaviour. The guest package is the
# new home; the legacy ``plugins.Base*`` modules stay allowed while the two
# trees coexist.
BASE_TO_FAMILY = {base: family for family, base in FAMILIES.items()}

CONTRACT_MODULES = (
    {f"plugins.{base}" for base in FAMILIES.values()}
    | {"guest", "guest.bases", "guest.box", "guest.sdk",
       "sandbox.guest", "sandbox.guest.bases", "sandbox.guest.box"}
)

# Pure stdlib: computation only, no way to reach the environment.
PURE_MODULES = {
    "__future__", "abc", "base64", "binascii", "bisect", "calendar",
    "collections", "colorsys", "copy", "dataclasses", "datetime", "decimal",
    "difflib", "enum", "fractions", "functools", "hashlib", "heapq", "hmac",
    "html", "itertools", "json", "math", "numbers", "operator", "random",
    "re", "statistics", "string", "textwrap", "types", "typing", "unicodedata",
    "urllib.parse", "uuid", "zoneinfo",
}

# Reaching for the environment directly. Each maps to the Request that does
# the same job through the gate.
EFFECT_MODULES = {
    "os": "sdk.fs / sdk.env",
    "io": "sdk.fs",
    "sys": "the SDK",
    "shutil": "sdk.fs.move / sdk.fs.delete",
    "pathlib": "sdk.fs (Path is fine for building paths, not for touching them)",
    "tempfile": "sdk.fs.temp",
    "subprocess": "sdk.proc.run",
    "socket": "sdk.net.http",
    "urllib": "sdk.net.http",
    "urllib.request": "sdk.net.http",
    "http": "sdk.net.http",
    "requests": "sdk.net.http",
    "httpx": "sdk.net.http",
    "sqlite3": "sdk.db.query / sdk.db.write",
    "threading": "the kernel schedules; a plugin should not",
    "multiprocessing": "the kernel schedules; a plugin should not",
    "ctypes": "nothing — this defeats the boundary entirely",
    "importlib": "sdk.plugin.register",
    "logging": "sdk.log",
}

BANNED_BUILTINS = {
    "open": "sdk.fs.read / sdk.fs.write",
    "eval": "nothing — build the value directly",
    "exec": "nothing — build the value directly",
    "compile": "nothing",
    "__import__": "sdk.plugin.register",
    "input": "sdk.ui.ask",
    "breakpoint": "nothing",
}

# Effect-shaped method names. Heuristic by nature: a linter, not a proof.
EFFECT_METHODS = {
    "read_text": "sdk.fs.read", "read_bytes": "sdk.fs.read",
    "write_text": "sdk.fs.write", "write_bytes": "sdk.fs.write",
    "unlink": "sdk.fs.delete", "rmdir": "sdk.fs.delete",
    "mkdir": "sdk.fs.write", "rename": "sdk.fs.move",
    "iterdir": "sdk.fs.list", "walk": "sdk.fs.list",
}

# Ceilings mirrored from the interpreter. Exceeding one is not an error —
# the plugin declares intent, the kernel clamps.
CEILINGS = {"load_timeout": 600.0, "timeout": 600.0, "max_calls": 25,
            "memory_mb": 4096, "batch_size": 1000}

# Declarations the kernel reads without importing, so they must be literals.
LITERAL_LISTS = ("dependencies_files", "dependencies_pip",
                 "requires_services", "requests", "exports")
LITERAL_STRINGS = ("name", "box", "isolation", "lifetime")

# Closed vocabularies. A typo here is silent otherwise: an unrecognised
# isolation reads as "unset" and the file quietly gets the default.
ENUMS = {"isolation": {"", "in_process", "subprocess"},
         "lifetime": {"", "ephemeral", "persistent"}}


@dataclass(frozen=True)
class Finding:
    """One thing wrong, and what to do instead."""
    level: str
    line: int
    message: str
    fix: str = ""

    def render(self) -> str:
        """One line, aimed at whoever has to fix it."""
        tail = f"  Use {self.fix} instead." if self.fix else ""
        return f"  line {self.line}: {self.message}{tail}"


@dataclass(frozen=True)
class Report:
    """The verdict on one file, carrying the bytes it was reached on.

    ``source`` is not a convenience: validating a *path* and then opening it
    again to run it means the code that ran is not the code that passed. The
    caller executes what is here.
    """
    filename: str
    findings: tuple = field(default_factory=tuple)
    source: str = ""
    declarations: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the file may load."""
        return not any(f.level == ERROR for f in self.findings)

    @property
    def disclaimed(self) -> bool:
        """Whether it loads, but with something the user should be told."""
        return any(f.level == WARNING for f in self.findings)

    def of(self, level: str) -> list:
        """Findings at one severity."""
        return [f for f in self.findings if f.level == level]

    def render(self) -> str:
        """The message handed back to the plugin's author."""
        if not self.findings:
            return f"{self.filename}: conforms."
        lines = []
        for level, heading in ((ERROR, "Will not load"),
                               (WARNING, "Loads with a disclaimer"),
                               (NOTE, "Advisory")):
            group = self.of(level)
            if group:
                lines.append(f"{heading}:")
                lines.extend(f.render() for f in group)
        return f"{self.filename}\n" + "\n".join(lines)


def _module_root(name: str) -> str:
    """Longest matching prefix present in the effect table, else the root."""
    if name in EFFECT_MODULES or name in PURE_MODULES:
        return name
    return name.split(".")[0]


class _Walker(ast.NodeVisitor):
    """Collects findings from one parsed file."""

    def __init__(self):
        self.findings: list[Finding] = []
        self.classes: list[ast.ClassDef] = []

    def add(self, level, node, message, fix=""):
        """Record a finding."""
        self.findings.append(
            Finding(level, getattr(node, "lineno", 0), message, fix))

    # ── imports ────────────────────────────────────────────────────

    def _check_module(self, name: str, node):
        """Classify one imported module."""
        if name in CONTRACT_MODULES:
            return
        key = _module_root(name)
        if key in PURE_MODULES or name in PURE_MODULES:
            return
        if key in EFFECT_MODULES:
            self.add(ERROR, node, f"imports {name!r}, which reaches the "
                                  f"environment directly", EFFECT_MODULES[key])
            return
        self.add(WARNING, node,
                 f"imports {name!r}, a foreign library. Its actions cannot be "
                 f"turned into Requests, so they are not mediated — run this "
                 f"plugin in a subprocess")

    def visit_Import(self, node):
        """import x"""
        for alias in node.names:
            self._check_module(alias.name, node)

    def visit_ImportFrom(self, node):
        """from x import y"""
        if node.level:
            return  # relative: a sibling plugin file, validated on its own
        if node.module:
            self._check_module(node.module, node)

    # ── calls and attributes ───────────────────────────────────────

    def visit_Call(self, node):
        """Direct calls to banned builtins."""
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in BANNED_BUILTINS:
            self.add(ERROR, node, f"calls {fn.id}()",
                     BANNED_BUILTINS[fn.id])
        elif isinstance(fn, ast.Attribute) and fn.attr in EFFECT_METHODS:
            if not self._is_sdk_call(fn):
                self.add(ERROR, node, f"calls .{fn.attr}(), which touches the "
                                      f"environment directly",
                         EFFECT_METHODS[fn.attr])
        self.generic_visit(node)

    @staticmethod
    def _is_sdk_call(attribute: ast.Attribute) -> bool:
        """Whether an attribute chain is rooted at the sdk handle."""
        node = attribute
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) and node.id == "sdk"

    def visit_ClassDef(self, node):
        """Remember plugin classes for the contract check."""
        self.classes.append(node)
        self.generic_visit(node)


def _literal(node):
    """Evaluate a literal, or return None if it is not one."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _plugin_classes(walker: _Walker):
    """Every class subclassing a known plugin base, with the family it names."""
    found = []
    for node in walker.classes:
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in BASE_TO_FAMILY:
                found.append((BASE_TO_FAMILY[base.id], base.id, node))
                break
    return found


def _check_contract(tree, walker: _Walker, filename: str, known_names):
    """Check the BasePlugin contract, declared metadata, and name collisions."""
    stem = Path(filename).stem
    prefix = stem.split("_")[0] if "_" in stem else ""
    declared = _plugin_classes(walker)

    if not declared:
        # A file *named* as a plugin has to be one. Anything else is a helper
        # or a script: effect checks only, no contract.
        if prefix in FAMILIES:
            walker.add(ERROR, tree.body[0] if tree.body else tree,
                       f"{stem}.py is named as a {prefix} plugin but no class "
                       f"subclasses {FAMILIES[prefix]}")
        return

    family, base, cls = declared[0]
    if len(declared) > 1:
        walker.add(ERROR, declared[1][2],
                   f"declares {len(declared)} plugin classes; a plugin file "
                   f"must declare exactly one")

    # Discovery is by file presence, so the filename *is* the declaration of
    # what family a file belongs to. A mismatch means the plugin silently
    # never loads.
    wanted = f"{family}_"
    if not stem.startswith(wanted) or len(stem) <= len(wanted):
        walker.add(ERROR, cls,
                   f"{stem}.py subclasses {base}, so it must be named "
                   f"{family}_<name>.py — discovery finds plugins by filename",
                   f"a {wanted}* filename")
    assigned = {}
    for item in cls.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = item
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            assigned[item.target.id] = item

    # The declared name: must exist, be a literal, and not already be taken.
    if "name" not in assigned:
        walker.add(ERROR, cls, f"{base} subclass declares no 'name'")
    else:
        node = assigned["name"]
        value = _literal(node.value)
        if not isinstance(value, str) or not value.strip():
            walker.add(ERROR, node,
                       "'name' must be a non-empty string literal — it is read "
                       "without importing the file")
        elif value in set(known_names):
            walker.add(ERROR, node, f"name {value!r} is already registered",
                       "a different name")

    # Declarations are read by AST, so they have to be literals.
    for key in LITERAL_LISTS:
        if key in assigned:
            value = _literal(assigned[key].value)
            if not isinstance(value, list) or not all(
                    isinstance(v, str) for v in value):
                walker.add(ERROR, assigned[key],
                           f"{key} must be a literal list of strings — the "
                           f"kernel reads it without importing")

    # Closed vocabularies: a typo would otherwise read as "unset".
    for key, allowed in ENUMS.items():
        if key in assigned:
            value = _literal(assigned[key].value)
            if value not in allowed:
                walker.add(ERROR, assigned[key],
                           f"{key}={value!r} is not one of "
                           f"{sorted(a for a in allowed if a)}")

    # Self-declared authority: allowed, then clamped.
    for key, ceiling in CEILINGS.items():
        if key in assigned:
            value = _literal(assigned[key].value)
            if isinstance(value, (int, float)) and value > ceiling:
                walker.add(NOTE, assigned[key],
                           f"{key}={value:g} exceeds the kernel ceiling of "
                           f"{ceiling:g} and will be clamped")


DECLARATION_KEYS = ("name", "box", "isolation", "lifetime", "timeout",
                    "memory_mb", "requests", "exports", "dependencies_files",
                    "dependencies_pip", "requires_services", "max_calls",
                    "background_safe")

# Reading declarations without importing means *inherited* defaults are
# invisible: ``class Counter(BaseService)`` never writes ``lifetime`` in the
# file, so the AST cannot see that BaseService sets it. Anything the kernel
# must know therefore has to be written in the file or derivable from the
# family — and the family is visible, because the base class is named in the
# source. Services and frontends hold state between calls by definition.
FAMILY_DEFAULTS = {
    "service": {"lifetime": "persistent"},
    "frontend": {"lifetime": "persistent"},
}


def _assignments(body):
    """Literal name = value assignments in one class or module body."""
    found = {}
    for item in body:
        targets = []
        if isinstance(item, ast.Assign):
            targets = [t for t in item.targets if isinstance(t, ast.Name)]
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target,
                                                            ast.Name):
            targets = [item.target]
        for target in targets:
            found[target.id] = item
    return found


def _collect_declarations(tree, walker: _Walker, filename: str) -> dict:
    """Read a file's declarations without importing it.

    Module level first, then the plugin class on top — so a helper can declare
    ``box`` at module scope and a plugin can declare everything on its class.
    """
    declared = {}
    scopes = [_assignments(tree.body)]
    classes = _plugin_classes(walker)
    if classes:
        family = classes[0][0]
        declared["family"] = family
        # Family defaults first, so anything written in the file wins.
        declared.update(FAMILY_DEFAULTS.get(family, {}))
        scopes.append(_assignments(classes[0][2].body))

    for scope in scopes:
        for key in DECLARATION_KEYS:
            if key in scope:
                value = _literal(scope[key].value)
                if value is not None:
                    declared[key] = value

    declared.setdefault("name", Path(filename).stem)
    return declared


def validate(source: str, *, filename: str = "<plugin>",
             known_names=()) -> Report:
    """Check one plugin file. Parses it; never imports or runs it.

    One pass yields both the verdict and the declarations, which is what lets
    the kernel resolve a file's execution context without ever importing it.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return Report(filename, (Finding(
            ERROR, exc.lineno or 0, f"does not parse: {exc.msg}"),), source)

    walker = _Walker()
    walker.visit(tree)
    _check_contract(tree, walker, filename, known_names)
    return Report(
        filename,
        tuple(sorted(walker.findings,
                     key=lambda f: (f.level != ERROR, f.line))),
        source,
        _collect_declarations(tree, walker, filename))


def validate_file(path, known_names=()) -> Report:
    """Validate the bytes on disk.

    The source is read once and returned with the report, so the caller can
    execute *those* bytes rather than re-opening the path — a file that changes
    between the check and the run was never checked.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8", errors="replace")
    return validate(source, filename=path.name, known_names=known_names)
