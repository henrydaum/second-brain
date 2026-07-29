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
import difflib
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .guest.requests import ALL_TYPES

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
    | {"guest", "guest.bases", "guest.box", "guest.sdk", "guest.forms", "guest.hooks",
       "guest.parsing", "guest.llm",
       "sandbox.guest", "sandbox.guest.bases", "sandbox.guest.box",
       "sandbox.guest.forms",
       "sandbox.guest.hooks", "sandbox.guest.parsing", "sandbox.guest.llm"}
)

# Pure stdlib: computation only, no way to reach the environment.
#
# Judgement calls worth knowing about:
#   - ``time`` is here because the clock is an SDK helper, not a Request; we
#     are not chasing replay determinism.
#   - ``email`` and ``csv`` parse things already in memory. The store's mail
#     and tabular plugins lean on them heavily and neither opens a file.
#   - ``ast`` and ``tokenize`` read source text, not files.
#   - ``io`` IS here, because ``BytesIO``/``StringIO`` are pure and are how
#     you hand bytes to a foreign decoder without giving it a path. Its one
#     dangerous name, ``io.open``, is caught as an attribute instead — banning
#     the module punished the pure use and taught nothing.
#   - ``xml`` is not here for the same reason: ``ElementTree.parse`` takes a
#     filename.
PURE_MODULES = {
    "__future__", "abc", "argparse", "array", "ast", "base64", "binascii",
    "bisect", "calendar", "cmath", "codecs", "collections", "colorsys",
    "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal",
    "difflib", "email", "enum", "fnmatch", "fractions", "functools",
    "graphlib", "hashlib", "heapq", "hmac", "html", "io", "itertools", "json",
    "keyword", "math", "mimetypes", "numbers", "operator", "posixpath",
    "pprint", "queue", "random", "re", "reprlib", "secrets", "statistics",
    "string", "struct", "textwrap", "time", "token", "tokenize", "traceback",
    "types", "typing", "unicodedata", "urllib.parse", "uuid", "warnings",
    "zoneinfo",
}

# Pure third-party packages the SDK vouches for. Not stdlib, but computation
# only — no disk, no network, no process — so importing one in-process is no
# more dangerous than arithmetic.
#
# Anything here has to be installed wherever guest code runs, which for a
# container means shipping it in the image. That is the cost of the list, and
# it is why the list is short and stays short: anything heavier counts as
# unmediated, which takes the disclaimer and a process boundary with it.
SDK_PACKAGES = {
    "croniter": "parsing and stepping cron expressions",
    "cron_descriptor": "describing cron expressions in English",
}

# Stdlib modules that perform their own I/O on a path the plugin names. They
# are not kernel-reaching and not foreign, but they cannot be mediated either
# — a tabular parser opening a user's ``.db`` read-only is a legitimate parse,
# and so is reading an archive. They get the foreign-library disclaimer.
#
# The dangerous case — reaching around the kernel's own database — is caught
# by DB_ATTRS as an ERROR regardless, so nothing is lost by allowing these.
UNMEDIATED_STDLIB = {
    "sqlite3": "opens a database file directly",
    "zipfile": "reads and extracts an archive directly",
    "tarfile": "reads and extracts an archive directly",
    # Not an allowance — asyncio already landed in the foreign-library branch
    # below, with the identical warning and the identical subprocess. What it
    # was missing was an accurate sentence: describing the stdlib's event loop
    # as "a foreign library" reads as a bug in the linter, and a finding an
    # author does not believe is a finding they route around. An async client
    # library is the honest reason to want it, and a frontend that owns a
    # transport loop is the case in front of us.
    "asyncio": "runs its own event loop, sockets and subprocesses",
}

# First-party kernel modules. Importing one is not a foreign-library problem
# — it is the boundary itself. Whatever the plugin wanted from it is a
# Request, and saying "this cannot be validated, subprocess it" would be both
# wrong and unhelpful.
KERNEL_MODULES = {
    "agent", "attachments", "config", "events", "paths", "pipeline",
    "plugins", "runtime", "state_machine",
}

# What a native plugin reached for on its context, and the SDK route that
# replaces it. This is the migration checklist for plugins that perform their
# effects through ``context`` rather than through stdlib calls.
CONTEXT_MAP = {
    "db": "sdk.db",
    "config": "sdk.config",
    "services": "sdk.services.call",
    "call_tool": "sdk.tools.call",
    "approve_command": "sdk.ui.approve",
    "request_user_input": "sdk.ui.ask",
    "tool_registry": "sdk.tools",
    "orchestrator": "sdk.pipeline",
    "command_registry": "sdk.tools.run_command",
    "runtime": "sdk.session / sdk.conv",
    "session_key": "sdk.session.get",
    "user_id": "sdk.users.read",
    "current_user": "sdk.users.read",
    "user_config": "sdk.config.read",
    "root_dir": "sdk.fs",
    "app_control": "sdk.app.stop",
}

# Reaching for the environment directly. Each maps to the Request that does
# the same job through the gate.
EFFECT_MODULES = {
    "os": "sdk.fs / sdk.env",
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
    "threading": "the kernel schedules; a plugin should not",
    "multiprocessing": "the kernel schedules; a plugin should not",
    "ctypes": "nothing — this defeats the boundary entirely",
    "importlib": "sdk.plugin.register",
    "logging": "sdk.log",
}

# Pure modules with one impure name. Importing them is fine; reaching for
# these is not. Cheaper and more teachable than banning the whole module.
PURE_MODULE_ATTRS = {
    ("io", "open"): "sdk.fs.read / sdk.fs.write",
    ("io", "FileIO"): "sdk.fs.read / sdk.fs.write",
}

BANNED_BUILTINS = {
    "open": "sdk.fs.read / sdk.fs.write",
    "eval": "nothing — build the value directly",
    "exec": "nothing — build the value directly",
    "compile": "nothing",
    "__import__": "sdk.plugin.register",
    # Two answers, because there are two questions. A plugin that wants to ask
    # a person something wants sdk.ui.ask. A *frontend* whose transport is the
    # terminal wants the console — and never input(), which would block its own
    # box and stop it rendering until the next keypress.
    "input": "sdk.ui.ask, or sdk.console.read_line in a console frontend",
    "breakpoint": "nothing",
    # Reaching the namespace is reaching everything in it. Banning ``open``
    # while leaving ``globals()["open"]`` open would be a rule that only stops
    # people who were not going around it anyway.
    "globals": "nothing — name what you need directly",
    "locals": "nothing — name what you need directly",
}

# Names that hand back a namespace rather than a value. Checked wherever they
# are *read*, not just called: ``getattr(__builtins__, "open")`` is a plain
# attribute fetch off a Name, so the call-shaped checks above never see it.
#
# This does not make the linter a proof — nothing does, and the contract says
# so. It closes the one escape that sits directly beside a rule already
# enforced, which is worth a line.
BANNED_NAMES = {
    "__builtins__": "nothing — name what you need directly",
}

# Database-shaped attribute names. Native plugins reach past the Database
# API into the raw sqlite connection in dozens of places, and every one of
# them has to become a Request. Bare ``execute`` and ``lock`` are deliberately
# absent: they appear on plenty of things that are not databases, and a linter
# that cries wolf gets worked around.
#
# What this is aimed at is reaching around *the kernel's* database, which a
# plugin does through a context or a service it was handed. It is not aimed at
# a parser opening some unrelated ``.db`` file of the user's, which is a
# legitimate parse and the reason ``sqlite3`` is in UNMEDIATED_STDLIB. Those
# two were in flat contradiction: the module was allowed with a disclaimer and
# then every method you would call on it was an ERROR, so the tabular parser
# the comment cites could not actually be written. See ``_is_own_connection``.
DB_ATTRS = {
    "conn": "sdk.db.query / sdk.db.write",
    "cursor": "sdk.db.query",
    "fetchone": "sdk.db.query",
    "fetchall": "sdk.db.query",
    "executemany": "sdk.db.write",
    "execute_write": "sdk.db.write",
    "ensure_output_table": "sdk.db.define",
}

# Effect-shaped method names. Heuristic by nature: a linter, not a proof.
EFFECT_METHODS = {
    "read_text": "sdk.fs.read", "read_bytes": "sdk.fs.read",
    "write_text": "sdk.fs.write", "write_bytes": "sdk.fs.write",
    "unlink": "sdk.fs.delete", "rmdir": "sdk.fs.delete",
    "mkdir": "sdk.fs.write", "rename": "sdk.fs.move",
    "iterdir": "sdk.fs.list",
}
# ``walk`` is deliberately absent. Every module that offers a dangerous one —
# ``os``, ``pathlib`` — is already refused at import, so the name could only
# ever fire on something harmless: ``email.Message.walk``, ``ast.walk``. A
# linter that cries wolf gets worked around, which costs more than it saves.

# Ceilings mirrored from the interpreter. Exceeding one is not an error —
# the plugin declares intent, the kernel clamps.
CEILINGS = {"load_timeout": 600.0, "timeout": 600.0, "max_calls": 25,
            "memory_mb": 4096, "batch_size": 1000,
            "poll_interval": 3600.0, "max_poll_failures": 100}

# Declarations the kernel reads without importing, so they must be literals.
LITERAL_LISTS = ("dependencies_files", "dependencies_pip",
                 "requires_services", "requests", "exports")
LITERAL_STRINGS = ("name", "box", "lifetime")

# Closed vocabularies. A typo here is silent otherwise: an unrecognised
# lifetime reads as "unset" and the file quietly gets the default.
ENUMS = {"lifetime": {"", "ephemeral", "persistent"}}


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
    # Imports the validator cannot see inside — foreign libraries, and stdlib
    # modules that do their own path I/O. This is what decides whether an
    # installed package needs a process boundary (``sandbox/isolation.py``),
    # so it is carried as a set rather than inferred from warning text.
    #
    # A syntax error yields an empty set, which is safe only because a file
    # that does not parse never runs at all.
    unmediated: frozenset = frozenset()
    #: SHA-256 of ``source``. Carrying the bytes was not enough on its own —
    #: every caller validated a *path* and then handed that path to a loader
    #: that opened it again, so the file could change in between and the claim
    #: above was not actually enforced anywhere. The loader re-hashes and
    #: refuses a mismatch. Last, so existing positional construction of a
    #: Report keeps meaning what it did.
    digest: str = ""

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


def digest_of(source: str) -> str:
    """Fingerprint the exact text that was checked.

    Computed over the decoded string rather than the raw file so it matches
    however the caller read it, which is the same thing the loader will do.
    """
    return hashlib.sha256((source or "").encode("utf-8")).hexdigest()


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
        # Names bound to a connection the plugin opened itself. A parser
        # reading a user's ``.db`` is doing exactly what ``sqlite3`` is
        # allowed (with a disclaimer) for; the DB_ATTRS rule is about reaching
        # around the *kernel's* database, and firing on both made the
        # allowance useless.
        self.own_connections: set[str] = set()
        # Modules whose behaviour the validator cannot see inside: foreign
        # libraries, and the stdlib modules that do their own path I/O. Kept
        # as data rather than left implicit in the warning text, because
        # isolation is decided from it — a security decision should not be
        # made by matching on a human-readable string.
        self.unmediated: set[str] = set()

    def add(self, level, node, message, fix=""):
        """Record a finding."""
        self.findings.append(
            Finding(level, getattr(node, "lineno", 0), message, fix))

    def _first_sighting(self, key: str) -> bool:
        """Record an unmediated module; True if this is the first time.

        The disclaimer is a fact about the *module*, not about the line — a
        frontend importing ``telegram`` in eight places got eight identical
        paragraphs saying the same thing, which is how a real warning turns
        into scrollback. Reported once, at the first import, and the isolation
        decision reads ``unmediated`` either way.
        """
        if key in self.unmediated:
            return False
        self.unmediated.add(key)
        return True

    # ── imports ────────────────────────────────────────────────────

    def _check_module(self, name: str, node):
        """Classify one imported module."""
        if name in CONTRACT_MODULES:
            return
        key = _module_root(name)
        if key in PURE_MODULES or name in PURE_MODULES:
            return
        if key in SDK_PACKAGES:
            return
        if key in UNMEDIATED_STDLIB:
            if self._first_sighting(key):
                self.add(WARNING, node,
                         f"imports {name!r}, which {UNMEDIATED_STDLIB[key]} "
                         f"and so cannot be mediated - this plugin runs in a "
                         f"subprocess")
            return
        if key in EFFECT_MODULES:
            self.add(ERROR, node, f"imports {name!r}, which reaches the "
                                  f"environment directly", EFFECT_MODULES[key])
            return
        if key in KERNEL_MODULES:
            self.add(ERROR, node,
                     f"imports {name!r}, which lives on the kernel side of "
                     f"the boundary", "a Request for whatever it needed")
            return
        if self._first_sighting(key):
            self.add(WARNING, node,
                     f"imports {name!r}, a foreign library. Its actions cannot "
                     f"be turned into Requests, so they are not mediated - "
                     f"this plugin runs in a subprocess")

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

    def visit_Assign(self, node):
        """Remember names bound to a connection this plugin opened itself."""
        if self._opens_own_connection(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.own_connections.add(target.id)
        self.generic_visit(node)

    @staticmethod
    def _opens_own_connection(value) -> bool:
        """Whether an expression is ``sqlite3.connect(...)``."""
        return (isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "connect"
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "sqlite3")

    def _is_own_connection(self, node) -> bool:
        """Whether an attribute hangs off a connection the plugin opened.

        Deliberately shallow — one level, on a name assigned in this file from
        ``sqlite3.connect``. It is a linter, not a proof, and the shallow
        version covers the shape a parser actually writes.
        """
        base = node.value
        # Alternating, not one pass each: ``conn.cursor().execute(q).fetchall``
        # is Attribute-over-Call-over-Attribute-over-Call, and unwinding only
        # one kind at a time stops halfway down.
        while isinstance(base, (ast.Call, ast.Attribute)):
            base = base.func if isinstance(base, ast.Call) else base.value
        return isinstance(base, ast.Name) and base.id in self.own_connections

    def visit_Call(self, node):
        """Direct calls to banned builtins."""
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "getattr":
            # ``getattr(context, "db", None)`` is how a native plugin asks for
            # an optional capability, and it is invisible to plain attribute
            # matching.
            self._check_getattr(node)
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

    def _check_getattr(self, node):
        """Flag getattr(context, "field") the way a direct access is flagged."""
        if len(node.args) < 2:
            return
        target, name = node.args[0], node.args[1]
        if not (isinstance(target, ast.Name) and target.id == "context"):
            return
        field_name = _literal(name)
        if isinstance(field_name, str) and field_name in CONTEXT_MAP:
            self.add(ERROR, node,
                     f"reads context.{field_name} via getattr; sandboxed code "
                     f"is handed an sdk, not a context",
                     CONTEXT_MAP[field_name])

    def visit_Name(self, node):
        """Namespace escapes, wherever the name is mentioned."""
        if isinstance(node.ctx, ast.Load) and node.id in BANNED_NAMES:
            self.add(ERROR, node, f"reaches {node.id}",
                     BANNED_NAMES[node.id])
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Catch effects performed through the old ``context`` object.

        A sandboxed plugin is never handed a context, so every ``context.x``
        is something that has to become a Request — and naming which one is
        the difference between a migration and a puzzle.
        """
        if (isinstance(node.value, ast.Name) and node.value.id == "context"
                and node.attr in CONTEXT_MAP):
            self.add(ERROR, node,
                     f"uses context.{node.attr}; sandboxed code is handed an "
                     f"sdk, not a context", CONTEXT_MAP[node.attr])
        elif (isinstance(node.value, ast.Name)
              and (node.value.id, node.attr) in PURE_MODULE_ATTRS):
            self.add(ERROR, node,
                     f"uses {node.value.id}.{node.attr}, which reaches the "
                     f"environment directly",
                     PURE_MODULE_ATTRS[(node.value.id, node.attr)])
        elif (node.attr in DB_ATTRS and not self._is_sdk_call(node)
                and not self._is_own_connection(node)):
            self.add(ERROR, node,
                     f"reaches the database directly via .{node.attr}",
                     DB_ATTRS[node.attr])
        self.generic_visit(node)

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


def _check_hooks(walker: _Walker, node, cls):
    """Check a ``hooks`` declaration names real moments and real methods.

    Both halves fail silently otherwise — an unknown moment is a doorway
    nobody stands at, and a missing method only raises when the turn reaches
    that doorway, which may be much later and somewhere confusing.
    """
    from .guest.hooks import MOMENTS

    value = _literal(node.value)
    if not isinstance(value, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        walker.add(ERROR, node,
                   "hooks must be a literal {moment: method_name} dict — the "
                   "kernel reads it without importing")
        return

    defined = {item.name for item in cls.body
               if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for moment, method in value.items():
        if moment not in MOMENTS:
            walker.add(ERROR, node,
                       f"{moment!r} is not a hook moment; "
                       f"they are {sorted(MOMENTS)}")
        if method not in defined:
            walker.add(ERROR, node,
                       f"hooks names {method!r} for {moment!r}, but this class "
                       f"defines no such method")


def _check_requests(walker: _Walker, node):
    """Check every name in a ``requests`` declaration is a real Request type.

    This declaration used to be documentation, and drifted the way
    documentation does. It is now load-bearing: it is exactly what a single
    user approval authorizes, so a name that matches nothing grants nothing
    and the misspelling surfaces as an approval dialog the user thought they
    had already answered.
    """
    value = _literal(node.value)
    if not isinstance(value, list):
        return          # already reported by the literal-list check
    for name in value:
        if not isinstance(name, str) or name in ALL_TYPES:
            continue
        close = difflib.get_close_matches(name, sorted(ALL_TYPES), 1, 0.7)
        walker.add(ERROR, node, f"{name!r} is not a Request type",
                   close[0] if close else "")


def _check_subscribed_channels(walker: _Walker, node, cls, family: str):
    """Check a ``subscribed_channels`` declaration can actually be honoured.

    Deliberately *not* checked: whether the channel names exist. The kernel's
    channels are listed in ``events/event_channels.py``, but that file is
    explicit that a plugin owns its own channels and must not register them
    there — so an allowlist would refuse the one case this feature is for, a
    plugin listening to another plugin.

    What is checked is the two ways a declaration silently does nothing: a
    family that cannot be delivered to, and a missing handler.
    """
    value = _literal(node.value)
    if not isinstance(value, list) or not all(
            isinstance(v, str) and v.strip() for v in value):
        walker.add(ERROR, node,
                   "subscribed_channels must be a literal list of non-empty "
                   "strings — the kernel reads it without importing")
        return
    if not value:
        return

    # Delivery needs something to deliver *to*. A tool is a call that ends, so
    # by the time an event arrived there would be no box holding the plugin.
    if family not in ("service", "frontend"):
        walker.add(ERROR, node,
                   f"a {family} cannot subscribe to the bus — it does not stay "
                   f"loaded, so nothing would ever be delivered; move the "
                   f"subscription to a service")
        return

    defined = {item.name for item in cls.body
               if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "on_event" not in defined:
        walker.add(ERROR, node,
                   "subscribed_channels is declared but this class defines no "
                   "on_event(self, sdk, channel, payload) — every delivery "
                   "would reach the base class and be discarded")


def _check_secret_names(walker: _Walker, node):
    """Warn about a config setting that looks like a credential but is not
    declared as one."""
    from .credentials import SECRET_PREFIX, looks_secret

    entries = _literal(node.value)
    if not isinstance(entries, (list, tuple)):
        return
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        key = entry[1]
        if not isinstance(key, str) or key.startswith(SECRET_PREFIX):
            continue
        if looks_secret(key):
            walker.add(NOTE, node,
                       f"setting {key!r} looks like a credential but is not "
                       f"declared as one, so it will be handed out in "
                       f"plaintext", f"{SECRET_PREFIX}{key}")


def _plugin_classes(walker: _Walker):
    """Every class subclassing a known plugin base, with the family it names."""
    found = []
    for node in walker.classes:
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in BASE_TO_FAMILY:
                found.append((BASE_TO_FAMILY[base.id], base.id, node))
                break
    return found


def _formstep_nodes(tree):
    """Imports/uses of the command-only guest form value."""
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module in {
                "guest.forms", "sandbox.guest.forms", "guest",
                "sandbox.guest",
            }
            and any(alias.name == "FormStep" for alias in node.names)
        ) or (
            isinstance(node, ast.Import)
            and any(alias.name in {
                "guest.forms", "sandbox.guest.forms",
            } for alias in node.names)
        ) or (
            isinstance(node, ast.Attribute)
            and node.attr == "FormStep"
            and isinstance(node.value, ast.Name)
            and node.value.id in {"guest", "forms"}
        ):
            found.append(node)
    return found


def _check_contract(tree, walker: _Walker, filename: str, known_names):
    """Check the BasePlugin contract, declared metadata, and name collisions."""
    stem = Path(filename).stem
    prefix = stem.split("_")[0] if "_" in stem else ""
    declared = _plugin_classes(walker)
    formstep_nodes = _formstep_nodes(tree)

    if not declared:
        for node in formstep_nodes:
            walker.add(
                ERROR,
                node,
                "FormStep is command-only; helper and scratch code cannot "
                "present a multi-step command form",
            )
        # A file *named* as a plugin has to be one. Anything else is a helper
        # or a script: effect checks only, no contract.
        if prefix in FAMILIES:
            walker.add(ERROR, tree.body[0] if tree.body else tree,
                       f"{stem}.py is named as a {prefix} plugin but no class "
                       f"subclasses {FAMILIES[prefix]}")
        return

    family, base, cls = declared[0]
    if family != "command":
        for node in formstep_nodes:
            walker.add(
                ERROR,
                node,
                f"FormStep is command-only; a {family} cannot present a "
                "multi-step command form",
            )
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

    if "poll_interval" in assigned:
        node = assigned["poll_interval"]
        interval = _literal(node.value)
        methods = {
            item.name
            for item in cls.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if family not in {"service", "frontend"}:
            walker.add(
                ERROR,
                node,
                "poll_interval is only valid for resident services and "
                "frontends",
            )
        elif not isinstance(interval, (int, float)) or interval < 0:
            walker.add(
                ERROR,
                node,
                "poll_interval must be a non-negative number",
            )
        elif interval > 0 and "poll" not in methods:
            walker.add(
                ERROR,
                node,
                "poll_interval is enabled but this class defines no "
                "poll(self, sdk)",
            )

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

    # ``requests`` is the grant an approval spends, so a typo in it is the one
    # that costs something: the misspelled Request is not in the set, and the
    # command stops mid-run asking about a capability the user already
    # approved. Checked against a closed vocabulary because, unlike bus
    # channels, a plugin cannot invent a Request type.
    if "requests" in assigned:
        _check_requests(walker, assigned["requests"])

    # Hooks are declared, not registered, so a typo here is silent: the shim
    # is never stood up and the doorway is simply never visited.
    if "hooks" in assigned:
        _check_hooks(walker, assigned["hooks"], cls)

    # Bus subscriptions are declared the same way and fail the same way: the
    # listener is simply never stood up, and the plugin waits forever for an
    # event that was never routed to it.
    if "subscribed_channels" in assigned:
        _check_subscribed_channels(walker, assigned["subscribed_channels"],
                                   cls, family)

    # A setting holding a credential is declared by its name. Catching the
    # omission here is the whole reason the name heuristic still exists: it
    # costs the author one message at authoring time instead of quietly
    # handing plaintext to a plugin that only ever needed a handle.
    if "config_settings" in assigned:
        _check_secret_names(walker, assigned["config_settings"])

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


# ``isolation`` is deliberately never collected. It was, and it made the code
# being contained the authority on its own containment — see
# ``sandbox/isolation.py``. The kernel now derives it from the file's tree, so
# reading it off the file would at best be ignored and at worst be believed.
#
# There was a ``DECLARATION_KEYS`` tuple here listing what to collect. It was
# dead: ``_collect_declarations`` takes *every* literal class attribute,
# because the dual-mode loader has to copy ``parameters``, ``description`` and
# whatever else a base class grows onto its adapter, and an allowlist that
# drifts silently drops a plugin's schema. A list nothing reads is worse than
# no list, since the next person to add a declaration will maintain it and
# wonder why it changed nothing.

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

    # Every literal class attribute, not just the documented ones: the
    # dual-mode loader has to copy ``parameters``, ``description`` and the
    # rest onto its adapter, or the registry advertises a plugin with no
    # schema.
    for scope in scopes:
        for key, node in scope.items():
            if key.startswith("_"):
                continue
            value = _literal(node.value)
            if value is not None:
                declared[key] = value

    # ``isolation`` is dropped rather than merely unused. Everything literal
    # gets collected here, not just the documented keys, so leaving it in
    # would put a value nothing honours within reach of anything that later
    # goes looking — and a stale declaration that reads as authoritative is
    # how this became a vulnerability in the first place. The author is told
    # once, at the line, rather than left to wonder why it does nothing.
    for scope in scopes:
        if (node := scope.get("isolation")) is not None:
            walker.add(NOTE, node,
                       "declares 'isolation', which is ignored",
                       "the kernel decides this from the plugin's tree: "
                       "sandbox_plugins is always subprocessed, plugins/ "
                       "always in-process, installed_plugins by whether it "
                       "imports a foreign library")
            break
    declared.pop("isolation", None)

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
            ERROR, exc.lineno or 0, f"does not parse: {exc.msg}"),), source,
            digest=digest_of(source))

    walker = _Walker()
    walker.visit(tree)
    _check_contract(tree, walker, filename, known_names)
    # Before the findings are frozen, not inline in the Report call.
    # ``_collect_declarations`` *adds* a finding — the advisory that
    # ``isolation`` is ignored — and Python evaluates arguments left to right,
    # so building the tuple in place meant that note was appended to a list
    # nobody read again. The one thing telling an author why their declaration
    # does nothing was itself doing nothing.
    declarations = _collect_declarations(tree, walker, filename)
    return Report(
        filename,
        tuple(sorted(walker.findings,
                     key=lambda f: (f.level != ERROR, f.line))),
        source,
        declarations,
        frozenset(walker.unmediated),
        digest_of(source))


def validate_file(path, known_names=()) -> Report:
    """Validate the bytes on disk.

    The source is read once and returned with the report, so the caller can
    execute *those* bytes rather than re-opening the path — a file that changes
    between the check and the run was never checked.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8", errors="replace")
    return validate(source, filename=path.name, known_names=known_names)
