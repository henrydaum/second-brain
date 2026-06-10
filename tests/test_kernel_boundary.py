"""The kernel boundary (CLAUDE.md's "one rule"), made executable.

Core code may lean on the plugin *substrate* — base classes, discovery, shared
path helpers, the command-registry adapter — but may hard-import exactly two
plugin *implementations*: ``service_llm`` and ``parser_registry``. Everything
else must arrive by discovery, so installing or uninstalling a package can
never break the kernel.

These tests AST-walk every core module (nothing is imported or executed) and
pin the complete set of ``plugins.*`` import edges, including lazy
function-local imports — a deferred import is still a hard dependency.
Widening the kernel then fails here, turning boundary drift into a deliberate
one-line decision in this file instead of an accident in commit #601.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Everything outside plugins/ that boots or runs the kernel.
CORE_DIRS = ("agent", "attachments", "config", "events", "pipeline",
             "runtime", "state_machine")
CORE_FILES = ("main.py", "main.pyw", "paths.py")

# The plugin *substrate*: infrastructure the plugin system itself is made of.
# Growing this set is sometimes right (a new base class, a new shared helper);
# do it here, on purpose, not by accident.
SUBSTRATE = frozenset({
    "plugins.BaseCommand",
    "plugins.BaseService",
    "plugins.BaseTask",
    "plugins.BaseTool",
    "plugins.plugin_discovery",
    "plugins.helpers.plugin_paths",
    "plugins.helpers.memory_paths",
    "plugins.frontends.helpers.command_registry",
})

# The two sanctioned plugin implementations, pinned to the exact core files
# allowed to import them. Any other core file wanting these must go through
# the services dict / discovery instead.
SANCTIONED = {
    "plugins.services.service_llm": {
        "runtime/conversation_loop.py",
    },
    "plugins.services.helpers.parser_registry": {
        "pipeline/orchestrator.py",
        "pipeline/watcher.py",
    },
}


def _iter_core_files():
    for name in CORE_FILES:
        path = ROOT / name
        if path.exists():
            yield path
    for dirname in CORE_DIRS:
        for path in sorted((ROOT / dirname).rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def _is_module(dotted):
    rel = Path(*dotted.split("."))
    return (ROOT / rel).is_dir() or (ROOT / rel).with_suffix(".py").exists()


def _plugin_imports(tree):
    """Yield every ``plugins.*`` module a parsed file imports, however deep."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "plugins" or alias.name.startswith("plugins."):
                    yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mod = node.module or ""
            if mod == "plugins" or mod.startswith("plugins."):
                for alias in node.names:
                    full = f"{mod}.{alias.name}"
                    # ``from plugins.services import service_llm`` imports a
                    # module; ``from plugins.BaseTool import BaseTool`` a name.
                    yield full if _is_module(full) else mod


def _collect_edges():
    edges = {}
    for path in _iter_core_files():
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for mod in _plugin_imports(tree):
            edges.setdefault(mod, set()).add(rel)
    return edges


# ── The one rule ─────────────────────────────────────────────────────

def test_core_imports_only_substrate_and_the_two_sanctioned_modules():
    edges = _collect_edges()
    allowed = SUBSTRATE | set(SANCTIONED)
    violations = {mod: sorted(files)
                  for mod, files in edges.items() if mod not in allowed}
    assert not violations, (
        "Core code grew a hard import of a plugin module:\n"
        + "\n".join(f"  {mod}  <-  {', '.join(files)}"
                    for mod, files in sorted(violations.items()))
        + "\nPlugin implementations must be reached via discovery (the"
        " services dict / registries), never imported from core. If this is"
        " genuinely new plugin *substrate*, add it to SUBSTRATE here and to"
        " the kernel-boundary section of CLAUDE.md — deliberately."
    )


def test_sanctioned_imports_do_not_spread_to_new_core_files():
    edges = _collect_edges()
    for mod, allowed_files in SANCTIONED.items():
        extra = edges.get(mod, set()) - allowed_files
        assert not extra, (
            f"{mod} is now imported from {sorted(extra)}. The sanctioned"
            f" call sites are {sorted(allowed_files)}; new core code should"
            " reach it through the services dict / parser service instead."
        )


def test_sanctioned_modules_resolve_in_the_kernel_tree():
    """CLAUDE.md: 'Keep these two resolvable in any kernel.' They are the
    only plugin files the kernel cannot boot without."""
    for mod in SANCTIONED:
        assert _is_module(mod), f"{mod} is missing from the built-in tree"


def test_scanner_still_sees_known_edges():
    """If the walker went blind (core dirs renamed, parse short-circuit),
    every test above would pass vacuously. Pin one known edge per group."""
    edges = _collect_edges()
    assert "plugins.plugin_discovery" in edges            # substrate
    assert "plugins.services.service_llm" in edges        # implementation
