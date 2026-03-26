"""
Build Plugin tool.

Allows the LLM agent to create, edit (via FIND/REPLACE), and delete
sandbox plugins (tools, tasks, services). Files are written to the
sandbox directories in DATA_DIR and registered immediately.
"""

import ast
import logging
import sys
from pathlib import Path

from Stage_3.BaseTool import BaseTool, ToolResult
from paths import SANDBOX_TOOLS, SANDBOX_TASKS, SANDBOX_SERVICES, ROOT_DIR

logger = logging.getLogger("BuildPlugin")

# Map plugin_type to (sandbox dir, naming prefix, base class name)
_PLUGIN_CONFIG = {
    "tool":    (SANDBOX_TOOLS,    "tool_",  "BaseTool"),
    "task":    (SANDBOX_TASKS,    "task_",  "BaseTask"),
    "service": (SANDBOX_SERVICES, None,     "BaseService"),
}

# Expected import statements for each plugin type
_EXPECTED_IMPORTS = {
    "tool":    ("Stage_3.BaseTool", {"BaseTool", "ToolResult"}),
    "task":    ("Stage_2.BaseTask", {"BaseTask", "TaskResult"}),
    "service": ("Stage_0.BaseService", {"BaseService"}),
}

# Baked-in source directories (read-only)
_BAKED_IN_DIRS = {
    "tool":    ROOT_DIR / "Stage_3" / "tools",
    "task":    ROOT_DIR / "Stage_2" / "tasks",
    "service": ROOT_DIR / "Stage_0" / "services",
}


class BuildPlugin(BaseTool):
    name = "build_plugin"
    description = (
        "Create, edit, or delete a sandbox plugin (tool, task, or service). "
        "Use action='create' with full source code to create a new plugin. "
        "Use action='edit' with search_block/replace_block for targeted edits. "
        "Use action='delete' to remove a plugin file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "plugin_type": {
                "type": "string",
                "enum": ["tool", "task", "service"],
                "description": "Type of plugin to create/edit/delete.",
            },
            "file_name": {
                "type": "string",
                "description": "File name (e.g. tool_get_weather.py). Must follow naming convention.",
            },
            "action": {
                "type": "string",
                "enum": ["create", "edit", "delete"],
                "description": "Action to perform on the plugin file.",
            },
            "code": {
                "type": "string",
                "description": "Complete Python source code. Required for 'create' action; leave blank for other actions.",
            },
            "search_block": {
                "type": "string",
                "description": (
                    "Exact text to find in the existing file. Required for 'edit' action; leave blank for other actions. "
                    "Whitespace and indentation must match exactly."
                ),
            },
            "replace_block": {
                "type": "string",
                "description": (
                    "Text to replace the search_block with. Required for 'edit' action; leave blank for other actions. "
                    "Can be empty string to delete the matched block."
                ),
            },
        },
        "required": ["plugin_type", "file_name", "action"],
    }
    requires_services = []
    agent_enabled = True
    max_calls = 10

    def run(self, context, **kwargs) -> ToolResult:
        plugin_type = kwargs.get("plugin_type", "")
        file_name = kwargs.get("file_name", "").strip()
        action = kwargs.get("action", "")

        if plugin_type not in _PLUGIN_CONFIG:
            return ToolResult.failed(f"Invalid plugin_type: '{plugin_type}'. Must be tool, task, or service.")
        if not file_name:
            return ToolResult.failed("file_name is required.")
        if action not in ("create", "edit", "delete"):
            return ToolResult.failed(f"Invalid action: '{action}'. Must be create, edit, or delete.")

        # Naming convention check
        err = _check_naming(plugin_type, file_name)
        if err:
            return ToolResult.failed(err)

        sandbox_dir = _PLUGIN_CONFIG[plugin_type][0]
        sandbox_path = sandbox_dir / file_name
        baked_in_path = _BAKED_IN_DIRS[plugin_type] / file_name

        # Baked-in protection
        if baked_in_path.exists():
            return ToolResult.failed(
                f"'{file_name}' exists as a baked-in plugin and cannot be modified. "
                f"Create a new plugin with a different name instead."
            )

        if action == "delete":
            return self._delete(sandbox_path, file_name, plugin_type, context)
        elif action == "create":
            return self._create(sandbox_path, file_name, plugin_type, kwargs.get("code"), context)
        elif action == "edit":
            return self._edit(sandbox_path, file_name, plugin_type,
                              kwargs.get("search_block"), kwargs.get("replace_block"), context)

    def _create(self, sandbox_path, file_name, plugin_type, code, context):
        if not code:
            return ToolResult.failed("'code' is required for action='create'.")
        if sandbox_path.exists():
            return ToolResult.failed(
                f"'{file_name}' already exists. Use action='edit' to modify it, "
                f"or action='delete' then action='create' to rewrite from scratch."
            )

        warnings = _validate_code(code, file_name, plugin_type, context)
        sandbox_path.write_text(code, encoding="utf-8")

        # Register immediately (only if no syntax errors)
        if not any("SyntaxError" in w for w in warnings):
            _try_register(sandbox_path, plugin_type, context, warnings)

        return _build_result(sandbox_path, warnings, "created")

    def _edit(self, sandbox_path, file_name, plugin_type, search_block, replace_block, context):
        if search_block is None:
            return ToolResult.failed("'search_block' is required for action='edit'.")
        if replace_block is None:
            return ToolResult.failed("'replace_block' is required for action='edit'.")
        if not sandbox_path.exists():
            return ToolResult.failed(f"'{file_name}' does not exist in the sandbox. Use action='create' first.")

        content = sandbox_path.read_text(encoding="utf-8")

        # Strict uniqueness check
        count = content.count(search_block)
        if count == 0:
            return ToolResult.failed(
                "Search block not found. Exact whitespace and indentation must match."
            )
        if count > 1:
            return ToolResult.failed(
                f"Match is not unique ({count} occurrences). "
                f"Add more surrounding lines to the search block for context."
            )

        # Apply the replacement
        new_content = content.replace(search_block, replace_block, 1)

        warnings = _validate_code(new_content, file_name, plugin_type, context)
        sandbox_path.write_text(new_content, encoding="utf-8")

        # Re-register immediately (only if no syntax errors)
        if not any("SyntaxError" in w for w in warnings):
            _try_register(sandbox_path, plugin_type, context, warnings)

        return _build_result(sandbox_path, warnings, "edited")

    def _delete(self, sandbox_path, file_name, plugin_type, context):
        if not sandbox_path.exists():
            return ToolResult.failed(f"'{file_name}' does not exist in the sandbox.")

        # Unregister before deleting
        plugin_name = _extract_plugin_name(sandbox_path, plugin_type)
        if plugin_name:
            from plugin_discovery import unload_plugin
            unload_plugin(plugin_type, plugin_name,
                          context.tool_registry, context.orchestrator, context.services)

        # Clean up sys.modules
        _cleanup_module(sandbox_path, plugin_type)

        sandbox_path.unlink()
        return ToolResult(
            llm_summary=f"Deleted and unregistered '{file_name}'.",
        )


# ── Helpers ──────────────────────────────────────────────────────────

def _check_naming(plugin_type: str, file_name: str) -> str | None:
    """Return an error string if naming convention is violated, else None."""
    if not file_name.endswith(".py"):
        return f"File name must end with .py, got '{file_name}'."

    if plugin_type == "tool" and not file_name.startswith("tool_"):
        return f"Tool files must start with 'tool_', got '{file_name}'."
    if plugin_type == "task" and not file_name.startswith("task_"):
        return f"Task files must start with 'task_', got '{file_name}'."
    if plugin_type == "service" and file_name.startswith("_"):
        return f"Service files must not start with '_', got '{file_name}'."
    return None


def _validate_code(code: str, file_name: str, plugin_type: str, context) -> list[str]:
    """
    Run validation checks on plugin code. Returns a list of warning strings.
    Does NOT block saving — the caller writes the file regardless.
    """
    warnings = []

    # 1. Syntax check
    try:
        compile(code, file_name, "exec")
    except SyntaxError as e:
        warnings.append(f"SyntaxError on line {e.lineno}: {e.msg}")
        return warnings  # Can't do AST checks if syntax is broken

    # 2. Structure check via AST
    try:
        tree = ast.parse(code, file_name)
    except SyntaxError:
        return warnings  # Already reported above

    base_class = _PLUGIN_CONFIG[plugin_type][2]
    _check_class_structure(tree, base_class, plugin_type, warnings)
    _check_import_statement(tree, plugin_type, warnings)

    if plugin_type in ("tool", "task"):
        _check_name_collision(tree, plugin_type, context, warnings)
    elif plugin_type == "service":
        _check_service_structure(tree, warnings)

    return warnings


def _check_class_structure(tree: ast.Module, base_class: str, plugin_type: str, warnings: list):
    """Check that the file has a class inheriting the correct base and a name attribute."""
    found_class = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            base_names = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(base.attr)
            if base_class in base_names:
                found_class = True
                # Check for name attribute
                has_name = False
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id == "name":
                                has_name = True
                if not has_name:
                    warnings.append(f"Class '{node.name}' is missing a 'name' attribute.")
                break
    if not found_class:
        warnings.append(f"No class inheriting {base_class} found. {plugin_type.title()}s must inherit {base_class}.")


def _check_service_structure(tree: ast.Module, warnings: list):
    """Check that the file has a build_services function."""
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_services":
            found = True
            break
    if not found:
        warnings.append("No build_services() function found. Services must define build_services(config).")


def _check_import_statement(tree: ast.Module, plugin_type: str, warnings: list):
    """Check that the code imports the correct Base class from the right stage folder."""
    expected_module, expected_names = _EXPECTED_IMPORTS[plugin_type]

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module == expected_module:
            imported_names = {alias.name for alias in node.names}
            if expected_names <= imported_names:
                return  # Correct import found

    names_str = ", ".join(sorted(expected_names))
    warnings.append(
        f"Missing required import: 'from {expected_module} import {names_str}'. "
        f"{plugin_type.title()}s must import their base class from the correct module."
    )


def _check_name_collision(tree: ast.Module, plugin_type: str, context, warnings: list):
    """Check if the plugin name collides with a baked-in plugin."""
    # Extract the name = "..." value from the class
    plugin_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "name":
                            if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                                plugin_name = item.value.value
    if not plugin_name:
        return

    # Check against baked-in registries
    if plugin_type == "tool" and hasattr(context, "call_tool"):
        # Check tool registry via the db's tool list isn't feasible here,
        # but we can check the baked-in directory for files
        baked_in_dir = _BAKED_IN_DIRS["tool"]
        for py_file in baked_in_dir.glob("tool_*.py"):
            try:
                source = py_file.read_text(encoding="utf-8")
                file_tree = ast.parse(source)
                for node in ast.walk(file_tree):
                    if isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, ast.Assign):
                                for target in item.targets:
                                    if isinstance(target, ast.Name) and target.id == "name":
                                        if isinstance(item.value, ast.Constant) and item.value.value == plugin_name:
                                            warnings.append(
                                                f"Name '{plugin_name}' collides with baked-in "
                                                f"{plugin_type} in {py_file.name}. Choose a unique name."
                                            )
                                            return
            except Exception:
                continue


def _try_register(sandbox_path: Path, plugin_type: str, context, warnings: list):
    """Attempt to load and register the plugin. Appends to warnings on failure."""
    from plugin_discovery import load_single_plugin
    name, error = load_single_plugin(
        plugin_type, sandbox_path,
        context.tool_registry, context.orchestrator,
        context.services, context.config,
    )
    if error:
        warnings.append(f"Registration failed: {error}")


def _extract_plugin_name(sandbox_path: Path, plugin_type: str) -> str | None:
    """Extract the plugin name from source via AST (without importing)."""
    try:
        source = sandbox_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None

    if plugin_type == "service":
        # Services use build_services() — name comes from dict keys, not class attr.
        # Best-effort: return the stem as identifier.
        return sandbox_path.stem

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "name":
                            if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                                return item.value.value
    return None


_SANDBOX_NS = {
    "tool":    "sandbox_tools_{stem}",
    "task":    "sandbox_tasks_{stem}",
    "service": "sandbox_services_{stem}",
}


def _cleanup_module(sandbox_path: Path, plugin_type: str):
    """Remove the sandbox module from sys.modules."""
    ns = _SANDBOX_NS.get(plugin_type, "")
    module_name = ns.format(stem=sandbox_path.stem)
    sys.modules.pop(module_name, None)


def _build_result(sandbox_path: Path, warnings: list, verb: str) -> ToolResult:
    """Build a ToolResult for create/edit actions."""
    if warnings:
        detail = "\n".join(f"  - {w}" for w in warnings)
        return ToolResult(
            success=True,
            llm_summary=(
                f"Plugin {verb} at {sandbox_path.name} but has validation warnings:\n"
                f"{detail}\n"
                f"The file is saved — fix the issues with action='edit'."
            ),
        )
    return ToolResult(
        llm_summary=f"Plugin {verb} and registered: {sandbox_path.name}. Ready to use.",
    )
