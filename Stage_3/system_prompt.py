"""
System prompt builder.

Assembles a system prompt for the agent that includes a brief identity,
actionable guidance, tool descriptions, and dynamic runtime state.
Built fresh each time the user sends a message so the LLM always sees
current state (e.g. newly registered plugins, service status changes).
"""

from pathlib import Path

from Stage_1.registry import get_supported_extensions

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_system_prompt(db, orchestrator, tool_registry, services: dict) -> str:
    sections = [
        _identity(),
        _available_tools(tool_registry),
        _authoring_guidance(),
        _sandbox_files(),
        _database_tables(db),
        _services_status(services),
        _pipeline_status(db, orchestrator),
        _file_inventory(db),
        _agent_memory(),
    ]
    return "\n\n".join(s for s in sections if s)


# ── Static sections ──────────────────────────────────────────────────

def _identity() -> str:
    return (
        "You are the Second Brain assistant — an AI embedded in a local file intelligence system. "
        "You have tools to search and query a SQLite database of the user's files, and to "
        "create new tools, tasks, and services via a sandbox plugin system.\n"
        "\n"
        "Guidelines:\n"
        "- Be concise. When answering questions about files, cite which files your answers come from.\n"
        "- Past conversations are stored in the database (conversations, conversation_messages tables) and can be recalled with sql_query.\n"
        "- If a question can't be answered with existing plugins, and it seems doable, suggest creating a new one with tool_build_plugin.\n"
        "- For architecture details and design philosophy, use read_file(path='README.md').\n"
        "- The tool call limit is per-message, not per-session. "
        "If you hit the limit on one tool, you can still call others."
    )


def _authoring_guidance() -> str:
    return (
        "## Building plugins\n"
        "You can extend the system by creating sandbox plugins (tools, tasks, services).\n\n"
        "Workflow:\n"
        "1. Read the template: read_file(path='templates/tool_template.py') (or task_template.py, service_template.py)\n"
        "2. Read similar existing plugins for reference (sandbox paths listed below, baked-in paths use read_file with relative paths like 'Stage_3/tools/tool_hybrid_search.py')\n"
        "3. Create: build_plugin(plugin_type='tool', file_name='tool_foo.py', action='create', code='...')\n"
        "4. Fix errors: build_plugin(action='edit', search_block='...', replace_block='...')\n"
        "5. Install packages if needed: run_command(command='pip install requests')\n"
        "6. If the plugin is a tool, test it out by calling it.\n\n"
        "Naming: tools are tool_<name>.py, tasks are task_<name>.py, services are only <name>.py\n"
        "Names must be unique — no collisions with baked-in plugins.\n\n"
        "Config settings:\n"
        "Plugins can declare config_settings to add user-configurable values to the Settings UI.\n"
        "Each entry: (title, variable_name, description, default, type_info).\n"
        "type_info options: {\"type\": \"text\"}, {\"type\": \"bool\"}, {\"type\": \"slider\", \"range\": (min, max, divs), \"is_float\": False}, {\"type\": \"json_list\"}.\n"
        "Values are stored in plugin_config.json and accessed via context.config.get(key).\n"
        "See the templates for full examples."
    )


# ── Dynamic sections ─────────────────────────────────────────────────

def _available_tools(tool_registry) -> str:
    if not tool_registry:
        return ""
    enabled = [t for t in tool_registry.tools.values() if t.agent_enabled]
    disabled = [t for t in tool_registry.tools.values() if not t.agent_enabled]
    lines = ["## Your tools (call via function calling)"]
    if enabled:
        for tool in enabled:
            desc = (tool.description or "").strip()
            tag = " [sandbox]" if getattr(tool, '_mutable', False) else ""
            lines.append(f"### {tool.name}{tag}")
            lines.append(desc)
    else:
        lines.append("No tools are currently enabled.")
    if disabled:
        lines.append("")
        lines.append(
            "Disabled tools (cannot call directly, used internally): "
            + ", ".join(t.name for t in disabled)
        )
    return "\n".join(lines)


def _sandbox_files() -> str:
    from paths import SANDBOX_TOOLS, SANDBOX_TASKS, SANDBOX_SERVICES

    sandbox_lines = []
    for label, sd in [("tools", SANDBOX_TOOLS), ("tasks", SANDBOX_TASKS), ("services", SANDBOX_SERVICES)]:
        if not sd.exists():
            continue
        for py_file in sorted(sd.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            sandbox_lines.append(f"  {py_file}")

    if not sandbox_lines:
        return "## Sandbox plugins\nNone yet. Use build_plugin to create one."

    lines = ["## Sandbox plugins (use these full paths with read_file)"]
    lines.extend(sandbox_lines)
    return "\n".join(lines)


def _database_tables(db) -> str:
    try:
        result = db.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = [row[0] for row in result["rows"]]
    except Exception:
        names = []

    if not names:
        return "## Database\nNo tables yet."

    return "## Database tables (use sql_query to explore)\n" + ", ".join(names)


def _pipeline_status(db, orchestrator) -> str:
    lines = ["## Task pipeline"]

    # DAG
    dag = orchestrator.dependency_pipeline_graph()
    if dag:
        lines.append(dag)

    # Per-task status counts
    stats = db.get_system_stats().get("tasks", {})
    if stats:
        lines.append("")
        lines.append("Status (P=pending, D=done, F=failed):")
        for name, counts in sorted(stats.items()):
            paused = " [PAUSED]" if name in orchestrator.paused else ""
            lines.append(
                f"  {name}: P:{counts['PENDING']} D:{counts['DONE']} F:{counts['FAILED']}{paused}"
            )

    return "\n".join(lines)


def _services_status(services: dict) -> str:
    if not services:
        return ""
    parts = []
    for name, svc in services.items():
        status = "loaded" if getattr(svc, "loaded", False) else "unloaded"
        parts.append(f"{name} ({status})")
    return "## Services\n" + ", ".join(parts)


def _file_inventory(db) -> str:
    file_stats = db.get_system_stats().get("files", {})
    total = sum(file_stats.values()) if file_stats else 0

    lines = ["## File inventory"]

    if file_stats:
        parts = [f"{count} {mod}" for mod, count in sorted(file_stats.items())]
        lines.append(", ".join(parts) + f" ({total} total)")
    else:
        lines.append("No files indexed yet.")

    # Supported extensions
    exts = sorted(get_supported_extensions())
    if exts:
        lines.append("Supported extensions: " + " ".join(exts))

    return "\n".join(lines)


def _agent_memory() -> str:
    """memory.md is developed by the agent using tool_update_memory"""
    from paths import DATA_DIR
    mem_path = DATA_DIR / "memory.md"
    if mem_path.exists():
        content = mem_path.read_text()
        return f"\n\n## Memory (from memory.md — please use tool_update_memory to save important facts, such as user info, to here)\n{content}"
    return ""