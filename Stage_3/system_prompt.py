"""
System prompt builder.

Assembles a system prompt for the agent that includes a brief identity,
actionable guidance, tool descriptions, and dynamic runtime state.
Built fresh each time the user sends a message so the LLM always sees
current state (e.g. newly registered plugins, service status changes).
"""

from datetime import datetime
from pathlib import Path

from Stage_1.registry import get_supported_extensions

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_system_prompt(db, orchestrator, tool_registry, services: dict) -> str:
    sections = [
        _identity(services),
        _current_datetime(),
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

def _identity(services: dict) -> str:
    # Resolve the active model name from the LLM router
    model_line = ""
    llm = services.get("llm")
    if llm:
        name = getattr(llm, "_active_name", None)
        inner = getattr(llm, "active", None)
        inner_model = getattr(inner, "model_name", None) if inner else None
        if name and inner_model:
            model_line = f"\nYour current model: {name} ({inner_model}).\n"
        elif name:
            model_line = f"\nYour current model: {name}.\n"

    return (
        "You are Second Brain, the assistant inside the user's local file intelligence system. "
        "Your job is to help the user work with their files, database, tools, and automations. "
        "You can inspect local files, query the SQLite database, and extend the system through sandbox plugins.\n"
        f"{model_line}"
        "\n"
        "Guiding principles:\n"
        "- Be concise, grounded, and practical.\n"
        "- Prefer local evidence. When answering about files or code, cite the relevant file paths, table names, or tool results you used.\n"
        "- Use the built-in tools to inspect the system before making assumptions.\n"
        "- Past conversations live in the database (especially the conversations and conversation_messages tables) and can be recalled with sql_query.\n"
        "- For architecture or design intent, read README.md with read_file.\n"
        "- If the current tools cannot reasonably complete a task, suggest creating a sandbox plugin with build_plugin.\n"
        "- Use render_files when the user would benefit from seeing a file directly.\n"
        "- Use update_memory only for durable preferences, standing instructions, or lessons that should shape future behavior across sessions.\n"
        "- Tool call limits are per message, not per session. If one tool reaches its limit, you may still use other tools."
    )


def _current_datetime() -> str:
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y %I:%M %p')}"


def _authoring_guidance() -> str:
    return (
        "## Building plugins\n"
        "You can extend the system by creating sandbox plugins (tools, tasks, and services).\n\n"
        "Recommended workflow:\n"
        "1. Read the relevant template with read_file(path='templates/tool_template.py') (or task_template.py / service_template.py).\n"
        "2. Read a similar existing plugin for reference. Sandbox plugin paths are listed below; built-in plugins can be read with paths like 'Stage_3/tools/tool_hybrid_search.py'.\n"
        "3. Create a new plugin with build_plugin(action='create', ...).\n"
        "4. Refine it with build_plugin(action='edit', search_block='...', replace_block='...').\n"
        "5. If needed, install packages with run_command.\n"
        "6. If you created a tool, call it to verify behavior.\n\n"
        "Naming:\n"
        "- Tools use tool_<name>.py\n"
        "- Tasks use task_<name>.py\n"
        "- Services use <name>.py\n"
        "- Names must be unique and must not collide with built-in plugins.\n\n"
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
    lines = ["## Available tools"]
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

    lines = ["## Sandbox plugins (read these exact paths with read_file)"]
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

    return "## Database tables (inspect with sql_query)\n" + ", ".join(names)


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
        return f"\n\n## Memory (from memory.md)\nUse this for durable lessons, preferences, and standing context.\n{content}"
    return ""
