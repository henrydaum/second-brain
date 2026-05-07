"""
System prompt builder.

Single entry point for assembling the agent system prompt. Sections are
gated on which tools the calling agent's registry actually exposes, so a
restricted profile doesn't get a wall of guidance about tools it can't
call. Also handles the scope-limits note and prompt suffixes so callers
don't have to stitch strings.

Built fresh each turn so the LLM always sees current state (newly
registered plugins, service status changes, current memory.md, etc.).
"""

from datetime import datetime
from pathlib import Path

from plugins.services.helpers.parser_registry import get_supported_extensions
from runtime.agent_scope import AgentScope

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_system_prompt(
    db,
    orchestrator,
    tool_registry,
    services: dict,
    *,
    scope: AgentScope | None = None,
    profile_name: str = "default",
    extra_suffix: str = "",
) -> str:
    r = tool_registry  # short alias for gating checks

    sections = [
        _identity(services, r),
        _current_datetime(),
        _available_tools(r),
        _authoring_guidance() if _has_tool(r, "register_plugin") else "",
        _sandbox_files() if _has_tool(r, "register_plugin") else "",
        _attachments() if _has_tool(r, "sql_query") else "",
        _database_tables(db) if _has_tool(r, "sql_query") else "",
        _services_status(services),
        _pipeline_status(db, orchestrator),
        _file_inventory(db) if _has_any(r, "read_file", "hybrid_search", "lexical_search", "semantic_search") else "",
        _agent_memory() if _has_tool(r, "update_memory") else "",
    ]
    prompt = "\n\n".join(s for s in sections if s)

    scope_note = _scope_prompt_note(profile_name, scope)
    if scope_note:
        prompt += "\n\n" + scope_note
    if scope and scope.prompt_suffix:
        prompt += "\n\n" + scope.prompt_suffix
    if extra_suffix:
        prompt += extra_suffix
    return prompt


# ── Tool gating helpers ──────────────────────────────────────────────

def _has_tool(registry, name: str) -> bool:
    return bool(registry) and name in getattr(registry, "tools", {})


def _has_any(registry, *names: str) -> bool:
    return any(_has_tool(registry, n) for n in names)


# ── Static sections ──────────────────────────────────────────────────

def _identity(services: dict, registry) -> str:
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

    from paths import DATA_DIR
    log_path = DATA_DIR / "app.log"

    # (required_tool, line). None means always include.
    habit_bullets: list[tuple[str | None, str]] = [
        (None, "- Use the built-in tools to inspect the system before making assumptions."),
        ("sql_query", "- Past conversations live in the database (conversations and conversation_messages tables) and can be recalled with sql_query."),
        ("read_file", "- For architecture or design intent, read README.md with read_file."),
        ("read_file", f"- A debug log for the current session is at {log_path} — read it with read_file if you need to investigate an error."),
        ("render_files", "- Use render_files when the user would benefit from seeing a file directly."),
        ("register_plugin", "- If the current tools cannot reasonably complete a task, suggest creating a sandbox plugin with register_plugin."),
        (None, "- Tool call limits are per message, not per session. If one tool reaches its limit, you may still use other tools."),
    ]

    memory_block = ""
    if _has_tool(registry, "update_memory"):
        memory_block = (
            "- Call update_memory proactively and often. Triggers:\n"
            "  * The user shares a fact about themselves, their role, or how they like to work → save it.\n"
            "  * You learn something non-obvious about a tool (a quirk, a common failure mode, the right argument pattern, a gotcha) → save the lesson so future sessions don't repeat the mistake.\n"
            "  * A tool call fails and you figure out what would have worked → save the correction.\n"
            "  * The user corrects your behavior or expresses a preference → save it as a standing instruction.\n"
            "  * You discover a useful pattern (e.g. which table to query for X, which plugin handles Y) → save it.\n"
            "  Be rigorous: err on the side of writing a note rather than skipping it. Memory is how you get smarter over time.\n"
        )

    habits = "\n".join(line for tool, line in habit_bullets if tool is None or _has_tool(registry, tool))

    capabilities = ["inspect local files", "query the SQLite database"]
    if _has_tool(registry, "register_plugin"):
        capabilities.append("extend the system through sandbox plugins")
    cap_line = ", ".join(capabilities[:-1]) + ", and " + capabilities[-1] if len(capabilities) > 1 else capabilities[0]

    return (
        "You are Second Brain, the assistant inside the user's local file intelligence system. "
        "Your job is to help the user work with their files, database, tools, and automations. "
        f"You can {cap_line}.\n"
        f"{model_line}"
        "\n"
        "Behavior:\n"
        "- Be concise, grounded, and practical.\n"
        "- Prefer local evidence. When answering about files, code, or system state, cite the file paths, table names, or tool results you used.\n"
        "- If a tool returns no results or an error, say so plainly — don't fabricate data to fill the gap. If you don't know a path, name, or value, use a tool to find it rather than guessing.\n"
        "- Pick the right tool for the job. Before acting, scan the available tools and choose the one that most directly fits the task; don't default to a familiar tool when a more specific one exists.\n"
        "\n"
        "Tool habits:\n"
        f"{habits}\n"
        f"{memory_block}"
    )


def _current_datetime() -> str:
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y %I:%M %p')}"


def _authoring_guidance() -> str:
    return (
        "## Building plugins\n"
        "You can extend the system by authoring sandbox plugins (tools, tasks, services, commands, frontends).\n\n"
        "Project layout:\n"
        "- Built-in plugins live under plugins/tools, plugins/tasks, plugins/services, plugins/commands, plugins/frontends.\n"
        "- Core app code lives under agent, pipeline, runtime, config, events.\n\n"
        "Workflow:\n"
        "1. Read the relevant template with read_file (templates/tool_template.py, task_template.py, service_template.py).\n"
        "2. Read a similar existing plugin for reference. Sandbox plugin paths are listed below.\n"
        "3. Write the file into the right sandbox directory using the file-editing tools.\n"
        "4. Activate it with register_plugin(plugin_type=..., action='register', file_name=...).\n"
        "5. To remove a plugin from the live system without deleting the file, call register_plugin(action='unregister', plugin_name=...).\n"
        "6. If you need extra packages, install them with run_command.\n\n"
        "Naming:\n"
        "- Tools: tool_<name>.py    Tasks: task_<name>.py    Commands: command_<name>.py    Frontends: frontend_<name>.py    Services: <name>.py\n"
        "- Names must be unique across baked-in and sandbox.\n\n"
        "Config settings:\n"
        "Plugins can declare config_settings to add user-configurable values.\n"
        "Each entry: (title, variable_name, description, default, type_info).\n"
        "type_info options: {\"type\": \"text\"}, {\"type\": \"bool\"}, {\"type\": \"slider\", \"range\": (min, max, divs), \"is_float\": False}, {\"type\": \"json_list\"}.\n"
        "Values are stored in plugin_config.json and read via context.config.get(key)."
    )


# ── Dynamic sections ─────────────────────────────────────────────────

def _available_tools(tool_registry) -> str:
    if not tool_registry:
        return ""
    tools = list(tool_registry.tools.values())
    lines = ["## Available tools"]
    if tools:
        for tool in tools:
            desc = (tool.description or "").strip()
            tag = " [sandbox]" if getattr(tool, '_mutable', False) else ""
            lines.append(f"### {tool.name}{tag}")
            lines.append(desc)
    else:
        lines.append("No tools are currently registered.")
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
        return "## Sandbox plugins\nNone yet. Use register_plugin to create one."

    lines = ["## Sandbox plugins (read these exact paths with read_file)"]
    lines.extend(sandbox_lines)
    return "\n".join(lines)


def _attachments() -> str:
    from paths import ATTACHMENT_CACHE
    return (
        "## Attachments\n"
        f"Files the user sends via a frontend (e.g. Telegram photos/documents) are persisted to {ATTACHMENT_CACHE} "
        "and indexed by the normal task pipeline (extracted, chunked, embedded, OCR'd, lexical-indexed). "
        "They're claimed ahead of regular files in the task queue (priority 100 vs. 0), so indexing typically finishes within seconds of upload.\n"
        "\n"
        "Finding recent attachments via sql_query:\n"
        f"  SELECT path, file_name, mtime FROM files WHERE path LIKE '{ATTACHMENT_CACHE}%' ORDER BY mtime DESC LIMIT 20\n"
        "Filenames start with a unix timestamp prefix (e.g. `1700000000_photo.jpg`), so ORDER BY mtime DESC gives newest first. "
        "For a specific conversation's attachments, cross-reference mtime against the conversation's message timestamps in the conversation_messages table.\n"
        "\n"
        "If a user sends a file with an extension the pipeline doesn't understand, the file is still saved to the cache — "
        "use read_file on the exact cache path, or create a new parser task plugin with register_plugin so future files of that type get indexed automatically."
    )


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
    dag = orchestrator.dependency_pipeline_graph() if orchestrator else None
    if dag:
        lines.append(dag)

    # Per-task status counts
    stats = db.get_system_stats().get("tasks", {})
    if stats:
        lines.append("")
        lines.append("Status (P=pending, D=done, F=failed):")
        paused_set = orchestrator.paused if orchestrator else set()
        for name, counts in sorted(stats.items()):
            paused = " [PAUSED]" if name in paused_set else ""
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
    header = (
        "## Memory (from memory.md)\n"
        "Durable notes that persist across sessions. Read them as standing context.\n"
        "Write to this file with update_memory whenever you learn something worth remembering:\n"
        "- Facts about the user (role, preferences, projects, how they work)\n"
        "- Tool lessons (quirks, failure modes, correct usage patterns you discovered the hard way)\n"
        "- Corrections from the user (behavior they want changed, don't repeat the mistake)\n"
        "- System knowledge (which table has what, which plugin to use for X)\n"
        "Be proactive. If you hesitate on whether a note is worth saving, save it. "
        "Keep each entry short and specific — the file has a 1000-character cap, so rewrite and compact when needed.\n"
    )
    if mem_path.exists():
        content = mem_path.read_text()
        return f"\n\n{header}\nCurrent contents:\n{content}"
    return f"\n\n{header}\nCurrent contents: (empty — the file will be created on first update_memory call)"


# ── Scope trailer ────────────────────────────────────────────────────

def _scope_prompt_note(profile_name: str, scope: AgentScope | None) -> str:
    if profile_name == "default" or not scope:
        return ""
    limits = []
    if scope.has_tool_filter:
        limits.append("tool access is limited to the tools exposed in this prompt")
    if not limits:
        return ""
    return (
        "## Agent profile limits\n"
        f"You are running under the '{profile_name}' agent profile.\n"
        f"{' and '.join(limits)}.\n"
        "If something is unavailable or denied, work within that scope instead of assuming full system access."
    )


