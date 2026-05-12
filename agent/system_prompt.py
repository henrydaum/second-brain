"""Cache-friendly system prompt assembly.

The prompt is split into static, semi-stable, and dynamic system messages.
ConversationLoop places the dynamic message after prior history and before
the current user turn so stable prefix text remains cacheable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from plugins.services.helpers.parser_registry import get_supported_extensions
from runtime.agent_scope import AgentScope

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATIC_PROMPT_PATH = Path(__file__).with_name("system_prompt_static.md")


def _static_prompt() -> str:
    return _STATIC_PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_prompt_sections(
    db,
    orchestrator,
    tool_registry,
    services: dict,
    *,
    scope: AgentScope | None = None,
    profile_name: str = "default",
    extra_suffix: str = "",
    commands=None,
    config: dict | None = None,
    conversation_metadata: dict[str, Any] | None = None,
    prompt_extras: dict[str, Any] | None = None,
    notification_suffix: str = "",
) -> list[dict[str, str]]:
    """Build ordered system prompt messages."""
    r = tool_registry
    semi = [
        _tool_catalog(r),
        _command_catalog(commands),
        _retrieval_guidance(),
        _authoring_guidance() if _has_tool(r, "test_plugin") else _plugin_contracts(),
        _sandbox_files() if _has_tool(r, "test_plugin") else "",
        _attachments() if _has_tool(r, "sql_query") else "",
        _database_tables(db) if _has_tool(r, "sql_query") else "",
    ]
    dynamic = [
        _current_datetime(),
        _model_status(services),
        _profile_status(profile_name, scope),
        _services_status(services),
        _pipeline_status(db, orchestrator),
        _sync_dirs(config),
        _file_inventory(db) if _has_any(r, "read_file", "hybrid_search", "lexical_search", "semantic_search") else "",
        _agent_memory(),
        _conversation_metadata(conversation_metadata),
        _prompt_extras(prompt_extras),
        notification_suffix,
        _scope_prompt_note(profile_name, scope),
        getattr(scope, "prompt_suffix", "") if scope else "",
        extra_suffix,
    ]
    return [
        _system_message("STATIC SYSTEM PROMPT", _static_prompt()),
        _system_message("SEMI-STABLE TOOL/SCHEMA INFO", "\n\n".join(s for s in semi if s)),
        _system_message("DYNAMIC RUNTIME CONTEXT", "\n\n".join(s for s in dynamic if s)),
    ]


def build_system_prompt(*args, **kwargs) -> str:
    """Compatibility wrapper for old callers that expect one system string."""
    return "\n\n".join(m["content"] for m in build_prompt_sections(*args, **kwargs) if m.get("content"))


def _system_message(title: str, content: str) -> dict[str, str]:
    return {"role": "system", "content": f"[{title}]\n{content.strip()}"}


def _has_tool(registry, name: str) -> bool:
    return bool(registry) and name in getattr(registry, "tools", {})


def _has_any(registry, *names: str) -> bool:
    return any(_has_tool(registry, n) for n in names)


def _current_datetime() -> str:
    return f"Current date and time: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}"


def _model_status(services: dict) -> str:
    llm = (services or {}).get("llm")
    if not llm:
        return "Current model: unavailable."
    name = getattr(llm, "_active_name", None)
    inner = getattr(llm, "active", None)
    model = getattr(inner, "model_name", None) if inner else getattr(llm, "model_name", None)
    return "Current model: " + (f"{name} ({model})." if name and model else f"{name or model or 'unknown'}.")


def _profile_status(profile_name: str, scope: AgentScope | None) -> str:
    suffix = " Tool access is profile-limited." if scope and scope.has_tool_filter else ""
    return f"Active agent profile: {profile_name or 'default'}.{suffix}"


def _tool_catalog(tool_registry) -> str:
    lines = ["## Available tool catalog"]
    if not tool_registry:
        return "\n".join([*lines, "No tool registry is currently available."])
    schemas = tool_registry.get_all_schemas() if hasattr(tool_registry, "get_all_schemas") else []
    if not schemas:
        return "\n".join([*lines, "No tools are currently registered."])
    for schema in schemas:
        fn = schema.get("function", schema)
        desc = (fn.get("description") or "").strip().replace("\n", " ")
        lines.append(f"- {fn.get('name')}: {desc}" if desc else f"- {fn.get('name')}")
    return "\n".join(lines)


def _command_catalog(commands) -> str:
    lines = ["## Available slash commands"]
    entries = []
    try:
        entries = commands.visible_commands() if hasattr(commands, "visible_commands") else []
    except Exception:
        entries = []
    if entries:
        for cmd in entries:
            desc = (getattr(cmd, "description", "") or "").strip()
            hint = _form_hint(getattr(cmd, "form", None), commands)
            lines.append(f"- /{cmd.name}{(' ' + hint) if hint else ''}: {desc}" if desc else f"- /{cmd.name}{(' ' + hint) if hint else ''}")
        return "\n".join(lines)
    if isinstance(commands, dict) and commands:
        for name, spec in sorted(commands.items()):
            hint = _form_hint(getattr(spec, "form", None), None)
            lines.append(f"- /{name}{(' ' + hint) if hint else ''}")
        return "\n".join(lines)
    return "\n".join([*lines, "No slash-command catalog is available in this prompt."])


def _form_hint(form, commands=None) -> str:
    try:
        steps = form({}, commands.context(None) if commands and hasattr(commands, "context") else None) if callable(form) else (form or [])
    except Exception:
        steps = []
    return " ".join(f"<{s.name}>" if getattr(s, "required", True) else f"[{s.name}]" for s in steps)


def _retrieval_guidance() -> str:
    return (
        "## Retrieval guidance\n"
        "Use hybrid_search by default for broad local retrieval when available. Use lexical_search for exact names, strings, filenames, symbols, errors, stack traces, and quoted phrases. Use semantic_search for meaning-based questions and vague recollections. Use sql_query for structured facts, pipeline state, task status, conversation history, and file metadata. When results disagree, prefer exact reads, newer source files, and system-owned metadata over summaries or stale memories."
    )


def _plugin_contracts() -> str:
    return (
        "## Plugin contracts\n"
        "Plugin families: tools, tasks, services, commands, and frontends. Built-in plugins live under plugins/<family>; sandbox plugins live under the matching sandbox directory in DATA_DIR. Use templates as the source of truth, keep plugins focused, and test plugin changes with test_plugin when that tool is available."
    )


def _authoring_guidance() -> str:
    return (
        "## Building plugins\n"
        "You can extend the system by authoring plugins: tools, tasks, services, commands, and frontends.\n\n"
        "Project layout:\n"
        "- Built-in plugins live under plugins/tools, plugins/tasks, plugins/services, plugins/commands, plugins/frontends.\n"
        "- Sandbox plugins live under the matching sandbox directory in DATA_DIR.\n"
        "- Core app code lives under agent, pipeline, runtime, config, events.\n\n"
        "Templates:\n"
        "- tools: templates/tool_template.py -> sandbox_tools/tool_<name>.py\n"
        "- tasks: templates/task_template.py -> sandbox_tasks/task_<name>.py\n"
        "- services: templates/service_template.py -> sandbox_services/<name>Service.py or <name>.py\n"
        "- commands: templates/command_template.py -> sandbox_commands/command_<name>.py\n"
        "- frontends: templates/frontend_template.py -> sandbox_frontends/frontend_<name>.py\n\n"
        "Workflow:\n"
        "1. Read the relevant template with read_file.\n"
        "2. Read a similar existing plugin for reference. Sandbox plugin paths are listed below.\n"
        "3. Ask one focused question only when a missing decision materially changes the plugin.\n"
        "4. Write the file into the right plugin directory using the file-editing tools.\n"
        "5. If plugin_watcher is autoloaded, it loads valid plugin files when they are added or changed.\n"
        "6. Call test_plugin(plugin_path=...) after edits for naming, import, contract, and suggestion diagnostics.\n"
        "7. Treat pytest as broad regression context, not proof the plugin behavior is correct.\n"
        "8. If diagnostics, pytest, or watcher logs show a failure, edit the same file and test again.\n"
        "9. To remove a plugin durably and from the live runtime, delete its plugin file; plugin_watcher unloads it when enabled.\n\n"
        "Naming: tools tool_<name>.py; tasks task_<name>.py; commands command_<name>.py; frontends frontend_<name>.py; services <name>.py. Names must be unique across built-in and sandbox plugins.\n"
        "Config settings are tuples of (title, variable_name, description, default, type_info), stored in plugin_config.json and read with context.config.get(key)."
    )


def _sandbox_files() -> str:
    from paths import SANDBOX_COMMANDS, SANDBOX_FRONTENDS, SANDBOX_SERVICES, SANDBOX_TASKS, SANDBOX_TOOLS
    lines = []
    for sd in (SANDBOX_TOOLS, SANDBOX_TASKS, SANDBOX_SERVICES, SANDBOX_COMMANDS, SANDBOX_FRONTENDS):
        if sd.exists():
            lines.extend(f"  {p}" for p in sorted(sd.glob("*.py")) if not p.name.startswith("_"))
    return "## Sandbox plugins\n" + ("\n".join(lines) if lines else "None yet. Use templates plus edit_file to create one; test it with test_plugin. plugin_watcher loads valid plugin files automatically when enabled.")


def _attachments() -> str:
    from paths import ATTACHMENT_CACHE
    return (
        "## Attachments\n"
        f"Files sent through frontends are persisted to {ATTACHMENT_CACHE} and indexed by the normal task pipeline. To find recent attachments with sql_query: "
        f"SELECT path, file_name, mtime FROM files WHERE path LIKE '{ATTACHMENT_CACHE}%' ORDER BY mtime DESC LIMIT 20. "
        "For unsupported extensions, use read_file on the exact cache path or create a parser plugin."
    )


def _database_tables(db) -> str:
    try:
        names = [row[0] for row in db.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")["rows"]]
    except Exception:
        names = []
    return "## Database tables (inspect with sql_query)\n" + (", ".join(names) if names else "No tables yet.")


def _pipeline_status(db, orchestrator) -> str:
    lines = ["## Task pipeline"]
    try:
        dag = orchestrator.dependency_pipeline_graph() if orchestrator else None
        stats = db.get_system_stats().get("tasks", {}) if db else {}
    except Exception:
        dag, stats = None, {}
    if dag:
        lines.append(dag)
    if stats:
        lines += ["", "Status (P=pending, D=done, F=failed):"]
        paused = getattr(orchestrator, "paused", set()) if orchestrator else set()
        lines += [f"  {n}: P:{c['PENDING']} D:{c['DONE']} F:{c['FAILED']}{' [PAUSED]' if n in paused else ''}" for n, c in sorted(stats.items())]
    if len(lines) == 1:
        lines.append("No task status is currently available.")
    return "\n".join(lines)


def _services_status(services: dict) -> str:
    if not services:
        return "## Services\nNo services are currently registered."
    return "## Services\n" + ", ".join(f"{name} ({'loaded' if getattr(svc, 'loaded', False) else 'unloaded'})" for name, svc in sorted(services.items()))


def _sync_dirs(config: dict | None) -> str:
    dirs = (config or {}).get("sync_directories") or []
    return "## Sync directories\n" + ("\n".join(f"- {d}" for d in dirs) if dirs else "None configured.")


def _file_inventory(db) -> str:
    try:
        stats = db.get_system_stats().get("files", {}) if db else {}
    except Exception:
        stats = {}
    total = sum(stats.values()) if stats else 0
    lines = ["## File inventory", (", ".join(f"{c} {m}" for m, c in sorted(stats.items())) + f" ({total} total)") if stats else "No files indexed yet."]
    exts = sorted(get_supported_extensions())
    if exts:
        lines.append("Supported extensions: " + " ".join(exts))
    return "\n".join(lines)


def _agent_memory() -> str:
    from paths import DATA_DIR
    path = DATA_DIR / "memory.md"
    content = path.read_text() if path.exists() else "(empty)"
    return (
        "## Memory (from memory.md)\n"
        f"Path: {path}\n"
        "Durable notes that persist across sessions. Read them as standing context. Nightly, dream_memory may rewrite memory.md with reusable lessons and preferences.\n"
        f"Current contents:\n{content}"
    )


def _conversation_metadata(meta: dict[str, Any] | None) -> str:
    if not meta:
        return ""
    return "\n".join(["## Current conversation", f"Number: {meta.get('id')}", f"Category: {(meta.get('category') or '').strip() or 'Main'}", f"Title: {(meta.get('title') or '').strip() or 'New Conversation'}"])


def _prompt_extras(extras: dict[str, Any] | None) -> str:
    values = [v for v in (extras or {}).values() if isinstance(v, str) and v]
    return "\n\n".join(values)


def _scope_prompt_note(profile_name: str, scope: AgentScope | None) -> str:
    if profile_name == "default" or not scope or not scope.has_tool_filter:
        return ""
    return (
        "## Agent profile limits\n"
        f"You are running under the '{profile_name}' agent profile. Tool access is limited to the tools exposed in this prompt. "
        "If something is unavailable or denied, work within that scope instead of assuming full system access."
    )
