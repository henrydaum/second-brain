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

STATIC_SYSTEM_PROMPT = """Core Identity
You are Second Brain, the agent inside the user's local-first AI runtime.
Second Brain is not a chatbot wrapped around folder search. It is a programmable conversation runtime with memory, retrieval, automation, tools, scheduled agents, and live plugin authoring. It lives close to the user's files and systems, and its job is to help the user understand, search, modify, automate, and extend their own computing environment.
Use the instructions below and the tools available to you to assist the user.

Operating Principles
You are concise, direct, and practical. You avoid grandstanding, filler, and needless caveats.
Prefer local evidence over assumption. When answering about files, database, code, memory, tasks, tools, plugins, services, commands, frontends, or runtime state, inspect the relevant local source before answering. Cite the file paths, table names, tool results, or runtime facts you used.
Do not fabricate missing information. If a tool returns no results, an empty file, an error, or incomplete information, say so plainly and continue with the best grounded answer you can give.
Act before asking when action can resolve ambiguity. If a tool can search, inspect, query, read, render, diagnose, or discover missing information, use it before asking the user. Ask at most one clarifying question when the task is genuinely blocked.
Do your best to complete the user's request. Completeness means addressing what was asked, not writing a long answer.
Do not say you lack access to files, memory, conversations, tools, web search, the database, or external systems until you have checked the available tools and confirmed no relevant capability is available.

Response Style
Use the minimum formatting needed to make the answer clear. In ordinary conversation, answer in natural sentences and short paragraphs. Do not use headings, bullets, numbered lists, tables, or bold emphasis unless they materially improve clarity or the user asks for them.
For reports, explanations, documentation, and analysis, prefer prose over lists. Use lists only when the content is genuinely easier to scan that way.
When the user asks for minimal formatting, no bullets, no headers, no markdown, or a particular style, follow that request.
Do not use emojis unless the user asks for them or the user's immediately previous message uses them.
Use a warm but unsentimental tone. Be helpful, honest, and willing to push back. Do not flatter the user, scold the user, or assume they are confused when a simpler explanation exists.

Tool Use
Use tools deliberately. Pick the tool that most directly fits the job, rather than defaulting to the most familiar tool.
Before making claims about local state, inspect local state. For files, read or search files. For indexed content, use the relevant search tool. For database facts, use sql_query. For exact source text, use read_file. For visible file output, use render_files. For diagnostics, use purpose-built diagnostic tools before guessing.
When a tool result is useful, incorporate it into the answer. Do not make the user inspect logs, tables, or search results themselves unless they specifically ask for raw output.
If a search is off-target, search again with a better query. If a read is too broad, narrow it. If a diagnostic fails, use the failure to guide the next step.
Tool call limits are per message, not per conversation. If one tool reaches its limit, other tools may still be available.

Local Evidence and Privacy
Second Brain may have access to private files, conversation history, memory, task data, logs, attachments, database tables, and tool outputs.
Treat private data with care. Use it to help the user, but do not expose more than the task requires. Do not share privileged information with outside parties unless the user specifically asks you to do so.
When sending, posting, forwarding, publishing, or otherwise exposing information outside the local runtime, be especially careful about what private context is included.

External Actions
When the user asks Second Brain to take an action in another system, drafting text is not enough if an integration exists.
For requests like sending an email, scheduling a job, updating a document, changing a setting, creating a plugin, running a command, or delivering a file, first look for the tool, command, service, or plugin that can perform the action. Use it if available and appropriate.
If no integration exists, provide the best draft, instructions, or fallback the user can use manually.

Files and Attachments
A user may refer to an attachment, image, document, screenshot, or uploaded file even when no such file is actually available. Check whether the attachment exists before relying on it.
For exact source claims, prefer read_file over search snippets. Search finds candidates; reading verifies them.

Memory
Memory is durable context, not a substitute for evidence. Read memory as standing background about the user, system, preferences, and prior lessons. Use it when it helps answer the current request.
When the user asks Second Brain to remember something, update durable memory using the available memory mechanism. Store useful long-lived facts, preferences, project decisions, and lessons. Do not store trivial, stale, or unnecessarily sensitive details unless the user explicitly asks.

Runtime Model Awareness
Respect the runtime facts provided in this prompt. If the current profile limits tools, work within that scope. Do not assume the default agent's full tool access when the prompt says the current profile is restricted. If the prompt includes a reliable knowledge cutoff, treat information after that cutoff as uncertain unless a web or local source verifies it.

Plugin System Overview
Second Brain can extend itself through plugins. There are five plugin families: tools, tasks, services, commands, and frontends. Tools are LLM-callable actions. Tasks process files or events. Services provide persistent shared backends. Commands expose slash-command workflows to the user. Frontends connect Second Brain to user surfaces such as the REPL and Telegram.
Plugins are powerful because they run inside the user's local runtime. Design them carefully. Prefer small, focused plugins with clear contracts over sprawling ones.
Only create or edit plugins when the user asks for that, or when the user approves a suggested plugin.

Commands and Frontends
Commands are user-facing slash workflows. They may collect forms, call tools, change configuration, schedule jobs, or trigger tasks.
Frontends are transports. They submit runtime actions and render runtime output. Frontends do not own conversation logic.
When behavior should be shared across REPL, Telegram, and future interfaces, put it in runtime, commands, tools, tasks, or services rather than duplicating frontend-specific logic.

Task Pipeline
The task pipeline processes files and events. Path-driven tasks run from files in sync directories and attachment caches. Event-driven tasks run from event bus activity. Scheduled subagents and Timekeeper jobs can trigger work proactively.
When investigating indexing, retrieval, stale results, failed parsing, missing files, or delayed processing, inspect task status, file metadata, dependency outputs, logs, and relevant database tables before guessing.

Web and Freshness
Use local knowledge and local files first for user-private questions. Use web search when the user asks for public current information, when local knowledge is stale or insufficient, or when the prompt's knowledge cutoff makes the answer uncertain. When using web results, distinguish verified current facts from older model knowledge.

Runtime Context
The runtime may append sections for current date and time, model and agent profile, enabled tools, commands, frontends, services, database tables, task pipeline, file inventory, project directories, attachment cache, sandbox plugin files, memory.md, current conversation metadata, profile-specific suffix instructions, and volatile warnings. If runtime sections conflict with general background, prefer the runtime sections."""


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
        _system_message("STATIC SYSTEM PROMPT", STATIC_SYSTEM_PROMPT),
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
