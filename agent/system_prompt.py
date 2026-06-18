"""Cache-friendly system prompt assembly.

Returns two messages:
- A combined ``system`` message (static + semi-stable) at position 0 —
  cacheable across turns.
- A ``user`` message tagged ``[SYSTEM CONTEXT UPDATE]`` carrying the
  dynamic runtime context. ConversationLoop merges this into the latest
  real user turn so the structure is one user message containing the
  context block followed by the user's actual content.

The user-role wrapper exists because some providers (MiniMax) reject
``system`` messages anywhere except position 0. Keeping the dynamic block
at the tail of the prompt also preserves the cacheable prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from runtime.agent_scope import AgentScope

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATIC_PROMPT_PATH = Path(__file__).with_name("system_prompt_static.md")

SYSTEM_CONTEXT_MARKER = "[SYSTEM CONTEXT UPDATE]"


@dataclass
class PromptContext:
    """Read-only bag passed to each plugin's ``agent_prompt_for``.

    Plugins read whatever they need (db/services/orchestrator/config/scope/
    frontend_name) to build their system-prompt contribution. Kept distinct from
    the heavier SecondBrainContext so the prompt builder stays a pure function.
    """
    db: Any = None
    services: dict = field(default_factory=dict)
    orchestrator: Any = None
    config: dict = field(default_factory=dict)
    scope: "AgentScope | None" = None
    profile_name: str = "default"
    frontend_name: str | None = None


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
    frontend_name: str | None = None,
    frontend=None,
    command_filter: Callable[[str], bool] | None = None,
) -> list[dict[str, str]]:
    """Build ordered system prompt messages.

    Optional per-plugin guidance is collected from whatever tools/services/tasks/
    commands/frontend are currently in scope (each plugin's ``agent_prompt_for``),
    so installed packages bring their own guidance and uninstalling removes it —
    the kernel no longer hardcodes prompt text for plugins it may not ship.
    """
    r = tool_registry
    pctx = PromptContext(
        db=db, services=services or {}, orchestrator=orchestrator,
        config=config or {}, scope=scope, profile_name=profile_name,
        frontend_name=frontend_name,
    )
    semi = [
        _tool_catalog(r),
        _command_catalog(commands, command_filter),
        _collect(_visible_tools_for_prompt(r), pctx),
        _collect(_loaded_services_for_prompt(services), pctx),
        _collect(_tasks_for_prompt(orchestrator), pctx),
        _collect(_visible_commands_for_prompt(commands, command_filter), pctx),
        _collect([frontend] if frontend is not None else [], pctx),
    ]
    dynamic = [
        _current_datetime(),
        _model_status(services),
        _profile_status(profile_name, scope),
        _services_status(services),
        _pipeline_status(db, orchestrator),
        _sync_dirs(config),
        _agent_memory(),
        _conversation_metadata(conversation_metadata),
        _prompt_extras(prompt_extras),
        notification_suffix,
        _scope_prompt_note(profile_name, scope),
        getattr(scope, "prompt_suffix", "") if scope else "",
        extra_suffix,
    ]
    static_block = _section("STATIC SYSTEM PROMPT", _static_prompt())
    semi_block = _section("SEMI-STABLE TOOL/SCHEMA INFO", "\n\n".join(s for s in semi if s))
    dynamic_block = _section(SYSTEM_CONTEXT_MARKER.strip("[]"), "\n\n".join(s for s in dynamic if s))
    return [
        {"role": "system", "content": f"{static_block}\n\n{semi_block}"},
        {"role": "user", "content": dynamic_block},
    ]


def build_system_prompt(*args, **kwargs) -> str:
    """Compatibility wrapper for old callers that expect one system string."""
    return "\n\n".join(m["content"] for m in build_prompt_sections(*args, **kwargs) if m.get("content"))


def _section(title: str, content: str) -> str:
    """Render a labeled section as a string."""
    return f"[{title}]\n{content.strip()}"


def _collect(plugins, ctx: PromptContext) -> str:
    """Join non-empty ``agent_prompt_for`` contributions from in-scope plugins."""
    parts = []
    for plugin in plugins:
        try:
            text = (plugin.agent_prompt_for(ctx) or "").strip()
        except Exception:
            text = ""
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _visible_tools_for_prompt(registry):
    """Tools the agent can currently see (profile-scoped), sorted by name."""
    if not registry or not hasattr(registry, "_visible_tools"):
        return []
    return sorted(registry._visible_tools(), key=lambda t: getattr(t, "name", ""))


def _loaded_services_for_prompt(services: dict):
    """Loaded service instances, sorted by registry name."""
    return [svc for _, svc in sorted((services or {}).items()) if getattr(svc, "loaded", False)]


def _tasks_for_prompt(orchestrator):
    """Registered task instances, sorted by name."""
    tasks = getattr(orchestrator, "tasks", {}) or {}
    return [tasks[name] for name in sorted(tasks)]


def _visible_commands_for_prompt(commands, command_filter):
    """Commands visible under the current frontend's policy, sorted by name."""
    if not commands or not hasattr(commands, "visible_commands"):
        return []
    try:
        return commands.visible_commands(command_filter)
    except Exception:
        return []


def _current_datetime() -> str:
    return f"Current date and time: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}"


def _model_status(services: dict) -> str:
    llm = (services or {}).get("llm")
    if not llm:
        return "Current model: unavailable."
    name = getattr(llm, "_active_name", None)
    inner = getattr(llm, "active", None)
    model = getattr(inner, "model_name", None) if inner else getattr(llm, "model_name", None)
    target = inner or llm
    caps = getattr(target, "capabilities", {}) or {}
    native = set(getattr(target, "native_attachment_modalities", set()) or set())
    parts = []
    for modality, label in (("image", "images"), ("audio", "audio"), ("video", "video")):
        parts.append(f"{label}: {'yes' if caps.get(modality) and modality in native else 'no'}")
    status = f"{name} ({model})." if name and model else f"{name or model or 'unknown'}."
    return (
        f"Current model: {status}\n"
        f"Native attachment processing: {'; '.join(parts)}. "
        "For unsupported modalities, rely only on parsed text or file pointers."
    )


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


def _command_catalog(commands, command_filter=None) -> str:
    lines = ["## Available slash commands"]
    entries = []
    try:
        entries = commands.visible_commands(command_filter) if hasattr(commands, "visible_commands") else []
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


def _agent_memory() -> str:
    """Memory section: the MEMORY.md index inlined, topics listed by name.

    Memory is a folder of per-topic markdown files plus an index
    (see ``plugins/helpers/memory_paths.py``). Only the index is inlined so
    prompt cost stays flat. How to read/write topics is the installed memory
    tool's business — its ``agent_prompt`` carries those instructions, keeping
    plugin guidance out of the kernel.
    """
    from plugins.helpers.memory_paths import INDEX_FILENAME, list_topics, memory_root

    root = memory_root()
    index_path = root / INDEX_FILENAME
    try:
        index = index_path.read_text(encoding="utf-8").strip() if index_path.exists() else ""
    except OSError:
        index = ""
    topics = [p.stem for p in list_topics()]

    lines = [
        "## Memory",
        f"Path: {root}",
        "Durable notes that persist across sessions. The index below is a map, not the content.",
        "",
        "Index (MEMORY.md):",
        index or "(empty)",
    ]
    if topics:
        lines += ["", "Topic files: " + ", ".join(topics)]
    return "\n".join(lines)


def _conversation_metadata(meta: dict[str, Any] | None) -> str:
    if not meta:
        return ""
    lines = "\n".join(["## Current conversation", f"Number: {meta.get('id')}", f"Category: {(meta.get('category') or '').strip() or 'Main'}", f"Title: {(meta.get('title') or '').strip() or 'New Conversation'}"])
    lines += "\nUse conversation IDs to query the 'conversations' and 'conversation_messages' tables. When a conversation gets too long, it will be compacted to save space. History prior to the compaction will still be available in the database, but won't be visible in the conversation context for new messages."
    return lines


def _prompt_extras(extras: dict[str, Any] | None) -> str:
    values = [v for v in (extras or {}).values() if isinstance(v, str) and v]
    return "\n\n".join(values)


def _scope_prompt_note(profile_name: str, scope: AgentScope | None) -> str:
    if profile_name == "default" or not scope or not scope.has_tool_filter:
        return ""
    return (
        f"""## Agent profile limits
You are running under the '{profile_name}' agent profile. Tool access is limited to the tools exposed in this prompt. """
    )
