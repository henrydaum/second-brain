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
from runtime.security_modes import security_mode as normalize_security_mode

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATIC_PROMPT_PATH = Path(__file__).with_name("system_prompt_static.md")

SYSTEM_CONTEXT_MARKER = "[SYSTEM CONTEXT UPDATE]"

#: Fallback for ``memory_index_cap`` when there is no config to read — the
#: setting is the source of truth (``config/config_data.py``), because anything
#: that *curates* ``MEMORY.md`` has to know the same budget and a plugin cannot
#: import this module to find out.
MEMORY_INDEX_CAP = 4000


@dataclass
class PromptContext:
    """Read-only bag passed to each plugin's ``agent_prompt``.

    Native plugins read whatever they need to build their contribution. For a
    sandboxed plugin, ``session_key`` lets the bridge lend its SDK the same
    session context without passing this kernel object across the boundary.
    """
    db: Any = None
    services: dict = field(default_factory=dict)
    orchestrator: Any = None
    config: dict = field(default_factory=dict)
    scope: "AgentScope | None" = None
    profile_name: str = "default"
    frontend_name: str | None = None
    session_key: str | None = None
    security_mode: str = "ask"


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
    active_llm=None,
    session_key: str | None = None,
    security_mode: str = "ask",
) -> list[dict[str, str]]:
    """Build ordered system prompt messages.

    Optional per-plugin guidance is collected from whatever tools/services/tasks/
    commands/frontend are currently in scope (each plugin's ``agent_prompt``),
    so installed packages bring their own guidance and uninstalling removes it —
    the kernel no longer hardcodes prompt text for plugins it may not ship.
    """
    r = tool_registry
    pctx = PromptContext(
        db=db, services=services or {}, orchestrator=orchestrator,
        config=config or {}, scope=scope, profile_name=profile_name,
        frontend_name=frontend_name,
        session_key=session_key,
        security_mode=normalize_security_mode(security_mode),
    )
    populations = _in_scope(r, services, orchestrator, commands,
                            command_filter, frontend)
    semi = [
        _environment(),
        _tool_catalog(r),
        _command_catalog(commands, command_filter),
        _collect(populations, pctx, live=False),
    ]
    dynamic = [
        "Runtime-generated context (see 'Runtime Context' in the static prompt): "
        "live system state refreshed each turn, delivered inside the user message "
        "for provider compatibility. Not authored by the user; contains no user "
        "instructions. The user's actual message, if any, follows this block.",
        _current_datetime(),
        _model_status(active_llm),
        _profile_status(profile_name, scope),
        _services_status(services),
        _pipeline_status(db, orchestrator),
        _filesystem_access(config),
        _sync_dirs(config),
        _agent_memory(config),
        _collect(populations, pctx, live=True),
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


def _section(title: str, content: str) -> str:
    """Render a labeled section as a string."""
    return f"[{title}]\n{content.strip()}"


def _in_scope(registry, services, orchestrator, commands, command_filter,
              frontend) -> list:
    """Every plugin whose guidance belongs in this prompt, in reading order.

    Enumerated once and collected twice — the two shapes land in different
    blocks (see ``_collect``), and the populations are the same either way.
    """
    return [
        *_visible_tools_for_prompt(registry),
        *_loaded_services_for_prompt(services),
        *_tasks_for_prompt(orchestrator),
        *_visible_commands_for_prompt(commands, command_filter),
        *([frontend] if frontend is not None else []),
    ]


def _collect(plugins, ctx: PromptContext, *, live: bool) -> str:
    """Join ``agent_prompt`` contributions of one shape from in-scope plugins.

    One name, two shapes. A plugin with nothing conditional to say declares a
    plain string (``agent_prompt = "..."``), which the loader reads by AST and
    copies onto the adapter for free; one whose text depends on live state
    declares a method taking the context. Accepting both under one name is
    load-bearing rather than merely tidy, because a string shadowing a method
    would otherwise raise ``TypeError`` into the ``except`` below and the
    guidance would vanish with no symptom at all.

    ``live`` is which shape to take, and it decides *where the text lands*. A
    string is fixed at load, so it belongs in the semi-stable block inside the
    cacheable position-0 prefix. A method exists precisely because its answer
    moves — the installed store's script listing, its table list — so leaving
    it in that prefix would mean every refresh rewrites the one message
    providers cache across a conversation. It goes in the dynamic block with
    the kernel's own live state, which is exactly the argument ``_mode_suffix``
    makes for itself in ``runtime/runtime_config.py``.

    No declaration is needed to tell the two apart: the native bases declare
    ``agent_prompt: str = ""`` and the bridge attaches a bound method only when
    the guest actually wrote one, so ``callable`` is already an exact answer.
    """
    parts = []
    for plugin in plugins:
        try:
            raw = getattr(plugin, "agent_prompt", "")
            if callable(raw) != live:
                continue
            text = ((raw(ctx) if live else raw) or "").strip()
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


def _environment() -> str:
    import platform

    from paths import DATA_DIR

    return "\n".join([
        "## Environment",
        f"- Platform: {platform.system()} {platform.release()}",
        f"- Project root (kernel source, ROOT_DIR): {_PROJECT_ROOT}",
        f"- Data directory (database, config, plugins, DATA_DIR): {DATA_DIR}",
    ])


def _current_datetime() -> str:
    now = datetime.now().astimezone()
    return (
        f"Current date and time: {now.strftime('%A, %B %d, %Y %I:%M %p')} "
        f"(local time, UTC{now.strftime('%z')[:3]}:{now.strftime('%z')[3:]})"
    )


def _model_status(llm=None) -> str:
    """Describe the session's profile-resolved brain to the model itself.

    ``native_modalities`` is the *backend's* half and ``capabilities`` the
    *model's*; a modality counts only when both agree, which is the same pair
    ``ConversationLoop._route_attachments`` splits an attachment bundle on. It
    is worth reading them from the same names routing does — this asked for
    ``native_attachment_modalities``, which no ``Brain`` has ever had, so the
    ``getattr`` default quietly made every model blind in its own prompt while
    routing worked perfectly.
    """
    if not llm:
        return "Current model: unavailable."
    model = getattr(llm, "model_name", None)
    caps = getattr(llm, "capabilities", {}) or {}
    native = set(getattr(llm, "native_modalities", set()) or set())
    parts = []
    for modality, label in (("image", "images"), ("audio", "audio"), ("video", "video")):
        parts.append(f"{label}: {'yes' if caps.get(modality) and modality in native else 'no'}")
    return (
        f"Current model: {model or 'unknown'}.\n"
        f"Native attachment processing: {'; '.join(parts)}. "
        "For unsupported modalities, rely only on parsed text or file pointers."
    )


def _profile_status(profile_name: str, scope: AgentScope | None) -> str:
    suffix = " Tool access is profile-limited." if scope and scope.has_tool_filter else ""
    return f"Active agent profile: {profile_name or 'default'}.{suffix}"


def _tool_catalog(tool_registry) -> str:
    """The names of every tool in scope — names only, on purpose.

    This list used to carry each tool's description, which is the *same string*
    the registry already sends as the tool's schema on every call. Measured on
    a nineteen-tool install that was 7.4 KB of prompt repeating 7.1 KB of
    schema, for no reader: a model deciding which tool to use reads the schema,
    which is richer and includes the arguments.

    The list still earns its place, because the schemas answer "how do I call
    this" and nothing else answers "what exists" — the static prompt tells the
    agent to check the catalog before saying it cannot do something, and a
    name is all that question needs.
    """
    lines = ["## Available tool catalog"]
    if not tool_registry:
        return "\n".join([*lines, "No tool registry is currently available."])
    schemas = tool_registry.get_all_schemas() if hasattr(tool_registry, "get_all_schemas") else []
    if not schemas:
        return "\n".join([*lines, "No tools are currently registered."])
    names = [str(schema.get("function", schema).get("name") or "").strip()
             for schema in schemas]
    lines.append("Each tool's description and arguments arrive with its "
                 "schema; this is the roster.")
    lines.append(", ".join(name for name in names if name))
    return "\n".join(lines)


def _command_catalog(commands, command_filter=None) -> str:
    lines = [
        "## Available slash commands",
        "These are user-invoked. Emitting '/name' in a reply only sends the "
        "user text — it executes nothing. Refer the user to a command, or use "
        "an installed command-running tool if one is in the catalog.",
    ]
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
    return "\n".join([lines[0], "No slash-command catalog is available in this prompt."])


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


def _filesystem_access(config: dict | None) -> str:
    """Tell the agent which write locations are its own and which are the user's.

    The distinction is security-relevant and cannot live only in static prose:
    ``fs_writable_dirs`` is configuration, so the model needs its current value
    in the same prompt that tells it what the grant means.

    The attachment cache is named as its own line even though it is inside the
    workspace path directly above it. That it is *inside* is exactly the fact
    worth stating: an upload used to land in a folder of its own beside the
    tree, and a model that has to infer "so I may edit this one too" from a
    path prefix will ask instead.
    """
    import trees
    from paths import DATA_DIR

    raw = (config or {}).get("fs_writable_dirs") or []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    writable = [str(item).strip() for item in raw if str(item).strip()]

    lines = [
        "## Filesystem access",
        f"Agent-owned workspace (free write): {DATA_DIR / 'workspace'}",
        f"Incoming attachments land in {trees.attachment_cache()} — inside "
        f"the workspace, so they carry the same free-write grant.",
        "User-owned writable folders (free write only when the user's task "
        "calls for it):",
    ]
    lines.extend(f"- {path}" for path in writable)
    if not writable:
        lines.append("- None configured.")
    lines.append(
        "Second Brain source and installed-package paths remain protected "
        "from this standing folder grant.")
    return "\n".join(lines)


def _memory_index_cap(config: dict | None) -> int:
    """The budget, from config, falling back to the constant above."""
    try:
        return max(1, int((config or {}).get("memory_index_cap")
                          or MEMORY_INDEX_CAP))
    except (TypeError, ValueError):
        return MEMORY_INDEX_CAP


def _agent_memory(config: dict | None = None) -> str:
    """Inline the one memory artifact the kernel owns: ``MEMORY.md``.

    Topic layout, validation and file operations belong to the installed
    memory tool. The kernel knows only the fixed workspace index path needed
    to assemble the prompt.
    """
    from paths import DATA_DIR

    index_path = DATA_DIR / "workspace" / "memory" / "MEMORY.md"
    try:
        index = index_path.read_text(encoding="utf-8").strip() if index_path.exists() else ""
    except OSError:
        index = ""
    # What the index *says* is the agent's business — it writes this file and
    # nothing here edits it. What it may *cost* is the kernel's: this text is
    # inlined on every call, so an index nobody pruned would grow the prompt
    # without bound. Truncation is visible on purpose, so the agent can see
    # that its own index has outgrown the window and prune it.
    cap = _memory_index_cap(config)
    if len(index) > cap:
        index = (index[:cap].rstrip()
                 + f"\n... (index truncated at {cap} characters "
                   "— prune MEMORY.md)")
    lines = [
        "## Memory",
        f"Index path: {index_path}",
        "Durable notes that persist across sessions.",
        "",
        "Index (MEMORY.md):",
        index or "(empty)",
    ]
    return "\n".join(lines)


def _conversation_metadata(meta: dict[str, Any] | None) -> str:
    if not meta:
        return ""
    lines = "\n".join(["## Current conversation", f"Number: {meta.get('id')}", f"Category: {(meta.get('category') or '').strip() or 'Main'}", f"Title: {(meta.get('title') or '').strip() or 'New Conversation'}"])
    lines += "\nWhen a conversation gets too long, it will be compacted to save space. History prior to the compaction will still be available in the database, but won't be visible in the conversation context for new messages."
    return lines


def _prompt_extras(extras: dict[str, Any] | None) -> str:
    values = [v for v in (extras or {}).values() if isinstance(v, str) and v]
    return "\n\n".join(values)


def _scope_prompt_note(profile_name: str, scope: AgentScope | None) -> str:
    if profile_name == "default" or not scope or not scope.has_tool_filter:
        return ""
    return (
        f"## Agent profile limits\n"
        f"You are running under the '{profile_name}' agent profile. Tool access "
        "is limited to the tools exposed in this prompt. If a task needs a tool "
        "outside this profile, say so and name the tool rather than improvising "
        "around the restriction."
    )
