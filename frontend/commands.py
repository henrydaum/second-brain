"""
Command registry for the slash-command system.

Each command is a CommandEntry with a name, description, argument hint,
and handler callable. The registry provides autocomplete filtering and
dispatch.

``register_core_commands()`` registers the shared commands used by the
frontends. Each frontend can add overrides on top.
"""

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger("Commands")


# Help section order — commands without a category fall into "Other".
_HELP_SECTIONS = [
    "Conversation",
    "Services & Tools",
    "Tasks",
    "Config & System",
    "Other",
]


@dataclass
class CommandEntry:
    name: str               # e.g. "services"
    description: str        # e.g. "List services and status"
    arg_hint: str = ""      # e.g. "<service_name>" — shown in autocomplete
    handler: Callable = None  # fn(arg: str) -> str | None
    arg_completions: Callable = None  # () -> list[str] — dynamic arg suggestions
    hide_from_help: bool = False      # if True, omit from /help and command menus
    category: str = "Other"           # grouping for /help output


class CommandRegistry:
    def __init__(self):
        self._commands: dict[str, CommandEntry] = {}

    def register(self, entry: CommandEntry):
        self._commands[entry.name] = entry

    def get_completions(self, prefix: str) -> list[CommandEntry]:
        """Return commands whose name starts with *prefix* (case-insensitive)."""
        prefix = prefix.lower()
        return sorted([
            cmd for cmd in self._commands.values()
            if cmd.name.startswith(prefix)
        ], key=lambda cmd: cmd.name)

    def dispatch(self, name: str, arg: str) -> str | None:
        """Look up a command by name and call its handler. Returns output string or None."""
        entry = self._commands.get(name)
        if entry is None:
            return f"Unknown command: '/{name}'. Run /help to see available commands."
        try:
            return entry.handler(arg)
        except Exception as e:
            logger.exception(f"Command '/{name}' handler raised")
            return f"Command '/{name}' failed: {e}"

    def all_commands(self) -> list[CommandEntry]:
        """Return all registered commands in alphabetical order."""
        return sorted(self._commands.values(), key=lambda cmd: cmd.name)

    def visible_commands(self) -> list[CommandEntry]:
        """Return help/menu-visible commands in alphabetical order."""
        return [
            cmd for cmd in self.all_commands()
            if not getattr(cmd, "hide_from_help", False)
        ]


def _build_help(registry: CommandRegistry) -> str:
    """Format the /help output grouped by category."""
    by_cat: dict[str, list[CommandEntry]] = {}
    for cmd in registry.visible_commands():
        by_cat.setdefault(cmd.category or "Other", []).append(cmd)

    ordered = [c for c in _HELP_SECTIONS if c in by_cat]
    ordered += [c for c in by_cat if c not in ordered]

    lines = ["Commands:"]
    for cat in ordered:
        lines.append("")
        lines.append(f"{cat}:")
        for cmd in by_cat[cat]:
            hint = f" {cmd.arg_hint}" if cmd.arg_hint else ""
            label = f"/{cmd.name}{hint}"
            lines.append(f"  {label:<26} {cmd.description}")
    return "\n".join(lines)


def _mask_api_key(value: str) -> str:
    """Mask an API key for display — env-var refs pass through."""
    if not isinstance(value, str) or not value:
        return value
    if value.startswith("$") or value.isupper():  # env-var-style reference
        return value
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def _describe_llm(model_name: str, profile: dict, default: bool, loaded: bool) -> str:
    """Render a single LLM line for /llm list."""
    marker = "*" if default else " "
    status = "default" if default else ("loaded" if loaded else "")
    cls = profile.get("llm_service_class") or "?"
    ctx = profile.get("llm_context_size") or 0
    if ctx and ctx >= 1000:
        ctx_str = f"ctx {ctx // 1000}k"
    elif ctx:
        ctx_str = f"ctx {ctx}"
    else:
        ctx_str = "ctx auto"
    endpoint = profile.get("llm_endpoint") or "(default)"
    return (f"  {marker} {model_name:<24} {status:<8} "
            f"{cls:<14} {ctx_str:<10} {endpoint}")


def _describe_agent_profile(name: str, profile: dict, active: bool) -> str:
    """Render a single agent profile line for /agent list."""
    marker = "*" if active else " "
    status = "active" if active else ""
    llm_ref = profile.get("llm") or "default"
    scope_parts = []
    tools_mode = profile.get("whitelist_or_blacklist_tools", "blacklist")
    tables_mode = profile.get("whitelist_or_blacklist_tables", "blacklist")
    folders_mode = profile.get("whitelist_or_blacklist_folders", "blacklist")
    scope_parts.append(f"tools{ '+' if tools_mode == 'whitelist' else '-' }{len(profile.get('tools_list') or [])}")
    scope_parts.append(f"tables{ '+' if tables_mode == 'whitelist' else '-' }{len(profile.get('tables_list') or [])}")
    scope_parts.append(f"folders{ '+' if folders_mode == 'whitelist' else '-' }{len(profile.get('folders_list') or [])}")
    if profile.get("prompt_suffix"):
        scope_parts.append("prompt+")
    scope_str = ("  [" + ",".join(scope_parts) + "]") if scope_parts else ""
    return (f"  {marker} {name:<16} {status:<8} llm={llm_ref}{scope_str}")


_AGENT_MODE_FIELDS = (
    "whitelist_or_blacklist_tools",
    "whitelist_or_blacklist_tables",
    "whitelist_or_blacklist_folders",
)
_AGENT_LIST_FIELDS = ("tools_list", "tables_list", "folders_list")
_AGENT_SCOPE_FIELDS = _AGENT_MODE_FIELDS + _AGENT_LIST_FIELDS
_AGENT_PROFILE_FIELDS = ("llm", "prompt_suffix") + _AGENT_SCOPE_FIELDS
_LLM_PROFILE_FIELDS = ("llm_endpoint", "llm_api_key", "llm_context_size", "llm_service_class")


def _normalize_agent_profile(profile: dict) -> dict:
    """Coerce profile fields into the canonical shape."""
    profile = dict(profile)
    for key in _AGENT_MODE_FIELDS:
        profile[key] = profile.get(key) or "blacklist"
    for key in _AGENT_LIST_FIELDS:
        val = profile.get(key, [])
        profile[key] = [] if val in (None, "", {}) else val
    return profile


def active_agent_name(config: dict) -> str:
    return config.get("active_agent_profile") or "default"


def active_agent_line(config: dict) -> str:
    return f"Agent: {active_agent_name(config)}."


def new_conversation_message(config: dict) -> str:
    return f"New conversation started. {active_agent_line(config)}"


def register_core_commands(registry: CommandRegistry, ctrl, services, tool_registry,
                           root_dir, get_agent=None, set_conversation_id=None,
                           refresh_agent=None, rescope_agents=None):
    """Register commands shared by Telegram and REPL.

    These are pure ctrl-wrapper commands with no UI-specific side effects.
    Each UI should call this first, then register/override its own commands.

    Parameters:
        get_agent:           Optional callable returning the current Agent instance
                             (or None). Used by /call, /new, /cancel, and /history.
        set_conversation_id: Optional callable(int | None) to update the current
                             conversation ID. Used by /history and /new.
        refresh_agent:       Optional callable() that rebuilds the frontend's Agent
                             instance, preserving conversation history. Used by
                             /refresh to recover from a stuck tool call.
        rescope_agents:      Optional callable() invoked after the active agent
                             profile changes, so existing session agents pick up
                             the new scope (tool filter, table filter, prompt
                             suffix) on their next message.
    """
    _set_conversation_id = set_conversation_id
    _refresh_agent = refresh_agent
    _rescope_agents = rescope_agents
    import json as _json
    import config.config_manager as _cm
    from config.config_data import SETTINGS_DATA as _SD
    from plugins.plugin_discovery import get_plugin_settings as _get_ps
    from agent.history_utils import heal_orphan_tool_calls
    from frontend.formatters import (
        format_services, format_tasks,
        format_tools, format_locations,
        format_tool_result, format_scheduled_jobs,
    )

    # Build a unified setting map (core + plugin) at registration time.
    # Plugin settings are added lazily on first use since discovery may
    # not be complete at import time.
    def _is_hidden(type_info):
        return isinstance(type_info, dict) and type_info.get("hidden") is True

    _core_map = {name: (title, desc) for title, name, desc, _, ti in _SD
                 if not _is_hidden(ti)}
    _WATCHER_KEYS = {"sync_directories", "ignored_extensions", "ignored_folders", "skip_hidden_folders"}

    def _all_setting_map():
        """Return a merged map of core + plugin settings (title, desc) by key.

        Settings flagged ``"hidden": True`` in their type_info are excluded —
        these keys (e.g. ``llm_profiles``, ``scheduled_jobs``) have dedicated
        commands (``/agent``, ``/schedule``) and should not appear in
        ``/config`` or ``/configure``.
        """
        merged = dict(_core_map)
        for title, name, desc, _, ti in _get_ps():
            if _is_hidden(ti):
                continue
            if name not in merged:
                merged[name] = (title, desc)
        return merged

    def _plugin_keys():
        return {entry[1] for entry in _get_ps()}

    def _cmd_config(arg):
        setting_map = _all_setting_map()
        arg = arg.strip()
        if arg:
            if arg not in setting_map:
                return f"Unknown setting: '{arg}'. Run /config to see all settings."
            title, desc = setting_map[arg]
            return f"{arg} = {ctrl.config.get(arg)}\n  {desc}"
        lines = [f"  {name} = {ctrl.config.get(name)}" for name in setting_map]
        return "Settings:\n" + "\n".join(lines)

    def _cmd_configure(arg):
        setting_map = _all_setting_map()
        parts = arg.split(None, 1)
        if len(parts) < 2:
            return ("Usage: /configure <setting_name> <value>\n"
                    "Example: /configure max_workers 8")
        key, raw = parts
        if key not in setting_map:
            return f"Unknown setting: '{key}'. Run /config to see all settings."
        try:
            value = _json.loads(raw)
        except _json.JSONDecodeError:
            value = raw

        old_val = ctrl.config.get(key)
        ctrl.config[key] = value
        _cm.save(ctrl.config)

        # Persist plugin config separately if this is a plugin key
        pk = _plugin_keys()
        if key in pk:
            existing = _cm.load_plugin_config()
            existing[key] = value
            _cm.save_plugin_config(existing)

        # Watcher rescan for filesystem-related keys
        if key in _WATCHER_KEYS and getattr(ctrl, 'watcher', None):
            ctrl.watcher.rescan()

        # Targeted service reload if the value actually changed
        feedback_parts = [f"Set {key} = {value}"]
        if value != old_val:
            svc_feedback = ctrl.reload_services_for_settings({key}, root_dir)
            feedback_parts.extend(f"  • {f}" for f in svc_feedback)
            runtime_feedback = ctrl.apply_runtime_config_changes({key})
            feedback_parts.extend(f"  • {f}" for f in runtime_feedback)

        return "\n".join(feedback_parts)

    def _cmd_trigger(arg):
        parts = arg.split(None, 1)
        if not parts:
            return ("Usage: /trigger <task_name> [json_payload]\n"
                    "Example: /trigger build_index {\"path\": \"/docs\"}")
        name = parts[0]
        task = ctrl.orchestrator.tasks.get(name)
        if task is None:
            return f"Unknown task: '{name}'. Run /tasks to see all tasks."
        if getattr(task, "trigger", "path") != "event":
            return (f"Task '{name}' is not event-triggered. "
                    f"Run /tasks to see event-driven tasks.")

        schema = getattr(task, "event_payload_schema", {}) or {}
        props = schema.get("properties", {})
        required = list(schema.get("required", []))
        payload = None
        if len(parts) > 1:
            try:
                payload = _json.loads(parts[1])
            except _json.JSONDecodeError:
                return "Payload must be valid JSON (or omit it)."
        elif required:
            example = {}
            for field in required:
                info = props.get(field, {})
                field_type = info.get("type", "string")
                if field_type == "array":
                    example[field] = ["value"]
                elif field_type == "object":
                    example[field] = {"key": "value"}
                elif field_type == "integer":
                    example[field] = 1
                elif field_type == "number":
                    example[field] = 1.0
                elif field_type == "boolean":
                    example[field] = True
                else:
                    example[field] = f"your {field}"

            lines = [f"Task '{name}' requires a payload."]
            lines.append(f"Required: {', '.join(required)}")
            optional = [field for field in props if field not in required]
            if optional:
                lines.append(f"Optional: {', '.join(optional)}")
            lines.append(f"Example: /trigger {name} {_json.dumps(example)}")
            return "\n".join(lines)
        return ctrl.trigger_event_task(name, payload)

    def _cmd_locations(arg):
        """Handler for /locations [tools|tasks|services]"""
        mode = (arg or "").strip().lower()
        filter_type = mode if mode in ("tools", "tasks", "services") else None
        if mode and filter_type is None:
            return ("Usage: /locations [tools|tasks|services]\n"
                    "Example: /locations tools")
        data = ctrl.list_locations(filter_type)
        return format_locations(data)

    def _cmd_call(arg):
        """Handler for /call <tool_name> {"arg": "value"}"""
        if not arg:
            return ("Usage: /call <tool_name> {json_args}\n"
                    "Example: /call sql_query {\"sql\": \"SELECT * FROM files LIMIT 5\"}")
        parts = arg.split(maxsplit=1)
        tool_name = parts[0]
        raw_args = parts[1] if len(parts) > 1 else "{}"
        try:
            kwargs = _json.loads(raw_args)
        except _json.JSONDecodeError as e:
            return (f"Invalid JSON arguments: {e}\n"
                    "Expected format: /call <tool_name> {\"key\": \"value\"}")
        return format_tool_result(ctrl.call_tool(tool_name, kwargs))

    def _cmd_history(arg):
        """Handler for /history — list or load past conversations."""
        from datetime import datetime

        arg = arg.strip()

        # /history <id> — load a specific conversation
        if arg.isdigit():
            conv_id = int(arg)
            messages = ctrl.db.get_conversation_messages(conv_id)
            if not messages:
                return f"Unknown conversation: {conv_id}. Run /history to see recent conversations."
            conversation = ctrl.db.get_conversation(conv_id)
            conversation_title = ((conversation or {}).get("title") or "").strip() or "New conversation"
            agent = get_agent() if get_agent else None
            if agent:
                agent_history = []
                for msg in messages:
                    role = msg["role"]
                    if role == "system":
                        continue
                    content = msg["content"] or ""
                    if role == "assistant":
                        try:
                            parsed = _json.loads(content)
                            if isinstance(parsed, dict) and "tool_calls" in parsed:
                                agent_history.append({
                                    "role": "assistant",
                                    "content": parsed.get("content"),
                                    "tool_calls": parsed["tool_calls"],
                                })
                                continue
                        except (_json.JSONDecodeError, TypeError):
                            pass
                        agent_history.append({"role": "assistant", "content": content})
                    elif role == "tool":
                        agent_history.append({
                            "role": "tool",
                            "tool_call_id": msg.get("tool_call_id"),
                            "content": content,
                        })
                    else:
                        agent_history.append({"role": role, "content": content})
                heal_orphan_tool_calls(agent_history)
                agent.history = agent_history
            # Update the conversation ref if a callback is available
            if _set_conversation_id:
                _set_conversation_id(conv_id)
            return f"Loaded conversation: {conversation_title}"

        # /history — list recent conversations
        conversations = ctrl.db.list_user_conversations(limit=10)
        if not conversations:
            return "No conversations yet."

        lines = ["Recent conversations:"]
        for conv in conversations:
            title = (conv["title"] or "New conversation").replace("\n", " ")[:50]
            ts = conv.get("updated_at")
            time_str = datetime.fromtimestamp(ts).strftime("%b %d, %I:%M %p") if ts else ""
            lines.append(f"  [{conv['id']}] {title}  ({time_str})")
        lines.append("\nRun /history <id> to load a conversation.")
        return "\n".join(lines)

    def _cmd_new(_arg):
        """Handler for /new — reset agent conversation history."""
        if _set_conversation_id:
            _set_conversation_id(None)
        agent = get_agent() if get_agent else None
        if agent:
            agent.reset()
        return new_conversation_message(ctrl.config)

    def _cmd_cancel(_arg):
        """Handler for /cancel — interrupt the agent at the next opportunity."""
        agent = get_agent() if get_agent else None
        if agent is None:
            return "No active agent to cancel."
        if not agent.running:
            return "Cancelled."
        agent.cancelled = True
        return "Cancelling..."

    def _cmd_refresh(_arg):
        """Handler for /refresh — forcibly rebuild the agent, preserving history.

        Use when /cancel isn't enough — e.g. a tool is wedged and the agent
        loop is stuck. Marks the old agent cancelled (best-effort), rebuilds
        a fresh agent via the frontend's refresh_agent callback, and heals
        any orphan tool_calls left behind by the interrupted turn.
        """
        old_agent = get_agent() if get_agent else None
        old_history = list(old_agent.history) if old_agent else []
        if old_agent:
            old_agent.cancelled = True
        if not _refresh_agent:
            return "Refresh is not supported in this frontend."
        _refresh_agent()
        new_agent = get_agent() if get_agent else None
        if new_agent is not None and old_history:
            new_agent.history = old_history
            heal_orphan_tool_calls(new_agent.history)
        return "Refreshed."

    def _cmd_restart(_arg):
        """Handler for /restart — hard restart: re-exec the whole process."""
        fn = getattr(ctrl, "restart", None)
        if fn is None:
            return "Restart is not supported in this frontend."
        fn()
        return "Restarting — the process will come back up in a few seconds."

    def _persist_config_key(key: str, value):
        """Save a config key. Mirrors plugin keys into plugin_config.json."""
        ctrl.config[key] = value
        _cm.save(ctrl.config)
        if key in _plugin_keys():
            existing = _cm.load_plugin_config()
            existing[key] = value
            _cm.save_plugin_config(existing)

    def _cmd_llm(arg):
        """Handler for /llm — manage LLM connection configs (model, endpoint, key, context)."""
        from plugins.services.llmService import LLMRouter

        router = services.get("llm")
        if not isinstance(router, LLMRouter):
            return "LLM service is not a router. Run /load llm to load it."

        parts = arg.strip().split(None, 1) if arg.strip() else []
        sub = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""

        if sub in ("list", "ls"):
            infos = router.list_llms()
            if not infos:
                return ("No LLMs configured. "
                        "Run /llm add <model_name> {json} to create one.")
            lines = ["LLMs:"]
            for info in infos:
                lines.append(_describe_llm(
                    info["model_name"],
                    {
                        "llm_service_class": info["class"],
                        "llm_context_size": info["context_size"],
                        "llm_endpoint": info["endpoint"],
                    },
                    default=info["default"],
                    loaded=info["loaded"],
                ))
            return "\n".join(lines)

        if sub == "default":
            model = rest.strip()
            if not model:
                return "Usage: /llm default <model_name>"
            profiles = ctrl.config.get("llm_profiles", {}) or {}
            if model not in profiles:
                return (f"Unknown LLM: '{model}'. "
                        f"Run /llm list to see all LLMs.")
            _persist_config_key("default_llm_profile", model)
            if _rescope_agents:
                try:
                    _rescope_agents()
                except Exception as e:
                    logger.warning(f"Rescope after /llm default failed: {e}")
            return f"Default LLM set to '{model}'."

        if sub == "add":
            add_parts = rest.split(None, 1)
            model = add_parts[0] if add_parts else ""
            json_str = add_parts[1] if len(add_parts) > 1 else ""
            if not model:
                return ("Usage: /llm add <model_name> {json}\n"
                        "Keys: llm_endpoint, llm_api_key, llm_context_size, "
                        "llm_service_class (OpenAILLM|LMStudioLLM).")
            if model == "default":
                return ("Cannot name an LLM 'default' — that string is the "
                        "sentinel agent profiles use to follow whatever LLM "
                        "is the current default. Pick a model name instead.")
            if not json_str:
                return ("Usage: /llm add <model_name> {json}\n"
                        "Example: /llm add gpt-4 "
                        '{\"llm_endpoint\": \"\", \"llm_api_key\": \"OPENAI_API_KEY\", '
                        '\"llm_context_size\": 128000, \"llm_service_class\": \"OpenAILLM\"}')
            try:
                profile = _json.loads(json_str)
            except _json.JSONDecodeError as e:
                return f"Invalid JSON: {e}"

            profiles = ctrl.config.setdefault("llm_profiles", {})
            profiles[model] = profile
            router.add_llm(model, profile)
            _persist_config_key("llm_profiles", profiles)

            # First LLM becomes the default automatically.
            if not ctrl.config.get("default_llm_profile"):
                _persist_config_key("default_llm_profile", model)
                return f"LLM '{model}' added and set as default."
            return f"LLM '{model}' added."

        if sub == "remove":
            model = rest.strip()
            if not model:
                return "Usage: /llm remove <model_name>"
            profiles = ctrl.config.get("llm_profiles", {}) or {}
            if model not in profiles:
                return (f"Unknown LLM: '{model}'. "
                        f"Run /llm list to see all LLMs.")
            if len(profiles) == 1:
                return "Refusing to remove the only LLM. Add another one first."
            was_default = ctrl.config.get("default_llm_profile") == model
            # Find agent profiles still referencing this LLM by name (the
            # "default" sentinel is fine — those resolve via default_llm_profile).
            agent_profiles = ctrl.config.get("agent_profiles", {}) or {}
            stale_refs = sorted(
                name for name, p in agent_profiles.items()
                if (p or {}).get("llm") == model
            )
            router.remove_llm(model)
            del profiles[model]
            _persist_config_key("llm_profiles", profiles)
            lines = []
            if was_default:
                new_default = next(iter(profiles))
                _persist_config_key("default_llm_profile", new_default)
                if _rescope_agents:
                    try:
                        _rescope_agents()
                    except Exception as e:
                        logger.warning(f"Rescope after /llm remove failed: {e}")
                lines.append(f"LLM '{model}' removed. Default is now '{new_default}'.")
            else:
                lines.append(f"LLM '{model}' removed.")
            if stale_refs:
                lines.append(
                    f"Warning: {len(stale_refs)} agent profile(s) still reference "
                    f"'{model}' by name: {', '.join(stale_refs)}. Until you edit "
                    "them, they will fall back to the default LLM."
                )
            return "\n".join(lines)

        if sub == "show":
            model = rest.strip()
            if not model:
                return "Usage: /llm show <model_name>"
            profiles = ctrl.config.get("llm_profiles", {}) or {}
            if model not in profiles:
                return (f"Unknown LLM: '{model}'. "
                        f"Run /llm list to see all LLMs.")
            display = dict(profiles[model])
            if "llm_api_key" in display:
                display["llm_api_key"] = _mask_api_key(display["llm_api_key"])
            return f"{model}:\n{_json.dumps(display, indent=2)}"

        if sub == "edit":
            edit_parts = rest.split(None, 2)
            if len(edit_parts) < 3:
                return ("Usage: /llm edit <model_name> <field> <value>\n"
                        f"Fields: {', '.join(_LLM_PROFILE_FIELDS)}")
            model, field, raw = edit_parts
            profiles = ctrl.config.get("llm_profiles", {}) or {}
            if model not in profiles:
                return (f"Unknown LLM: '{model}'. "
                        f"Run /llm list to see all LLMs.")
            if field not in _LLM_PROFILE_FIELDS:
                return (f"Unknown field: '{field}'. "
                        f"Allowed: {', '.join(_LLM_PROFILE_FIELDS)}.")
            try:
                value = _json.loads(raw)
            except _json.JSONDecodeError:
                value = raw
            if field == "llm_context_size":
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    return "llm_context_size must be an integer."
            if field == "llm_service_class" and value not in ("OpenAILLM", "LMStudioLLM"):
                return "llm_service_class must be 'OpenAILLM' or 'LMStudioLLM'."
            profiles[model][field] = value
            # Rebuild the registered LLM so the change takes effect immediately.
            try:
                if model in router.services and getattr(router.services[model], "loaded", False):
                    router.services[model].unload()
            except Exception as e:
                logger.warning(f"Unload during /llm edit failed: {e}")
            router.add_llm(model, profiles[model])
            _persist_config_key("llm_profiles", profiles)
            if _rescope_agents:
                try:
                    _rescope_agents()
                except Exception as e:
                    logger.warning(f"Rescope after /llm edit failed: {e}")
            return f"Updated {model}.{field}."

        return (f"Unknown subcommand: '{sub}'. "
                f"Use: list, add, edit, remove, show, default.")

    def _cmd_agent(arg):
        """Handler for /agent — manage agent profiles (LLM reference + optional scope)."""
        from runtime.agent_scope import load_scope

        parts = arg.strip().split(None, 1) if arg.strip() else []
        sub = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""

        agent_profiles = ctrl.config.get("agent_profiles", {}) or {}
        active_name = active_agent_name(ctrl.config)

        if sub in ("list", "ls"):
            if not agent_profiles:
                return ("No agent profiles configured. "
                        "Run /agent add <name> {json} to create one.")
            lines = ["Agent Profiles:"]
            for name, profile in agent_profiles.items():
                lines.append(_describe_agent_profile(
                    name, profile, active=(name == active_name),
                ))
            return "\n".join(lines)

        if sub == "switch":
            name = rest.strip()
            if not name:
                return "Usage: /agent switch <profile_name>"
            if name not in agent_profiles:
                return (f"Unknown agent profile: '{name}'. "
                        f"Run /agent list to see all profiles.")
            try:
                load_scope(name, ctrl.config)
            except ValueError as e:
                return f"Cannot switch to '{name}': {e}"
            _persist_config_key("active_agent_profile", name)
            if _rescope_agents:
                try:
                    _rescope_agents()
                except Exception as e:
                    logger.warning(f"Rescope after /agent switch failed: {e}")
            return f"Switched to agent profile '{name}'."

        if sub == "add":
            add_parts = rest.split(None, 1)
            name = add_parts[0] if add_parts else ""
            json_str = add_parts[1] if len(add_parts) > 1 else ""
            if not name:
                return ("Usage: /agent add <profile_name> {json}\n"
                        "Keys: llm (model_name or 'default'), prompt_suffix, "
                        "whitelist_or_blacklist_tools, tools_list, "
                        "whitelist_or_blacklist_tables, tables_list, "
                        "whitelist_or_blacklist_folders, folders_list.")
            if not json_str:
                return ("Usage: /agent add <profile_name> {json}\n"
                        "Example: /agent add researcher "
                        '{\"llm\": \"default\", \"prompt_suffix\": \"\", '
                        '\"whitelist_or_blacklist_tools\": \"whitelist\", '
                        '\"tools_list\": [\"sql_query\", \"read_file\"], '
                        '\"whitelist_or_blacklist_tables\": \"blacklist\", '
                        '\"tables_list\": [], '
                        '\"whitelist_or_blacklist_folders\": \"blacklist\", '
                        '\"folders_list\": []}')
            try:
                profile = _json.loads(json_str)
            except _json.JSONDecodeError as e:
                return f"Invalid JSON: {e}"
            profile = _normalize_agent_profile(profile)

            llm_ref = profile.get("llm") or "default"
            llm_profiles = ctrl.config.get("llm_profiles", {}) or {}
            if llm_ref != "default" and llm_ref not in llm_profiles:
                return (f"Profile references unknown LLM '{llm_ref}'. "
                        f"Run /llm list to see all LLMs.")

            profiles = ctrl.config.setdefault("agent_profiles", {})
            profiles[name] = profile
            try:
                load_scope(name, ctrl.config)
            except ValueError as e:
                del profiles[name]
                return f"Invalid scope for '{name}': {e}"
            _persist_config_key("agent_profiles", profiles)
            return f"Agent profile '{name}' added."

        if sub == "remove":
            name = rest.strip()
            if not name:
                return "Usage: /agent remove <profile_name>"
            if name == "default":
                return "Refusing to remove the 'default' agent profile."
            if name not in agent_profiles:
                return (f"Unknown agent profile: '{name}'. "
                        f"Run /agent list to see all profiles.")
            was_active = active_name == name
            del agent_profiles[name]
            _persist_config_key("agent_profiles", agent_profiles)
            if was_active:
                _persist_config_key("active_agent_profile", "default")
                if _rescope_agents:
                    try:
                        _rescope_agents()
                    except Exception as e:
                        logger.warning(f"Rescope after /agent remove failed: {e}")
                return f"Agent profile '{name}' removed. Active is now 'default'."
            return f"Agent profile '{name}' removed."

        if sub == "show":
            name = rest.strip()
            if not name:
                return "Usage: /agent show <profile_name>"
            if name not in agent_profiles:
                return (f"Unknown agent profile: '{name}'. "
                        f"Run /agent list to see all profiles.")
            return f"{name}:\n{_json.dumps(agent_profiles[name], indent=2)}"

        if sub == "edit":
            edit_parts = rest.split(None, 2)
            if len(edit_parts) < 3:
                return ("Usage: /agent edit <profile_name> <field> <value>\n"
                        f"Fields: {', '.join(_AGENT_PROFILE_FIELDS)}")
            name, field, raw = edit_parts
            if name not in agent_profiles:
                return (f"Unknown agent profile: '{name}'. "
                        f"Run /agent list to see all profiles.")
            if field not in _AGENT_PROFILE_FIELDS:
                return (f"Unknown field: '{field}'. "
                        f"Allowed: {', '.join(_AGENT_PROFILE_FIELDS)}.")
            try:
                value = _json.loads(raw)
            except _json.JSONDecodeError:
                value = raw
            if field == "llm":
                if not isinstance(value, str) or not value:
                    return "llm must be a string ('default' or a model name)."
                llm_profiles = ctrl.config.get("llm_profiles", {}) or {}
                if value != "default" and value not in llm_profiles:
                    return (f"Unknown LLM '{value}'. "
                            f"Run /llm list to see all LLMs.")
            elif field in _AGENT_MODE_FIELDS:
                if value not in ("whitelist", "blacklist"):
                    return f"{field} must be 'whitelist' or 'blacklist'."
            elif field in _AGENT_LIST_FIELDS:
                if value in (None, "", {}):
                    value = []
                elif not isinstance(value, list):
                    return f"{field} must be a JSON array."
            elif field == "prompt_suffix":
                if value is None:
                    value = ""
                value = str(value)

            old_value = agent_profiles[name].get(field)
            agent_profiles[name][field] = value
            try:
                load_scope(name, ctrl.config)
            except ValueError as e:
                agent_profiles[name][field] = old_value
                return f"Edit rejected: {e}"
            _persist_config_key("agent_profiles", agent_profiles)
            if name == active_name and _rescope_agents:
                try:
                    _rescope_agents()
                except Exception as e:
                    logger.warning(f"Rescope after /agent edit failed: {e}")
            return f"Updated {name}.{field}."

        return (f"Unknown subcommand: '{sub}'. "
                f"Use: list, switch, add, edit, remove, show.")

    def _cmd_schedule(arg):
        """Handler for /schedule — manage scheduled jobs (timekeeper)."""
        timekeeper = services.get("timekeeper")
        if timekeeper is None or not getattr(timekeeper, "loaded", False):
            return ("Timekeeper service is not loaded. "
                    "Run /load timekeeper to start it.")

        parts = arg.strip().split(None, 1) if arg.strip() else []
        sub = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""

        if sub in ("list", "ls"):
            jobs = timekeeper.list_jobs()
            return format_scheduled_jobs(jobs, timekeeper)

        if sub == "show":
            name = rest.strip()
            if not name:
                return "Usage: /schedule show <job_name>"
            job = timekeeper.get_job(name)
            if job is None:
                return (f"Unknown job: '{name}'. "
                        f"Run /schedule list to see all jobs.")
            return f"{name}:\n{_json.dumps(job, indent=2, default=str)}"

        if sub in ("enable", "disable"):
            name = rest.strip()
            if not name:
                return f"Usage: /schedule {sub} <job_name>"
            if timekeeper.get_job(name) is None:
                return (f"Unknown job: '{name}'. "
                        f"Run /schedule list to see all jobs.")
            timekeeper.enable_job(name, sub == "enable")
            return f"Job '{name}' {sub}d."

        if sub == "delete":
            name = rest.strip()
            if not name:
                return "Usage: /schedule delete <job_name>"
            if not timekeeper.remove_job(name):
                return (f"Unknown job: '{name}'. "
                        f"Run /schedule list to see all jobs.")
            return f"Job '{name}' deleted."

        if sub == "run":
            name = rest.strip()
            if not name:
                return "Usage: /schedule run <job_name>"
            job = timekeeper.get_job(name)
            if job is None:
                return (f"Unknown job: '{name}'. "
                        f"Run /schedule list to see all jobs.")
            from events.event_bus import bus
            from datetime import datetime
            payload = dict(job.get("payload", {}))
            payload["_timekeeper"] = {
                "job_name": name,
                "scheduled_for": datetime.now().astimezone().isoformat(),
                "emitted_at": datetime.now().astimezone().isoformat(),
                "one_time": bool(job.get("one_time")),
                "source": "timekeeper_manual",
            }
            bus.emit(job["channel"], payload)
            return f"Job '{name}' fired manually."

        if sub == "create":
            create_parts = rest.split(None, 1)
            name = create_parts[0] if create_parts else ""
            json_str = create_parts[1] if len(create_parts) > 1 else ""
            if not name or not json_str:
                return ("Usage: /schedule create <job_name> {json}\n"
                        "Example: /schedule create daily_digest "
                        '{\"channel\": \"subagent.run\", \"cron\": \"0 8 * * *\", '
                        '\"payload\": {\"prompt\": \"Summarize overnight activity\", '
                        '\"notifications\": \"all\", \"title\": \"Daily digest\"}}')
            try:
                definition = _json.loads(json_str)
            except _json.JSONDecodeError as e:
                return f"Invalid JSON: {e}"
            try:
                timekeeper.create_job(name, definition)
            except ValueError as e:
                return f"Failed to create job '{name}': {e}"
            return f"Job '{name}' created."

        return (f"Unknown subcommand: '{sub}'. "
                f"Use: list, show, enable, disable, delete, run, create.")

    _agent_completions = lambda: (
        ["list", "switch", "add", "edit", "remove", "show"]
        + sorted((ctrl.config.get("agent_profiles", {}) or {}).keys())
    )

    _llm_completions = lambda: (
        ["list", "add", "edit", "remove", "show", "default"]
        + sorted((ctrl.config.get("llm_profiles", {}) or {}).keys())
    )

    def _schedule_completions():
        actions = ["list", "show", "enable", "disable", "delete", "run", "create"]
        timekeeper = services.get("timekeeper")
        if timekeeper and getattr(timekeeper, "loaded", False):
            try:
                actions += sorted(timekeeper.list_jobs().keys())
            except Exception:
                pass
        return actions

    # Lambdas (not static lists) so completions reflect hot-reloaded plugins.
    _all_task_names = lambda: ctrl.list_task_names()
    _path_task_names = lambda: ctrl.list_task_names(trigger="path")
    _event_task_names = lambda: ctrl.list_task_names(trigger="event")
    _service_names = lambda: sorted(services.keys())
    _tool_names = lambda: sorted(tool_registry.tools.keys())
    _retry_names = lambda: ["all"] + _path_task_names()
    _render_tasks = lambda: format_tasks(ctrl.list_tasks())

    for entry in [
        # ── Conversation ───────────────────────────────────────────
        CommandEntry("help", "Show available commands",
                     handler=lambda _: _build_help(registry),
                     category="Conversation"),
        CommandEntry("new", "Start a new conversation",
                     handler=_cmd_new,
                     category="Conversation"),
        CommandEntry("cancel", "Interrupt the agent",
                     handler=_cmd_cancel,
                     category="Conversation"),
        CommandEntry("refresh", "Rebuild the agent when a tool is stuck (try /restart if a service is hung)",
                     handler=_cmd_refresh,
                     category="Conversation"),
        CommandEntry("restart", "Restart the whole app — escape hatch when /refresh isn't enough",
                     handler=_cmd_restart,
                     category="Conversation"),
        CommandEntry("history", "List or load past conversations", "[id]",
                     handler=_cmd_history,
                     category="Conversation"),

        # ── Services & Tools ───────────────────────────────────────
        CommandEntry("services", "List services and status",
                     handler=lambda _: format_services(ctrl.list_services()),
                     category="Services & Tools"),
        CommandEntry("load", "Load a service", "<service_name>",
                     handler=lambda a: ctrl.load_service(a) if a
                             else "Usage: /load <service_name>",
                     arg_completions=_service_names,
                     category="Services & Tools"),
        CommandEntry("unload", "Unload a service", "<service_name>",
                     handler=lambda a: ctrl.unload_service(a) if a
                             else "Usage: /unload <service_name>",
                     arg_completions=_service_names,
                     category="Services & Tools"),
        CommandEntry("tools", "List registered tools",
                     handler=lambda _: format_tools(ctrl.list_tools()),
                     category="Services & Tools"),
        CommandEntry("call", "Call a tool directly", "<tool_name> {json_args}",
                     handler=_cmd_call, arg_completions=_tool_names,
                     category="Services & Tools"),

        # ── Tasks ──────────────────────────────────────────────────
        CommandEntry("tasks", "List path-driven and event-driven tasks",
                     handler=lambda _: _render_tasks(),
                     category="Tasks"),
        CommandEntry("pipeline", "Show the path-driven task dependency graph",
                     handler=lambda _: ctrl.orchestrator.dependency_pipeline_graph(),
                     category="Tasks"),
        CommandEntry("pause", "Pause a task", "<task_name>",
                     handler=lambda a: ctrl.pause_task(a) if a
                             else "Usage: /pause <task_name>",
                     arg_completions=_all_task_names,
                     category="Tasks"),
        CommandEntry("unpause", "Unpause a task", "<task_name>",
                     handler=lambda a: ctrl.unpause_task(a) if a
                             else "Usage: /unpause <task_name>",
                     arg_completions=_all_task_names,
                     category="Tasks"),
        CommandEntry("reset", "Reset a path-driven task to Pending", "<task_name>",
                     handler=lambda a: ctrl.reset_task(a) if a
                             else "Usage: /reset <task_name>",
                     arg_completions=_path_task_names,
                     category="Tasks"),
        CommandEntry("retry", "Retry failed path-driven task entries",
                     "<task_name>|all",
                     handler=lambda a: ctrl.retry_all() if a and a.lower() == "all"
                             else ctrl.retry_task(a) if a
                             else "Usage: /retry <task_name>|all",
                     arg_completions=_retry_names,
                     category="Tasks"),
        CommandEntry("trigger", "Manually fire an event-triggered task",
                     "<task_name> [json_payload]",
                     handler=_cmd_trigger,
                     arg_completions=_event_task_names,
                     category="Tasks"),

        # ── Config & System ────────────────────────────────────────
        CommandEntry("config", "Show config settings", "[setting_name]",
                     handler=_cmd_config,
                     category="Config & System"),
        CommandEntry("configure", "Update a config setting",
                     "<setting_name> <value>",
                     handler=_cmd_configure,
                     category="Config & System"),
        CommandEntry("llm", "Manage LLM connection configs",
                     "[list|add|edit|remove|show|default] [args]",
                     handler=_cmd_llm, arg_completions=_llm_completions,
                     category="Config & System"),
        CommandEntry("agent", "Manage scoped agent profiles",
                     "[list|switch|add|edit|remove|show] [args]",
                     handler=_cmd_agent, arg_completions=_agent_completions,
                     category="Config & System"),
        CommandEntry("schedule", "Manage scheduled jobs",
                     "[list|show|enable|disable|delete|run|create] [args]",
                     handler=_cmd_schedule,
                     arg_completions=_schedule_completions,
                     category="Config & System"),
        CommandEntry("locations", "List file-system locations",
                     "[tools|tasks|services]",
                     handler=_cmd_locations,
                     arg_completions=lambda: ["tools", "tasks", "services"],
                     category="Config & System"),
        CommandEntry("reload", "Hot-reload tasks and tools",
                     handler=lambda _: ctrl.reload_plugins(root_dir),
                     category="Config & System"),
    ]:
        registry.register(entry)
