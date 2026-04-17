"""
Command registry for the slash-command system.

Each command is a CommandEntry with a name, description, argument hint,
and handler callable. The registry provides autocomplete filtering and
dispatch.

``register_core_commands()`` registers the shared commands used by the
frontends and API. Each frontend can add overrides on top.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class CommandEntry:
    name: str               # e.g. "services"
    description: str        # e.g. "List services and status"
    arg_hint: str = ""      # e.g. "<service_name>" — shown in autocomplete
    handler: Callable = None  # fn(arg: str) -> str | None
    arg_completions: Callable = None  # () -> list[str] — dynamic arg suggestions
    hide_from_help: bool = False      # if True, omit from /help output


class CommandRegistry:
    def __init__(self):
        self._commands: dict[str, CommandEntry] = {}

    def register(self, entry: CommandEntry):
        self._commands[entry.name] = entry

    def get_completions(self, prefix: str) -> list[CommandEntry]:
        """Return commands whose name starts with *prefix* (case-insensitive)."""
        prefix = prefix.lower()
        return [
            cmd for cmd in self._commands.values()
            if cmd.name.startswith(prefix)
        ]

    def dispatch(self, name: str, arg: str) -> str | None:
        """Look up a command by name and call its handler. Returns output string or None."""
        entry = self._commands.get(name)
        if entry is None:
            return f"Unknown command: '/{name}'. Type /help for available commands."
        return entry.handler(arg)

    def all_commands(self) -> list[CommandEntry]:
        """Return all registered commands in insertion order."""
        return list(self._commands.values())


def _build_help(registry: CommandRegistry) -> str:
    """Format the /help output from all registered commands."""
    lines = ["Commands:"]
    for cmd in registry.all_commands():
        if getattr(cmd, "hide_from_help", False):
            continue
        hint = f" {cmd.arg_hint}" if cmd.arg_hint else ""
        label = f"/{cmd.name}{hint}"
        lines.append(f"  {label:<22} {cmd.description}")
    return "\n".join(lines)


def register_core_commands(registry: CommandRegistry, ctrl, services, tool_registry,
                           root_dir, get_agent=None, set_conversation_id=None):
    """Register commands shared by Telegram, REPL, and API.

    These are pure ctrl-wrapper commands with no UI-specific side effects.
    Each UI should call this first, then register/override its own commands.

    Parameters:
        get_agent:           Optional callable returning the current Agent instance
                             (or None). Used by /call, /new, /cancel, and /history.
        set_conversation_id: Optional callable(int | None) to update the current
                             conversation ID. Used by /history and /new.
    """
    _set_conversation_id = set_conversation_id
    import json as _json
    import config_manager as _cm
    from config_data import SETTINGS_DATA as _SD
    from plugin_discovery import get_plugin_settings as _get_ps
    from frontend.formatters import (
        format_services, format_tasks,
        format_stats, format_tools, format_locations,
        format_tool_result,
    )

    # Build a unified setting map (core + plugin) at registration time.
    # Plugin settings are added lazily on first use since discovery may
    # not be complete at import time.
    _core_map = {name: (title, desc) for title, name, desc, _, __ in _SD}
    _WATCHER_KEYS = {"sync_directories", "ignored_extensions", "ignored_folders", "skip_hidden_folders"}

    def _all_setting_map():
        """Return a merged map of core + plugin settings (title, desc) by key."""
        merged = dict(_core_map)
        for title, name, desc, _, __ in _get_ps():
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
                return f"Unknown setting '{arg}'. Run /config to see all settings."
            title, desc = setting_map[arg]
            return f"{arg} = {ctrl.config.get(arg)}\n  {desc}"
        lines = [f"  {name} = {ctrl.config.get(name)}" for name in setting_map]
        return "\n".join(lines)

    def _cmd_configure(arg):
        setting_map = _all_setting_map()
        parts = arg.split(None, 1)
        if len(parts) < 2:
            return "Usage: /configure <key> <value>"
        key, raw = parts
        if key not in setting_map:
            return f"Unknown setting '{key}'. Run /config to see all settings."
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
            return "Usage: /trigger <task> [json_payload]"
        name = parts[0]
        payload = None
        if len(parts) > 1:
            try:
                payload = _json.loads(parts[1])
            except _json.JSONDecodeError:
                return "Payload must be valid JSON (or omit it)."
        return ctrl.trigger_event_task(name, payload)

    def _cmd_runs(arg):
        parts = arg.split()
        task_name = None
        limit = 50
        for p in parts:
            if p.isdigit():
                limit = int(p)
            else:
                task_name = p
        rows = ctrl.list_runs(task_name=task_name, limit=limit)
        if not rows:
            return "No runs."
        lines = [f"{r.get('run_id','?')}  {r.get('task_name','?')}  {r.get('status','?')}  "
                 f"by={r.get('triggered_by','?')}" for r in rows]
        return "\n".join(lines)

    def _cmd_locations(arg):
        """Handler for /locations [tools|tasks|services]"""
        mode = (arg or "").strip().lower()
        filter_type = mode if mode in ("tools", "tasks", "services") else None
        if mode and filter_type is None:
            return "Usage: /locations [tools|tasks|services]"
        data = ctrl.list_locations(filter_type)
        return format_locations(data)

    def _cmd_call(arg):
        """Handler for /call <tool_name> {"arg": "value"}"""
        if not arg:
            return ("Usage: /call <tool_name> {\"arg\": \"value\"}\n"
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
                return f"Conversation {conv_id} not found or is empty."
            agent = get_agent() if get_agent else None
            if agent:
                agent_history = []
                for msg in messages:
                    role = msg["role"]
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
                agent.history = agent_history
            # Update the conversation ref if a callback is available
            if _set_conversation_id:
                _set_conversation_id(conv_id)
            return f"(loaded conversation)"

        # /history — list recent conversations
        conversations = ctrl.db.list_conversations(limit=10)
        if not conversations:
            return "No conversations yet."

        lines = ["Recent conversations:"]
        for conv in conversations:
            title = (conv["title"] or "New conversation").replace("\n", " ")[:50]
            ts = conv.get("updated_at")
            time_str = datetime.fromtimestamp(ts).strftime("%b %d, %I:%M %p") if ts else ""
            lines.append(f"  [{conv['id']}] {title}  ({time_str})")
        lines.append("\nUse /history <id> to load a conversation.")
        return "\n".join(lines)

    def _cmd_new(_arg):
        """Handler for /new — reset agent conversation history."""
        if _set_conversation_id:
            _set_conversation_id(None)
        agent = get_agent() if get_agent else None
        if agent:
            agent.reset()
        return "(new conversation started)"

    def _cmd_cancel(_arg):
        """Handler for /cancel — interrupt the agent at the next opportunity."""
        agent = get_agent() if get_agent else None
        if agent:
            agent.cancelled = True
            return "(cancelling...)"
        return "No active agent to cancel."

    def _cmd_model(arg):
        """Handler for /model — manage LLM profiles."""
        from Stage_0.services.llmService import LLMRouter

        router = services.get("llm")
        if not isinstance(router, LLMRouter):
            return "LLM service is not a profile router."

        parts = arg.strip().split(None, 1) if arg.strip() else []
        sub = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""

        if sub in ("list", "ls"):
            infos = router.list_profiles()
            if not infos:
                return "No LLM profiles configured. Use /model add <name> <json> to create one."
            lines = ["LLM Profiles:"]
            for p in infos:
                marker = " *" if p["active"] else ""
                status = "loaded" if p["loaded"] else "unloaded"
                lines.append(f"  {p['name']}{marker}  ({p['class']}: {p['model']}) [{status}]")
            return "\n".join(lines)

        elif sub == "switch":
            name = rest.strip()
            if not name:
                return "Usage: /model switch <profile_name>"
            result = router.switch(name)
            ctrl.config["active_llm_profile"] = name
            _cm.save(ctrl.config)
            pk = _plugin_keys()
            if "active_llm_profile" in pk:
                existing = _cm.load_plugin_config()
                existing["active_llm_profile"] = name
                _cm.save_plugin_config(existing)
            return result

        elif sub == "add":
            add_parts = rest.split(None, 1)
            name = add_parts[0] if add_parts else ""
            json_str = add_parts[1] if len(add_parts) > 1 else ""
            if not name:
                return ("Usage: /model add <name> {json}\n"
                        "Keys: llm_model_name, llm_endpoint, llm_api_key, "
                        "llm_context_size, llm_service_class (OpenAILLM|LMStudioLLM)")
            if not json_str:
                return ("Usage: /model add <name> {json}\n"
                        "Example: /model add mymodel "
                        '{\"llm_model_name\": \"gpt-4\", \"llm_endpoint\": \"\", '
                        '\"llm_api_key\": \"OPENAI_API_KEY\", \"llm_context_size\": 0, '
                        '\"llm_service_class\": \"OpenAILLM\"}')
            try:
                profile = _json.loads(json_str)
            except _json.JSONDecodeError as e:
                return f"Invalid JSON: {e}"

            profiles = ctrl.config.setdefault("llm_profiles", {})
            profiles[name] = profile
            router.add_profile(name, profile)
            _cm.save(ctrl.config)
            pk = _plugin_keys()
            if "llm_profiles" in pk:
                existing = _cm.load_plugin_config()
                existing["llm_profiles"] = profiles
                _cm.save_plugin_config(existing)

            # If first profile or no active, make it active
            if not ctrl.config.get("active_llm_profile"):
                ctrl.config["active_llm_profile"] = name
                router.switch(name)
                _cm.save(ctrl.config)
                return f"Profile '{name}' added and set as active."
            return f"Profile '{name}' added."

        elif sub == "remove":
            name = rest.strip()
            if not name:
                return "Usage: /model remove <profile_name>"
            profiles = ctrl.config.get("llm_profiles", {})
            if name not in profiles:
                return f"Unknown profile: '{name}'"
            router.remove_profile(name)
            del profiles[name]
            if ctrl.config.get("active_llm_profile") == name:
                remaining = list(profiles.keys())
                if remaining:
                    ctrl.config["active_llm_profile"] = remaining[0]
                    router.switch(remaining[0])
                else:
                    ctrl.config["active_llm_profile"] = ""
            _cm.save(ctrl.config)
            pk = _plugin_keys()
            if "llm_profiles" in pk:
                existing = _cm.load_plugin_config()
                existing["llm_profiles"] = profiles
                _cm.save_plugin_config(existing)
            return f"Profile '{name}' removed."

        elif sub == "show":
            name = rest.strip()
            if not name:
                return "Usage: /model show <profile_name>"
            profiles = ctrl.config.get("llm_profiles", {})
            if name not in profiles:
                return f"Unknown profile: '{name}'"
            return f"{name}:\n{_json.dumps(profiles[name], indent=2)}"

        return f"Unknown subcommand '{sub}'. Use: list, switch, add, remove, show."

    _model_completions = lambda: (
        ["list", "switch", "add", "remove", "show"]
        + list(ctrl.config.get("llm_profiles", {}).keys())
    )

    # Lambdas (not static lists) so completions reflect hot-reloaded plugins.
    _task_names = lambda: list(ctrl.orchestrator.tasks.keys())
    _service_names = lambda: list(services.keys())
    _tool_names = lambda: list(tool_registry.tools.keys())
    _retry_names = lambda: _task_names() + ["all"]

    for entry in [
        CommandEntry("help",     "Show available commands",
                     handler=lambda _: _build_help(registry)),
        CommandEntry("services", "List services and status",
                     handler=lambda _: format_services(ctrl.list_services())),
        CommandEntry("load",     "Load a service",        "<service>",
                     handler=lambda a: ctrl.load_service(a) if a else "Usage: /load <service>",
                     arg_completions=_service_names),
        CommandEntry("unload",   "Unload a service",      "<service>",
                     handler=lambda a: ctrl.unload_service(a) if a else "Usage: /unload <service>",
                     arg_completions=_service_names),
        CommandEntry("tasks",    "List tasks with status counts",
                     handler=lambda _: format_tasks(ctrl.list_tasks())),
        CommandEntry("pipeline", "Show task dependency graph",
                     handler=lambda _: ctrl.orchestrator.dependency_pipeline_graph()),
        CommandEntry("pause",    "Pause a task",          "<task>",
                     handler=lambda a: ctrl.pause_task(a) if a else "Usage: /pause <task>",
                     arg_completions=_task_names),
        CommandEntry("unpause",  "Unpause a task",        "<task>",
                     handler=lambda a: ctrl.unpause_task(a) if a else "Usage: /unpause <task>",
                     arg_completions=_task_names),
        CommandEntry("reset",    "Reset a task to PENDING", "<task>",
                     handler=lambda a: ctrl.reset_task(a) if a else "Usage: /reset <task>",
                     arg_completions=_task_names),
        CommandEntry("retry",    "Retry failed entries",  "<task>|all",
                     handler=lambda a: ctrl.retry_all() if a and a.lower() == "all"
                             else ctrl.retry_task(a) if a else "Usage: /retry <task>|all",
                     arg_completions=_retry_names),
        CommandEntry("trigger",  "Manually fire an event-triggered task", "<task> [json]",
                     handler=lambda a: _cmd_trigger(a),
                     arg_completions=_task_names),
        CommandEntry("runs",     "List recent event-task runs", "[task] [limit]",
                     handler=lambda a: _cmd_runs(a),
                     arg_completions=_task_names),
        CommandEntry("tools",    "List registered tools",
                     handler=lambda _: format_tools(ctrl.list_tools())),
        CommandEntry("enable",   "Enable a tool for agent use", "<tool>",
                     handler=lambda a: ctrl.enable_tool(a) if a else "Usage: /enable <tool>",
                     arg_completions=_tool_names),
        CommandEntry("disable",  "Disable a tool",        "<tool>",
                     handler=lambda a: ctrl.disable_tool(a) if a else "Usage: /disable <tool>",
                     arg_completions=_tool_names),
        CommandEntry("reload",    "Hot-reload tasks and tools",
                     handler=lambda _: ctrl.reload_plugins(root_dir)),
        CommandEntry("stats",     "System overview",
                     handler=lambda _: format_stats(ctrl.stats())),
        CommandEntry("locations", "List file system locations",
                 "[tools|tasks|services]",
                 handler=lambda a: _cmd_locations(a),
                 arg_completions=lambda: ["tools", "tasks", "services"]),
        CommandEntry("config",    "Show config settings",      "[key]",
                     handler=_cmd_config),
        CommandEntry("configure", "Update a config setting",   "<key> <value>",
                     handler=_cmd_configure),
        CommandEntry("call",      "Call a tool directly",   "<tool> {json}",
                     handler=_cmd_call, arg_completions=_tool_names),
        CommandEntry("history",   "List or load past conversations", "[id]",
                     handler=_cmd_history),
        CommandEntry("new",       "Start a new conversation",
                     handler=_cmd_new),
        CommandEntry("cancel",    "Interrupt the agent",
                     handler=_cmd_cancel),
        CommandEntry("model",     "Manage LLM profiles",
                     "[list|switch|add|remove|show] [args]",
                     handler=_cmd_model, arg_completions=_model_completions),
    ]:
        registry.register(entry)
