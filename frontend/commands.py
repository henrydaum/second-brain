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


def _describe_profile(name: str, profile: dict, active: bool, loaded: bool = False) -> str:
    """Render a single profile line for /model list."""
    marker = "*" if active else " "
    status = "active" if active else ("loaded" if loaded else "")
    cls = profile.get("llm_service_class") or "?"
    model = profile.get("llm_model_name") or "?"
    ctx = profile.get("llm_context_size") or 0
    if ctx and ctx >= 1000:
        ctx_str = f"ctx {ctx // 1000}k"
    elif ctx:
        ctx_str = f"ctx {ctx}"
    else:
        ctx_str = "ctx auto"
    return (f"  {marker} {name:<16} {status:<8} "
            f"{cls:<14} {model:<24} ({ctx_str})")


def register_core_commands(registry: CommandRegistry, ctrl, services, tool_registry,
                           root_dir, get_agent=None, set_conversation_id=None,
                           restart_agent=None):
    """Register commands shared by Telegram and REPL.

    These are pure ctrl-wrapper commands with no UI-specific side effects.
    Each UI should call this first, then register/override its own commands.

    Parameters:
        get_agent:           Optional callable returning the current Agent instance
                             (or None). Used by /call, /new, /cancel, and /history.
        set_conversation_id: Optional callable(int | None) to update the current
                             conversation ID. Used by /history and /new.
        restart_agent:       Optional callable() that rebuilds the frontend's Agent
                             instance, preserving conversation history. Used by
                             /restart to recover from a stuck tool call.
    """
    _set_conversation_id = set_conversation_id
    _restart_agent = restart_agent
    import json as _json
    import config_manager as _cm
    from config_data import SETTINGS_DATA as _SD
    from plugin_discovery import get_plugin_settings as _get_ps
    from Stage_3.history_utils import heal_orphan_tool_calls
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
        commands (``/model``, ``/schedule``) and should not appear in
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
        return "New conversation started."

    def _cmd_cancel(_arg):
        """Handler for /cancel — interrupt the agent at the next opportunity."""
        agent = get_agent() if get_agent else None
        if agent:
            agent.cancelled = True
            return "Cancelling..."
        return "No active agent to cancel."

    def _cmd_restart(_arg):
        """Handler for /restart — forcibly rebuild the agent, preserving history.

        Use when /cancel isn't enough — e.g. a tool is wedged and the agent
        loop is stuck. Marks the old agent cancelled (best-effort), rebuilds
        a fresh agent via the frontend's restart_agent callback, and heals
        any orphan tool_calls left behind by the interrupted turn.
        """
        old_agent = get_agent() if get_agent else None
        old_history = list(old_agent.history) if old_agent else []
        if old_agent:
            old_agent.cancelled = True
        if not _restart_agent:
            return "Restart is not supported in this frontend."
        _restart_agent()
        new_agent = get_agent() if get_agent else None
        if new_agent is not None and old_history:
            new_agent.history = old_history
            heal_orphan_tool_calls(new_agent.history)
        return "Restarted. Previous tool call was abandoned."

    def _cmd_model(arg):
        """Handler for /model — manage LLM profiles."""
        from Stage_0.services.llmService import LLMRouter

        router = services.get("llm")
        if not isinstance(router, LLMRouter):
            return "LLM service is not a profile router. Run /load llm to load it."

        parts = arg.strip().split(None, 1) if arg.strip() else []
        sub = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""

        if sub in ("list", "ls"):
            infos = router.list_profiles()
            if not infos:
                return ("No LLM profiles configured. "
                        "Run /model add <name> {json} to create one.")
            active_name = ctrl.config.get("active_llm_profile")
            lines = ["LLM Profiles:"]
            for p in infos:
                profile = ctrl.config.get("llm_profiles", {}).get(p["name"], {})
                lines.append(_describe_profile(
                    p["name"], profile,
                    active=(p["name"] == active_name),
                    loaded=p.get("loaded", False),
                ))
            return "\n".join(lines)

        elif sub == "switch":
            name = rest.strip()
            if not name:
                return "Usage: /model switch <profile_name>"
            profiles = ctrl.config.get("llm_profiles", {})
            if name not in profiles:
                return (f"Unknown profile: '{name}'. "
                        f"Run /model list to see all profiles.")
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
                return ("Usage: /model add <profile_name> {json}\n"
                        "Keys: llm_model_name, llm_endpoint, llm_api_key, "
                        "llm_context_size, llm_service_class (OpenAILLM|LMStudioLLM)")
            if not json_str:
                return ("Usage: /model add <profile_name> {json}\n"
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
                return (f"Unknown profile: '{name}'. "
                        f"Run /model list to see all profiles.")
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
                return (f"Unknown profile: '{name}'. "
                        f"Run /model list to see all profiles.")
            # Mask API keys
            display = dict(profiles[name])
            if "llm_api_key" in display:
                display["llm_api_key"] = _mask_api_key(display["llm_api_key"])
            return f"{name}:\n{_json.dumps(display, indent=2)}"

        return (f"Unknown subcommand: '{sub}'. "
                f"Use: list, switch, add, remove, show.")

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
            from event_bus import bus
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
                        '{\"channel\": \"subagent_run\", \"cron\": \"0 8 * * *\", '
                        '\"payload\": {\"prompt\": \"Summarize overnight activity\", '
                        '\"title\": \"Daily digest\"}}')
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

    _model_completions = lambda: (
        ["list", "switch", "add", "remove", "show"]
        + sorted(ctrl.config.get("llm_profiles", {}).keys())
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
        CommandEntry("restart", "Force-rebuild the agent when a tool is stuck",
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
        CommandEntry("enable", "Enable a tool for agent use", "<tool_name>",
                     handler=lambda a: ctrl.enable_tool(a) if a
                             else "Usage: /enable <tool_name>",
                     arg_completions=_tool_names,
                     category="Services & Tools"),
        CommandEntry("disable", "Disable a tool", "<tool_name>",
                     handler=lambda a: ctrl.disable_tool(a) if a
                             else "Usage: /disable <tool_name>",
                     arg_completions=_tool_names,
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
        CommandEntry("model", "Manage LLM profiles",
                     "[list|switch|add|remove|show] [args]",
                     handler=_cmd_model, arg_completions=_model_completions,
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
