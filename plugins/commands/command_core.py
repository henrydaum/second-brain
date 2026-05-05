from __future__ import annotations

import json
import logging
import subprocess
import threading
from datetime import datetime

from plugins.BaseCommand import BaseCommand
from state_machine.conversationClass import FormStep

logger = logging.getLogger("Commands")

_AGENT_MODE_FIELDS = ("whitelist_or_blacklist_tools",)
_AGENT_LIST_FIELDS = ("tools_list",)
_AGENT_SCOPE_FIELDS = _AGENT_MODE_FIELDS + _AGENT_LIST_FIELDS
_AGENT_PROFILE_FIELDS = ("llm", "prompt_suffix") + _AGENT_SCOPE_FIELDS
_LLM_PROFILE_FIELDS = ("llm_endpoint", "llm_api_key", "llm_context_size", "llm_service_class")
_WATCHER_KEYS = {"sync_directories", "ignored_extensions", "ignored_folders", "skip_hidden_folders"}


def _ctrl(c):
    return c.controller


def _rescope(c):
    runtime = getattr(c, "runtime", None)
    if runtime:
        runtime.refresh_session_specs()


def _plugin_keys():
    from plugins.plugin_discovery import get_plugin_settings
    return {entry[1] for entry in get_plugin_settings()}


def _setting_map():
    from config.config_data import SETTINGS_DATA
    from plugins.plugin_discovery import get_plugin_settings

    def hidden(type_info):
        return isinstance(type_info, dict) and type_info.get("hidden") is True

    merged = {name: (title, desc) for title, name, desc, _, ti in SETTINGS_DATA if not hidden(ti)}
    for title, name, desc, _, ti in get_plugin_settings():
        if not hidden(ti) and name not in merged:
            merged[name] = (title, desc)
    return merged


def _save_config(c, key, value):
    import config.config_manager as cm
    c.config[key] = value
    cm.save(c.config)
    if key in _plugin_keys():
        plugin_config = cm.load_plugin_config()
        plugin_config[key] = value
        cm.save_plugin_config(plugin_config)


def _mask_api_key(value: str) -> str:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith("$") or value.isupper():
        return value
    return "****" if len(value) <= 8 else f"{value[:4]}...{value[-4:]}"


def _describe_llm(model_name: str, profile: dict, default: bool, loaded: bool) -> str:
    marker = "*" if default else " "
    status = "default" if default else ("loaded" if loaded else "")
    cls = profile.get("llm_service_class") or "?"
    ctx = profile.get("llm_context_size") or 0
    ctx_str = f"ctx {ctx // 1000}k" if ctx and ctx >= 1000 else f"ctx {ctx}" if ctx else "ctx auto"
    return f"  {marker} {model_name:<24} {status:<8} {cls:<14} {ctx_str:<10} {profile.get('llm_endpoint') or '(default)'}"


def _describe_agent_profile(name: str, profile: dict, active: bool) -> str:
    marker = "*" if active else " "
    mode = profile.get("whitelist_or_blacklist_tools", "blacklist")
    scope = [f"tools{'+' if mode == 'whitelist' else '-'}{len(profile.get('tools_list') or [])}"]
    if profile.get("prompt_suffix"):
        scope.append("prompt+")
    return f"  {marker} {name:<16} {'active' if active else '':<8} llm={profile.get('llm') or 'default'}  [{','.join(scope)}]"


def _normalize_agent_profile(profile: dict) -> dict:
    val = profile.get("tools_list", [])
    return {
        "llm": profile.get("llm") or "default",
        "prompt_suffix": str(profile.get("prompt_suffix") or ""),
        "whitelist_or_blacklist_tools": profile.get("whitelist_or_blacklist_tools") or "blacklist",
        "tools_list": [] if val in (None, "", {}) else val,
    }


def _schema_fields(schema):
    from state_machine.forms import schema_to_form_steps
    fields = schema_to_form_steps(schema)
    for field in fields:
        if not field.required:
            field.prompt_when_missing = True
    return fields


class HelpCommand(BaseCommand):
    name = "help"; description = "Show available commands"; category = "Conversation"
    def run(self, args, c): return c.command_registry.help_text()


class NewCommand(BaseCommand):
    name = "new"; description = "Start a new conversation"; category = "Conversation"
    def run(self, args, c): return f"New conversation started. Agent: {c.config.get('active_agent_profile') or 'default'}."


class CancelCommand(BaseCommand):
    name = "cancel"; description = "Interrupt the agent"; category = "Conversation"
    def run(self, args, c): return "Cancelled."


class RefreshCommand(BaseCommand):
    name = "refresh"; description = "Rebuild the agent when a tool is stuck (try /restart if a service is hung)"; category = "Conversation"
    def run(self, args, c): return "Refresh is handled by the active frontend runtime."


class RestartCommand(BaseCommand):
    name = "restart"; description = "Restart the whole app - escape hatch when /refresh is not enough"; category = "Conversation"
    def run(self, args, c):
        fn = getattr(_ctrl(c), "restart", None)
        if fn is None:
            return "Restart is not supported in this frontend."
        threading.Timer(0.75, fn).start()
        return "Restarting - Second Brain will be back in a few seconds."


class HistoryCommand(BaseCommand):
    name = "history"; description = "List or load past conversations"; category = "Conversation"
    def form(self, args, c): return [FormStep("id", "Conversation id", False, default="")]
    def run(self, args, c):
        conv_id = args.get("id")
        if str(conv_id or "").isdigit():
            messages = c.db.get_conversation_messages(int(conv_id))
            if not messages:
                return f"Unknown conversation: {conv_id}. Run /history to see recent conversations."
            conversation = c.db.get_conversation(int(conv_id))
            return f"Loaded conversation: {((conversation or {}).get('title') or '').strip() or 'New conversation'}"
        conversations = c.db.list_user_conversations(limit=10)
        if not conversations:
            return "No conversations yet."
        lines = ["Recent conversations:"]
        for conv in conversations:
            title = (conv["title"] or "New conversation").replace("\n", " ")[:50]
            ts = conv.get("updated_at")
            lines.append(f"  [{conv['id']}] {title}  ({datetime.fromtimestamp(ts).strftime('%b %d, %I:%M %p') if ts else ''})")
        return "\n".join(lines + ["", "Run /history <id> to load a conversation."])


class ServicesCommand(BaseCommand):
    name = "services"; description = "List services and status"; category = "Services & Tools"
    def run(self, args, c):
        from plugins.frontends.helpers.formatters import format_services
        return format_services(_ctrl(c).list_services())


class LoadCommand(BaseCommand):
    name = "load"; description = "Load a service"; category = "Services & Tools"
    def form(self, args, c): return [FormStep("service_name", "Service name", True, enum=sorted(c.services))]
    def arg_completions(self, c): return sorted(c.services)
    def run(self, args, c):
        name = (args.get("service_name") or "").strip()
        return _ctrl(c).load_service(name) if name else "Usage: /load <service_name>"


class UnloadCommand(LoadCommand):
    name = "unload"; description = "Unload a service"
    def run(self, args, c):
        name = (args.get("service_name") or "").strip()
        return _ctrl(c).unload_service(name) if name else "Usage: /unload <service_name>"


class ToolsCommand(BaseCommand):
    name = "tools"; description = "List registered tools"; category = "Services & Tools"
    def run(self, args, c):
        from plugins.frontends.helpers.formatters import format_tools
        return format_tools(_ctrl(c).list_tools())


class CallCommand(BaseCommand):
    name = "call"; description = "Call a tool directly"; category = "Services & Tools"
    def form(self, args, c):
        tools = c.tool_registry
        steps = [FormStep("tool_name", "Tool name", True, enum=sorted((getattr(tools, "tools", {}) or {}).keys()))]
        schema = tools.get_schema(args.get("tool_name")) if tools and args.get("tool_name") else None
        return steps + (_schema_fields(schema.get("function", schema).get("parameters")) if schema else [])
    def arg_completions(self, c): return sorted((getattr(c.tool_registry, "tools", {}) or {}).keys())
    def run(self, args, c):
        from plugins.frontends.helpers.formatters import format_tool_result
        name = args.get("tool_name")
        if not name:
            return "Usage: /call <tool_name> {json_args}"
        return format_tool_result(_ctrl(c).call_tool(name, {k: v for k, v in args.items() if k != "tool_name"}))


class TasksCommand(BaseCommand):
    name = "tasks"; description = "List path-driven and event-driven tasks"; category = "Tasks"
    def run(self, args, c):
        from plugins.frontends.helpers.formatters import format_tasks
        return format_tasks(_ctrl(c).list_tasks())


class PipelineCommand(BaseCommand):
    name = "pipeline"; description = "Show the path-driven task dependency graph"; category = "Tasks"
    def run(self, args, c): return c.orchestrator.dependency_pipeline_graph()


class _TaskNameCommand(BaseCommand):
    category = "Tasks"
    trigger = None
    usage = ""
    def form(self, args, c): return [FormStep("task_name", "Task name", True, enum=_ctrl(c).list_task_names(trigger=self.trigger) if self.trigger else _ctrl(c).list_task_names())]
    def arg_completions(self, c): return _ctrl(c).list_task_names(trigger=self.trigger) if self.trigger else _ctrl(c).list_task_names()


class PauseCommand(_TaskNameCommand):
    name = "pause"; description = "Pause a task"; usage = "Usage: /pause <task_name>"
    def run(self, args, c): return _ctrl(c).pause_task(args.get("task_name")) if args.get("task_name") else self.usage


class UnpauseCommand(_TaskNameCommand):
    name = "unpause"; description = "Unpause a task"; usage = "Usage: /unpause <task_name>"
    def run(self, args, c): return _ctrl(c).unpause_task(args.get("task_name")) if args.get("task_name") else self.usage


class ResetCommand(_TaskNameCommand):
    name = "reset"; description = "Reset a path-driven task to Pending"; trigger = "path"; usage = "Usage: /reset <task_name>"
    def run(self, args, c): return _ctrl(c).reset_task(args.get("task_name")) if args.get("task_name") else self.usage


class RetryCommand(_TaskNameCommand):
    name = "retry"; description = "Retry failed path-driven task entries"; trigger = "path"
    def form(self, args, c): return [FormStep("task_name", "Task name or all", True, enum=["all"] + _ctrl(c).list_task_names(trigger="path"))]
    def arg_completions(self, c): return ["all"] + _ctrl(c).list_task_names(trigger="path")
    def run(self, args, c):
        name = args.get("task_name")
        return _ctrl(c).retry_all() if name and name.lower() == "all" else _ctrl(c).retry_task(name) if name else "Usage: /retry <task_name>|all"


class TriggerCommand(BaseCommand):
    name = "trigger"; description = "Manually fire an event-triggered task"; category = "Tasks"
    def form(self, args, c):
        steps = [FormStep("task_name", "Event task name", True, enum=_ctrl(c).list_task_names(trigger="event"))]
        task = c.orchestrator.tasks.get(args.get("task_name")) if args.get("task_name") else None
        schema = getattr(task, "event_payload_schema", None) if task else None
        return steps + (_schema_fields(schema) if schema else [])
    def arg_completions(self, c): return _ctrl(c).list_task_names(trigger="event")
    def run(self, args, c):
        name = args.get("task_name")
        task = c.orchestrator.tasks.get(name) if name else None
        if not name:
            return "Usage: /trigger <task_name> [json_payload]"
        if task is None:
            return f"Unknown task: '{name}'. Run /tasks to see all tasks."
        if getattr(task, "trigger", "path") != "event":
            return f"Task '{name}' is not event-triggered. Run /tasks to see event-driven tasks."
        return _ctrl(c).trigger_event_task(name, {k: v for k, v in args.items() if k != "task_name"} or None)


class ConfigCommand(BaseCommand):
    name = "config"; description = "Show config settings"; category = "Config & System"
    def form(self, args, c): return [FormStep("setting_name", "Setting name", False, enum=sorted(_setting_map()), default="")]
    def run(self, args, c):
        settings, key = _setting_map(), (args.get("setting_name") or "").strip()
        if key:
            if key not in settings:
                return f"Unknown setting: '{key}'. Run /config to see all settings."
            return f"{key} = {c.config.get(key)}\n  {settings[key][1]}"
        return "Settings:\n" + "\n".join(f"  {name} = {c.config.get(name)}" for name in settings)


class ConfigureCommand(BaseCommand):
    name = "configure"; description = "Update a config setting"; category = "Config & System"
    def form(self, args, c): return [FormStep("setting_name", "Setting name", True, enum=sorted(_setting_map())), FormStep("value", "New value", True)]
    def run(self, args, c):
        key, raw = args.get("setting_name"), args.get("value")
        if not key:
            return "Usage: /configure <setting_name> <value>"
        if key not in _setting_map():
            return f"Unknown setting: '{key}'. Run /config to see all settings."
        if isinstance(raw, str):
            try: value = json.loads(raw)
            except json.JSONDecodeError: value = raw
        else:
            value = raw
        old = c.config.get(key)
        _save_config(c, key, value)
        parts = [f"Set {key} = {value}"]
        if key in _WATCHER_KEYS and getattr(_ctrl(c), "watcher", None):
            _ctrl(c).watcher.rescan()
        if value != old:
            parts += [f"  - {x}" for x in _ctrl(c).reload_services_for_settings({key}, c.root_dir)]
            parts += [f"  - {x}" for x in _ctrl(c).apply_runtime_config_changes({key})]
        return "\n".join(parts)


class LlmCommand(BaseCommand):
    name = "llm"; description = "Manage LLM connection configs"; category = "Config & System"
    def form(self, args, c):
        sub = args.get("subcommand")
        steps = [FormStep("subcommand", "Subcommand", True, enum=["list", "add", "edit", "remove", "show", "default"])]
        if sub in {None, "", "list"}: return steps
        if sub in {"default", "show", "remove"}: return steps + [FormStep("model_name", "Model name", True)]
        if sub == "edit": return steps + [FormStep("model_name", "Model name", True), FormStep("field", "Field to edit", True, enum=list(_LLM_PROFILE_FIELDS)), FormStep("value", "New value", True)]
        if sub == "add": return steps + [FormStep("model_name", "Model name", True), FormStep("llm_service_class", "LLM service class", True, enum=["OpenAILLM", "LMStudioLLM"]), FormStep("llm_endpoint", "Endpoint", False, default="", prompt_when_missing=True), FormStep("llm_api_key", "API key or env var", False, default="", prompt_when_missing=True), FormStep("llm_context_size", "Context size", False, "integer", default=0, prompt_when_missing=True)]
        return steps
    def arg_completions(self, c): return ["list", "add", "edit", "remove", "show", "default"] + sorted((c.config.get("llm_profiles", {}) or {}))
    def run(self, args, c):
        from plugins.services.llmService import LLMRouter
        router, sub = c.services.get("llm"), (args.get("subcommand") or "list").lower()
        if not isinstance(router, LLMRouter): return "LLM service is not a router. Run /load llm to load it."
        profiles = c.config.setdefault("llm_profiles", {})
        if sub in ("list", "ls"):
            infos = router.list_llms()
            return "No LLMs configured. Run /llm add <model_name> {json} to create one." if not infos else "\n".join(["LLMs:"] + [_describe_llm(i["model_name"], {"llm_service_class": i["class"], "llm_context_size": i["context_size"], "llm_endpoint": i["endpoint"]}, i["default"], i["loaded"]) for i in infos])
        model = args.get("model_name")
        if sub == "default":
            if not model: return "Usage: /llm default <model_name>"
            if model not in profiles: return f"Unknown LLM: '{model}'. Run /llm list to see all LLMs."
            _save_config(c, "default_llm_profile", model); _rescope(c); return f"Default LLM set to '{model}'."
        if sub == "add":
            if not model: return "Usage: /llm add <model_name> {json}"
            if model == "default": return "Cannot name an LLM 'default' - that string is the sentinel agent profiles use to follow whatever LLM is the current default. Pick a model name instead."
            profile = {k: args.get(k) for k in _LLM_PROFILE_FIELDS if k in args}
            profiles[model] = profile; router.add_llm(model, profile); _save_config(c, "llm_profiles", profiles)
            if not c.config.get("default_llm_profile"):
                _save_config(c, "default_llm_profile", model); return f"LLM '{model}' added and set as default."
            return f"LLM '{model}' added."
        if sub == "remove":
            if not model: return "Usage: /llm remove <model_name>"
            if model not in profiles: return f"Unknown LLM: '{model}'. Run /llm list to see all LLMs."
            if len(profiles) == 1: return "Refusing to remove the only LLM. Add another one first."
            was_default = c.config.get("default_llm_profile") == model
            stale = sorted(n for n, p in (c.config.get("agent_profiles", {}) or {}).items() if (p or {}).get("llm") == model)
            router.remove_llm(model); del profiles[model]; _save_config(c, "llm_profiles", profiles)
            lines = []
            if was_default:
                new_default = next(iter(profiles)); _save_config(c, "default_llm_profile", new_default); _rescope(c); lines.append(f"LLM '{model}' removed. Default is now '{new_default}'.")
            else:
                lines.append(f"LLM '{model}' removed.")
            if stale: lines.append(f"Warning: {len(stale)} agent profile(s) still reference '{model}' by name: {', '.join(stale)}. Until you edit them, they will fall back to the default LLM.")
            return "\n".join(lines)
        if sub == "show":
            if not model: return "Usage: /llm show <model_name>"
            if model not in profiles: return f"Unknown LLM: '{model}'. Run /llm list to see all LLMs."
            display = dict(profiles[model]); display["llm_api_key"] = _mask_api_key(display.get("llm_api_key"))
            return f"{model}:\n{json.dumps(display, indent=2)}"
        if sub == "edit":
            field, value = args.get("field"), args.get("value")
            if not model or not field: return f"Usage: /llm edit <model_name> <field> <value>\nFields: {', '.join(_LLM_PROFILE_FIELDS)}"
            if model not in profiles: return f"Unknown LLM: '{model}'. Run /llm list to see all LLMs."
            if field not in _LLM_PROFILE_FIELDS: return f"Unknown field: '{field}'. Allowed: {', '.join(_LLM_PROFILE_FIELDS)}."
            if field == "llm_context_size":
                try: value = int(value)
                except (ValueError, TypeError): return "llm_context_size must be an integer."
            if field == "llm_service_class" and value not in ("OpenAILLM", "LMStudioLLM"): return "llm_service_class must be 'OpenAILLM' or 'LMStudioLLM'."
            profiles[model][field] = value
            try:
                if model in router.services and getattr(router.services[model], "loaded", False): router.services[model].unload()
            except Exception as e: logger.warning(f"Unload during /llm edit failed: {e}")
            router.add_llm(model, profiles[model]); _save_config(c, "llm_profiles", profiles); _rescope(c)
            return f"Updated {model}.{field}."
        return f"Unknown subcommand: '{sub}'. Use: list, add, edit, remove, show, default."


class AgentCommand(BaseCommand):
    name = "agent"; description = "Manage scoped agent profiles"; category = "Config & System"
    def form(self, args, c):
        sub = args.get("subcommand")
        steps = [FormStep("subcommand", "Subcommand", True, enum=["list", "switch", "add", "edit", "remove", "show"])]
        if sub in {None, "", "list"}: return steps
        if sub in {"switch", "remove", "show"}: return steps + [FormStep("profile_name", "Agent profile name", True)]
        if sub == "edit": return steps + [FormStep("profile_name", "Agent profile name", True), FormStep("field", "Field to edit", True, enum=list(_AGENT_PROFILE_FIELDS)), FormStep("value", "New value", True)]
        if sub == "add": return steps + [FormStep("profile_name", "Agent profile name", True), FormStep("llm", "LLM name or default", False, default="default", prompt_when_missing=True), FormStep("prompt_suffix", "Prompt suffix", False, default="", prompt_when_missing=True), FormStep("whitelist_or_blacklist_tools", "Tool filter mode", False, default="blacklist", enum=["whitelist", "blacklist"], prompt_when_missing=True), FormStep("tools_list", "Tool names JSON array", False, "array", default=[], prompt_when_missing=True)]
        return steps
    def arg_completions(self, c): return ["list", "switch", "add", "edit", "remove", "show"] + sorted((c.config.get("agent_profiles", {}) or {}))
    def run(self, args, c):
        from runtime.agent_scope import load_scope
        sub = (args.get("subcommand") or "list").lower()
        profiles = c.config.setdefault("agent_profiles", {})
        active = c.config.get("active_agent_profile") or "default"
        if sub in ("list", "ls"):
            return "No agent profiles configured. Run /agent add <name> {json} to create one." if not profiles else "\n".join(["Agent Profiles:"] + [_describe_agent_profile(n, p, n == active) for n, p in profiles.items()])
        name = args.get("profile_name")
        if sub == "switch":
            if not name: return "Usage: /agent switch <profile_name>"
            if name not in profiles: return f"Unknown agent profile: '{name}'. Run /agent list to see all profiles."
            try: load_scope(name, c.config)
            except ValueError as e: return f"Cannot switch to '{name}': {e}"
            _save_config(c, "active_agent_profile", name); _rescope(c); return f"Switched to agent profile '{name}'."
        if sub == "add":
            if not name: return "Usage: /agent add <profile_name> {json}"
            profile = _normalize_agent_profile(args)
            llm_ref = profile.get("llm") or "default"
            if llm_ref != "default" and llm_ref not in (c.config.get("llm_profiles", {}) or {}): return f"Profile references unknown LLM '{llm_ref}'. Run /llm list to see all LLMs."
            profiles[name] = profile
            try: load_scope(name, c.config)
            except ValueError as e: del profiles[name]; return f"Invalid scope for '{name}': {e}"
            _save_config(c, "agent_profiles", profiles); return f"Agent profile '{name}' added."
        if sub == "remove":
            if not name: return "Usage: /agent remove <profile_name>"
            if name == "default": return "Refusing to remove the 'default' agent profile."
            if name not in profiles: return f"Unknown agent profile: '{name}'. Run /agent list to see all profiles."
            was_active = active == name; del profiles[name]; _save_config(c, "agent_profiles", profiles)
            if was_active: _save_config(c, "active_agent_profile", "default"); _rescope(c); return f"Agent profile '{name}' removed. Active is now 'default'."
            return f"Agent profile '{name}' removed."
        if sub == "show":
            if not name: return "Usage: /agent show <profile_name>"
            if name not in profiles: return f"Unknown agent profile: '{name}'. Run /agent list to see all profiles."
            return f"{name}:\n{json.dumps(_normalize_agent_profile(profiles[name]), indent=2)}"
        if sub == "edit":
            field, value = args.get("field"), args.get("value")
            if not name or not field: return f"Usage: /agent edit <profile_name> <field> <value>\nFields: {', '.join(_AGENT_PROFILE_FIELDS)}"
            if name not in profiles: return f"Unknown agent profile: '{name}'. Run /agent list to see all profiles."
            if field not in _AGENT_PROFILE_FIELDS: return f"Unknown field: '{field}'. Allowed: {', '.join(_AGENT_PROFILE_FIELDS)}."
            if field == "llm" and (not isinstance(value, str) or not value): return "llm must be a string ('default' or a model name)."
            if field == "llm" and value != "default" and value not in (c.config.get("llm_profiles", {}) or {}): return f"Unknown LLM '{value}'. Run /llm list to see all LLMs."
            if field in _AGENT_MODE_FIELDS and value not in ("whitelist", "blacklist"): return f"{field} must be 'whitelist' or 'blacklist'."
            if field in _AGENT_LIST_FIELDS and value in (None, "", {}): value = []
            if field in _AGENT_LIST_FIELDS and not isinstance(value, list): return f"{field} must be a JSON array."
            if field == "prompt_suffix": value = "" if value is None else str(value)
            old = profiles[name].get(field); profiles[name][field] = value; profiles[name] = _normalize_agent_profile(profiles[name])
            try: load_scope(name, c.config)
            except ValueError as e: profiles[name][field] = old; return f"Edit rejected: {e}"
            _save_config(c, "agent_profiles", profiles)
            if name == active: _rescope(c)
            return f"Updated {name}.{field}."
        return f"Unknown subcommand: '{sub}'. Use: list, switch, add, edit, remove, show."


class ScheduleCommand(BaseCommand):
    name = "schedule"; description = "Manage scheduled jobs"; category = "Config & System"
    def form(self, args, c):
        sub = args.get("subcommand")
        steps = [FormStep("subcommand", "Subcommand", True, enum=["list", "show", "enable", "disable", "delete", "run", "create"])]
        if sub in {None, "", "list"}: return steps
        if sub in {"show", "enable", "disable", "delete", "run"}: return steps + [FormStep("job_name", "Job name", True)]
        if sub == "create":
            time_field = "run_at" if args.get("one_time") else "cron"
            return steps + [FormStep("job_name", "Job name", True), FormStep("channel", "Event channel", True), FormStep("one_time", "One-time job?", False, "boolean", default=False, prompt_when_missing=True), FormStep(time_field, "Run at ISO datetime" if args.get("one_time") else "Cron expression", True), FormStep("payload", "Payload JSON object", False, "object", default={}, prompt_when_missing=True)]
        return steps
    def arg_completions(self, c):
        tk = c.services.get("timekeeper"); jobs = sorted(tk.list_jobs()) if tk and getattr(tk, "loaded", False) else []
        return ["list", "show", "enable", "disable", "delete", "run", "create"] + jobs
    def run(self, args, c):
        tk, sub = c.services.get("timekeeper"), (args.get("subcommand") or "list").lower()
        if tk is None or not getattr(tk, "loaded", False): return "Timekeeper service is not loaded. Run /load timekeeper to start it."
        if sub in ("list", "ls"):
            from plugins.frontends.helpers.formatters import format_scheduled_jobs
            return format_scheduled_jobs(tk.list_jobs(), tk)
        name = args.get("job_name")
        if sub == "show":
            if not name: return "Usage: /schedule show <job_name>"
            job = tk.get_job(name) if name else None
            return f"{name}:\n{json.dumps(job, indent=2, default=str)}" if job else f"Unknown job: '{name}'. Run /schedule list to see all jobs."
        if sub in ("enable", "disable"):
            if not name: return f"Usage: /schedule {sub} <job_name>"
            if tk.get_job(name) is None: return f"Unknown job: '{name}'. Run /schedule list to see all jobs."
            tk.enable_job(name, sub == "enable"); return f"Job '{name}' {sub}d."
        if sub == "delete":
            if not name: return "Usage: /schedule delete <job_name>"
            return f"Job '{name}' deleted." if tk.remove_job(name) else f"Unknown job: '{name}'. Run /schedule list to see all jobs."
        if sub == "run":
            if not name: return "Usage: /schedule run <job_name>"
            job = tk.get_job(name) if name else None
            if not job: return f"Unknown job: '{name}'. Run /schedule list to see all jobs."
            from events.event_bus import bus
            payload = dict(job.get("payload", {})); now = datetime.now().astimezone().isoformat()
            payload["_timekeeper"] = {"job_name": name, "scheduled_for": now, "emitted_at": now, "one_time": bool(job.get("one_time")), "source": "timekeeper_manual"}
            bus.emit(job["channel"], payload); return f"Job '{name}' fired manually."
        if sub == "create":
            if not name or not args.get("channel"): return "Usage: /schedule create <job_name> {json}"
            definition = {k: v for k, v in args.items() if k not in {"subcommand", "job_name"} and v not in (None, "")}
            try: tk.create_job(name, definition)
            except ValueError as e: return f"Failed to create job '{name}': {e}"
            return f"Job '{name}' created."
        return f"Unknown subcommand: '{sub}'. Use: list, show, enable, disable, delete, run, create."


class MessageCommand(BaseCommand):
    name = "message"; description = "Leave a message for a scheduled subagent"; category = "Config & System"
    def form(self, args, c): return [FormStep("job_name", "Subagent job name", True), FormStep("notify", "Notification override", False, enum=["all", "important", "off"], default="", prompt_when_missing=True), FormStep("text", "Message text", True)]
    def arg_completions(self, c):
        tk = c.services.get("timekeeper")
        if tk is None or not getattr(tk, "loaded", False): return ["--notify=all", "--notify=important", "--notify=off"]
        from events.event_channels import SUBAGENT_RUN
        return sorted(n for n, j in tk.list_jobs().items() if (j.get("channel") or "").strip() == SUBAGENT_RUN) + ["--notify=all", "--notify=important", "--notify=off"]
    def run(self, args, c):
        from events.event_channels import SUBAGENT_RUN
        tk, job_name, text = c.services.get("timekeeper"), args.get("job_name"), (args.get("text") or "").strip()
        if tk is None or not getattr(tk, "loaded", False): return "Timekeeper service is not loaded. Run /load timekeeper to start it."
        if not job_name or not text: return "Usage: /message <job_name> [--notify=all|important|off] <text>"
        job = tk.get_job(job_name)
        if job is None or (job.get("channel") or "").strip() != SUBAGENT_RUN: return f"Unknown subagent job: '{job_name}'. Run /schedule list to see scheduled subagents."
        runtime = c.runtime
        if runtime is None: return "Conversation runtime is not available."
        payload, new_payload = job.get("payload") or {}, None
        conv_id = payload.get("conversation_id")
        if conv_id is None or c.db.get_conversation(int(conv_id)) is None:
            conv_id = runtime.create_conversation((payload.get("title") or job_name or "Scheduled subagent")[:200], kind="subagent")
            new_payload = dict(payload, conversation_id=conv_id)
        if args.get("notify"):
            new_payload = dict(new_payload if new_payload is not None else payload, next_notifications=args["notify"])
        if new_payload is not None:
            try: tk.update_job(job_name, {"payload": new_payload})
            except ValueError as e: return f"Failed to update job '{job_name}': {e}"
        key, existed = runtime.subagent_session_key(job_name), runtime.subagent_session_key(job_name) in getattr(runtime, "sessions", {})
        runtime.inject_user_message(key, text, conversation_id=int(conv_id))
        if not existed: runtime.unload_conversation(key)
        pending = c.db.count_pending_inbox(int(conv_id))
        suffix = f" (next run notifications: {args.get('notify')})" if args.get("notify") else ""
        return f"Queued for '{job_name}'{suffix}. {pending} pending message{'s' if pending != 1 else ''} until next wake."


class LocationsCommand(BaseCommand):
    name = "locations"; description = "List file-system locations"; category = "Config & System"
    def form(self, args, c): return [FormStep("mode", "Location type", False, enum=["tools", "tasks", "services"], default="")]
    def arg_completions(self, c): return ["tools", "tasks", "services"]
    def run(self, args, c):
        from plugins.frontends.helpers.formatters import format_locations
        mode = (args.get("mode") or "").lower()
        if mode and mode not in ("tools", "tasks", "services"):
            return "Usage: /locations [tools|tasks|services]\nExample: /locations tools"
        return format_locations(_ctrl(c).list_locations(mode if mode in ("tools", "tasks", "services") else None))


class ReloadCommand(BaseCommand):
    name = "reload"; description = "Hot-reload tasks, tools, and commands"; category = "Config & System"
    def run(self, args, c): return _ctrl(c).reload_plugins(c.root_dir)


class UpdateCommand(BaseCommand):
    name = "update"; description = "Pull latest changes from the Second Brain repo"; category = "Config & System"
    def run(self, args, c):
        try:
            result = subprocess.run(["git", "pull"], capture_output=True, text=True, timeout=60, cwd=c.root_dir)
        except Exception as e:
            return f"Update failed: {e}"
        out, err = (result.stdout or "").strip(), (result.stderr or "").strip()
        if result.returncode == 0:
            if not out or out.lower().startswith("already up to date"):
                return out or "Already up to date."
            return f"{out}\n\n/restart to take effect"
        return f"git pull failed (exit {result.returncode}):\n{err or out}"
