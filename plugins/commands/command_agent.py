import json

from config import config_manager
from plugins.BaseCommand import BaseCommand
from state_machine.conversation import FormStep


ACTIONS = ["switch", "edit", "remove"]
FIELDS = ["llm", "prompt_suffix", "whitelist_or_blacklist_tools", "tools_list"]


class AgentCommand(BaseCommand):
    name = "agent"
    description = "Select an agent profile, then switch, edit, or remove it"
    category = "System"

    def form(self, args, context):
        profiles = context.config.get("agent_profiles", {}) or {}
        steps = [FormStep("profile_name", "Agent", True, enum=[*sorted(profiles), "add"])]
        llms = ["default", *sorted((context.config.get("llm_profiles", {}) or {}).keys())]
        tools = sorted(getattr(getattr(context, "tool_registry", None), "tools", {}))
        if args.get("profile_name") == "add":
            return steps + [
                FormStep("new_profile_name", "Profile name", True),
                FormStep("llm", "LLM", True, enum=llms, default="default"),
                FormStep("prompt_suffix", "Prompt suffix", False, default="", prompt_when_missing=True),
                FormStep("whitelist_or_blacklist_tools", "Tool mode", True, enum=["blacklist", "whitelist"], default="blacklist"),
                FormStep("tools_list", f"Tools JSON array. Available: {', '.join(tools) or '(none)'}", False, "array", default=[], prompt_when_missing=True),
            ]
        if args.get("profile_name"):
            steps.append(FormStep("action", _describe(context, args["profile_name"]), True, enum=ACTIONS))
        if args.get("action") == "edit":
            steps += [FormStep("field", "Field", True, enum=FIELDS), FormStep("value", "Value", True)]
        return steps

    def run(self, args, context):
        profiles = context.config.setdefault("agent_profiles", {})
        name = args.get("profile_name")
        if name == "add":
            name = args.get("new_profile_name", "").strip()
            if not name:
                return "Profile name is required."
            profiles[name] = _profile(args)
            _save(context.config)
            return f"Added agent profile: {name}"
        if name not in profiles:
            return "Unknown agent profile."
        if args.get("action") == "switch":
            runtime, session_key = getattr(context, "runtime", None), getattr(context, "session_key", None)
            if runtime and session_key and runtime.set_agent_profile(session_key, name):
                return f"Switched agent profile to: {name}"
            context.config["active_agent_profile"] = name
            _save(context.config)
            return f"Active agent profile set to: {name}"
        if args.get("action") == "edit":
            field = args.get("field")
            profiles[name][field] = _coerce(field, args.get("value"))
            _save(context.config)
            _refresh(context)
            return f"Updated agent profile: {name}"
        if args.get("action") == "remove":
            if name == "default":
                return "Cannot remove the default agent profile."
            profiles.pop(name, None)
            if context.config.get("active_agent_profile") == name:
                context.config["active_agent_profile"] = "default"
            _save(context.config)
            _refresh(context)
            return f"Removed agent profile: {name}"
        return f"Unknown action: {args.get('action')}"


def _profile(args):
    return {f: _coerce(f, args.get(f)) for f in FIELDS}


def _coerce(field, value):
    if field == "tools_list":
        return value if isinstance(value, list) else json.loads(value or "[]")
    return "" if value is None else str(value)


def _describe(context, name):
    p = (context.config.get("agent_profiles", {}) or {}).get(name)
    if not p:
        return "Action"
    active = " active" if context.config.get("active_agent_profile") == name else ""
    return f"{name}{active}\nLLM: {p.get('llm', 'default')}\nTool mode: {p.get('whitelist_or_blacklist_tools', 'blacklist')}\nTools: {', '.join(p.get('tools_list') or []) or '(none)'}"


def _save(config):
    config_manager.save(config)


def _refresh(context):
    runtime = getattr(context, "runtime", None)
    if runtime and hasattr(runtime, "refresh_session_specs"):
        runtime.refresh_session_specs()
