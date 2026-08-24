"""Slash command plugin for `/agent`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


ACTIONS = ["switch", "edit", "remove"]
PROFILE_FIELDS = [
    "llm", "prompt_suffix", "whitelist_or_blacklist_tools", "tools_list"]
FIELDS = ["agent_profile_name", *PROFILE_FIELDS]
FIELD_LABELS = [
    "Profile name", "LLM", "Prompt suffix", "Tool mode", "Tool list"]


class AgentCommand(BaseCommand):
    """Inspect, switch, and manage agent profiles."""

    name = "agent"
    description = "Select an agent profile, then switch, edit, or remove it"
    category = "System"
    requests = ["config.read", "config.write", "tool.list"]
    def form(self, sdk, args):
        profiles = sdk.config.read("agent_profiles") or {}
        active = sdk.config.read("active_agent_profile") or "default"
        names = [*sorted(profiles), "add"]
        steps = [FormStep(
            "profile_name",
            "Select an agent profile, or add a new one.",
            True,
            enum=names,
            enum_labels=[_profile_label(active, item) for item in names],
        )]
        llms = [
            "default",
            *sorted(sdk.config.read("llm_profiles", keys=True)),
        ]
        tools = sorted(sdk.tools.list())
        if args.get("profile_name") == "add":
            return steps + [
                FormStep(
                    "new_profile_name",
                    "Enter a short name for the new agent profile.",
                    True),
                FormStep(
                    "llm",
                    "Choose the LLM this agent should use. Select default to "
                    "follow the current default LLM.",
                    True, enum=llms, default="default"),
                FormStep(
                    "prompt_suffix",
                    "Optional extra instructions to append to this agent's "
                    "system prompt.",
                    False, default="", prompt_when_missing=True),
                FormStep(
                    "whitelist_or_blacklist_tools",
                    "Choose how this profile should treat the tool list.",
                    True, enum=["blacklist", "whitelist"],
                    default="blacklist",
                    enum_labels=["Blacklist tools", "Whitelist tools"]),
                FormStep(
                    "tools_list",
                    f"Optional tool names. Available: "
                    f"{', '.join(tools) or '(none)'}",
                    False, "array", default=[], prompt_when_missing=True),
            ]
        name = args.get("profile_name")
        if name:
            steps.append(FormStep(
                "action",
                "What do you want to do with this agent profile?\n\n"
                + _describe(sdk, profiles, active, name),
                True,
                enum=ACTIONS,
                enum_labels=["Switch to it", "Edit it", "Remove it"],
            ))
        if args.get("action") == "edit":
            field = args.get("field")
            steps += [
                FormStep(
                    "field",
                    "Choose which part of the agent profile to edit.",
                    True, enum=FIELDS, enum_labels=FIELD_LABELS),
                FormStep(
                    "value", _value_prompt(field), True,
                    "array" if field == "tools_list" else "string"),
            ]
        return steps

    def run(self, sdk, args):
        profiles = sdk.config.read("agent_profiles") or {}
        active = sdk.config.read("active_agent_profile") or "default"
        name = args.get("profile_name")
        if name == "add":
            name = (args.get("new_profile_name") or "").strip()
            if not name:
                return "Profile name is required."
            profiles[name] = _profile(args)
            sdk.config.write("agent_profiles", profiles)
            return f"Added agent profile: {name}"
        if name not in profiles:
            return "Unknown agent profile."
        action = args.get("action")
        if action == "switch":
            sdk.config.write("active_agent_profile", name)
            return f"Switched agent profile to: {name}"
        if action == "edit":
            field = args.get("field")
            if field == "agent_profile_name":
                new_name = _coerce(field, args.get("value")).strip()
                if not new_name:
                    return "Profile name is required."
                if new_name != name and new_name in profiles:
                    return f"Agent profile already exists: {new_name}"
                profiles[new_name] = profiles.pop(name)
                sdk.config.write("agent_profiles", profiles)
                if active == name:
                    sdk.config.write("active_agent_profile", new_name)
                name = new_name
            else:
                profiles[name][field] = _coerce(
                    field, args.get("value"))
                sdk.config.write("agent_profiles", profiles)
            return f"Updated agent profile: {name}"
        if action == "remove":
            if name == "default":
                return "Cannot remove the default agent profile."
            profiles.pop(name, None)
            sdk.config.write("agent_profiles", profiles)
            if active == name:
                sdk.config.write("active_agent_profile", "default")
            return f"Removed agent profile: {name}"
        return f"Unknown action: {action}"


def _profile(args):
    return {
        field: _coerce(field, args.get(field))
        for field in PROFILE_FIELDS
    }


def _coerce(field, value):
    if field == "tools_list":
        if isinstance(value, list):
            return value
        return [
            item.strip() for item in str(value or "").splitlines()
            if item.strip()
        ]
    return "" if value is None else str(value)


def _describe(sdk, profiles, active, name):
    profile = profiles.get(name)
    if not profile:
        return "Action"
    suffix = " (active)" if active == name else ""
    return sdk.md.card(f"{name}{suffix}", [
        ("LLM", profile.get("llm", "default")),
        (
            "Tool mode",
            profile.get("whitelist_or_blacklist_tools", "blacklist"),
        ),
        (
            "Tool list",
            ", ".join(profile.get("tools_list") or []) or "(none)",
        ),
    ])


def _profile_label(active, name):
    if name == "add":
        return "Add profile"
    return f"{name} (active)" if active == name else name


def _value_prompt(field):
    return {
        "agent_profile_name": "Enter the new profile name.",
        "llm": "Enter the LLM profile name, or default.",
        "prompt_suffix": (
            "Enter the extra system-prompt instructions for this agent."),
        "whitelist_or_blacklist_tools": (
            "Enter 'blacklist' to block listed tools, or 'whitelist' to allow "
            "only listed tools."),
        "tools_list": "Enter tool names.",
    }.get(field, "Enter the new value.")
