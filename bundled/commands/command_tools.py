"""Slash command plugin for `/tools`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


ACTIONS = ["call"]
ACTION_LABELS = ["Call tool"]


class ToolsCommand(BaseCommand):
    """Inspect and invoke tools through the kernel-owned registry."""

    name = "tools"
    description = "Select a tool, then call it"
    category = "Capabilities"
    requests = [
        "tool.list", "tool.call", "config.read", "config.write",
    ]

    def form(self, sdk, args):
        """Build the dependent tool, action, argument, and setting steps."""
        tools = sdk.tools.list(details=True)
        steps = [FormStep(
            "tool_name",
            "Select a tool to inspect or call.",
            True,
            enum=[tool["name"] for tool in tools],
            columns=2,
        )]
        tool = _find(tools, args.get("tool_name"))
        if tool:
            links, labels = sdk.forms.setting_actions(
                tool.get("config_settings"))
            steps.append(FormStep(
                "action",
                "What do you want to do with this tool?\n\n"
                + _describe(sdk, tool),
                True,
                enum=ACTIONS + links,
                enum_labels=ACTION_LABELS + labels,
            ))
        if tool and args.get("action") == "call":
            steps += sdk.forms.from_schema(
                tool.get("parameters"), prompt_optional=True)
        setting = sdk.forms.setting_for_action(
            (tool or {}).get("config_settings"), args.get("action"))
        if setting:
            steps.append(sdk.forms.setting_value_step(setting))
        return steps

    def run(self, sdk, args):
        """Execute `/tools` for the active session."""
        tools = sdk.tools.list(details=True)
        name = args.get("tool_name")
        if not name:
            return sdk.md.tools(tools)
        tool = _find(tools, name)
        if not tool:
            return "Unknown tool."

        setting = sdk.forms.setting_for_action(
            tool.get("config_settings"), args.get("action"))
        if setting:
            try:
                sdk.config.write(setting["key"], args.get("value"))
            except sdk.Failed as exc:
                if "user settings are not available" in exc.error.lower():
                    return "User settings are not available in this context."
                raise
            return (
                f"Set {setting['key']} = "
                f"{sdk.text.value(args.get('value'))}"
            )

        if args.get("action") == "call":
            fields = (tool.get("parameters") or {}).get(
                "properties", {}).keys()
            result = sdk.tools.call(
                name,
                _result=True,
                _user_initiated=True,
                **{key: args[key] for key in fields if key in args},
            )
            return sdk.md.tool_result(result)
        return f"Unknown action: {args.get('action')}"


def _find(tools, name):
    return next((tool for tool in tools if tool["name"] == name), None)


def _describe(sdk, tool):
    params = tool.get("parameters") or {}
    required = set(params.get("required") or [])
    fields = [
        f"{name}{'*' if name in required else ''}"
        for name in (params.get("properties") or {})
    ]
    pairs = [("Args", ", ".join(fields) or "(none)")]
    pairs += [
        (setting["title"], sdk.text.value(setting.get("current")))
        for setting in tool.get("config_settings") or []
    ]
    card = sdk.md.card(tool["name"], pairs)
    desc = (tool.get("description") or "").strip()
    return f"{card}\n\n{sdk.md.quote(desc)}" if desc else card
