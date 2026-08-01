"""Slash command plugin for `/frontends`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


FIELDS = [
    "agent_profile",
    "whitelist_or_blacklist_commands",
    "commands_list",
]
FIELD_LABELS = ["Agent profile", "Command mode", "Command list"]
DEFAULT_PROFILE = {
    "agent_profile": "default",
    "whitelist_or_blacklist_commands": "blacklist",
    "commands_list": [],
}


class FrontendsCommand(BaseCommand):
    """Enable frontends and configure their access profiles."""

    name = "frontends"
    description = "Enable/disable a frontend or configure its access profile"
    category = "Capabilities"
    requests = [
        "plugin.list", "config.read", "config.write", "command.list",
    ]

    def form(self, sdk, args):
        """Build frontend, action, quicklink, profile-field, and value steps."""
        frontends = _frontends(sdk)
        enabled = set(sdk.config.read("enabled_frontends") or [])
        # The status table goes in the *prompt*, the way ``/services`` and
        # ``/schedule`` do it. ``_show`` renders exactly this and was reachable
        # only from the no-argument path — i.e. never from the menu, because
        # this step is required.
        steps = [FormStep(
            "frontend_name",
            _show(sdk, frontends) + "\n\nSelect a frontend.",
            True,
            enum=[frontend["name"] for frontend in frontends],
            columns=2,
        )]
        name = args.get("frontend_name")
        frontend = _find(frontends, name)
        if name:
            settings = (
                frontend.get("config_settings") if frontend else [])
            links, labels = sdk.forms.setting_actions(settings)
            actions, action_labels = _actions_for(enabled, name)
            steps.append(FormStep(
                "action",
                "What do you want to do with this frontend?\n\n"
                + _describe(sdk, name, frontend),
                True,
                enum=actions + links,
                enum_labels=action_labels + labels,
            ))

        setting = sdk.forms.setting_for_action(
            (frontend or {}).get("config_settings"),
            args.get("action"),
        )
        if setting:
            steps.append(sdk.forms.setting_value_step(setting))
        if args.get("action") == "configure":
            steps.append(FormStep(
                "field",
                "Choose which part of the frontend profile to edit.",
                True,
                enum=FIELDS,
                enum_labels=FIELD_LABELS,
            ))
            field = args.get("field")
            if field:
                steps.append(_value_step(sdk, field))
        return steps

    def run(self, sdk, args):
        """Execute the selected frontend action."""
        action = args.get("action")
        name = args.get("frontend_name")
        frontends = _frontends(sdk)
        if not name:
            return _show(sdk, frontends)

        frontend = _find(frontends, name)
        setting = sdk.forms.setting_for_action(
            (frontend or {}).get("config_settings"), action)
        if setting:
            old = setting.get("current")
            sdk.config.write(setting["key"], args.get("value"))
            changed = args.get("value") != old
            suffix = ". Restart required." if changed else ""
            return (
                f"Set {setting['key']} = "
                f"{sdk.text.value(args.get('value'))}{suffix}"
            )
        if action in ("enable", "disable"):
            return _toggle(sdk, name, action)
        if action == "configure":
            return _configure(
                sdk, name, args.get("field"), args.get("value"))
        return f"Unknown action: {action}"


def _frontends(sdk):
    return sdk.plugins.list(
        category="frontends", details=True)


def _actions_for(enabled, name):
    """Only the half of the toggle that does anything.

    Both were always offered, so half of every menu was a no-op the user had
    to know to avoid — and picking the wrong one looked like a broken command
    rather than a redundant option. ``/services`` has always got this right;
    the pattern is one stable action *value* with a state-dependent label,
    which keeps ``run`` branching on a single name.
    """
    if name in enabled:
        return ["configure", "disable"], ["Edit", "Disable it"]
    return ["configure", "enable"], ["Edit", "Enable it"]


def _find(frontends, name):
    return next(
        (frontend for frontend in frontends
         if frontend["name"] == name),
        None,
    )


def _toggle(sdk, name, action):
    names = set(sdk.config.read("enabled_frontends") or [])
    if action == "enable":
        names.add(name)
    else:
        if name in names and len(names) == 1:
            return "Cannot disable the last enabled frontend."
        names.discard(name)
    sdk.config.write("enabled_frontends", sorted(names))
    verb = "Enabled" if action == "enable" else "Disabled"
    return f"{verb} frontend: {name}. Restart required."


def _configure(sdk, name, field, value):
    if field not in FIELDS:
        return f"Unknown field: {field}"
    if (
        field == "whitelist_or_blacklist_commands"
        and value not in ("whitelist", "blacklist")
    ):
        return "Command mode must be 'whitelist' or 'blacklist'."

    profiles = sdk.config.read("frontend_profiles") or {}
    profile = dict(profiles.get(name) or DEFAULT_PROFILE)
    profile[field] = _coerce(field, value)
    sdk.config.write(
        "frontend_profiles", {name: profile}, merge=True)
    note = ""
    if (
        field == "whitelist_or_blacklist_commands"
        and value == "whitelist"
        and not profile.get("commands_list")
    ):
        note = (
            "\nNote: whitelist is empty — every command is now blocked "
            "on this frontend."
        )
    label = FIELD_LABELS[FIELDS.index(field)]
    return (
        f"Updated {name} profile: {label} → "
        f"{_render_value(field, profile[field])}{note}"
    )


def _coerce(field, value):
    if field == "commands_list":
        return value if isinstance(value, list) else []
    return "" if value is None else str(value)


def _value_step(sdk, field):
    if field == "agent_profile":
        names = sdk.config.read("agent_profiles", keys=True)
        profiles = ["default", *[name for name in names if name != "default"]]
        return FormStep(
            "value",
            "Choose the agent profile sessions on this frontend should use. "
            "'default' follows the global active profile.",
            True,
            enum=profiles,
            default="default",
        )
    if field == "whitelist_or_blacklist_commands":
        return FormStep(
            "value",
            "Blacklist blocks the listed commands; whitelist allows only "
            "the listed commands.",
            True,
            enum=["blacklist", "whitelist"],
            enum_labels=["Blacklist commands", "Whitelist commands"],
            default="blacklist",
        )
    commands = sdk.commands.list()
    return FormStep(
        "value",
        "Command names for the list. Available: "
        f"{', '.join(commands) or '(none)'}",
        False,
        "array",
        default=[],
        prompt_when_missing=True,
    )


def _show(sdk, frontends):
    enabled = set(sdk.config.read("enabled_frontends") or [])
    profiles = sdk.config.read("frontend_profiles") or {}
    rows = [
        (
            frontend["name"],
            _status(frontend, enabled),
            _profile_summary(profiles.get(frontend["name"])),
        )
        for frontend in frontends
    ]
    return "Frontends:\n\n" + sdk.md.table(
        ["Frontend", "Status", "Access"],
        rows,
        leading_blank=False,
    )


def _describe(sdk, name, frontend=None):
    enabled = set(sdk.config.read("enabled_frontends") or [])
    profiles = sdk.config.read("frontend_profiles") or {}
    pairs = [
        ("Status", _status(frontend or {"name": name}, enabled)),
        ("Profile", _profile_summary(profiles.get(name))),
    ]
    pairs += [
        (setting["title"], sdk.text.value(setting.get("current")))
        for setting in (frontend or {}).get("config_settings") or []
    ]
    card = sdk.md.card(name, pairs)
    description = ((frontend or {}).get("description") or "").strip()
    return f"{card}\n\n{sdk.md.quote(description)}" if description else card


def _status(frontend, enabled):
    """Enabled, disabled, or enabled-but-not-running.

    The payload has carried ``loaded`` all along and nothing read it, so a
    frontend that was switched on and then failed to start looked exactly like
    one that was serving traffic.
    """
    if frontend.get("name") not in enabled:
        return "Disabled"
    return "Enabled" if frontend.get("loaded", True) else "Enabled (not running)"


def _profile_summary(profile):
    if not profile:
        return "agent default, all commands"
    agent = profile.get("agent_profile") or "default"
    mode = profile.get(
        "whitelist_or_blacklist_commands", "blacklist")
    listed = profile.get("commands_list") or []
    if listed:
        commands = f"{mode} {', '.join(listed)}"
    elif mode == "whitelist":
        commands = "whitelist (none → all blocked)"
    else:
        commands = "all commands"
    return f"agent {agent}, {commands}"


def _render_value(field, value):
    if field == "commands_list":
        return ", ".join(value) or "(none)"
    return str(value)
