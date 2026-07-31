"""Slash command plugin for `/config`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


ACTIONS = ["edit"]
CATEGORIES = ["kernel", "plugin", "user", "all"]
CATEGORY_LABELS = {
    "kernel": "Kernel Settings",
    "plugin": "Plugin Settings",
    "user": "User Settings",
    "all": "All Settings",
}


class ConfigCommand(BaseCommand):
    """Inspect and edit kernel, plugin, and user settings."""

    name = "config"
    description = "Select a config setting, then edit it"
    category = "System"
    requests = ["config.read", "config.write"]

    def form(self, sdk, args):
        """Build the category, plugin, setting, action, and value steps."""
        settings = sdk.config.read(details=True)
        category = args.get("category")
        steps = []
        if not args.get("setting_name"):
            steps.append(FormStep(
                "category",
                "Which settings do you want to browse?",
                True,
                enum=CATEGORIES,
                enum_labels=[CATEGORY_LABELS[item] for item in CATEGORIES],
            ))
            if category == "plugin":
                groups = _plugin_groups(settings)
                names = sorted(groups)
                steps.append(FormStep(
                    "plugin_name",
                    "Which plugin's settings do you want to browse?",
                    False,
                    enum=names,
                    enum_labels=names,
                    columns=2,
                    prompt_when_missing=True,
                ))
        available = _filter(
            settings, category, args.get("plugin_name"))
        steps.append(FormStep(
            "setting_name",
            "Select a setting to inspect or edit.",
            True,
            enum=sorted(item["key"] for item in available),
            columns=2,
        ))
        setting = _find(settings, args.get("setting_name"))
        if setting:
            steps.append(FormStep(
                "action",
                "What do you want to do with this setting?\n\n"
                + _describe(sdk, setting),
                True,
                enum=ACTIONS,
                enum_labels=["Edit setting"],
            ))
        if setting and args.get("action") == "edit":
            steps.append(sdk.forms.setting_value_step(setting))
        return steps

    def run(self, sdk, args):
        """Execute the selected config action."""
        settings = sdk.config.read(details=True)
        key = args.get("setting_name")
        if not key:
            return _list(
                sdk, settings, args.get("category"),
                args.get("plugin_name"))
        setting = _find(settings, key)
        if not setting:
            return f"Unknown setting: {key}"
        if args.get("action") != "edit":
            return _describe(sdk, setting)
        value = args.get("value")
        old = setting.get("current")
        sdk.config.write(key, value)
        suffix = (
            ". Restart required."
            if setting.get("restart_required") and value != old
            else ""
        )
        return f"Set {key} = {sdk.text.value(value)}{suffix}"


def _find(settings, key):
    return next(
        (setting for setting in settings if setting["key"] == key), None)


def _filter(settings, category=None, plugin_name=None):
    if category not in {"kernel", "plugin", "user"}:
        return settings
    selected = [
        setting for setting in settings
        if setting["category"] == category
    ]
    if category == "plugin" and plugin_name:
        selected = [
            setting for setting in selected
            if plugin_name in setting.get("owners", [])
        ]
    return selected


def _plugin_groups(settings):
    groups = {}
    for setting in _filter(settings, "plugin"):
        for owner in setting.get("owners") or ["(unknown)"]:
            groups.setdefault(owner, []).append(setting)
    return groups


def _describe(sdk, setting):
    owners = setting.get("owners") or []
    title = setting["title"]
    if setting.get("scope") == "user":
        title += " (per-user)"
    card = sdk.md.card(title, [
        (setting["key"], sdk.text.value(setting.get("current"))),
        ("Used by", ", ".join(owners) if owners else "kernel"),
    ])
    description = (setting.get("description") or "").strip()
    output = card + (
        f"\n\n{sdk.md.quote(description)}" if description else "")
    if len(owners) > 1:
        output += (
            "\n\n⚠ Shared setting — changing this also affects: "
            + ", ".join(owners) + "."
        )
    return output


def _table(sdk, settings):
    return sdk.md.table(
        ["Setting", "Value"],
        [
            (item["key"], sdk.text.value(item.get("current")))
            for item in settings
        ],
        leading_blank=False,
    )


def _list(sdk, settings, category=None, plugin_name=None):
    if category == "plugin" and not plugin_name:
        groups = _plugin_groups(settings)
        if not groups:
            return "No plugin settings found."
        return "\n\n".join(
            f"{name}:\n\n{_table(sdk, sorted(
                groups[name], key=lambda item: item['key']))}"
            for name in sorted(groups)
        )
    if category == "plugin" and plugin_name:
        selected = sorted(
            _filter(settings, "plugin", plugin_name),
            key=lambda item: item["key"],
        )
        if not selected:
            return f"No settings for plugin: {plugin_name}"
        return f"{plugin_name}:\n\n{_table(sdk, selected)}"
    categories = (
        [category]
        if category in {"kernel", "plugin", "user"}
        else ["kernel", "plugin", "user"]
    )
    sections = []
    for item in categories:
        selected = sorted(
            _filter(settings, item), key=lambda setting: setting["key"])
        if selected:
            sections.append(
                f"{CATEGORY_LABELS[item]} ({selected[0]['storage']}):"
                f"\n\n{_table(sdk, selected)}"
            )
    return "\n\n".join(sections) or "No settings found."
