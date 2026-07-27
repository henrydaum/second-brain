"""Slash command plugin for `/services`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


_EDIT_PREFIX = "edit_setting:"


class ServicesCommand(BaseCommand):
    """Inspect services and manage their kernel-owned lifecycle."""

    name = "services"
    description = "Inspect services and load or unload managed ones"
    category = "System"
    requests = [
        "service.list", "service.load", "service.unload",
        "config.read", "config.write",
    ]

    def form(self, sdk, args):
        """Build the dependent service, action, and setting-value steps."""
        services = sdk.services.list(details=True)
        steps = [FormStep(
            "service_name", "Select a service.", True,
            enum=[service["name"] for service in services], columns=2)]
        service = _find(services, args.get("service_name"))
        if service is None:
            return steps

        actions, labels = _actions_for(sdk, service)
        if actions:
            steps.append(FormStep(
                "action",
                "What do you want to do with this service?\n\n"
                + _describe(service),
                True,
                enum=actions,
                enum_labels=labels,
            ))
        setting = _setting_for_action(service, args.get("action"))
        if setting is not None:
            steps.append(FormStep(
                "value",
                _value_prompt(setting),
                True,
                _value_type(setting),
            ))
        return steps

    def run(self, sdk, args):
        """Execute `/services` for the active session."""
        services = sdk.services.list(details=True)
        action = args.get("action")
        name = args.get("service_name")
        if not name:
            return _show(services)
        service = _find(services, name)
        if service is None:
            return "Unknown service."
        if not action:
            return _describe(service)

        setting = _setting_for_action(service, action)
        if setting is not None:
            sdk.config.write(setting["key"], args.get("value"))
            return f"Set {setting['key']} = {_format_value(args.get('value'))}"

        if service["lifecycle"] != "managed":
            return (
                f"{name} is an installed extension and is loaded "
                "automatically."
            )
        if action in {"toggle_loaded", "load", "unload"}:
            load = (
                not service["loaded"]
                if action == "toggle_loaded"
                else action == "load"
            )
            if load:
                if sdk.services.load(name) is False:
                    return f"Failed to load service: {name}"
                return f"Loaded service: {name}"
            sdk.services.unload(name)
            return f"Unloaded service: {name}"
        if action == "toggle_autoload":
            return _toggle_autoload(sdk, name)
        return f"Unknown action: {action}"


def _find(services, name):
    return next(
        (service for service in services if service["name"] == name), None)


def _actions_for(sdk, service):
    actions = []
    labels = []
    if service["lifecycle"] == "managed":
        config = sdk.config.read() or {}
        autoloaded = service["name"] in (
            config.get("autoload_services") or [])
        actions += ["toggle_loaded", "toggle_autoload"]
        labels += [
            "Unload it" if service["loaded"] else "Load it",
            (
                "Don't autoload on startup"
                if autoloaded
                else "Autoload on startup"
            ),
        ]
    for setting in service["config_settings"]:
        actions.append(_EDIT_PREFIX + setting["key"])
        labels.append("Edit " + setting["title"])
    return actions, labels


def _setting_for_action(service, action):
    if not isinstance(action, str) or not action.startswith(_EDIT_PREFIX):
        return None
    key = action[len(_EDIT_PREFIX):]
    return next(
        (
            setting for setting in service["config_settings"]
            if setting["key"] == key
        ),
        None,
    )


def _toggle_autoload(sdk, name):
    config = sdk.config.read() or {}
    names = [str(item) for item in (config.get("autoload_services") or [])]
    enabled = name not in names
    names = (
        sorted(set(names) | {name})
        if enabled
        else [item for item in names if item != name]
    )
    sdk.config.write("autoload_services", names)
    return (
        f"{name} will {'now' if enabled else 'no longer'} "
        "load automatically on startup."
    )


def _show(services):
    if not services:
        return "No services registered."
    rows = [
        (
            service["name"],
            (
                "Extension"
                if service["lifecycle"] == "extension"
                else "Loaded" if service["loaded"] else "Unloaded"
            ),
            service["model_name"],
        )
        for service in services
    ]
    return "Services:\n\n" + _md_table(
        ["Service", "Status", "Model"], rows)


def _describe(service):
    status = (
        "Extension"
        if service["lifecycle"] == "extension"
        else "Loaded" if service["loaded"] else "Unloaded"
    )
    pairs = [
        ("Status", status),
        ("Model", service["model_name"] or "-"),
    ]
    pairs += [
        (setting["title"], _format_value(setting["current"]))
        for setting in service["config_settings"]
    ]
    return _md_table([service["name"], ""], pairs)


def _value_type(setting):
    default = setting["default"]
    info = setting["info"]
    type_name = info.get("type")
    if type_name in {"path", "path_list"}:
        return type_name
    if type_name == "json_list":
        return "array"
    if type_name == "json_dict":
        return "object"
    if type_name in {"bool", "boolean"}:
        return "boolean"
    if type_name == "slider":
        return "number" if info.get("is_float") else "integer"
    if isinstance(default, list):
        return "array"
    if isinstance(default, dict):
        return "object"
    return "string"


def _value_prompt(setting):
    value_type = _value_type(setting)
    if value_type == "path_list":
        return (
            "Enter one folder path per line. / and \\ are both accepted; "
            "each folder must already exist. Example:\n\n"
            "C:\\Users\\you\\Notes\nD:\\Archive"
        )
    if value_type == "path":
        return (
            "Enter a path. / and \\ are both accepted; the parent folder "
            "must exist."
        )
    if value_type == "array":
        return (
            "Enter a list of items, one on each line, like so:\n\n"
            "item 1\nitem 2"
        )
    return "Enter the new value."


def _format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "(none)" if not value else ", ".join(map(str, value))
    return str(value)


def _md_table(headers, rows):
    def cell(value):
        return str("" if value is None else value).replace(
            "\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "|" + "|".join(" --- " for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)
