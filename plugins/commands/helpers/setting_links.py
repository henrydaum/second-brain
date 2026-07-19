"""Quicklinks from plugin commands to /config setting edits.

Any plugin (tool, task, service, frontend) that declares ``config_settings``
gets "Edit <Setting Title>" entries appended to its action step in the
/tools, /tasks, /services, and /frontends commands. Choosing one runs the
same edit flow as ``/config <key> edit`` — the value step and the write
logic are shared with command_config, so scope rules (user vs global),
plugin-config persistence, and watcher rescans behave identically.

Action values are encoded as ``edit_setting:<key>`` so they coexist with
each command's own actions in one enum.
"""

_PREFIX = "edit_setting:"


def _entries(plugin) -> list[tuple]:
    settings = getattr(plugin, "config_settings", None) or []
    out = []
    for entry in settings:
        if not isinstance(entry, (list, tuple)) or len(entry) != 5:
            continue
        info = entry[4] if isinstance(entry[4], dict) else {}
        if info.get("hidden") is True:
            continue
        out.append(tuple(entry))
    return out


def quicklinks(plugin) -> tuple[list[str], list[str]]:
    """(enum values, labels) linking each of *plugin*'s settings to an edit."""
    entries = _entries(plugin)
    return ([f"{_PREFIX}{e[1]}" for e in entries],
            [f"Edit {e[0]}" for e in entries])


def setting_rows(plugin, context) -> list[tuple]:
    """(Setting title, current-value string) rows for *plugin*'s settings, for
    describe cards — so the user sees what's currently set before choosing a new
    value. Values come through the same scope-aware reader /config uses."""
    from plugins.commands.command_config import _current_value, _format_value
    return [(e[0], _format_value(_current_value(context, e[1]))) for e in _entries(plugin)]


def quicklink_value_steps(action, context) -> list:
    """The /config value step when *action* is a quicklink, else []."""
    if not isinstance(action, str) or not action.startswith(_PREFIX):
        return []
    from plugins.commands.command_config import value_step_for
    return [value_step_for(action[len(_PREFIX):], context)]


def quicklink_run(action, args, context):
    """Apply a quicklink edit; None when *action* isn't a quicklink."""
    if not isinstance(action, str) or not action.startswith(_PREFIX):
        return None
    from plugins.commands.command_config import apply_edit
    return apply_edit(context, action[len(_PREFIX):], args.get("value"))
