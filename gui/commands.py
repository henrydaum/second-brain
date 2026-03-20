"""
Command registry for the slash-command system.

Each command is a CommandEntry with a name, description, argument hint,
and handler callable. The registry provides autocomplete filtering and
dispatch.

``register_core_commands()`` registers the shared commands that are
identical between the GUI and REPL. Each UI then adds its own
UI-specific commands (or overrides) on top.
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
        hint = f" {cmd.arg_hint}" if cmd.arg_hint else ""
        lines.append(f"  /{cmd.name}{hint:<20}  {cmd.description}")
    return "\n".join(lines)


def register_core_commands(registry: CommandRegistry, ctrl, services, tool_registry, root_dir):
    """Register commands shared by both GUI and REPL.

    These are pure ctrl-wrapper commands with no UI-specific side effects.
    Each UI should call this first, then register/override its own commands.
    """
    from gui.formatters import (
        format_services, format_tasks,
        format_stats, format_tools,
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
        CommandEntry("tools",    "List registered tools",
                     handler=lambda _: format_tools(ctrl.list_tools())),
        CommandEntry("enable",   "Enable a tool for agent use", "<tool>",
                     handler=lambda a: ctrl.enable_tool(a) if a else "Usage: /enable <tool>",
                     arg_completions=_tool_names),
        CommandEntry("disable",  "Disable a tool",        "<tool>",
                     handler=lambda a: ctrl.disable_tool(a) if a else "Usage: /disable <tool>",
                     arg_completions=_tool_names),
        CommandEntry("reload",   "Hot-reload tasks and tools",
                     handler=lambda _: ctrl.reload_plugins(root_dir)),
        CommandEntry("stats",    "System overview",
                     handler=lambda _: format_stats(ctrl.stats())),
    ]:
        registry.register(entry)
