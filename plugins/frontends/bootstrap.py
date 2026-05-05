from __future__ import annotations

import logging
import threading

from agent.system_prompt import build_system_prompt
from events.event_bus import bus
from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.command_registry import CommandRegistry
from plugins.plugin_discovery import discover_commands
from plugins.frontends.repl_frontend import ReplFrontend
from plugins.frontends.telegram_frontend import TelegramFrontend
from runtime.context import build_context
from runtime.agent_scope import load_scope, scoped_registry
from state_machine.runtime import ConversationRuntime

logger = logging.getLogger("Frontends")


class _HostCommand(BaseCommand):
    category = "Conversation"
    require_approval = True
    approval_actor_id = "user"

    def __init__(self, name: str, description: str, callback):
        self.name = name
        self.description = description
        self.callback = callback

    def run(self, _args, _context):
        return self.callback() or None


def _restart(scaffold):
    fn = getattr(scaffold, "restart", None)
    if fn is None:
        return "Restart is not supported in this frontend."
    threading.Timer(0.75, fn).start()
    return "Restarting - Second Brain will be back in a few seconds."


def _quit(shutdown_fn):
    threading.Timer(0.75, shutdown_fn).start()
    return "Shutting down."


def start_frontends(frontends: set[str], scaffold, shutdown_fn, shutdown_event,
                    tool_registry, services, config, root_dir):
    runtime = _conversation_runtime(scaffold, shutdown_fn, tool_registry, services, config, root_dir) if frontends & {"repl", "telegram"} else None
    adapters, threads = {}, []
    if runtime and "repl" in frontends:
        repl = ReplFrontend(shutdown_fn, shutdown_event)
        repl.bind(runtime, runtime.command_registry, config)
        _start("repl", repl.start, adapters, threads, repl)
    if runtime and "telegram" in frontends:
        telegram = TelegramFrontend(shutdown_event, services)
        telegram.bind(runtime, runtime.command_registry, config)
        _start("telegram", telegram.start, adapters, threads, telegram)
    for name in sorted(frontends - {"repl", "telegram"}):
        logger.warning(f"Unknown frontend '{name}' - skipping.")
    return runtime, adapters, threads


def _conversation_runtime(scaffold, shutdown_fn, tool_registry, services, config, root_dir):
    ref = {}
    registry = CommandRegistry(
        lambda session_key=None: build_context(
            scaffold.db, config, services, tool_registry=tool_registry,
            orchestrator=scaffold.orchestrator, runtime=ref.get("runtime"),
            root_dir=root_dir, session_key=session_key,
        )
    )
    discover_commands(root_dir, registry, config)
    registry.register(_HostCommand("quit", "Shutdown", lambda: _quit(shutdown_fn)))
    registry.register(_HostCommand("restart", "Restart the app", lambda: _restart(scaffold)))

    def prompt():
        profile = config.get("active_agent_profile") or "default"
        scope = _scope(profile, config)
        registry_for_prompt = scoped_registry(tool_registry, scope, db=scaffold.db) if scope else tool_registry
        return build_system_prompt(scaffold.db, scaffold.orchestrator, registry_for_prompt, services, scope=scope, profile_name=profile)

    runtime = ConversationRuntime(
        db=scaffold.db,
        services=services,
        config=config,
        tool_registry=tool_registry,
        system_prompt=prompt,
        commands=registry.to_callable_specs(),
        emit_event=lambda channel, payload: bus.emit(channel, payload),
    )
    runtime.command_registry = registry
    runtime._orchestrator_ref = scaffold.orchestrator
    ref["runtime"] = runtime
    # Tasks running through the orchestrator (scheduled subagents in
    # particular) reach the runtime via context.runtime.
    if scaffold.orchestrator is not None:
        scaffold.orchestrator.runtime = runtime
    if tool_registry is not None:
        tool_registry.runtime = runtime
        tool_registry.command_registry = registry
    return runtime


def _scope(profile, config):
    try:
        scope = load_scope(profile, config)
    except ValueError as e:
        logger.warning(f"Invalid scope for profile '{profile}': {e}")
        return None
    return scope if scope.has_tool_filter or scope.prompt_suffix else None


def _start(name, target, adapters, threads, adapter):
    thread = threading.Thread(target=target, daemon=True, name=f"{name}-frontend")
    thread.start()
    adapters[name] = adapter
    threads.append(thread)
