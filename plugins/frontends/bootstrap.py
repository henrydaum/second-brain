from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class _HostCommand(BaseCommand):
    name: str
    description: str
    callback: object
    category: str = "Conversation"

    def run(self, _args, _context):
        return self.callback() or None


def start_frontends(frontends: set[str], ctrl, shutdown_fn, shutdown_event,
                    tool_registry, services, config, root_dir):
    runtime = _conversation_runtime(ctrl, shutdown_fn, tool_registry, services, config, root_dir) if frontends & {"repl", "telegram"} else None
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


def _conversation_runtime(ctrl, shutdown_fn, tool_registry, services, config, root_dir):
    ref = {}
    registry = CommandRegistry(
        lambda session_key=None: build_context(
            ctrl.db, config, services, tool_registry=tool_registry,
            orchestrator=ctrl.orchestrator, runtime=ref.get("runtime"),
            controller=ctrl, root_dir=root_dir,
        )
    )
    discover_commands(root_dir, registry, config)
    registry.register(_HostCommand("quit", "Shutdown", shutdown_fn))
    registry.register(_HostCommand("exit", "Shutdown", shutdown_fn))

    def prompt():
        profile = config.get("active_agent_profile") or "default"
        scope = _scope(profile, config)
        registry_for_prompt = scoped_registry(tool_registry, scope, db=ctrl.db) if scope else tool_registry
        return build_system_prompt(ctrl.db, ctrl.orchestrator, registry_for_prompt, services, scope=scope, profile_name=profile)

    runtime = ConversationRuntime(
        db=ctrl.db,
        services=services,
        config=config,
        tool_registry=tool_registry,
        system_prompt=prompt,
        commands=registry.to_callable_specs(),
        emit_event=lambda channel, payload: bus.emit(channel, payload),
        title_callback=ctrl.maybe_generate_conversation_title_async,
    )
    runtime.command_registry = registry
    runtime._orchestrator_ref = getattr(ctrl, "orchestrator", None)
    ref["runtime"] = runtime
    # Tasks running through the orchestrator (scheduled subagents in
    # particular) reach the runtime via context.runtime.
    if getattr(ctrl, "orchestrator", None) is not None:
        ctrl.orchestrator.runtime = runtime
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
