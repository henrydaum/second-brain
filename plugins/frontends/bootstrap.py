from __future__ import annotations

import logging
import threading

from agent.system_prompt import build_system_prompt
from events.event_bus import bus
from plugins.frontends.helpers.command_registry import CommandEntry, CommandRegistry, register_core_commands
from plugins.frontends.repl_frontend import ReplFrontend
from plugins.frontends.telegram_frontend import TelegramFrontend
from runtime.agent_scope import load_scope, scoped_registry
from state_machine.runtime import ConversationRuntime

logger = logging.getLogger("Frontends")


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
    registry = CommandRegistry()
    ref = {}
    register_core_commands(
        registry, ctrl, services, tool_registry, root_dir,
        rescope_agents=lambda: ref["runtime"].refresh_session_specs(),
    )
    registry.register(CommandEntry("quit", "Shutdown", handler=lambda _arg: shutdown_fn() or None, category="Conversation"))
    registry.register(CommandEntry("exit", "Shutdown", handler=lambda _arg: shutdown_fn() or None, category="Conversation"))

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
    ref["runtime"] = runtime
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
