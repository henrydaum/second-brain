"""Frontend plugin for bootstrap."""

from __future__ import annotations

import logging
import threading

from agent.system_prompt import build_prompt_sections
from config import config_manager
from events.event_bus import bus
from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.command_registry import CommandRegistry
from plugins.plugin_discovery import discover_commands, discover_frontends, get_plugin_settings
from runtime.context import build_context
from runtime.agent_scope import load_scope, scoped_registry
from runtime.conversation_runtime import ConversationRuntime

logger = logging.getLogger("Frontends")


class _HostCommand(BaseCommand):
    """Host command."""
    category = "Conversation"
    require_approval = True
    approval_actor_id = "user"

    def __init__(self, name: str, description: str, callback):
        """Initialize the host command."""
        self.name = name
        self.description = description
        self.callback = callback

    def run(self, _args, _context):
        """Execute `/bootstrap` for the active session."""
        return self.callback() or None


def _restart(scaffold):
    """Internal helper to handle restart."""
    fn = getattr(scaffold, "restart", None)
    if fn is None:
        return "Restart is not supported in this frontend."
    threading.Timer(0.75, fn).start()
    return "Restarting - Second Brain will be back in a few seconds."


def _quit(shutdown_fn):
    """Internal helper to handle quit."""
    threading.Timer(0.75, shutdown_fn).start()
    return "Shutting down."


class FrontendManager:
    """Holds running frontend instances. Supports register/unregister at runtime.

    Each frontend's transport-specific constructor args come from a factory
    registered via ``set_factory(name, factory)``. When ``register(cls)`` is
    called for a known frontend name, the factory builds the instance, the
    base class binds it to the runtime + command registry, and it's started
    on a daemon thread.
    """

    def __init__(self, runtime, command_registry, config: dict):
        """Initialize the frontend manager."""
        self.runtime = runtime
        self.command_registry = command_registry
        self.config = config
        self._adapters: dict[str, object] = {}
        self._threads: list[threading.Thread] = []
        self._factories: dict[str, callable] = {}

    def set_factory(self, name: str, factory) -> None:
        """Set factory."""
        self._factories[name] = factory

    @property
    def adapters(self) -> dict:
        """Handle adapters."""
        return self._adapters

    @property
    def threads(self) -> list:
        """Handle threads."""
        return self._threads

    def register(self, cls) -> str | None:
        """Register frontend manager."""
        name = getattr(cls, "name", "")
        if not name:
            return "Frontend class has no name"
        if name in self._adapters:
            return f"Frontend '{name}' already running"
        factory = self._factories.get(name)
        try:
            adapter = factory(cls) if factory else cls()
        except Exception as e:
            logger.exception(f"Frontend '{name}' instantiation failed")
            return f"Frontend '{name}' instantiation failed: {e}"
        try:
            adapter.bind(self.runtime, self.command_registry, self.config)
        except Exception as e:
            logger.exception(f"Frontend '{name}' bind failed")
            return f"Frontend '{name}' bind failed: {e}"
        thread = threading.Thread(target=adapter.start, daemon=True, name=f"{name}-frontend")
        thread.start()
        self._adapters[name] = adapter
        self._threads.append(thread)
        return None

    def unregister(self, name: str) -> str | None:
        """Unregister frontend manager."""
        adapter = self._adapters.pop(name, None)
        if adapter is None:
            return f"Frontend '{name}' is not running"
        try:
            if hasattr(adapter, "unbind"):
                adapter.unbind()
            if hasattr(adapter, "stop"):
                adapter.stop()
        except Exception:
            logger.exception(f"Frontend '{name}' stop failed")
        return None


def start_frontends(frontends: set[str], scaffold, shutdown_fn, shutdown_event,
                    tool_registry, services, config, root_dir):
    """Start frontends."""
    if not frontends:
        return None, {}, []

    runtime = _conversation_runtime(scaffold, shutdown_fn, tool_registry, services, config, root_dir)
    classes = discover_frontends(root_dir, config)
    config_manager.reconcile_plugin_config(config, get_plugin_settings())
    manager = FrontendManager(runtime, runtime.command_registry, config)

    # Transport-specific constructor args: discovery returns the class, the
    # bootstrap supplies what each frontend needs to talk to the host.
    manager.set_factory("repl", lambda cls: cls(shutdown_fn, shutdown_event))
    for name in sorted(frontends):
        cls = classes.get(name)
        if cls is None:
            logger.warning(f"Unknown frontend '{name}' - skipping.")
            continue
        err = manager.register(cls)
        if err:
            logger.warning(err)

    runtime.frontend_manager = manager
    return runtime, manager.adapters, manager.threads


def _conversation_runtime(scaffold, shutdown_fn, tool_registry, services, config, root_dir):
    """Internal helper to handle conversation runtime."""
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
        """Handle prompt."""
        profile = config.get("active_agent_profile") or "default"
        scope = _scope(profile, config)
        registry_for_prompt = scoped_registry(tool_registry, scope, db=scaffold.db) if scope else tool_registry
        return build_prompt_sections(scaffold.db, scaffold.orchestrator, registry_for_prompt, services, scope=scope, profile_name=profile, commands=registry, config=config)

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
    # Tasks running through the orchestrator reach the runtime via
    # context.runtime.
    if scaffold.orchestrator is not None:
        scaffold.orchestrator.runtime = runtime
    if tool_registry is not None:
        tool_registry.runtime = runtime
        tool_registry.command_registry = registry
    return runtime


def _scope(profile, config):
    """Internal helper to handle scope."""
    try:
        scope = load_scope(profile, config)
    except ValueError as e:
        logger.warning(f"Invalid scope for profile '{profile}': {e}")
        return None
    return scope if scope.has_tool_filter or scope.prompt_suffix else None
