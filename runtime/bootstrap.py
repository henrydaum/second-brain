"""Application composition root.

Builds the ``ConversationRuntime`` (commands, context, system prompt, agent
scope) and the ``FrontendManager`` that starts/stops transport plugins, then
wires them together. This is the top of the dependency graph — ``main.pyw``
calls ``start_frontends`` here after services and the pipeline are up.
"""

from __future__ import annotations

import logging
import threading

from agent.system_prompt import build_prompt_sections
from config import config_manager
from events.event_bus import bus
from events.event_channels import CONFIG_CHANGED
from plugins.frontends.helpers.command_registry import CommandRegistry
from plugins.plugin_discovery import discover_commands, discover_frontends, get_plugin_settings
from runtime.context import build_context, set_kernel_parts
from runtime.agent_scope import load_scope, scoped_registry
from runtime.conversation_runtime import ConversationRuntime
from runtime.notifications import announce_config_change

logger = logging.getLogger("Bootstrap")


class _AppControl:
    """Stopping and restarting the process, for the ``app.stop`` Request.

    The two callables exist only here, at the top of the dependency graph, so
    this is the object the handler is answered from — reached through
    ``context.app_control`` like every other host resource. ``/quit`` and
    ``/restart`` are ordinary sandboxed commands on the other side of it.

    Both defer briefly so the answer reaches the frontend before the process
    goes away; otherwise the user is told nothing about why it ended.
    """

    def __init__(self, shutdown_fn, scaffold):
        """Initialize the app control."""
        self._shutdown_fn = shutdown_fn
        self._scaffold = scaffold

    def stop(self) -> str:
        """Shut down."""
        threading.Timer(0.75, self._shutdown_fn).start()
        return "Shutting down."

    def restart(self):
        """Restart, or None when this frontend cannot.

        None rather than a message: the handler turns a missing capability into
        a failure, and inventing a successful-looking string here would make a
        restart that never happened read as one that did.
        """
        fn = getattr(self._scaffold, "restart", None)
        if fn is None:
            return None
        threading.Timer(0.75, fn).start()
        return "Restarting - Second Brain will be back in a few seconds."


class FrontendManager:
    """Holds running frontend instances. Supports register/unregister at runtime.

    Construction is plugin-agnostic: a frontend asks for host resources by
    naming them as constructor parameters (``services``, ``shutdown_fn``,
    ``shutdown_event``, ...), and ``register(cls)`` supplies whatever the
    signature requests from ``host_kwargs`` — the kernel never needs to know
    a specific frontend's name. ``set_factory(name, factory)`` remains as an
    explicit override for kernel-owned frontends with bespoke wiring. After
    construction the base class binds the instance to the runtime + command
    registry and it's started on a daemon thread.
    """

    def __init__(self, runtime, command_registry, config: dict):
        """Initialize the frontend manager."""
        self.runtime = runtime
        self.command_registry = command_registry
        self.config = config
        self._adapters: dict[str, object] = {}
        self._threads: list[threading.Thread] = []
        self._factories: dict[str, callable] = {}
        # name -> zero-arg callable producing the value; callables so
        # per-instance resources (e.g. a fresh shutdown Event) aren't shared.
        self.host_kwargs: dict[str, callable] = {}
        self.available_frontends: set[str] = set()

    def set_factory(self, name: str, factory) -> None:
        """Set factory."""
        self._factories[name] = factory

    def _construct(self, cls):
        """Build a frontend by matching its constructor params to host_kwargs."""
        import inspect
        try:
            params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            return cls()
        kwargs = {}
        for name, provide in self.host_kwargs.items():
            param = params.get(name)
            if param is not None and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                kwargs[name] = provide()
        return cls(**kwargs)

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
        self.available_frontends.add(name)
        if name in self._adapters:
            return f"Frontend '{name}' already running"
        factory = self._factories.get(name)
        try:
            adapter = factory(cls) if factory else self._construct(cls)
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
    manager.available_frontends.update(classes)

    # Host resources any frontend (kernel, sandbox, or installed) can request
    # by naming them as constructor parameters. shutdown_event is a fresh
    # per-instance Event so one frontend's stop() never signals another's.
    manager.host_kwargs = {
        "shutdown_fn": lambda: shutdown_fn,
        "shutdown_event": lambda: threading.Event(),
        "services": lambda: services,
    }
    # The REPL is kernel-owned and observes the app-wide shutdown event (it
    # owns the terminal and must exit with the app), so it keeps an explicit
    # factory. It never sets that event itself: stop() uses a private signal
    # so a hot-reload deregistration can't take the whole app down.
    manager.set_factory(
        "repl", lambda cls: cls(shutdown_event=shutdown_event)
    )

    # A name discovery cannot resolve is a store frontend that is no longer
    # installed — Telegram, most often. Config normalization deliberately keeps
    # unknown names (that is what lets an installed store frontend survive load
    # order), so the stale entry is dropped here instead, where discovery has
    # already run and the answer is known. Otherwise it warns on every boot
    # forever about a package the user removed.
    stale = []
    for name in sorted(frontends):
        cls = classes.get(name)
        if cls is None:
            stale.append(name)
            continue
        err = manager.register(cls)
        if err:
            logger.warning(err)
    if stale:
        _forget_frontends(config, stale)

    runtime.frontend_manager = manager
    return runtime, manager.adapters, manager.threads


def _forget_frontends(config: dict, names: list) -> None:
    """Drop frontends discovery could not resolve from the saved config.

    Best-effort: failing to tidy the config is not a reason to fail a boot.
    """
    enabled = [n for n in (config.get("enabled_frontends") or [])
               if n not in set(names)]
    config["enabled_frontends"] = enabled
    try:
        config_manager.save(config)
    except Exception:
        logger.exception("could not persist enabled_frontends")
        return
    logger.info(
        f"Removed {', '.join(sorted(names))} from enabled_frontends - "
        f"not installed."
    )


def _conversation_runtime(scaffold, shutdown_fn, tool_registry, services, config, root_dir):
    """Internal helper to handle conversation runtime."""
    ref = {}
    app_control = _AppControl(shutdown_fn, scaffold)
    # ``call_tool`` is what /tools' "Call tool" action needs: without it
    # ``sdk.tools.call`` is refused for every command, because the handler only
    # checks whether the context carries the callable.
    registry = CommandRegistry(
        lambda session_key=None: build_context(
            scaffold.db, config, services, call_tool=tool_registry.call,
            tool_registry=tool_registry,
            orchestrator=scaffold.orchestrator, runtime=ref.get("runtime"),
            app_control=app_control,
            root_dir=root_dir, session_key=session_key,
        )
    )
    discover_commands(root_dir, registry, config)

    def prompt():
        """Handle prompt."""
        profile = config.get("active_agent_profile") or "default"
        scope = _scope(profile, config)
        registry_for_prompt = scoped_registry(tool_registry, scope, db=scaffold.db) if scope else tool_registry
        from runtime.agent_scope import resolve_agent_llm
        return build_prompt_sections(scaffold.db, scaffold.orchestrator, registry_for_prompt, services, scope=scope, profile_name=profile, commands=registry, config=config, active_llm=resolve_agent_llm(profile, config, services))

    # Action-ledger wiring: config saves get audit rows, and the single
    # data-retention knob is applied once at startup (then opportunistically
    # on ledger writes).
    if scaffold.db is not None:
        config_manager.set_ledger_db(scaffold.db)
        scaffold.db.retention_days = int(config.get("data_retention_days") or 0)
        scaffold.db.prune_expired(scaffold.db.retention_days)

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
    # Begin serving scheduled spawns. Immediate ones go straight to the
    # registry through the agent.spawn Request; this is only the bus half.
    runtime.subagents.start()
    # Give sandboxed plugins somebody to ask. Until this runs, the sandbox has
    # no approver and refuses every unsafe Request outright — plugins are
    # discovered and loaded long before a runtime exists, so the wiring cannot
    # happen at construction. Approval then flows the kernel's usual way:
    # vet_permission hooks, the user's trusted list, then a dialog.
    try:
        from sandbox.bridge import get_sandbox
        get_sandbox().bind_runtime(runtime)
    except Exception:
        logger.exception("could not wire sandbox approval to the runtime")
    # The last two pieces of the kernel context. Everything a resident service
    # reaches through sdk.session / sdk.conv / sdk.commands hangs off these,
    # and neither exists until here.
    set_kernel_parts(runtime=runtime, command_registry=registry,
                     app_control=app_control)
    # Tasks running through the orchestrator reach the runtime via
    # context.runtime.
    if scaffold.orchestrator is not None:
        scaffold.orchestrator.runtime = runtime
    if tool_registry is not None:
        tool_registry.runtime = runtime
        tool_registry.command_registry = registry
    # Settings changes are announced in chat whether or not they needed
    # approval — a command the user typed writes config without a dialog, so
    # this is what keeps it visible. Subscribed here rather than in
    # config_manager: the kernel's foundational config module should not know
    # about sessions, and a notice must never be able to fail a write.
    bus.subscribe(
        CONFIG_CHANGED,
        lambda payload: announce_config_change(
            payload, session_key=runtime.active_session_key),
    )
    return runtime


def _scope(profile, config):
    """Internal helper to handle scope."""
    try:
        scope = load_scope(profile, config)
    except ValueError as e:
        logger.warning(f"Invalid scope for profile '{profile}': {e}")
        return None
    return scope if scope.has_tool_filter or scope.prompt_suffix else None
