"""Resident boxes wearing a native face — services and frontends.

The other half of :mod:`sandbox.bridge`, split off because a residency is not
a call. A tool, task or command is adapted into a function that opens nothing
and returns; a service or a frontend is adapted into an object that *holds a
process* for its whole life, and everything that follows from that lives
here: the refcounted box two services can share, the hooks a service stands
at, the bus channels it listens to, the poll loop the kernel drives it with,
and the inverted start/poll/render loop a frontend runs inside.

Both halves build the same kind of thing — a native-looking subclass whose
methods forward into a box — so the construction machinery stays in
``bridge`` and is imported from here. The one call in the other direction is
``bridge.adapt`` reaching for the two entry points below, which it does with
a function-local import; a residency is reached *through* the bridge and
never instead of it.

Two shapes are worth knowing before reading:

- **A service box closes on a refcount, not on the first unload.** Services
  are the one family that may share a file, and when they do they share a
  box — ``service_embed.py`` holds a text and an image embedder so one torch
  import serves both. The kernel loads and unloads each by name with no idea
  they are neighbours, so the naive mapping kills a live model with no
  symptom beyond the survivor's calls failing.
- **A frontend inverts its loop.** A native frontend blocks in ``start()``
  forever, and a box takes one call at a time, so a guest that never returned
  would hold its box and no render could get in. The guest's ``start`` sets up
  and returns; the kernel calls ``poll`` repeatedly on the daemon thread
  ``FrontendManager`` already provides, and a render lands between polls.
"""

from __future__ import annotations

import logging
import threading
import types
from pathlib import Path

from .bridge import (NOT_CARRIED, _box_prompt, _build, _cached_prompt,
                     _defines, _make_response, _prompt_method, get_sandbox)
from .policy import Chain

logger = logging.getLogger("Sandbox")


def _sync_hooks(service) -> None:
    """Stand this service's declared hooks at their doorways, exactly once.

    Called from both ``bind_runtime`` and ``_load`` because the two arrive in
    different orders at boot and on reload, and neither alone is enough: a
    hook needs a runtime to register with and a box to call into.
    """
    declared = getattr(service, "_hooks", None) or {}
    runtime = getattr(service, "_runtime", None)
    registry = getattr(runtime, "hooks", None)
    if not declared or registry is None or service._shims is not None:
        return

    from .hooks import build_shim

    shims = []
    for moment, method in declared.items():
        try:
            shim = build_shim(service, moment, method, _make_response)
        except ValueError as exc:
            logger.error("service %s: %s", service.name, exc)
            continue
        registry.add(moment, shim)
        shims.append(shim)
    service._shims = shims
    if shims:
        logger.info("service %s stands at %s", service.name,
                    ", ".join(sorted(declared)))


def _unhook(service) -> None:
    """Walk this service away from every doorway.

    A hook outliving its plugin is a leak with no symptom — it keeps being
    consulted and keeps abstaining — so unload has to be thorough.
    """
    registry = getattr(getattr(service, "_runtime", None), "hooks", None)
    for shim in getattr(service, "_shims", None) or []:
        if registry is not None:
            try:
                registry.remove(shim)
            except Exception:
                logger.exception("could not remove a hook for %s", service.name)
    service._shims = None


def _listen(service) -> None:
    """Subscribe this service to every channel it declared.

    Called only from ``_load``, unlike ``_sync_hooks``: a subscription needs a
    box to deliver into but no runtime, so there is only one moment it can
    happen and no ordering problem to be idempotent about.
    """
    channels = getattr(service, "_channels", None) or []
    if not channels or service._listeners is not None:
        return

    from .events import subscribe_all

    def deliver(channel: str, payload):
        """Carry one event into the box, if the box is still there."""
        box = getattr(service, "_sandbox_box", None)
        if box is None or not box.alive:
            return
        result = box.call("__event__", channel=channel, payload=payload)
        if not result.ok:
            # A subscriber that raises is the publisher's problem only if we
            # make it one, and the bus contract says we must not.
            logger.warning("%s failed handling %s: %s",
                           service.name, channel, result.error)

    service._listeners = subscribe_all(service, channels, deliver)


def _deafen(service) -> None:
    """Drop every subscription. Symmetrical with ``_unhook``."""
    from .events import unsubscribe_all

    unsubscribe_all(getattr(service, "_listeners", None))
    service._listeners = None


class ServiceCallFailed(RuntimeError):
    """An exported service method failed inside its box.

    Raised rather than returned because native callers reach a service by
    attribute access — ``services.get("embedder").embed(texts)`` — and expect
    a value or an exception, not a Result they have to unwrap.
    """


MIN_POLL_INTERVAL = 0.01
MAX_POLL_INTERVAL = 3600.0
DEFAULT_POLL_FAILURES = 5


def _poll_settings(declarations: dict, default_interval: float) -> tuple:
    """Clamp a resident plugin's polling wishes to kernel-owned limits."""
    try:
        raw = float(declarations.get("poll_interval", default_interval))
    except (TypeError, ValueError):
        raw = default_interval
    interval = (
        0.0
        if raw <= 0
        else max(MIN_POLL_INTERVAL, min(raw, MAX_POLL_INTERVAL))
    )
    try:
        failures = int(
            declarations.get("max_poll_failures", DEFAULT_POLL_FAILURES)
        )
    except (TypeError, ValueError):
        failures = DEFAULT_POLL_FAILURES
    return interval, max(1, min(failures, 100))


def _drive_polls(
    *,
    family: str,
    name: str,
    box,
    stopping,
    interval: float,
    max_failures: int,
    done=None,
):
    """Drive one resident plugin's kernel-owned poll loop.

    Returns whether the loop ended because it was asked to. A box that dies
    under the loop used to end it in complete silence — the ``while``
    condition simply went false — which is indistinguishable from a clean
    stop and was how a starved REPL disappeared without a word.
    """
    failures = 0
    while not stopping.is_set() and not (callable(done) and done()):
        if not box.alive:
            logger.error("%s %s stopped: its box is no longer running",
                         family, name)
            return False
        outcome = box.call("poll")
        if outcome.ok:
            failures = 0
            # Truthy means work remains: drain it before sleeping.
            if not outcome.data:
                stopping.wait(interval)
            continue

        failures += 1
        logger.warning(
            "%s %s poll failed (%d/%d): %s",
            family,
            name,
            failures,
            max_failures,
            outcome.error,
        )
        if failures >= max_failures:
            logger.error(
                "%s %s stopped polling after repeated failures",
                family,
                name,
            )
            return False
        stopping.wait(interval)
    return True


class _Occupant:
    """One service's view of a box it may be sharing.

    Everything that calls into a resident service — the export forwarders, the
    poll loop, the bus deliverer, the hook shims, the prompt collector —
    reaches ``service._sandbox_box`` and calls ``.call(method, ...)``. Handing
    each adapter a handle that already knows its own target keeps every one of
    those sites written exactly as it was, instead of threading a ``target``
    argument through five call paths that have no other reason to know a box
    can hold more than one thing.
    """

    def __init__(self, box, target: str):
        self._box = box
        self._target = target

    @property
    def alive(self) -> bool:
        """Whether the underlying box can still take calls."""
        return self._box is not None and self._box.alive

    @property
    def execution(self):
        """The box's execution, for callers that adjust its context."""
        return self._box.execution

    def call(self, method: str, *args, **kwargs):
        """Invoke a method on *this* occupant."""
        return self._box.call(method, *args, target=self._target, **kwargs)


class _Residency:
    """The one box a file's services share, and who is still using it.

    A file holding two services registers as two services: the kernel loads
    and unloads each by name, with no idea they are neighbours. But there is
    one process behind them, and it exists precisely *because* they share
    something expensive — so the naive mapping, where each adapter's
    ``unload`` closes the box, means unloading the text embedder kills the
    image embedder's model too, with no symptom beyond calls suddenly failing.

    Hence a refcount. The first adapter to load opens the box; the last to
    unload closes it. Everything in between joins what is already there.
    """

    def __init__(self, source_path: str, box_name: str, entries: list):
        self.source_path = source_path
        self.box_name = box_name
        self.entries = list(entries)
        self.box = None
        self._users: set = set()
        self._lock = threading.Lock()

    def join(self, name: str, chain):
        """Open the box if nobody has yet, and record this user."""
        with self._lock:
            if self.box is None or not self.box.alive:
                self.box = get_sandbox().open(
                    self.source_path,
                    self.entries[0] if self.entries else "",
                    name=self.box_name,
                    chain=chain,
                    # Empty for a lone occupant, so a one-service file opens
                    # exactly the box it always did.
                    entries=self.entries if len(self.entries) > 1 else (),
                )
            self._users.add(name)
            return self.box

    def leave(self, name: str):
        """Drop this user, and close the box once the last one has gone."""
        with self._lock:
            self._users.discard(name)
            if self._users or self.box is None:
                return
            self.box = None
        try:
            get_sandbox().close(self.box_name)
        except Exception:
            logger.exception("failed to close box %s", self.box_name)


def _adapt_service(
    path,
    entry: str,
    base,
    declarations: dict,
    box_name: str,
    source: str,
):
    """Build native-looking services backed by one resident box.

    The other families are *calls*: run once, translate the answer, tear down.
    A service is a *residency*, so the adapter maps the native lifecycle onto
    the box lifecycle instead:

        _load()   ->  open the box (start() runs inside it)
        unload()  ->  close the box (stop() runs inside it)

    and every method the plugin lists in ``exports`` becomes a real method on
    the adapter, because native callers reach services by attribute access
    rather than through ``service.call``.

    A file may declare several service classes — the shape ``build_services``
    has always supported, and the reason it returns a dict. They share one box
    and are told apart by the call's ``target``, so a heavy library is
    imported once and an accelerator context is created once. Everything below
    is built per class; only :class:`_Residency` is shared.
    """
    source_path = str(path.resolve())
    classes = list(declarations.get("classes") or [])
    if not classes:
        # A file the validator did not describe per class — a hand-built
        # declarations dict in a test, most likely. Treat it as the one class
        # it names, which is what this function always assumed.
        classes = [{"entry": entry, **declarations}]

    residency = _Residency(source_path, box_name,
                           [c.get("entry") or entry for c in classes])

    adapters = {}
    module = None
    for spec in classes:
        adapter, module = _service_adapter(
            path, spec.get("entry") or entry, base, spec, residency, source,
            module)
        adapters[adapter.name] = adapter

    def build_services(config: dict) -> dict:
        """Services are discovered by calling this, not by scanning classes."""
        return {name: adapter() for name, adapter in adapters.items()}

    module.build_services = build_services
    return module


def _service_adapter(
    path,
    entry: str,
    base,
    declarations: dict,
    residency: "_Residency",
    source: str,
    module=None,
):
    """One service class's adapter. See :func:`_adapt_service`."""
    source_path = residency.source_path
    box_name = residency.box_name
    name = declarations.get("name") or path.stem.split("_", 1)[-1]
    exports = list(declarations.get("exports") or [])
    interval, max_failures = _poll_settings(declarations, 0.0)
    polls = interval > 0 and _defines(source, entry, "poll")

    if not exports:
        # Not fatal — a service may exist only for its side effects — but it
        # is nearly always a forgotten declaration, and the symptom (every
        # call failing as "not exported") points nowhere near the cause.
        logger.warning("sandboxed service %s declares no exports; nothing "
                       "will be able to call it", path.name)

    def __init__(self):
        """Initialize native adapter state without importing guest code."""
        base.__init__(self)
        self._poll_stop = threading.Event()
        self._poll_thread = None

    def _load(self) -> bool:
        """Open the resident box. Its start() runs inside."""
        self._poll_stop.clear()
        # A residency's prompt text is only knowable while it is resident, so
        # the cache is scoped to one lifetime rather than to the adapter. Asked
        # before load, the answer is "nothing" — and that must not be the answer
        # forever after.
        self._prompt_text = None
        try:
            box = residency.join(name, Chain(root=f"service:{name}"))
        except Exception as exc:
            logger.error("service %s did not start: %s", name, exc)
            return False
        # ``target`` is this service's own name, and empty when it is the box's
        # only occupant — which is what keeps a one-service file's wire
        # identical to what it always sent.
        self._sandbox_box = _Occupant(
            box, name if len(residency.entries) > 1 else "")
        self.loaded = True
        # Binding and loading happen in either order depending on whether this
        # is boot or a live reload, so both ends call the same idempotent sync.
        _sync_hooks(self)
        _listen(self)
        if polls:
            box = self._sandbox_box
            self._poll_thread = threading.Thread(
                target=_drive_polls,
                kwargs={
                    "family": "service",
                    "name": name,
                    "box": box,
                    "stopping": self._poll_stop,
                    "interval": interval,
                    "max_failures": max_failures,
                },
                daemon=True,
                name=f"{name}-poll",
            )
            self._poll_thread.start()
        return True

    def unload(self):
        """Close the box and step away from every doorway and channel."""
        self.loaded = False
        self._prompt_text = None
        self._poll_stop.set()
        thread, self._poll_thread = self._poll_thread, None
        if (
            thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=5.0)
        _unhook(self)
        _deafen(self)
        self._sandbox_box = None
        # The box closes when the *last* service in this file unloads. See
        # ``_Residency``: unloading one of a pair must not take the other's
        # loaded model down with it.
        residency.leave(name)

    def bind_runtime(self, *, runtime=None, **_):
        """Receive the runtime. Idempotent, and may arrive before or after load."""
        if runtime is not None:
            self._runtime = runtime
        _sync_hooks(self)

    def agent_prompt(self, ctx):
        """Prompt contribution, answered in the resident box.

        No spawn to pay for here, but still cached: a resident box serializes
        its calls and prompt collection runs on the turn thread, so queuing
        behind a long export every turn would be a stall for no gain.
        """
        return _cached_prompt(self, lambda: _box_prompt(self, name, _prompt_name))

    def _export(method: str):
        """One forwarding method, so callers see an ordinary service."""
        def call(self, *args, **kwargs):
            """Invoke an exported method inside the box."""
            box = getattr(self, "_sandbox_box", None)
            if box is None or not box.alive:
                raise ServiceCallFailed(
                    f"service {name!r} is not loaded")
            result = box.call(method, *args, **kwargs)
            if not result.ok:
                raise ServiceCallFailed(f"{name}.{method}: {result.error}")
            return result.data
        call.__name__ = method
        call.__qualname__ = f"{entry}.{method}"
        call.__doc__ = f"Call {name}.{method} inside its sandbox box."
        return call

    attributes = {
        "__doc__": f"Sandboxed service loaded from {path.name}.",
        "_source_path": source_path,
        "_sandboxed": True,
        "_sandbox_box": None,
        "_runtime": None,
        "_shims": None,
        "_listeners": None,
        "_hooks": dict(declarations.get("hooks") or {}),
        "_channels": list(declarations.get("subscribed_channels") or []),
        "_box": box_name,
        "_entry": entry,
        "name": name,
        "__init__": __init__,
        "_load": _load,
        "unload": unload,
    }
    for key, value in declarations.items():
        if key not in NOT_CARRIED:
            attributes[key] = value
    # exports rides along on purpose: handlers._service_call reads it to
    # refuse anything unexported, so the declaration keeps working when the
    # caller is other sandboxed code rather than the kernel.
    attributes["exports"] = exports
    attributes["bind_runtime"] = bind_runtime
    _prompt_name = _prompt_method(source, entry)
    if _prompt_name:
        attributes["agent_prompt"] = agent_prompt
    for method in exports:
        attributes[method] = _export(method)

    return _build(entry, base, attributes, path, source_path, module)


def _adapt_frontend(path, entry: str, base, declarations: dict, box_name: str,
                    source: str = ""):
    """Build a native-looking frontend backed by a resident box.

    Like a service this is a residency, but a frontend is the family the kernel
    calls *into*, and that changes two things.

    **The loop inverts.** A native frontend blocks in ``start()`` forever. A box
    serializes one call at a time, so a guest that never returned from ``start``
    would hold its box and no ``render`` could get in — the frontend would go
    deaf the moment it started listening. So the guest's ``start`` sets up and
    returns, and this adapter runs the loop on the daemon thread the frontend
    manager already gives it, calling ``poll`` over and over. Between polls is
    when a render lands.

    **Nine render methods become one call.** ``BaseFrontend`` hands subclasses
    nine typed methods; the box gets one ``render(kind, payload)``. Nine wire
    methods for one concept is surface with no payoff, and it lets a guest
    handle the kinds its transport can show and ignore the rest.
    """
    from .frontends import park, project_payload, unpark

    source_path = str(Path(path).resolve())
    name = declarations.get("name") or path.stem.split("_", 1)[-1]
    interval, max_failures = _poll_settings(declarations, 0.05)

    wants_console = bool(declarations.get("uses_console"))
    restore_on_start = bool(declarations.get("restore_on_start"))

    def __init__(self, shutdown_event=None):
        """Take the host's shutdown Event, if the manager offers one.

        Named as a constructor parameter because that is how ``FrontendManager``
        supplies host resources — it matches parameters against what it has.
        """
        base.__init__(self)
        self._sandbox_box = None
        self._token = ""
        self._prompt_text = None
        self._stopping = threading.Event()
        self._shutdown_event = shutdown_event

    def _done(self) -> bool:
        """Whether the loop should stop."""
        return self._stopping.is_set() or (
            self._shutdown_event is not None
            and self._shutdown_event.is_set())

    def start(self):
        """Open the box, hand it its authority, then drive it until stopped.

        Runs on the frontend manager's daemon thread, so blocking here is
        correct — this is the loop the guest is no longer allowed to write.
        """
        try:
            self._sandbox_box = get_sandbox().open(
                source_path,
                entry,
                name=box_name,
                chain=Chain(root=f"frontend:{name}"),
                manage_lifecycle=False,
            )
        except Exception as exc:
            logger.error("frontend %s did not start: %s", name, exc)
            return

        # Persistent frontend Requests need the same host context native
        # commands receive. The execution object stays host-side; only Request
        # results cross into the box.
        try:
            context_for = getattr(self.commands, "context", None)
            context = (
                context_for(None) if callable(context_for)
                else types.SimpleNamespace(runtime=self.runtime,
                                           session_key=None)
            )
            self._sandbox_box.execution.context = context
        except Exception:
            logger.exception("frontend %s could not bind sandbox context", name)
            self.stop()
            return

        # The desk opens before the guest's start(), so a frontend can submit
        # from its very first line — restoring a session, say.
        self._token = park(self)
        box = self._sandbox_box

        # The console is exclusive. Losing the claim is not fatal — a frontend
        # that cannot read the keyboard may still have something to show — but
        # it must be loud, because the symptom is a frontend that ignores
        # everything typed at it.
        if wants_console:
            from .console import CONSOLE

            if not CONSOLE.claim(self._token):
                logger.error("frontend %s wants the console but %s already "
                             "has it; it will not receive input", name,
                             "another frontend")
        result = box.call("__bind__", token=self._token)
        if not result.ok:
            logger.error("frontend %s could not be bound: %s", name,
                         result.error)
            self.stop()
            return

        if not box.call("start").ok:
            logger.error("frontend %s refused to start", name)
            self.stop()
            return

        # Restoration may synchronously emit form/approval renders. It must
        # happen between guest calls, after start() has released the box.
        if restore_on_start:
            try:
                key = self.session_key(None)
                notice = self.runtime.restore_last_active(key)
                if notice:
                    self.render_messages(key, [notice])
            except Exception:
                logger.exception("frontend %s restore_last_active failed", name)

        asked = _drive_polls(
            family="frontend",
            name=name,
            box=box,
            stopping=self._stopping,
            interval=interval,
            max_failures=max_failures,
            done=self._done,
        )
        if not asked:
            # The guest cannot say this — its box is the thing that died — so
            # the host says it, on the console it already owns. Without this a
            # dead console frontend is a terminal that echoes nothing and
            # explains nothing, which is exactly how the deadlock presented.
            logger.error("frontend %s died; it is no longer accepting input",
                         name)
            if wants_console:
                from .console import CONSOLE
                CONSOLE.write(
                    f"\n[frontend {name} stopped: see app.log. Restart the "
                    f"app to recover.]")

        self.stop()

    def stop(self):
        """Stop the loop, close the box, and take the frontend's authority.

        Idempotent: the loop calls it on the way out and the manager calls it
        on unregister, and either may be first.
        """
        self._stopping.set()
        # Scoped to one residency, for the reason ``_load`` gives: asked while
        # the box is shut, the honest answer is "nothing", and it must not
        # outlive the shutdown that made it true.
        self._prompt_text = None
        box, self._sandbox_box = self._sandbox_box, None
        if box is not None and box.alive:
            try:
                box.call("stop")
            except Exception:
                logger.exception("frontend %s stop() failed", name)
        # Revoked before the box is closed, so nothing can submit during
        # teardown on a frontend that is already going away. Releasing the
        # console names the token, so a frontend that already lost the claim
        # cannot take it from whoever holds it now.
        if wants_console:
            from .console import CONSOLE

            CONSOLE.release(self._token)
        unpark(self._token)
        self._token = ""
        if box is not None:
            try:
                get_sandbox().close(box_name)
            except Exception:
                logger.exception("failed to close box %s", box_name)

    def _render(self, session_key: str, kind: str, payload=None):
        """One render, forwarded into the box.

        A frontend that cannot show something is not an error the kernel needs
        to hear about — the turn carries on either way — so failures are logged
        and swallowed, the same policy hooks have.
        """
        box = self._sandbox_box
        if box is None or not box.alive:
            return
        result = box.call("render", session_key=session_key, kind=kind,
                          payload=project_payload(kind, payload))
        if not result.ok:
            logger.warning("frontend %s could not render %s: %s", name, kind,
                           result.error)

    def session_key(self, ctx):
        """Ask the box to name a session.

        A transport context is the frontend's own object and cannot cross, so
        what goes in is whatever of it is plain data. Most frontends key off a
        string or a couple of ids, and one that cannot answer falls back to a
        single session rather than losing the message.
        """
        box = self._sandbox_box
        if box is None or not box.alive:
            return "default"
        result = box.call("session_key",
                          ctx=project_payload("session_key", ctx))
        return str(result.data) if result.ok and result.data else "default"

    def _live_session_keys(self):
        """Only the sessions this frontend actually owns.

        The native default is *every* session the runtime knows about, which
        works because each native frontend overrides it — the REPL answers
        ``["default"]``. A sandboxed frontend cannot override a native method,
        and inheriting that default would mean rendering another frontend's
        conversation to this one's transport.

        The tag is already there: ``_tag_session`` stamps ``frontend_name`` on
        every session a frontend submits for, so ownership can be read rather
        than declared. Untagged sessions are included, because a session that
        nobody has claimed is one this frontend may still be about to receive
        — dropping it would lose the first message of a conversation.
        """
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return []
        return [key for key, session in (getattr(runtime, "sessions", None)
                                         or {}).items()
                if getattr(session, "frontend_name", None) in (None, self.name)]

    def _renderer(kind: str):
        """One native render method, funnelled into the single box call."""
        def render(self, session_key, payload=None):
            """Show something."""
            self._render(session_key, kind, payload)
        render.__name__ = f"render_{kind}"
        render.__doc__ = f"Forward a {kind} render into the box."
        return render

    attributes = {
        "__doc__": f"Sandboxed frontend loaded from {path.name}.",
        "_source_path": source_path,
        "_sandboxed": True,
        "_box": box_name,
        "_entry": entry,
        "name": name,
        "__init__": __init__,
        "_done": _done,
        "_render": _render,
        "_live_session_keys": _live_session_keys,
        "start": start,
        "stop": stop,
        "session_key": session_key,
    }
    for key, value in declarations.items():
        if key not in NOT_CARRIED:
            attributes[key] = value

    # The native names differ from the wire kinds in three places, because the
    # kernel named them for what they are and the wire names them for what a
    # frontend does with them.
    native_names = {"messages": "render_messages",
                    "attachments": "render_attachments",
                    "form_field": "render_form_field",
                    "approval": "render_approval_request",
                    "buttons": "render_buttons",
                    "error": "render_error",
                    "typing": "render_typing",
                    "tool_status": "render_tool_status",
                    "stream_delta": "render_stream_delta"}
    for kind, method in native_names.items():
        attributes[method] = _renderer(kind)

    # ``capabilities`` is declared as a plain dict because a box cannot hold a
    # dataclass; the native side reads attributes off one. Same rebuild as
    # ``_form_step`` does for a command's form.
    attributes["capabilities"] = _capabilities(declarations.get("capabilities"))

    _prompt_name = _prompt_method(source, entry)
    if _prompt_name:
        attributes["agent_prompt"] = (
            lambda self, ctx, _m=_prompt_name: _cached_prompt(
                self, lambda: _box_prompt(self, name, _m)))

    adapter, module = _build(entry, base, attributes, path, source_path)
    return module


def _capabilities(declared):
    """Rebuild a FrontendCapabilities from the literal dict a guest declares.

    Unknown keys are dropped rather than raising: a frontend claiming a
    capability this kernel has never heard of should lose that claim, not fail
    to load.
    """
    from plugins.native.frontend import FrontendCapabilities

    if not isinstance(declared, dict):
        return FrontendCapabilities()
    allowed = set(FrontendCapabilities.__dataclass_fields__)
    unknown = set(declared) - allowed
    if unknown:
        logger.warning("frontend declares unknown capabilities: %s",
                       ", ".join(sorted(unknown)))
    return FrontendCapabilities(**{k: v for k, v in declared.items()
                                   if k in allowed})
