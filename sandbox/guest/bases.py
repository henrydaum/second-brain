"""The plugin contracts, rewritten for the Requests system.

These live in the guest because every plugin subclasses one, so they are part
of the shippable SDK rather than of the kernel. A container image copies
``guest/`` and has everything a plugin needs to be written against.

What changed from ``plugins/Base*.py``:

- **There is a common ancestor.** The five families were five unrelated
  classes that happened to share six attributes. Those attributes now live on
  :class:`BasePlugin` and are declared once.
- **``context`` became ``sdk``.** A plugin no longer receives a bag of live
  kernel objects; it receives a handle that can only ask. This is the entire
  difference, and it is why the base classes had to move.
- **Boxes.** A plugin declares which execution context it shares with its
  helpers (see :mod:`guest.box`).
- **Services declare ``exports``.** Only listed methods are reachable through
  ``service.call``, which answers "which service methods can a plugin call?"
  by declaration rather than by guesswork.

**None of this is required to use the sandbox.** An arbitrary script — an
agent's scratch computation, a helper module — needs no base class and no
declaration at all. It is a file with functions that take ``sdk``. The classes
here exist to describe things the *kernel* has to register and schedule; code
that only wants to compute should ignore them.

Every attribute is a declaration of *intent*. The kernel reads them without
importing the file, resolves them, and clamps them. Declaring a longer timeout
or a wider box does not grant one.
"""

from __future__ import annotations

from .box import EPHEMERAL, IN_PROCESS, PERSISTENT, SUBPROCESS, Membership

TOOL = "tool"
TASK = "task"
SERVICE = "service"
COMMAND = "command"
FRONTEND = "frontend"


class BasePlugin:
    """What every plugin family has in common.

    Subclass one of the five families below, never this directly.
    """

    # ── identity ───────────────────────────────────────────────────
    name: str = ""
    description: str = ""
    family: str = ""          # set by the family class, not by the author

    # ── packaging ──────────────────────────────────────────────────
    dependencies_files: list = []
    dependencies_pip: list = []
    requires_services: list = []
    config_settings: list = []

    # ── execution ──────────────────────────────────────────────────
    # All intent. The kernel resolves and clamps; see guest.box.
    box: str = ""             # "" means a box of its own
    isolation: str = ""       # "" means the kernel's default
    lifetime: str = ""        # "" means ephemeral
    timeout: float = 0.0      # 0 means the kernel's default
    memory_mb: int = 0

    # Requests this plugin expects to make. Advisory: it lets the user see at
    # install time what a plugin wants, and lets the validator flag a Request
    # the author forgot to declare. It grants nothing — the gate decides.
    requests: list = []

    # ── the bus, inbound ───────────────────────────────────────────
    # Channels this plugin wants to hear. ``sdk.events.emit`` is the outbound
    # half and needs no declaration; receiving does, for the same reason
    # ``hooks`` does — a subscription the plugin never registered is one it
    # cannot forget to undo, and uninstalling the file takes it away.
    #
    # Only a plugin that *stays loaded* can be delivered to: a tool is a call
    # that ends. Services and frontends, then.
    #
    # Channel names are not a closed vocabulary. The kernel's are in
    # ``events/event_channels.py``, but a plugin owns its own and may listen to
    # another plugin's, so nothing here validates the string.
    subscribed_channels: list = []

    def on_event(self, sdk, channel: str, payload):
        """One bus event on a channel this plugin declared. Return nothing.

        Handlers run on whichever thread emitted, so this must be quick and
        must not raise — the bus is fire-and-forget, and a slow subscriber
        slows down the code that published.
        """
        return None

    def __event__(self, sdk, channel: str, payload=None):
        """Receive one delivery. The kernel calls this, never an author.

        Undeclared channels are refused here as well as host-side. The host
        only subscribes to what was declared, so this is belt-and-braces —
        but the declaration is the security story, and a story that is only
        told in one place is one edit away from not being true.
        """
        if channel not in (self.subscribed_channels or []):
            raise ValueError(
                f"{self.name}: {channel!r} was not declared in "
                f"subscribed_channels")
        return self.on_event(sdk, channel, payload)

    # ── prompt contribution ────────────────────────────────────────
    agent_prompt: str = ""

    def agent_prompt_for(self, sdk) -> str:
        """Guidance to add to the system prompt, or "" to stay silent.

        Called per turn, so it must be cheap and stable — the result lands in
        a cacheable block. Do not make Requests here.
        """
        return self.agent_prompt

    # ── introspection ──────────────────────────────────────────────

    @classmethod
    def membership(cls, source: str) -> Membership:
        """This plugin's declared wishes about its execution context."""
        return Membership(
            source=source, box=cls.box, isolation=cls.isolation,
            lifetime=cls.lifetime, timeout=cls.timeout,
            memory_mb=cls.memory_mb)

    @classmethod
    def declared(cls) -> dict:
        """Every declaration, for the kernel to resolve and clamp.

        The kernel normally reads these by parsing the file rather than
        importing it; this is the same data for callers that already hold the
        class.
        """
        return {
            "name": cls.name, "family": cls.family,
            "description": cls.description,
            "dependencies_files": list(cls.dependencies_files),
            "dependencies_pip": list(cls.dependencies_pip),
            "requires_services": list(cls.requires_services),
            "requests": list(cls.requests),
            "subscribed_channels": list(cls.subscribed_channels),
            "box": cls.box, "isolation": cls.isolation,
            "lifetime": cls.lifetime, "timeout": cls.timeout,
            "memory_mb": cls.memory_mb,
        }


class BaseTool(BasePlugin):
    """Something the agent can call during a turn."""

    family = TOOL

    parameters: dict = {}      # JSON schema for the arguments
    dependencies_tools: list = []
    max_calls: int = 3         # per message; clamped
    background_safe: bool = True   # authority-bearing: clamped, not trusted
    auto_register: bool = True

    def run(self, sdk, **kwargs):
        """Do the work and return a Result.

        ``sdk.ok(...)`` and ``sdk.fail(...)`` build one.
        """
        raise NotImplementedError


class BaseTask(BasePlugin):
    """Pipeline work: triggered by files appearing or by an event."""

    family = TASK

    trigger: str = "path"          # "path" | "event"
    trigger_channels: list = []
    event_payload_schema: dict = {}
    default_jobs: dict = {}

    modalities: list = []
    reads: list = []               # input tables; dependencies derived
    writes: list = []              # output tables
    require_all_inputs: bool = True
    output_schema: str = ""
    batch_size: int = 1
    max_workers: int = 0           # 0 means the orchestrator's default

    def run(self, sdk, paths):
        """Process a batch of paths and return a Result."""
        raise NotImplementedError


class BaseService(BasePlugin):
    """A long-lived capability other plugins call.

    Services are the natural persistent box: state loaded once — a model, a
    connection pool, a cache — and kept. That state stays *inside* the box. It
    is never handed across the boundary, so callers get simple data and the
    thing itself never leaves.
    """

    family = SERVICE
    lifetime = PERSISTENT

    shared: bool = True
    is_llm_backend: bool = False
    poll_interval: float = 0.0
    max_poll_failures: int = 5

    # Methods reachable through ``service.call``. Anything not listed is
    # internal. Declaring the surface explicitly is what makes "which service
    # methods can a plugin call?" answerable without guessing, and every
    # listed method must return simple data.
    exports: list = []

    # Doorways this service stands at: ``{moment: method_name}``. Declared
    # rather than registered because a hook is *inbound* — the kernel calls
    # it — so there is nothing for the plugin to call at load time, and
    # nothing it can forget to undo at unload. See :mod:`guest.hooks`.
    hooks: dict = {}

    def start(self, sdk):
        """Acquire whatever this service holds. Return True on success."""
        raise NotImplementedError

    def stop(self, sdk):
        """Release it. Must tolerate never having started."""
        return None

    def poll(self, sdk):
        """Perform one small unit of periodic work."""
        return False

    def __hook__(self, sdk, moment: str, handler: str, ctx: dict,
                 payload=None, token: str = ""):
        """Receive one doorway visit. The kernel calls this, never an author.

        It exists so hook methods can be written in ordinary Python — real
        objects with attributes, returning a verdict — while the wire between
        the kernel and this box carries nothing but data. Rehydrating here
        rather than in the shim keeps that translation on the guest side,
        where the dataclasses live.
        """
        from .hooks import HookContext, unwrap, wrap

        fn = getattr(self, handler, None)
        if not callable(fn):
            raise AttributeError(
                f"{self.name}: hooks names {handler!r} for {moment!r}, "
                f"but there is no such method")
        envelope = HookContext(**{k: v for k, v in dict(ctx or {}).items()
                                  if k in HookContext.__dataclass_fields__})
        # An escort's ``sdk.model.proceed()`` has to name the call it is
        # placing. Carrying that on the sdk rather than on the payload keeps
        # it off the author's hands: a rewritten request still proceeds.
        sdk._hook_token = token
        try:
            return unwrap(fn(sdk, envelope, wrap(moment, payload)))
        finally:
            sdk._hook_token = ""


class BaseCommand(BasePlugin):
    """A slash command a person types."""

    family = COMMAND

    category: str = "Other"
    hide_from_help: bool = False
    require_approval: bool = False   # authority-bearing: clamped
    approval_actions: tuple[str, ...] = ()
    approval_action_prefixes: tuple[str, ...] = ()
    approval_actor_id: str = ""

    def form(self, sdk, args: dict) -> list:
        """Steps to collect missing arguments, or [] to run immediately."""
        return []

    def arg_completions(self, sdk) -> list:
        """Completion candidates for the first argument."""
        return []

    def run(self, sdk, args: dict):
        """Execute and return markdown, or None."""
        raise NotImplementedError


class BaseFrontend(BasePlugin):
    """A surface a person talks to.

    Frontends are the one family the kernel calls *into* rather than up from,
    and that inverts the usual shape twice.

    **There is no main loop.** A native frontend blocks in ``start()`` forever.
    That cannot work here: a box serializes one call at a time, so code that
    never returns from ``start`` holds the box and no ``render`` ever gets in —
    the frontend would go deaf the moment it started listening. So ``start``
    sets up and *returns*, and the kernel calls ``poll`` over and over on a
    thread it owns. You are not driving; you are being asked.

    **Showing things is not a Request.** ``render`` is called on you when the
    kernel has something for a person to see. Carrying what that person does
    back the other way *is* — see ``sdk.frontend``.
    """

    family = FRONTEND
    lifetime = PERSISTENT

    user_binding: str = "single"     # "single" | "per_user"
    default_user_id: int = 1
    capabilities: dict = {}

    # How long the kernel waits after a poll that found nothing. Only paid
    # when idle: a poll that reports work is called straight back.
    poll_interval: float = 0.05
    max_poll_failures: int = 5

    # Whether this frontend's transport is the machine's own console. The
    # kernel lends it to exactly one frontend — two readers would split a
    # person's keystrokes between them — so a second claimant is refused and
    # ``sdk.console`` reaches nothing for it. See :mod:`sandbox.console`.
    uses_console: bool = False

    # Submissions can synchronously render back into this serialized box.
    # Terminal-style frontends therefore ask the host to submit off poll().
    background_submit: bool = False

    # Restore after start() releases the box, since restoration may render.
    restore_on_start: bool = False

    def start(self, sdk):
        """Open the transport. **Must return** — do not loop here.

        Return True on success. Anything else and the frontend does not run.
        """
        raise NotImplementedError

    def poll(self, sdk):
        """Take whatever input is waiting and submit it. Called repeatedly.

        **Must return promptly.** Between polls is the only time the kernel
        can call ``render``, so a slow poll is a frozen display. Return truthy
        if you did something — you will be called straight back — and falsy if
        there was nothing, which earns a ``poll_interval`` pause.

        A long-poll with a short server-side timeout is the right shape. An
        unbounded wait is not.
        """
        raise NotImplementedError

    def stop(self, sdk):
        """Close the transport. Must tolerate never having started."""
        return None

    def render(self, sdk, session_key: str, kind: str, payload):
        """Show one thing to a person. ``kind`` says what.

        ``messages`` (list[str] of markdown) · ``attachments`` (list of paths)
        · ``form_field`` · ``approval`` · ``buttons`` · ``error`` · ``typing``
        (bool) · ``tool_status`` · ``stream_delta``.

        Handle the kinds your transport can show and ignore the rest — a
        frontend that only renders ``messages`` is a working frontend.
        Answer an ``approval`` with ``sdk.frontend.resolve``.
        """
        raise NotImplementedError

    def session_key(self, sdk, ctx):
        """Name the session some transport context belongs to.

        One string per conversational surface: a chat, a socket, a thread.
        The kernel treats two contexts with the same key as the same person
        in the same place.
        """
        raise NotImplementedError

    def __bind__(self, sdk, token: str = ""):
        """Receive the handle on this frontend's own adapter. Kernel-called.

        Held on the ``sdk`` rather than on the plugin so ``sdk.frontend``
        works without the author ever seeing a token — and so a frontend
        cannot hand its authority to anything by passing itself around.
        """
        sdk._frontend_token = token
        return True


FAMILIES = {TOOL: BaseTool, TASK: BaseTask, SERVICE: BaseService,
            COMMAND: BaseCommand, FRONTEND: BaseFrontend}


def entry_for(obj, method: str = "run"):
    """Resolve a module attribute to the callable a runner should invoke.

    A plugin's entry point is a method on a class; an arbitrary script's is
    just a function. Both runners resolve through here so a file needs a base
    class only when the kernel has to register it — never merely to run.

    ``method`` names which one, because a plugin has more than one entry: a
    command answers with ``run`` but collects its arguments with ``form``.
    """
    if isinstance(obj, type) and issubclass(obj, BasePlugin):
        instance = obj()
        return (getattr(instance, method, None)
                or getattr(instance, "run", None)
                or getattr(instance, "start"))
    return obj

__all__ = ["BasePlugin", "BaseTool", "BaseTask", "BaseService", "BaseCommand",
           "BaseFrontend", "FAMILIES", "TOOL", "TASK", "SERVICE", "COMMAND",
           "FRONTEND", "EPHEMERAL", "PERSISTENT", "IN_PROCESS", "SUBPROCESS"]
