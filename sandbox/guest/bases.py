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

    # Methods reachable through ``service.call``. Anything not listed is
    # internal. Declaring the surface explicitly is what makes "which service
    # methods can a plugin call?" answerable without guessing, and every
    # listed method must return simple data.
    exports: list = []

    def start(self, sdk):
        """Acquire whatever this service holds. Return True on success."""
        raise NotImplementedError

    def stop(self, sdk):
        """Release it. Must tolerate never having started."""
        return None


class BaseCommand(BasePlugin):
    """A slash command a person types."""

    family = COMMAND

    category: str = "Other"
    hide_from_help: bool = False
    require_approval: bool = False   # authority-bearing: clamped
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

    Deliberately the thinnest of the five. Frontends are inbound-driven — the
    kernel calls *them* — which the Request model does not describe yet, and
    they are the last family to migrate. Treat this as the shape, not the
    finished contract.
    """

    family = FRONTEND
    lifetime = PERSISTENT

    user_binding: str = "single"     # "single" | "per_user"
    default_user_id: int = 1
    capabilities: dict = {}

    def start(self, sdk):
        """Begin accepting input. Returns when the frontend stops."""
        raise NotImplementedError

    def stop(self, sdk):
        """Stop accepting input."""
        return None

    def render(self, sdk, session_key: str, payload: dict):
        """Show something to the user."""
        raise NotImplementedError


FAMILIES = {TOOL: BaseTool, TASK: BaseTask, SERVICE: BaseService,
            COMMAND: BaseCommand, FRONTEND: BaseFrontend}


def entry_for(obj):
    """Resolve a module attribute to the callable a runner should invoke.

    A plugin's entry point is a method on a class; an arbitrary script's is
    just a function. Both runners resolve through here so a file needs a base
    class only when the kernel has to register it — never merely to run.
    """
    if isinstance(obj, type) and issubclass(obj, BasePlugin):
        instance = obj()
        entry = getattr(instance, "run", None) or getattr(instance, "start")
        return entry
    return obj

__all__ = ["BasePlugin", "BaseTool", "BaseTask", "BaseService", "BaseCommand",
           "BaseFrontend", "FAMILIES", "TOOL", "TASK", "SERVICE", "COMMAND",
           "FRONTEND", "EPHEMERAL", "PERSISTENT", "IN_PROCESS", "SUBPROCESS"]
