"""Boxes — which files share an execution context.

A *box* is one running sandbox: one process (or one thread), one memory space,
one lifetime. Files in the same box can import each other and share module
state, exactly as they do today. Files in different boxes cannot reach each
other at all — they have to go through the gate, via ``service.call`` or
``tool.call``, and pay a Request for the privilege.

That is the whole grouping rule, and it falls out of what a box *is* rather
than being a separate policy.

**Why grouping is needed.** A plugin and its helpers are one unit: the helpers
exist to be imported. A service and the tools that front it often share
in-memory state. Forcing each file into its own container would turn an
attribute access into an IPC round trip and make ordinary code unwritable.

**Membership.** A file declares ``box = "name"`` (module level, or as a class
attribute on a plugin). Undeclared files get a box of their own, named after
the file — so the default is maximum isolation and grouping is a deliberate
act.

**Lifetime.** An ephemeral box is torn down when its work finishes; a
persistent one stays open and keeps its state, which is what a loaded service
needs. Persistence is a property of the box, not of any one file in it.

**Resolution is most-restrictive-wins.** A box's settings come from every
member, and the tightest value carries. Joining a box can therefore only ever
narrow what the joiner may do — a careless file cannot loosen a box by moving
into it, which is the LibOS invariant applied to grouping.

Pure declaration and arithmetic. Nothing here performs an effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Isolation, loosest first. The index is the restrictiveness ordering.
IN_PROCESS = "in_process"
SUBPROCESS = "subprocess"
ISOLATION_ORDER = (IN_PROCESS, SUBPROCESS)

# Lifetime.
EPHEMERAL = "ephemeral"
PERSISTENT = "persistent"

DEFAULT_ISOLATION = IN_PROCESS
DEFAULT_LIFETIME = EPHEMERAL


@dataclass(frozen=True)
class Membership:
    """One file's declared wishes about the box it runs in.

    Every field is *intent*. The kernel resolves and clamps; nothing here
    grants anything.
    """
    source: str
    box: str = ""
    isolation: str = ""
    lifetime: str = ""
    timeout: float = 0.0
    memory_mb: int = 0

    @property
    def box_name(self) -> str:
        """The box this file joins — its own, unless it says otherwise."""
        return self.box or self.source


@dataclass(frozen=True)
class BoxSpec:
    """A resolved execution context: what actually gets run, and how."""
    name: str
    members: tuple = field(default_factory=tuple)
    isolation: str = DEFAULT_ISOLATION
    lifetime: str = DEFAULT_LIFETIME
    timeout: float = 0.0
    memory_mb: int = 0

    @property
    def persistent(self) -> bool:
        """Whether this box stays open and keeps its state."""
        return self.lifetime == PERSISTENT

    @property
    def isolated(self) -> bool:
        """Whether this box runs behind a process boundary."""
        return self.isolation == SUBPROCESS


def _tightest_isolation(values) -> str:
    """The most restrictive isolation any member asked for."""
    ranks = [ISOLATION_ORDER.index(v) for v in values if v in ISOLATION_ORDER]
    return ISOLATION_ORDER[max(ranks)] if ranks else DEFAULT_ISOLATION


def _tightest_limit(values) -> float:
    """The smallest positive limit any member asked for; 0 means unset."""
    positive = [v for v in values if v and v > 0]
    return min(positive) if positive else 0.0


def resolve(memberships) -> dict:
    """Group memberships into boxes, tightest declaration winning.

    Returns ``{box_name: BoxSpec}``. A file that declared nothing gets its own
    box with kernel defaults, so isolation is what you get by saying nothing.
    """
    grouped: dict = {}
    for m in memberships:
        grouped.setdefault(m.box_name, []).append(m)

    boxes = {}
    for name, members in grouped.items():
        boxes[name] = BoxSpec(
            name=name,
            members=tuple(sorted(m.source for m in members)),
            isolation=_tightest_isolation(m.isolation for m in members),
            # Any member needing to keep state makes the whole box persistent:
            # a box cannot be half torn down.
            lifetime=(PERSISTENT
                      if any(m.lifetime == PERSISTENT for m in members)
                      else DEFAULT_LIFETIME),
            timeout=_tightest_limit(m.timeout for m in members),
            memory_mb=int(_tightest_limit(m.memory_mb for m in members)),
        )
    return boxes


def same_box(a: Membership, b: Membership) -> bool:
    """Whether two files may import each other.

    The import rule and the isolation rule are the same rule: if two files do
    not share a box, the only way between them is a Request.
    """
    return a.box_name == b.box_name
