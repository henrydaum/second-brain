"""The guest side of the sandbox — everything that runs *inside* the boundary.

This package is self-contained and stdlib-only. It knows the Request
vocabulary, the wire format, the SDK, and the plugin contracts, and it knows
nothing about the gate, the policy function, or the handlers. That is the
point: guest code physically cannot reach the kernel's decision-making,
because it is not importable from here.

It is also the shippable unit. A container image copies this directory and
nothing else, and runs ``python -m guest.child``. Because the modules are
relative-imported within the package, ``guest`` works equally as a top-level
package (in an image) or as ``sandbox.guest`` (in the repo).

**A base class is optional.** Plugins subclass one so the kernel can register
and schedule them, but the sandbox runs any file with functions that take
``sdk`` — an agent's scratch computation, a helper module, a one-off script.
Nothing here is required to compute.

The host imports *from* here — the Request vocabulary and wire format are
shared. Nothing here imports back.
"""

from .bases import (COMMAND, FRONTEND, SERVICE, TASK, TOOL, BaseCommand,
                    BaseFrontend, BasePlugin, BaseService, BaseTask, BaseTool)
from .box import (EPHEMERAL, IN_PROCESS, PERSISTENT, SUBPROCESS, BoxSpec,
                  Membership, resolve, same_box)
from .requests import Request, Result
from .sdk import SDK

__all__ = [
    "SDK", "Request", "Result",
    "BasePlugin", "BaseTool", "BaseTask", "BaseService", "BaseCommand",
    "BaseFrontend", "TOOL", "TASK", "SERVICE", "COMMAND", "FRONTEND",
    "BoxSpec", "Membership", "resolve", "same_box",
    "EPHEMERAL", "PERSISTENT", "IN_PROCESS", "SUBPROCESS",
]
