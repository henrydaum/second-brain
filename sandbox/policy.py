"""The kernel policy function — the one place a security level is decided.

``classify`` is the whole authorization surface. Sandboxed code never
participates: a plugin declares *intent* by making a Request, and the kernel
alone decides what that Request is permitted to affect.

The level is computed from three inputs, per the security contract:

- **what** — the Request's type and arguments
- **who** — the chain of provenance, rooted in what caused the work
- **where** — the destination: a path, a host, a table, a user

Two levels. SAFE executes immediately; UNSAFE goes to the user for approval.
Nothing here is a property *of* a Request type — ``fs.write`` to scratch is
safe, the same Request aimed at ``main.pyw`` is not.

**One rule runs through the whole catalogue: widening capability is unsafe,
narrowing it is safe.** Adding a tool, injecting prompt text, enabling a job,
or loading a service changes what the agent may do next; the reverse never
does. Where a family had no obvious answer, that rule gave one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .guest.requests import (AGENT_SCHEDULE, CONFIG_READ, CONV_DELETE,
                             ENV_READ, FS_DELETE, FS_MOVE, FS_TEMP, FS_WRITE,
                             NET_HTTP, PROC_RUN, SESSION_ADD_PROMPT,
                             SESSION_ADD_TOOL, UI_ASK, Request)
from .secrets import is_secret

SAFE = "safe"
UNSAFE = "unsafe"


@dataclass(frozen=True)
class Decision:
    """A classification, with the reason that produced it.

    The reason is not decoration: it is what the approval dialog shows the
    user, and what lands in the ledger row.
    """
    level: str
    reason: str = ""

    @property
    def safe(self) -> bool:
        """Whether this may execute without asking."""
        return self.level == SAFE


@dataclass(frozen=True)
class Chain:
    """The chain of provenance for a running piece of code.

    ``root`` is what *caused* the work — a user turn, a cron job, a subagent,
    a frontend event. It is the part that makes an approval dialog answerable:
    "cron:nightly_index -> task_index -> net.http" tells the user everything;
    "task_index -> net.http" tells them nothing.

    ``links`` is who called whom, innermost last. The kernel maintains this as
    its own stack, so plugins can neither report nor misstate their identity.
    """
    root: str = "user"
    links: tuple = ()

    def push(self, name: str) -> "Chain":
        """Descend into a nested call."""
        return Chain(root=self.root, links=self.links + (name,))

    @property
    def depth(self) -> int:
        """How deep the call stack currently is."""
        return len(self.links)

    @property
    def attended(self) -> bool:
        """Whether a human is plausibly present to answer a dialog."""
        return self.root == "user"

    @property
    def cyclic(self) -> bool:
        """Whether something in this chain is calling itself.

        A tool that reaches itself, directly or through two others, is nearly
        always a careless mistake rather than intent — and left alone it
        recurses until something breaks. The chain is the call stack, so
        detecting this costs a set comparison.
        """
        return len(set(self.links)) != len(self.links)

    def render(self) -> str:
        """Human-readable chain for dialogs and ledger rows."""
        return " -> ".join((self.root,) + self.links)


# ──────────────────────────────────────────────────────────────────────
# Path scoping.
# ──────────────────────────────────────────────────────────────────────

def _scratch_roots() -> list:
    """Directories a plugin may write to without asking.

    Resolved from the kernel's own path constants rather than the
    environment, so this agrees with where the kernel actually keeps things.
    Falls back to the system temp directory when the kernel is absent, which
    is the case in tests and in a bare container.
    """
    import tempfile

    roots = [Path(tempfile.gettempdir())]
    try:
        from paths import ATTACHMENT_CACHE, SCRATCH_DIR
        roots.extend([Path(SCRATCH_DIR), Path(ATTACHMENT_CACHE)])
    except Exception:
        pass
    return [r for r in roots if r]


def _within(path, roots) -> bool:
    """Whether a path resolves inside any of roots.

    A missing path is not inside anything, which sends it down the approval
    path rather than raising inside the policy function.
    """
    if not path:
        return False
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError, TypeError):
        return False
    for root in roots:
        try:
            resolved.relative_to(Path(root).resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


# ──────────────────────────────────────────────────────────────────────
# What each family costs.
# ──────────────────────────────────────────────────────────────────────

MAX_DEPTH = 8

# Requests that change state the kernel owns. Safe to *read*, never safe to
# perform without asking, whatever the arguments.
ALWAYS_UNSAFE = {
    # Widening what the agent may do next.
    "session.add_tool", "session.add_prompt_extra", "session.set_profile",
    # The literal subject of the LibOS quote: the agent extending itself.
    "plugin.register", "plugin.unregister", "plugin.reload",
    "plugin.install", "plugin.uninstall",
    "service.load", "service.unload",
    # Recurring unattended work.
    "cron.create", "cron.update", "cron.remove",
    "agent.schedule",
    # Identity and settings.
    "config.write", "user.write", "user.list",
    # Destructive.
    "conv.delete", "fs.delete",
}

# Requests that narrow capability, or only ever affect this execution.
ALWAYS_SAFE = {
    "self.respond", "fs.temp", "fs.read", "fs.list", "fs.search",
    "db.query", "conv.read", "conv.list", "conv.create", "conv.append",
    "conv.set_title", "conv.set_category",
    "session.get", "session.list", "session.push", "session.cancel",
    "session.state_get", "session.state_set", "session.remove_tool",
    "session.remove_prompt_extra",
    "ui.approve", "ui.render",
    "user.read",
    "plugin.list", "plugin.describe", "service.list", "service.call",
    "tool.list", "tool.call", "command.list", "command.call",
    "cron.enable",
    "agent.complete", "agent.spawn",
    "cron.list", "cron.get",
    "event.emit", "event.request",
    "task.enqueue", "task.status", "task.output",
    "file.register", "file.list",
    "parse.file", "parse.modality",
    "ledger.record", "ledger.read",
    "db.write", "db.define",
}


def classify(request: Request, chain: Chain) -> Decision:
    """Decide whether a Request may execute without asking the user.

    This is the complete authorization surface for sandboxed code.
    """
    # Runaway nesting is stupidity, not malice, and it is caught here rather
    # than by exhausting the machine. Both checks read the chain, which is the
    # second job it does: it is the call stack, so it is also the cycle
    # detector.
    if chain.depth > MAX_DEPTH:
        return Decision(UNSAFE, f"call chain deeper than {MAX_DEPTH}")
    if chain.cyclic:
        return Decision(UNSAFE, f"call cycle: {chain.render()}")

    kind = request.type
    args = request.args

    # ── egress: checked regardless of verb ────────────────────────
    if kind == NET_HTTP:
        # A GET with data in the query string is exfiltration exactly as much
        # as a POST body is. This is the single control that makes generous
        # filesystem and database reads safe, so it gets no exceptions.
        return Decision(UNSAFE, f"outbound request to {args.get('url', '')}")

    # ── the shell ─────────────────────────────────────────────────
    if kind == PROC_RUN:
        argv = args.get("argv")
        shown = argv if isinstance(argv, str) else " ".join(map(str, argv or []))
        return Decision(UNSAFE, f"shell command: {shown[:200]}")

    # ── filesystem writes depend entirely on where ────────────────
    if kind == FS_WRITE:
        if _within(args.get("path"), _scratch_roots()):
            return Decision(SAFE, "write to scratch")
        return Decision(UNSAFE, f"write to {args.get('path')}")

    if kind == FS_MOVE:
        roots = _scratch_roots()
        if _within(args.get("src"), roots) and _within(args.get("dst"), roots):
            return Decision(SAFE, "move within scratch")
        return Decision(UNSAFE,
                        f"move {args.get('src')} to {args.get('dst')}")

    if kind == FS_DELETE:
        if _within(args.get("path"), _scratch_roots()):
            return Decision(SAFE, "delete from scratch")
        return Decision(UNSAFE, f"delete {args.get('path')}")

    if kind == FS_TEMP:
        return Decision(SAFE, "scratch space")

    # ── secrets are readable as handles, so reading them is safe ──
    if kind in (CONFIG_READ, ENV_READ):
        name = args.get("key") or args.get("name") or ""
        if is_secret(name):
            return Decision(SAFE, f"{name} (returned as a handle)")
        return Decision(SAFE, "read setting")

    # ── asking a human is only possible when one is there ─────────
    if kind == UI_ASK:
        if not chain.attended:
            return Decision(UNSAFE,
                            "nobody is present to answer this question")
        return Decision(SAFE, "asking the user")

    # ── unattended work gets less benefit of the doubt ────────────
    if kind in (SESSION_ADD_TOOL, SESSION_ADD_PROMPT, AGENT_SCHEDULE,
                CONV_DELETE):
        return Decision(UNSAFE, f"{kind} ({chain.render()})")

    if kind in ALWAYS_UNSAFE:
        return Decision(UNSAFE, f"{kind} changes what the system can do")

    if kind in ALWAYS_SAFE:
        return Decision(SAFE, kind)

    # Anything not classified is refused. A new Request type is unsafe until
    # somebody decides otherwise, which is the right direction to fail.
    return Decision(UNSAFE, f"unclassified request: {kind}")
