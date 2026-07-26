"""Asking the user — the sandbox's approver, wired to the kernel's doorway.

The policy function decides *whether* to ask. This decides *how*, and it
deliberately reuses machinery the kernel already has rather than growing a
second permission system beside it:

1. **``vet_permission`` hooks first.** The kernel already has a doorway where
   policy plugins stand — plan mode refuses everything there, and a trust
   plugin could allow. A gate that has an opinion wins; every gate abstaining
   falls through.
2. **The user's trusted list.** ``skip_permissions`` is the user-scoped list
   of things they have already decided about, and it is consulted for the
   *root* of the chain rather than the leaf, because that is what they
   actually approved.
3. **The dialog.** ``runtime.request_input`` renders it, exactly as tool
   approvals are rendered today.
4. **Nobody home.** An unattended session refuses rather than blocking, which
   is the kernel's own default when every gate abstains at the
   ``unattended_call`` stage.

The dialog is built from the chain, not the leaf. "service_web wants to make
an HTTP request" is a question nobody can answer; "the summarize tool you just
ran wants to POST to example.com" can be answered in a second. That is the
whole reason provenance exists, and this is the only place it is shown.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("Sandbox")

DIALOG_TIMEOUT = 300.0
STAGE_APPROVAL = "approval"
STAGE_UNATTENDED = "unattended_call"


def describe(chain, request, decision) -> tuple:
    """Build the title and body a person is shown.

    Returns ``(title, body)``. The body leads with what will happen, then
    where it came from — a chain the user can trace back to something they
    did, or to the cron job that did it for them.
    """
    title = "Sandboxed code requests approval"
    action = _action_line(request)
    body = (
        f"{action}\n\n"
        f"**From:** `{chain.render()}`\n"
        f"**Why it needs asking:** {decision.reason}"
    )
    return title, body


def _action_line(request) -> str:
    """One line naming the effect, in the user's terms rather than ours."""
    args = request.args
    kind = request.type
    if kind == "net.http":
        method = (args.get("method") or "GET").upper()
        return f"Send a **{method}** request to `{args.get('url', '?')}`"
    if kind == "proc.run":
        argv = args.get("argv")
        shown = argv if isinstance(argv, str) else " ".join(map(str, argv or []))
        return f"Run a shell command:\n```\n{shown}\n```"
    if kind.startswith("fs."):
        target = args.get("path") or args.get("src") or "?"
        verb = {"fs.write": "Write to", "fs.delete": "Delete",
                "fs.move": "Move"}.get(kind, "Touch")
        return f"{verb} `{target}`"
    if kind == "config.write":
        return f"Change the setting `{args.get('key')}`"
    if kind.startswith("plugin."):
        return f"`{kind}` — change what this system can do (`{args.get('name') or args.get('stem') or ''}`)"
    if kind.startswith("cron.") or kind == "agent.schedule":
        return f"`{kind}` — create or change unattended work"
    return f"`{kind}`"


def _trusted(runtime, session_key, chain) -> bool:
    """Whether the user has already decided about the thing that caused this.

    Consulted for the chain's *root* rather than its leaf: a user who trusted
    a tool trusted what that tool does, including through a service it calls.
    """
    if runtime is None or not session_key:
        return False
    reader = getattr(runtime, "user_setting", None)
    if reader is None:
        return False
    try:
        trusted = reader(session_key, "skip_permissions") or []
    except Exception:
        return False
    names = set(chain.links) | {chain.root}
    return bool(names & set(trusted))


def build_approver(runtime, session_key=None, timeout: float = DIALOG_TIMEOUT):
    """Build the ``approve`` callable an :class:`Interpreter` takes.

    Returns ``callable(chain, request, decision) -> bool``. Absent a runtime
    there is nobody to ask, so everything unsafe is refused — the same safe
    default the kernel uses when every permission hook abstains.
    """

    def approve(chain, request, decision) -> bool:
        """Decide whether one unsafe Request may proceed."""
        if runtime is None:
            return False

        key = session_key or getattr(runtime, "active_session_key", None)
        session = (getattr(runtime, "sessions", None) or {}).get(key)
        attended = _attended(runtime, key, chain)

        # 1. Policy plugins get first say, at the stage that matches whether
        #    anyone is present to be asked.
        hooks = getattr(runtime, "hooks", None)
        if hooks is not None:
            try:
                verdict = hooks.vet_permission(
                    session, _leaf(chain), _command_text(request),
                    runtime=runtime,
                    stage=STAGE_APPROVAL if attended else STAGE_UNATTENDED,
                    # The doorway was built for tools asking to run commands.
                    # A Request is a different question, so it arrives whole —
                    # typed, classified, and carrying what caused it — rather
                    # than flattened into the command string.
                    origin="request", request=request, chain=chain,
                    decision=decision)
            except Exception:
                logger.exception("permission gate raised; treating as abstain")
                verdict = None
            if verdict is not None:
                return bool(verdict.allow)

        # 2. Things the user already decided about.
        if _trusted(runtime, key, chain):
            return True

        # 3. Nobody to ask means no.
        if not attended:
            logger.info("refusing %s from unattended %s",
                        request.type, chain.render())
            return False

        # 4. Ask.
        title, body = describe(chain, request, decision)
        try:
            pending = runtime.request_input(key, title, body, type="boolean")
        except Exception:
            logger.exception("could not render an approval dialog")
            return False

        if not pending.wait(timeout=timeout):
            pending.metadata["timed_out"] = True
            try:
                runtime.answer_request(key, pending.id, False)
            except Exception:
                pass
            return False
        if pending.metadata.get("cancelled"):
            return False
        return bool(pending.approved)

    return approve


def _leaf(chain) -> str:
    """The innermost link, which is what the hook contract calls a tool."""
    return chain.links[-1] if chain.links else chain.root


def _command_text(request) -> str:
    """A short rendering of the Request for hooks that match on text."""
    args = request.args
    detail = (args.get("url") or args.get("path") or args.get("key")
              or args.get("name") or "")
    return f"{request.type} {detail}".strip()


def _attended(runtime, session_key, chain) -> bool:
    """Whether a human is present to answer.

    Asks the kernel's single reader when it exists, so a frontend that owns
    its own attendance policy is respected. Falls back to the chain's root,
    which is the sandbox's own notion of what caused the work.
    """
    reader = getattr(runtime, "is_attended", None)
    if reader is not None and session_key:
        try:
            return bool(reader(session_key))
        except Exception:
            pass
    return chain.attended
