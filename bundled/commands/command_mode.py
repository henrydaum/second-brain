"""Slash command plugin for `/mode` — how this conversation answers approvals.

Three modes, and they are the three answers a person can give to "may this
run?" before being asked it: ``lockdown`` (no), ``ask`` (ask me, the default),
``yolo`` (yes). The mode is what stands in for you at the approval dialog, and
nothing else — so what it can reach is exactly what would have interrupted you.

Kernel rather than a store package, for the reason ``/permissions`` is: a
safety surface that stops working when a package is uninstalled is worse than
none. It sits in the same category for the same reason — between them they are
the two commands that answer "what is allowed here", one standing on
destinations and one on time.

Ephemeral by design, and per conversation: the mode is not persisted anywhere,
so a restart returns to ``ask``, and it is scoped to the conversation it was
set in, so ``/new`` does too. A forgotten ``yolo`` that outlives the work it
was for is the one failure this must not have.
"""

from guest.bases import BaseCommand
from guest.forms import FormStep

LOCKDOWN = "lockdown"
ASK = "ask"
YOLO = "yolo"
MODES = (LOCKDOWN, ASK, YOLO)

#: What each mode does, in the words the dialog it replaces would have used.
BLURBS = {
    LOCKDOWN: "Refuse anything that needs approval, without asking.",
    ASK: "Ask you about anything that needs approval.",
    YOLO: "Approve anything that needs approval, without asking.",
}

#: Button text. Each carries its own blurb, which is why the picker prompt
#: does not also print the table — see ``_picker_prompt``.
LABELS = {
    LOCKDOWN: "Lockdown - refuse without asking",
    ASK: "Ask - the default, ask me each time",
    YOLO: "YOLO - approve without asking",
}


class ModeCommand(BaseCommand):
    """Set how this conversation answers permission requests."""

    name = "mode"
    description = "Set how this conversation answers permission requests"
    category = "Capabilities"
    # Only the loosening direction is gated. Tightening to lockdown narrows
    # and needs no dialog; switching back to ask is what lifts lockdown and
    # must never be gated by the thing it lifts. ``yolo`` is the one act here
    # that hands away a decision, so it is the one that gets stated and
    # answered up front — the state machine's path, not a mid-run Request.
    # A literal, not ``(YOLO,)``. Declarations are read by AST without
    # importing the file, so a module constant here reads as nothing at all
    # and the gate silently does not exist — which is exactly what
    # ``tests/test_command_approval_declarations.py`` catches.
    approval_actions = ("yolo",)
    approval_actor_id = "user"
    requests = ["session.get", "session.set_mode"]

    agent_prompt = (
        "The user controls how this conversation answers permission requests "
        "with /mode (lockdown, ask, yolo). You are told which mode is active "
        "when it is not the default. If something of yours is refused because "
        "of the mode, say so and name /mode as the fix — do not switch it "
        "yourself and do not look for another route to the same effect."
    )

    def form(self, sdk, args):
        """One step: all three modes, as buttons, with the current one marked.

        **Required**, and that word is the whole of whether this command has a
        UI. ``_missing`` keeps a step only when ``required or
        prompt_when_missing``, so an optional one is never missing, the form
        never suspends, and ``run`` is reached immediately with no action —
        which renders the text fallback and no buttons at all. It fails
        silently in both directions: nothing errors, and the command still
        prints something plausible.

        All three are offered rather than the two you are not in. The current
        one answers "already in this mode", which costs a line; hiding it would
        move the other two buttons depending on where you are, which costs the
        ability to click the same place twice.

        Named ``action`` because ``approval_actions`` is matched against
        ``args["action"]`` and nothing else, which is also what makes
        ``/mode yolo`` parse positionally.
        """
        if args.get("action"):
            return []
        current = _current(sdk)
        return [FormStep(
            "action", _picker_prompt(current), True, enum=list(MODES),
            enum_labels=[_label(mode, current) for mode in MODES], columns=1)]

    def run(self, sdk, args):
        """Switch to the mode named, or describe where we are."""
        action = (args.get("action") or "").strip().lower()
        if not action:
            return _overview(sdk, _current(sdk))
        if action not in MODES:
            return f"Unknown mode: {action}. Pick one of: " + ", ".join(MODES)

        current = _current(sdk)
        if action == current:
            return f"Already in **{action}** mode. {BLURBS[action]}"
        try:
            sdk.session.set_mode(action)
        except sdk.Failed as exc:
            return f"Could not switch mode: {exc.error}"
        return _confirmation(action, current)


def _current(sdk) -> str:
    """The mode in force, or ``ask`` when there is nothing to ask.

    A session that cannot be read is reported as the default rather than as an
    error: the reader is the runtime's, so a missing one means no conversation
    is in any mode, which is what ``ask`` says.
    """
    try:
        session = sdk.session.get()
    except sdk.Failed:
        return ASK
    return ((session or {}).get("mode") or ASK)


def _label(mode: str, current: str) -> str:
    """Button text, marking where we are.

    Worth the four characters: the buttons are what you are looking at, so a
    picker whose options give no sign which one is already in force makes you
    read the prompt to find out, every time.
    """
    return LABELS[mode] + (" (current)" if mode == current else "")


def _picker_prompt(current: str) -> str:
    """What the buttons cannot say, and deliberately nothing they can.

    No table of what each mode does: the three labels beside this each carry
    their own blurb, and a prompt repeating them is the approval dialog's old
    "Run shell commands / Run shell commands" bug in a new place. What is left
    is the part no button has room for — where we are, and the limit that
    makes the permissive option answerable.
    """
    return (
        f"This conversation is in **{current}** mode.\n\n"
        "A mode answers the approval dialogs you would have seen, and nothing "
        "else: it grants no more than you could have approved yourself, and "
        "never applies to work nobody is watching."
    )


def _overview(sdk, current: str) -> str:
    """The text fallback, for a surface with no buttons on it.

    Reached when the form was skipped — a cancelled picker, or a caller that
    supplied no action. It *does* print the table the picker leaves out, which
    is not a second voice saying the same thing: there are no labels here to
    carry the blurbs, so this is the one place they have to be written down.
    """
    rows = [
        [f"**{mode}**" if mode == current else mode, BLURBS[mode]]
        for mode in MODES
    ]
    return "\n\n".join([
        f"This conversation is in **{current}** mode.",
        sdk.md.table(["Mode", "What it does"], rows),
        _always(),
        "Switch with `/mode lockdown`, `/mode ask` or `/mode yolo`.",
    ])


def _confirmation(mode: str, previous: str) -> str:
    """What changed, and — for yolo — what did not.

    The permissive answer is the one that has to say what it does *not* cover,
    for the same reason ``/permissions`` ends with its own footnote: a view
    that states only the grant leaves "what can it reach" wrong in the
    direction that matters.
    """
    line = f"Switched from **{previous}** to **{mode}**. {BLURBS[mode]}"
    if mode == YOLO:
        return line + "\n\n" + _always()
    if mode == LOCKDOWN:
        return line + (
            "\n\nReading, searching and questions to you still work — only "
            "things that would have raised a dialog are refused. `/mode ask` "
            "lifts it."
        )
    return line


def _always() -> str:
    """What no mode changes. The limits of the grant, stated where it is made."""
    return (
        "**YOLO is not root.** It answers the dialogs you would have seen, "
        "and nothing else. Reaching another user's data, editing Second "
        "Brain's own program files, and anything the kernel refuses outright "
        "stay refused — those never asked you in the first place, so there is "
        "no answer to stand in for.\n\n"
        "**It does not reach unattended work.** A scheduled job, a background "
        "service or a subagent is refused anything consequential whatever "
        "this conversation is set to. Every conversation starts at `ask`, and "
        "this one returns there when it changes or when the app restarts."
    )
