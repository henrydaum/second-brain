"""Asking the user — the sandbox's approver, wired to the kernel's doorway.

The policy function decides *whether* to ask. This decides *how*, and it
deliberately reuses machinery the kernel already has rather than growing a
second permission system beside it:

1. **``vet_permission`` hooks first.** The kernel already has a doorway where
   policy plugins stand — plan mode refuses everything there, and a trust
   plugin could allow. A gate that has an opinion wins; every gate abstaining
   falls through.
2. **The dialog.** ``runtime.request_input`` renders it, and its options are
   where a "yes" can be kept — see :mod:`sandbox.options`.
3. **Nobody home.** An unattended session refuses rather than blocking, which
   is the kernel's own default when every gate abstains at the
   ``unattended_call`` stage.

There was a step between 1 and 2: ``skip_permissions``, a user-scoped list of
plugin *names* whose dialogs were auto-approved. It was the only durable
answer the system had, which is why its unit had to be that broad — trusting
``web_search`` trusted every host it would ever reach, forever, through any
service it called. Now that an answer can be kept at the grain the question
was asked at, a whole-plugin bypass is strictly worse than the thing it was
standing in for, so it is gone rather than left as a blunter option.

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

    Returns ``(title, body)``. The body leads with what will happen, then who
    asked, then — only when there is one — the part of the *why* the action
    line cannot carry.

    It used to print ``decision.reason`` unconditionally under "Why it needs
    asking", which said the same thing twice: both lines are built from the
    same arguments by different code, so a shell approval read "Run shell
    commands: `git pull`" and then "run shell command: git pull (in Z:\\...)".
    ``Decision.say`` is the half written for a person; see its docstring.

    The title lost its one constant string for the same reason. "Sandboxed
    code requests approval" was identical for every askable Request type, and
    both shipped frontends print it above the body — a whole line of screen
    that never varied and never told anybody anything.
    """
    # The title is the bare phrase and the body carries only what the phrase
    # cannot: the arguments. Both shipped frontends print the title above the
    # body, so a body that opened with the action line said it twice —
    #
    #     Run shell commands
    #
    #     Run shell commands
    #     ```echo ~/Desktop```
    #
    # which is the duplication this rewrite existed to remove, reintroduced
    # the moment the title stopped being one constant string.
    phrase = phrase_for(request.type)
    title = phrase[:1].upper() + phrase[1:]
    lines = []
    if detail := _detail(request.type, request.args).strip():
        lines.append(detail)
    if asker := describe_asker(chain):
        lines.append(f"Asked by {asker}")
    if say := (decision.say or "").strip():
        lines.append(say)
    return title, "\n\n".join(lines)


def describe_asker(chain) -> str:
    """The plugin that made this Request, or "" when there is none.

    The leaf, and nothing else — because **the root of a chain that reaches
    this function is never worth naming.** Only an attended chain gets a
    dialog at all (step 3 refuses the rest), and the attended roots are
    exactly ``user``, ``user:command``, and a session key. The first two mean
    "you did this", which is what being asked already means. The third names
    the session the dialog is *delivered to* (step 4), so printing it tells
    somebody reading Telegram that they are in Telegram — and it is a
    frontend-built identifier, so printing it raw put a ten-digit number,
    twice, over a question it had nothing to do with.

    Everything a root could interestingly say — a cron schedule, a background
    agent, a service acting on its own — belongs to work nobody is watching,
    which is refused rather than asked. Those clauses were written and then
    deleted, because a dialog cannot render them. The chain is still recorded
    whole in the ledger, which is where unattended provenance is read.
    """
    links = tuple(getattr(chain, "links", ()) or ())
    return links[-1] if links else ""


def _action_line(request) -> str:
    """One line naming the effect, in the user's terms rather than ours.

    **One vocabulary, two levels of detail.** The phrase comes from
    :func:`phrase_for` — the same table the grant dialog reads — and this adds
    the arguments when it has any worth showing. So a capability is described
    the same way whether the user meets it up front ("/update wants to: run
    shell commands") or at the moment it happens ("Run shell commands:
    `git pull`").

    Before, this had its own hand-written branches and fell through to a bare
    dotted name for **69 of the 97** Request types. That was invisible while
    nothing wired an approver, because this dialog never rendered; the moment
    it did, most approvals would have asked the user to authorise
    ``session.add_tool``. The grant table is total and test-enforced, so
    deriving from it makes this total too.
    """
    args = request.args
    kind = request.type
    phrase = phrase_for(kind)
    headline = phrase[:1].upper() + phrase[1:]

    if detail := _detail(kind, args):
        # A detail that opens its own block is appended, not introduced: a
        # trailing "run shell commands: " with the command on the next line
        # reads as a colon somebody forgot to finish.
        joiner = "" if detail.startswith("\n") else ": "
        return f"{headline}{joiner}{detail}"
    return headline


def _detail(kind: str, args: dict) -> str:
    """The concrete part, when the arguments carry one worth showing.

    Returns ``""`` when they do not — a Request with nothing but a session key
    reads better as its phrase alone than as a phrase followed by noise.
    """
    if kind in ("proc.run", "proc.start"):
        # The same renderer the policy's reason and the ledger row use, so the
        # command a person approves is the command that gets recorded.
        from .shell import render_command

        shown = render_command(args)
        return f"\n```\n{shown}\n```" if shown else ""
    if kind == "net.http":
        method = (args.get("method") or "GET").upper()
        return f"**{method}** `{args.get('url', '?')}`"
    if kind.startswith("fs."):
        target = args.get("path") or args.get("src") or ""
        if target and (destination := args.get("dst")):
            # ASCII arrow, like ``Chain.render``: this lands on a Windows
            # console under cp1252, where a unicode arrow raises rather than
            # renders.
            return f"`{target}` -> `{destination}`"
        return f"`{target}`" if target else ""
    if kind == "ui.approve":
        # The plugin's own words for what it is about to do. Its
        # ``justification`` is deliberately not here — that is the *reason*,
        # and the body prints it one line down under "Why it needs asking".
        return f"`{args.get('action') or '?'}`"
    if kind == "secret.reveal":
        return (f"`{args.get('name')}` - sandboxed code will then hold it "
                f"directly")
    if kind in ("config.read", "config.write"):
        return f"`{args.get('key')}`" if args.get("key") else ""
    if kind == "session.set_mode":
        # The mode *and* what it means, because the word alone is the one
        # thing a person cannot check: "yolo" tells you it is permissive and
        # nothing about what stops being asked.
        from runtime.security_modes import MODE_BLURBS, security_mode

        mode = security_mode(args.get("mode"))
        return f"`{mode}` - {MODE_BLURBS.get(mode, '')}"
    if kind == "agent.schedule":
        # The two things being authorised are *when* it fires and *what it is
        # told to do*, and neither was shown: the generic scan below reaches
        # for ``name``, which defaults to empty, so the dialog said "Schedule
        # unattended work" and nothing else. Approving that is approving a
        # blank cheque.
        return _schedule_detail(
            args.get("cron"), args.get("one_time"), None,
            args.get("title"), args.get("prompt"))
    if kind in ("cron.create", "cron.update"):
        job = args.get("job") or args.get("patch") or {}
        payload = job.get("payload") if isinstance(job, dict) else None
        payload = payload if isinstance(payload, dict) else {}
        detail = _schedule_detail(
            job.get("cron"), job.get("one_time"), job.get("run_at"),
            payload.get("title"), payload.get("prompt"))
        if name := args.get("name"):
            return f"`{name}`{detail}"
        return detail
    # Everything else names its subject the same way, so a new Request gets
    # a useful line without a branch being added for it.
    for field in ("name", "stem", "package_id", "tool", "key", "channel",
                  "id", "path"):
        if value := args.get(field):
            return f"`{value}`"
    return ""


PROMPT_PREVIEW_CHARS = 600


def _schedule_detail(cron, one_time, run_at, title, prompt) -> str:
    """When it fires and what it is told to do, for a person to check.

    Scheduling is the one grant whose whole substance is in its arguments: a
    cron expression is not readable at a glance and the instructions the agent
    will follow unattended are the only thing worth reading twice. Both are
    rendered here rather than by the timekeeper, because this runs on the gate
    and calling into a guest service mid-classification is the wrong
    direction — ``cron_descriptor`` is an SDK package the app already has.
    """
    lines = []
    if when := _when(cron, one_time, run_at):
        lines.append(f"\n**When:** {when}")
    if title := (title or "").strip():
        lines.append(f"\n**Title:** {title}")
    if prompt := (prompt or "").strip():
        shown = prompt[:PROMPT_PREVIEW_CHARS]
        if len(prompt) > PROMPT_PREVIEW_CHARS:
            shown += "\n..."
        lines.append(f"\n**It will be told:**\n```\n{shown}\n```")
    return "".join(lines)


def _when(cron, one_time, run_at) -> str:
    """A schedule in a person's words, falling back to the raw expression."""
    if run_at:
        return f"once at {run_at}"
    cron = (cron or "").strip()
    if not cron:
        return ""
    english = _cron_to_text(cron)
    if english == cron:
        # The description fell back to the expression itself — printing it
        # twice tells the user nothing and reads like a bug.
        return f"once, at the next `{cron}`" if one_time else f"`{cron}`"
    if one_time:
        return f"once, at the next `{cron}` ({english})"
    return f"{english} (`{cron}`)"


def _cron_to_text(cron: str) -> str:
    """Describe a cron expression in English.

    The locale is pinned rather than inherited. Left to itself the library
    follows the machine's, so a Spanish-locale desktop rendered an English
    dialog with a Spanish schedule in the middle of it — and the accented
    output cannot be printed to the REPL's cp1252 console at all.

    Guarded twice over besides: the package may be absent from a minimal
    install, and it raises on an expression the user is about to be shown
    *because* it might be wrong. Either way the raw cron is still worth
    printing.
    """
    try:
        from cron_descriptor import ExpressionDescriptor

        text = ExpressionDescriptor(cron, locale_code="en_US").get_description()
        return text.encode("ascii", "ignore").decode() or cron
    except Exception:
        return cron


# ──────────────────────────────────────────────────────────────────────
# Grants: the type-level question, asked once before a command runs.
# ──────────────────────────────────────────────────────────────────────

# What a Request type means to a person, with no arguments to lean on.
# Deliberately coarser than ``_action_line``: that renders one concrete
# effect at the moment it happens, this summarises a whole capability class
# before anything has run. Phrases are plural and verb-first so they read as
# a list under "wants to:".
GRANT_PHRASES = {
    "app.stop": "shut Second Brain down or restart it",
    # Distinct from the ``ui`` family's "ask you questions": this one is the
    # plugin stopping to get permission, not collecting a value it needs.
    "ui.approve": "act only with your approval",
    "proc.run": "run shell commands",
    "proc.start": "start background processes that keep running",
    # The three that only ever speak about a process already started. They are
    # safe, so they reach a dialog only as part of a command's declared grant —
    # where saying so is still worth a line.
    "proc.status": "check on background processes",
    "proc.stop": "stop background processes",
    "proc.list": "list background processes",
    # Deliberately not phrased as "run code". Everything a script does comes
    # back through the gate on its own, so what is being granted is much
    # narrower than the words "run code" would have a person imagine — and a
    # dialog that overstates a grant erodes trust in the dialog exactly as fast
    # as one that understates it erodes safety.
    "script.run": "run its own sandboxed scripts",
    "net.http": "make network requests",
    "secret.reveal": "read your credentials in plaintext",
    "config.write": "change settings",
    "fs.write": "write files", "fs.write_bytes": "write files",
    "fs.delete": "delete files", "fs.move": "move files",
    "fs.read": "read files", "fs.read_bytes": "read files",
    "fs.list": "list folders", "fs.search": "search files",
    "env.read": "read environment variables",
    "paths.get": "look up application folders",
    "db.write": "write to the database", "db.query": "read the database",
    "db.define": "create database tables",
    "conv.delete": "delete conversations",
    "agent.schedule": "schedule unattended work",
    "agent.spawn": "start a subagent",
    "agent.collect": "wait for subagents it started",
    "agent.stop": "stop a subagent it started",
    "user.write": "change user accounts",
    # Read-only members of families whose fallback phrase is a write. Without
    # these, declaring ``plugin.list`` would be announced as "install, remove
    # or reload plugins" — overstating the grant, which erodes trust in the
    # dialog exactly as fast as understating it erodes safety.
    "plugin.list": "see what plugins are installed",
    "plugin.describe": "see what plugins are installed",
    "plugin.validate": "check plugin code for problems",
    "cron.list": "see scheduled jobs", "cron.get": "see scheduled jobs",
    "tool.list": "see what tools exist",
    "command.list": "see what commands exist",
    "service.list": "see what services exist",
    "task.list": "see background work", "task.status": "see background work",
    "task.graph": "see background work",
    # The write half of the family, split from the reads for the same reason
    # ``plugin.*`` and ``cron.*`` were: one phrase for both would either
    # overstate a listing or understate a reset.
    "task.reset": "re-run work already done",
    "task.pause": "pause or resume background work",
    "task.trigger": "start background work now",
    "conv.list": "list conversations", "conv.read": "read conversations",
    "user.read": "read user accounts", "user.list": "read user accounts",
    "config.read": "read settings",
    # The write half of those same families, split for the same reason in the
    # other direction. The fallback lumps install/remove/reload together, so
    # an *update* was announced as "install, remove or reload plugins" — which
    # names a removal that is not going to happen. These matter more now that
    # the phrase is also what the execution-time dialog leads with.
    "plugin.install": "install a package",
    "plugin.uninstall": "remove a package",
    "plugin.update": "update installed packages",
    "plugin.reload": "reload a plugin",
    "plugin.register": "add a plugin", "plugin.unregister": "remove a plugin",
    "service.load": "start a service", "service.unload": "stop a service",
    # Widening what the agent may do next is the thing the policy singles out
    # as always unsafe; "change this session" was too mild to answer.
    "session.add_tool": "give the agent another tool",
    "session.add_prompt_extra": "add instructions to the agent's prompt",
    "session.remove_tool": "take a tool away from the agent",
    "session.remove_prompt_extra": "remove instructions from the prompt",
    "session.add_attachment": "show the agent a file",
    # Named for its effect rather than its mechanism. "Change the security
    # mode" describes a setting; what is actually being granted is the power
    # to stop the user being consulted, and only the second is answerable.
    "session.set_mode": "decide for you whether future actions get approved",
    # Machinery. Neither is something a plugin declares, but the table is
    # total so that a new Request cannot quietly render as a dotted name.
    "agent.complete": "finish the agent's turn",
    "self.respond": "answer its own request",
}

# Families that share one phrase when no specific entry matched. Keeps a new
# Request from rendering as a bare dotted name in front of a user.
GRANT_FAMILIES = {
    "plugin": "install, remove or reload plugins",
    "cron": "create or change scheduled jobs",
    "conv": "read and change conversations",
    "session": "change this session",
    "service": "call other services",
    "tool": "call other tools",
    "command": "run other commands",
    "task": "queue background work",
    "ledger": "read the action log",
    "user": "read user accounts",
    "ui": "ask you questions",
    "frontend": "act as a frontend",
    "console": "use the terminal",
    "event": "publish events",
    "llm": "talk to the model",
    "parse": "parse files",
    "file": "register files",
    "db": "use the database",
    "fs": "use the filesystem",
}


def phrase_for(kind: str) -> str:
    """One human phrase for a Request *type*, arguments unknown."""
    if kind in GRANT_PHRASES:
        return GRANT_PHRASES[kind]
    family = kind.split(".", 1)[0]
    return GRANT_FAMILIES.get(family, kind)


def describe_grant(name: str, requests) -> str:
    """The prompt for approving a command, naming what the yes covers.

    A single approval authorizes exactly the Request types a command
    declared, so this is the sentence that makes the grant answerable: the
    user is told the scope rather than just the name. Ordered by how much
    the capability is worth thinking about, not alphabetically — a person
    skimming should meet the shell and the network first.

    Falls back to the bare question when a command declares nothing — which
    is a command that performs no consequential effect, or one whose author
    forgot to list them. ``tests/test_command_approval_declarations.py``
    catches the second.
    """
    seen, phrases = set(), []
    for kind in sorted(set(requests or ()), key=_grant_rank):
        phrase = phrase_for(kind)
        if phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    if not phrases:
        return f"Approve /{name}?"
    return f"/{name} wants to:\n" + "\n".join(f"  - {p}" for p in phrases)


# Ordering for the list above. The two that carry the most consequence lead,
# then everything that changes state, then reads.
_GRANT_ORDER = ("proc.run", "proc.start", "net.http", "secret.reveal",
                "app.stop")


def _grant_rank(kind: str) -> tuple:
    """Sort key: consequential first, then writes, then reads, then name."""
    if kind in _GRANT_ORDER:
        return (0, _GRANT_ORDER.index(kind), kind)
    verb = kind.split(".", 1)[-1]
    writes = verb.startswith(("write", "delete", "move", "install",
                             "uninstall", "create", "register", "set_",
                             "add_", "remove", "update", "spawn", "schedule",
                             # ``script.run``. ``proc.run`` never reaches here
                             # — it leads the explicit order above — so this
                             # only ever means the contained kind, which still
                             # belongs above the reads.
                             "run"))
    return (1 if writes else 2, 0, kind)


def build_approver(runtime, session_key=None, timeout: float = DIALOG_TIMEOUT):
    """Build the ``approve`` callable an :class:`Interpreter` takes.

    Returns ``callable(chain, request, decision) -> bool``. Absent a runtime
    there is nobody to ask, so everything unsafe is refused — the same safe
    default the kernel uses when every permission hook abstains.
    """

    from .options import DENY, chosen, options_for
    from .policy import chain_session

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

        # 2. A plugin asking for its own credential. Configuring a key *for*
        #    a service is the consent; re-asking on every load would be the
        #    approval fatigue this whole design is trying to avoid. Another
        #    plugin asking for that same key still gets a dialog, which is
        #    the part actually worth a question.
        if _owns_secret(chain, request):
            logger.debug("%s revealing its own %s", _leaf(chain),
                         request.args.get("name"))
            return True

        # 3. Nobody to ask means no.
        if not attended:
            logger.info("refusing %s from unattended %s",
                        request.type, chain.render())
            return False

        # The session the work belongs to, which is not always the one this
        # approver was built with. ``build_approver`` is wired with no key, so
        # ``key`` is whatever happens to be active; the chain names the session
        # the agent was actually acting in. Reaching here at all means
        # attendance was resolved against that same session, so both the mode
        # below and the dialog after it read from — and land on — the
        # conversation the person can actually answer for.
        target = chain_session(chain) or key

        # 4. The mode this conversation is in, which is a standing answer to
        #    the dialog step 5 would otherwise draw. Deliberately *here* rather
        #    than at the top:
        #
        #    - after attendance, so ``yolo`` never reaches work nobody is
        #      watching. A cron job or a subagent cannot spend a grant given
        #      for a foreground task, and ``sandbox/policy.py`` rests the
        #      safety of ``agent.spawn`` on exactly that.
        #    - after 1 and 2, so ``lockdown`` answers only what would have
        #      reached the person. It does not countermand a plugin gate that
        #      positively allowed something, and it does not countermand a
        #      service reading the credential it was configured with. Lockdown
        #      means "stop asking me, the answer is no" — not "break the
        #      plugins I already set up", which is a different and much worse
        #      promise to make.
        #
        #    What it cannot touch either way is everything the policy settled
        #    without a dialog: a structural refusal produced no question, so
        #    there is no answer here to stand in for it.
        standing = _standing_answer(runtime, target)
        if standing is not None:
            logger.info("%s %s for %s (mode)",
                        "allowing" if standing else "refusing",
                        request.type, chain.render())
            return standing

        # 5. Ask.
        title, body = describe(chain, request, decision)
        options = options_for(chain, request, decision)
        try:
            # ``type="string"`` and not ``"boolean"``: ``AnswerApproval._coerce``
            # short-circuits into a lenient yes/no parser before it ever looks
            # at the enum, so a boolean request with choices silently ignores
            # them. The options carry their own deny, so nothing is lost.
            pending = runtime.request_input(
                target, title, body, type="string",
                enum=[option.value for option in options],
                enum_labels=[option.label for option in options])
        except Exception:
            logger.exception("could not render an approval dialog")
            return False

        if not pending.wait(timeout=timeout):
            pending.metadata["timed_out"] = True
            try:
                # Deny *by name*. ``False`` fails ``match_enum`` on a string
                # request, and a failed coercion never reaches ``pop_phase`` —
                # so the session would sit in ``approving_request`` forever,
                # where every ordinary keystroke comes back ``invalid_action``,
                # over a dialog that has already expired.
                runtime.answer_request(target, pending.id, DENY.value)
            except Exception:
                pass
            return False
        if pending.metadata.get("cancelled"):
            return False

        # ``.value``, never ``.approved`` — the latter is ``bool(self.value)``,
        # which is True for the string "deny".
        answer = chosen(options, getattr(pending, "value", None))
        if answer is None:
            logger.warning("unrecognised approval answer %r for %s",
                           getattr(pending, "value", None), request.type)
            return False
        if answer.remember is not None:
            try:
                # Run it for a denying option too. Nothing declares one today,
                # and that is exactly why: "Deny forever" has to be a registry
                # entry later, not an edit to this function.
                answer.remember()
            except Exception:
                # Failing to write the grant down must not turn the person's
                # yes into a no. They answered about *this* Request.
                logger.exception("could not remember %s", answer.value)
        return answer.allow

    return approve


def _standing_answer(runtime, session_key):
    """What the conversation's mode answers, or ``None`` to ask the person.

    Asked of the runtime rather than imported, exactly as attendance is: the
    approver already holds the runtime, the mode is per conversation, and a
    reader is the only thing that knows whether a mode set against one
    conversation still applies to the one this Request came from.

    Every failure is ``None``. A runtime too old to have the reader, a missing
    session, a raiser — all of them mean "ask", which is the behaviour this
    feature is invisible against.
    """
    reader = getattr(runtime, "security_mode", None)
    if reader is None or not session_key:
        return None
    try:
        from runtime.security_modes import standing_answer

        return standing_answer(reader(session_key))
    except Exception:
        logger.exception("could not read the security mode; asking")
        return None


def _owns_secret(chain, request) -> bool:
    """Whether the plugin asking for a secret is the one that declared it.

    Plugins declare their ``config_settings``, so the kernel already knows who
    a key belongs to. A service reading the credential it was configured with
    is the setup the user already agreed to; a *different* plugin reaching for
    it is a genuinely different question, and only that one is asked.

    Anything the kernel cannot answer falls through to the dialog.
    """
    if request.type != "secret.reveal":
        return False
    name = request.args.get("name")
    if not name:
        return False
    try:
        from plugins.plugin_discovery import get_setting_plugin_names
        owners = set(get_setting_plugin_names(name))
    except Exception:
        return False
    if not owners:
        return False
    return bool(owners & (set(chain.links) | {chain.root}))


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
    """Whether a human is present to answer *this* work.

    **The chain decides which session is asked; the runtime decides whether
    anybody is at it.** The chain says what caused the work — and, when the
    cause was an agent, *which session* it was acting in, which is the only
    thing separating a foreground turn from a subagent's. The runtime's reader
    answers the rest. Neither half is sufficient alone, and the whole of it is
    :func:`policy.attended_now`.

    It was once the other way round, and the consequence was ugly.
    ``is_attended`` won whenever a session key existed, and ``build_approver``
    is wired with none (``runtime/bootstrap.py``), so the key fell back to
    ``active_session_key`` — trivially attended, by definition. An unsafe
    Request from a scheduled subagent therefore raised a dialog on the
    foreground session, pushing it into ``approving_request``, where the only
    legal actions are answering or cancelling. Every ordinary keystroke came
    back ``invalid_action``, once per firing, about work the person could not
    see and had never started.

    The correction was ``chain.attended`` as a hard floor, which overshot in
    the other direction: that property is True only for a root of ``user``,
    and an agent's own tool call roots at the session key instead. So every
    unsafe Request any tool made was refused without a dialog, in a session
    the person was watching. Handing the chain's *root* to ``is_attended``
    restores both cases at once — a subagent's root is a session key that is
    not the active one, so it still approves nothing.

    ``sandbox/policy.py`` rests the safety of ``agent.spawn`` on that.
    """
    from .policy import attended_now

    return attended_now(chain, runtime, session_key)
