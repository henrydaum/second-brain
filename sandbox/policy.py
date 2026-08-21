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

from .guest import requests as R
from .guest.requests import (AGENT_SCHEDULE, CONFIG_READ, CONV_DELETE,
                             ENV_READ, FS_DELETE, FS_MOVE, FS_TEMP, FS_WRITE,
                             NET_HTTP, PROC_RUN, SESSION_ADD_PROMPT,
                             SESSION_ADD_TOOL, UI_ASK, Request)
from .credentials import is_secret, redact

SAFE = "safe"
UNSAFE = "unsafe"


@dataclass(frozen=True)
class Decision:
    """A classification, with the reason that produced it.

    Two strings, because ``reason`` had three readers with opposite needs: the
    ledger row (greppable, stable, machine-shaped), the refusal handed back to
    a *model* (`interpreter._settle`), and the approval dialog shown to a
    *person*. Written for the first two it read badly to the third — and worse,
    it duplicated the dialog's own action line, which is built from the same
    arguments by different code. "Run shell commands: `git pull`" over "run
    shell command: git pull (in Z:\\...)" is the same sentence twice.

    So ``reason`` keeps the first two readers and no longer reaches the dialog.
    ``say`` is the human half: the part a person needs that the action line
    cannot carry — *why this is being asked at all*. It is deliberately absent
    from most branches, because most branches have nothing to add beyond the
    arguments the dialog already renders, and a line that restates the question
    is worse than no line.
    """
    level: str
    reason: str = ""
    say: str = ""

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

    ``approved`` is what the user said yes to: a *set of Request types*, never
    a boolean, taken from the command's own declared ``requests``. Why the
    declaration is the only decidable scope is in CLAUDE.md, "An approval is
    scoped to what the command declared".

    ``None`` means nothing was approved. An empty set means a command was
    approved but declared no Requests, which grants nothing — deliberately
    different from ``None`` only in intent, not in effect.
    """
    root: str = "user"
    links: tuple = ()
    approved: frozenset | None = None

    def push(self, name: str) -> "Chain":
        """Descend into a nested call.

        The grant is copied down unchanged. It was fixed when the user
        answered, so a callee can only ever spend what the approved command
        was given — it can never widen the set by declaring more itself.
        """
        return Chain(
            root=self.root,
            links=self.links + (name,),
            approved=self.approved,
        )

    @property
    def depth(self) -> int:
        """How deep the call stack currently is."""
        return len(self.links)

    @property
    def attended(self) -> bool:
        """Whether a human is plausibly present to answer a dialog.

        ``user`` on its own, or qualified with what the person did
        (``user:command``) — the qualifier narrows who is asking, never whether
        anyone is there.
        """
        return self.root == "user" or self.root.startswith("user:")

    @property
    def typed_command(self) -> bool:
        """Whether this is a slash command a person typed, and nothing deeper.

        Depth 1 is the command's own code. Anything it calls sits below that and
        is judged on its own, so a command cannot lend its standing to a tool or
        service it reaches.
        """
        return self.root == "user:command" and self.depth <= 1

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


def setting_entries(raw) -> list:
    """A list setting's entries, from either shape it can be stored in.

    Every user-maintained grant list documents a comma-separated string as
    well as a JSON array, because both are things a person types into
    ``/config``. Four readers used to each carry their own copy of this —
    ``_allowed_hosts``, ``_writable_dirs``, ``shell._allowed_prefixes`` and
    ``options.remember`` — which is three chances for one of them to disagree
    with the others about what a stored value means. A grant that the policy
    honours and the listing does not show, or the reverse, is the failure that
    duplication buys.

    Public because it crosses modules; ``/permissions`` keeps a fifth copy on
    purpose, since a command is guest code and cannot import the kernel.
    """
    if isinstance(raw, str):
        raw = raw.split(",")
    return [text for item in (raw or []) if (text := str(item).strip())]


def kernel_list(key: str) -> list:
    """One kernel list setting's entries, read live.

    Live on every call rather than cached, so a ``/config`` edit or a
    ``/permissions`` revocation takes effect on the next Request instead of at
    the next restart. Removing a grant has to be as immediate as adding one.

    Absent kernel (tests, a bare container) answers empty, which grants
    nothing — the only safe direction.
    """
    try:
        from runtime.context import kernel_config

        return setting_entries((kernel_config() or {}).get(key))
    except Exception:
        return []


def _security_mode(value) -> str:
    """Normalize a requested security mode, for the dialog and the branch.

    Lazy like every other kernel reach in this file, so ``policy`` stays
    importable with no kernel behind it. An absent one answers with whatever
    was asked for, which is only ever used to *render* — the decision below
    rests on :func:`_tightens`, which fails the other way.
    """
    try:
        from runtime.security_modes import security_mode

        return security_mode(value)
    except Exception:
        return str(value or "")


def _tightens(mode) -> bool:
    """Whether moving to ``mode`` can only narrow. False is the safe answer.

    Absent kernel says no, so an unresolvable mode is asked about rather than
    waved through — the same direction :func:`kernel_list` fails in.
    """
    try:
        from runtime.security_modes import tightens

        return bool(tightens(mode))
    except Exception:
        return False


def chain_session(chain) -> str:
    """The session key a chain's root names, or "" if it names none.

    ``bridge._root_for`` roots an *agent-caused* call at the session key it
    happened in, because which session it was is the only thing separating a
    foreground turn from a subagent's. A person's own action roots at ``user``
    or ``user:command`` instead, which names no session — the caller supplies
    that.

    Roots that are not sessions (``kernel``, ``agent``, ``service:x``,
    ``cron:x``) are handed back unchanged rather than filtered out here. Every
    one of them matches no live session and no active key, so the reader
    answers False for all of them on its own; an allowlist would be a second
    list to keep in step with ``_root_for`` for no decision's sake.
    """
    root = str(getattr(chain, "root", "") or "")
    if root == "user" or root.startswith("user:"):
        return ""
    return root


def attended_now(chain, runtime=None, session_key=None) -> bool:
    """Whether a human is present for the work ``chain`` describes.

    :attr:`Chain.attended` answers who *caused* the work. This answers whether
    anybody is there to be asked, which is the question an approval dialog
    actually needs — and the two come apart in the most ordinary case there
    is. An agent's tool call during a foreground turn roots at the session
    key, so the chain reads unattended while a person sits watching the turn.
    Read as a floor, that refused every unsafe Request a tool ever made:
    egress, the whole ``proc.*`` family, ``plugin.install``, and ``ui.ask``
    with the reason "nobody is present to answer this question".

    The root goes to ``runtime.is_attended`` unchanged, which is what keeps
    the fix from widening anything. A subagent's root is a real session key
    too and still comes back False, for exactly the reason that makes a
    subagent safe: its session is not the active one. Everything with no
    session at all — a service's poll tick, a bus delivery, a cron-fired task,
    a frontend's own loop — fails closed the same way, so attendance stays a
    fact about *why* something is running rather than a rule per plugin family.

    Absent a runtime nobody can be asked, so the chain's own verdict stands.
    """
    if runtime is None:
        try:
            from runtime.context import kernel_runtime

            runtime = kernel_runtime()
        except Exception:
            runtime = None
    reader = getattr(runtime, "is_attended", None)
    if reader is None:
        return bool(chain.attended)
    # A user root names no session, so the caller's own key is the one to ask
    # about — a frontend that owns its attendance policy still gets to say
    # nobody is there for work the person started.
    key = chain_session(chain) or (session_key if chain.attended else "")
    if not key:
        return bool(chain.attended)
    try:
        return bool(reader(key))
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# Path scoping.
# ──────────────────────────────────────────────────────────────────────

def _scratch_roots() -> list:
    """Directories a plugin may write to without asking.

    Resolved from the kernel's own path constants rather than the
    environment, so this agrees with where the kernel actually keeps things.
    Scratch belongs to Second Brain under ``workspace/temp``; the operating
    system's shared temp tree is deliberately not writable by plugins.

    The ``workspace`` tree is here for a different reason than the rest, and
    it is the point of the whole boundary — see :func:`_authoring_root`.

    The attachment cache used to be listed here as a third root, because it
    sat at ``DATA_DIR/attachment_cache`` and a frontend has to be able to save
    into it. That was a grant with a seam: writing an incoming file was free
    and *reading, moving or deleting* one was a dialog, so the folder holding
    the file a person had just handed the agent was the folder the agent could
    do least with. It lives under ``workspace`` now
    (``trees.attachment_cache()``), so the entry is gone rather than moved —
    the authoring root below already covers it, and one rule that covers a
    case beats a rule plus an exception that happens to agree.
    """
    roots = []
    if (authoring := _authoring_root()) is not None:
        roots.append(authoring / "temp")
        roots.append(authoring)
    return [r for r in roots if r]


def _authoring_root():
    """The tree an agent may write code into freely, or None.

    This is what the process boundary is *for*. Every file under
    ``workspace`` runs in a subprocess — not because it asked to, but
    because of where it is (``sandbox/isolation.py``) — so code the agent
    writes there is contained before it ever runs. Asking a human to approve
    each edit would buy nothing that containment has not already bought, and
    would cost the thing that makes an authoring agent useful: writing a
    plugin is a dozen edits, and a dialog on each is a dozen interruptions to
    approve something that cannot act unmediated anyway.

    What this does *not* grant is worth being precise about, because it is the
    LibOS invariant exactly. Writing a file changes what the system can *ask*.
    It does not change what it may *affect*: the new plugin's own Requests are
    classified like anything else's, and it inherits no authority from having
    been written without a dialog. Free authorship, unchanged authorization.
    """
    try:
        import trees
        return Path(trees.tree("workspace").path)
    except Exception:
        return None


def _writable_dirs() -> list:
    """Folders the *user* has opened to the agent, from kernel config.

    The filesystem counterpart to :func:`_allowed_hosts`, and config rather
    than a plugin declaration for the same reason egress is: a person deciding
    what code may reach is a different act from code claiming its own reach.

    ``~`` is expanded because a person typing a path will write one. Absent
    kernel means an empty list, which grants nothing.
    """
    dirs = []
    for entry in kernel_list("fs_writable_dirs"):
        try:
            dirs.append(Path(entry).expanduser())
        except (OSError, ValueError, RuntimeError):
            continue
    return dirs


def _protected(path) -> bool:
    """Whether ``path`` is code this app runs, which no user grant may open.

    **The carve-out that makes the setting safe to have.** The natural thing
    to put in ``fs_writable_dirs`` is the folder you keep projects in — and on
    a developer's machine that folder plausibly *contains Second Brain's own
    source*. Without this, listing it would make ``sandbox/policy.py`` freely
    writable, and the agent could edit the classifier that decides what it is
    allowed to do. The same reasoning covers ``DATA_DIR/installed``: a free
    write there is a way around ``plugin.install``, which is ALWAYS_UNSAFE
    precisely to gate what code gets to run.

    So this is checked against the *target*, never against the listed folder.
    A parent directory is the whole problem — filtering the list would miss
    the case where somebody grants ``Z:\\My Code`` and the app lives inside it.

    It removes the *free* grant and nothing else. These paths fall through to
    the dialog exactly as they do today, so editing the app's own source is
    still possible, one answered question at a time.

    ``workspace`` is the deliberate hole in ``DATA_DIR``: it is the agent's own
    tree and free authorship there is the point of the whole boundary. It does
    not depend on this function being right, though — it is a scratch root in
    its own right, and this is only consulted for the user's list.

    Fails closed. Unable to locate the app means treating everything as
    protected, which costs a dialog rather than a grant.
    """
    try:
        from paths import DATA_DIR, ROOT_DIR
    except Exception:
        return True
    if _within(path, [ROOT_DIR]):
        return True
    if not _within(path, [DATA_DIR]):
        return False
    authoring = _authoring_root()
    return not (authoring is not None and _within(path, [authoring]))


def _freely_writable(path) -> bool:
    """Whether writing to ``path`` needs no dialog.

    Two grants, and the carve-out applies to exactly one of them. The built-in
    scratch roots are the kernel's own and are never second-guessed;
    :func:`_writable_dirs` is the user's, and it stops at the app's code.
    """
    if _within(path, _scratch_roots()):
        return True
    return _within(path, _writable_dirs()) and not _protected(path)


def _write_reason(path, verb: str = "write") -> str:
    """Why an allowed write was allowed — scratch, authoring, or the user's.

    Three different grants land in the same branch and the ledger should not
    have to guess which: "somewhere to put a temporary file", "the agent is
    writing a plugin", and "a folder the person opened to it". Only the last
    two are interesting when reading back what happened overnight, and the
    third is the one whose blast radius is somebody's actual work.

    Ordered narrowest first — the authoring tree is itself a scratch root, and
    a listed folder can contain either.
    """
    root = _authoring_root()
    scratch = root / "temp" if root is not None else None
    if scratch is not None and _within(path, [scratch]):
        return f"{verb} in scratch"
    if root is not None and _within(path, [root]):
        return f"{verb} in the agent's own tree"
    if _within(path, _scratch_roots()):
        return f"{verb} in scratch"
    return f"{verb} in a directory you allowed"


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
    R.SESSION_ADD_TOOL,
    # The literal subject of the LibOS quote: the agent extending itself.
    R.PLUGIN_REGISTER, R.PLUGIN_UNREGISTER, R.PLUGIN_RELOAD,
    R.PLUGIN_INSTALL, R.PLUGIN_UNINSTALL, R.PLUGIN_UPDATE,
    R.SERVICE_LOAD, R.SERVICE_UNLOAD,
    # Opening a brain is the same act one subsystem over: a pool of real
    # processes starts, each holding a provider SDK and an API key. Closing one
    # ends calls that may be in flight. Reading the list is safe and lives
    # below with the other listings.
    R.LLM_LOAD, R.LLM_UNLOAD,
    # Recurring unattended work.
    R.CRON_CREATE, R.CRON_UPDATE, R.CRON_REMOVE,
    R.AGENT_SCHEDULE,
    # Identity and settings.
    R.USER_WRITE, R.USER_LIST,
    # Destructive.
    R.CONV_DELETE, R.FS_DELETE,
    # Irreversible, which is the criterion the db-write branch below states for
    # itself: the thing worth asking about is the write that cannot be undone
    # by writing again. Nothing anywhere removes a compaction marker —
    # ``latest_compaction`` finds it and ``messages_to_history`` honours it
    # forever — so a conversation, once folded into a summary, has no way back
    # to being read in full.
    #
    # The rows themselves survive, which makes this *less* destructive than
    # CONV_CLEAR one set over. It is here anyway, and with no
    # ``chain.typed_command`` exemption, because the question is not how much
    # data is lost but whether the loss is recoverable. A person typing
    # ``/compact`` answers one dialog; that is the price of nothing else in the
    # system being able to rewrite their conversation without saying so.
    R.SESSION_COMPACT,
    # Ending the process. Unconditional, including the restart variant: coming
    # back up is not a mitigation, since everything in flight still dies.
    R.APP_STOP,
}

# Requests that narrow capability, or only ever affect this execution.
ALWAYS_SAFE = {
    R.SELF_RESPOND, R.FS_TEMP, R.FS_READ, R.FS_READ_BYTES, R.FS_STAT,
    R.FS_LIST, R.FS_SEARCH,
    R.DB_QUERY, R.DB_DEFINE,
    R.CONV_READ, R.CONV_LIST, R.CONV_CREATE, R.CONV_APPEND,
    R.CONV_SET_TITLE, R.CONV_SET_CATEGORY, R.CONV_SET_NOTIFICATION_MODE,
    # CONV_NEW is the counterpart to CONV_LOAD — one binds a session to a
    # conversation, the other lets go of it — and is safe for the stronger of
    # the two reasons: it touches no row at all. Nothing is created (a
    # conversation now comes from the first message), nothing is deleted, and
    # what was open stays in the list to be loaded again.
    R.CONV_LOAD, R.CONV_NEW, R.CONV_CLEAR,
    R.SESSION_GET, R.SESSION_LIST, R.SESSION_PUSH, R.SESSION_CANCEL,
    # Reading and settling notifications the kernel wrote *about this user*,
    # scoped in SQL by the context's own ``user_id`` rather than by an argument
    # anybody could name. LIST reads; MARK_READ writes one timestamp whose only
    # effect is that a panel stops highlighting a row. Neither is reach: the
    # contents were already delivered to this user's own surfaces.
    #
    # SESSION_PUSH covers *raising* one, and stays exactly where it was. It
    # grew a ``notify`` argument, not a capability — showing the user text is
    # the same act whichever surface it lands on, and the argument that made it
    # safe (you can reach a person, you cannot reach anything of theirs) is
    # untouched by which pane draws it.
    R.NOTIFICATION_LIST, R.NOTIFICATION_MARK_READ,
    R.SESSION_STATE_GET, R.SESSION_STATE_SET, R.SESSION_REMOVE_TOOL,
    R.SESSION_REMOVE_PROMPT,
    # Staging a file for the model to look at is safe for the reason FS_READ
    # is, and the handler applies the same ``protected.reason_for`` guard. The
    # guest could already read those bytes and hand them back as an
    # ``llm_summary``; staging is that path with fewer hops, not new reach. It
    # sits apart from its ADD_ siblings deliberately — those widen what the
    # agent may *do*, where this only changes what it can see.
    R.SESSION_ADD_ATTACHMENT,
    # UI_APPROVE is *not* here. Asking permission is a policy decision, so it
    # is always unsafe and the approver asks it — see the branch below.
    R.UI_RENDER,
    # A line of text on the call the person is already watching, and only when
    # one is running — the handler abstains otherwise, so there is no reach to
    # decide about. A dialog would be worse than pointless: progress is emitted
    # in a loop, so asking would make narrating cost more interruptions than
    # the work it describes.
    R.UI_PROGRESS,
    R.USER_READ,
    # PLUGIN_VALIDATE sits with the listings, not with REGISTER and friends,
    # because it changes nothing: the validator is a pure AST walk that never
    # imports the file it reads. It is how an agent authoring a plugin checks
    # its own work, and putting a dialog in that loop would only teach the
    # agent to skip the check.
    R.PLUGIN_LIST, R.PLUGIN_DESCRIBE, R.PLUGIN_VALIDATE,
    R.SERVICE_LIST, R.SERVICE_CALL,
    # Which model profiles exist and whether each is open. Names, endpoints and
    # context sizes — never the key, which is a ``secret_*`` setting and comes
    # back as a handle like every other one.
    R.LLM_LIST,
    R.PATH_GET,
    # TOOL_CALL is safe for the reason SERVICE_CALL is: the callee's own
    # Requests are classified with the caller still in the chain, so routing
    # through a tool launders nothing, and the scope already decides which
    # tools exist. COMMAND_CALL is not — see the branch below.
    R.TOOL_LIST, R.TOOL_CALL, R.COMMAND_LIST,
    # A subagent is safe because it can approve nothing. Its turn runs on a
    # session key that is never the active one, so its Requests build a chain
    # rooted there rather than at ``user`` — ``Chain.attended`` is False, which
    # refuses ``ui.ask`` and every unsafe Request outright, and the parent's
    # ``approved`` grant is not inherited. A child therefore reaches strictly
    # less than whoever started it.
    #
    # (This used to be justified as "the child's Requests are classified with
    # the parent still in the chain", which is not what happens: a turn is not
    # a nested call and its chain is fresh. The verdict was right; the reason
    # was wrong, and the real one is stronger.)
    #
    # COLLECT reads a report the child already produced. STOP is here for the
    # reason SESSION_CANCEL and PROC_STOP are: it narrows, and an agent that
    # needs a dialog to end something it started will leave it running.
    R.AGENT_COMPLETE, R.AGENT_SPAWN, R.AGENT_COLLECT, R.AGENT_STOP,
    # Safe because it widens nothing: the kernel handed this escort a call it
    # had already decided to place, and proceeding is placing that one. The
    # token is what limits it — code with no token reaches no call at all.
    R.LLM_PROCEED,
    # Same reachability argument, one call further in: the sink is parked for
    # the duration of one ``chat`` call, so a backend can only push text into
    # the call it was asked to place. It is also write-only and one-way —
    # nothing comes back, so it cannot be used to read anything.
    R.LLM_DELTA,
    R.CRON_LIST, R.CRON_GET, R.CRON_ENABLE,
    R.EVENT_EMIT, R.EVENT_REQUEST,
    # Safe for the same reason LLM_PROCEED is: the limit is reachability,
    # not a verdict. Each resolves to the calling frontend's *own* adapter
    # through a token parked when its box opened, so code that is not a loaded
    # frontend reaches nothing and is refused. Carrying a person's input into
    # the state machine is the entire job of a frontend — a dialog on every
    # keystroke would be nonsense, and FRONTEND_BIND is here rather than with
    # USER_WRITE for the same reason: asking a user to approve their own login
    # would make a per_user frontend unusable, and binding sessions is already
    # what the kernel lets a native frontend do. The token is what stops one
    # frontend doing it on another's behalf.
    R.FRONTEND_SUBMIT, R.FRONTEND_CANCEL, R.FRONTEND_BIND, R.FRONTEND_ATTEND,
    R.FRONTEND_RESOLVE, R.FRONTEND_PENDING,
    # FRONTEND_ACT is safe *as a Request* and buys nothing on its own: it runs
    # an inner Request, and that one is classified here like any other. What it
    # changes is the chain — from ``frontend:<name>``, which names no session
    # and is therefore unattended forever, to a session the frontend owns. The
    # authority that follows is not this type's; it is whatever
    # ``runtime.is_attended`` says about that session, which is the frontend's
    # own declaration through FRONTEND_ATTEND. So the widening is exactly "act
    # as a session you own while somebody is watching it", and it ends when the
    # frontend says nobody is. Ownership is checked host-side against the
    # adapter's live sessions, which the guest cannot state about itself.
    # FRONTEND_COLLECT only takes an answer already produced.
    R.FRONTEND_ACT, R.FRONTEND_COLLECT,
    # The console is scoped harder still: not merely "a frontend", but the one
    # frontend that claimed it. Reading takes only what a person already typed
    # at this machine's own keyboard, and writing puts text on the screen in
    # front of them — neither reaches past the console, and gating them would
    # mean asking permission to show the prompt that asks permission.
    R.CONSOLE_READ, R.CONSOLE_WRITE,
    # The port is scoped exactly like the console, and the reachability is what
    # does the work: the kernel opened the socket, on loopback, because a
    # frontend the user enabled declared it — and only the token that claimed
    # it reaches these at all. Draining takes requests already accepted;
    # responding, pushing and closing write back down a connection the kernel
    # is holding. None of the four reaches past that socket, and none of them
    # can open one: binding is the kernel's act, not a Request. Gating them
    # would mean a dialog per SSE frame, which is the same nonsense as asking
    # permission to show the prompt that asks permission.
    #
    # Worth being explicit that this is *not* an inbound ``net.http``. That
    # Request dials out and is classified on where it is dialling; there is no
    # destination here to classify, because the client came to us.
    R.HTTP_DRAIN, R.HTTP_RESPOND, R.HTTP_PUSH, R.HTTP_CLOSE,
    R.TASK_ENQUEUE, R.TASK_STATUS, R.TASK_OUTPUT,
    R.TASK_LIST, R.TASK_GRAPH, R.TASK_TRIGGER,
    # The three ways of speaking about a process this system already started.
    # Listing and asking after one read a registry the kernel owns, and it
    # holds nothing that was not approved at ``proc.start``. Stopping is here
    # for the same reason SESSION_REMOVE_TOOL is: it narrows. A dev server the
    # agent cannot kill without a dialog is a dev server the agent will not
    # start, and the alternative to stopping it is leaving it running.
    R.PROC_STATUS, R.PROC_LIST, R.PROC_STOP,
    # And the same two ways of speaking about a *script* this system already
    # started, on exactly the argument the three above make. ``script.run`` is
    # itself branched — where the file lives and what it imports is the whole
    # question — but that question was already answered when the run started.
    # Collecting reads a result the script has already produced; stopping
    # narrows, and a fan-out the caller cannot abandon without a dialog is one
    # it will not start.
    R.SCRIPT_COLLECT, R.SCRIPT_STOP,
    R.FILE_REGISTER, R.FILE_LIST,
    R.PARSE_FILE, R.PARSE_MODALITY,
    R.LEDGER_RECORD, R.LEDGER_READ,
    # Asking the kernel how long this execution has left. It reads a clock the
    # kernel keeps *about the caller itself* — no other execution is visible
    # through it — and the answer only ever causes the guest to do less. A
    # deadline nobody can see is one that can only be discovered by being
    # killed by it.
    R.SELF_BUDGET,
}

# Every Request must appear in exactly one of the two sets or be handled by a
# branch above. Checked at import so a new Request cannot be added without a
# decision being made about it — silently defaulting to "unsafe" would look
# like policy when it is really an oversight.
_BRANCHED = {NET_HTTP, PROC_RUN, R.PROC_START, R.SCRIPT_RUN,
             R.DB_WRITE,
             FS_WRITE, R.FS_WRITE_BYTES,
             FS_MOVE, FS_DELETE, FS_TEMP,
             CONFIG_READ, R.CONFIG_WRITE, ENV_READ, R.SECRET_REVEAL,
             R.COMMAND_CALL,
             UI_ASK, R.UI_APPROVE, SESSION_ADD_TOOL, R.SESSION_COMPACT,
             SESSION_ADD_PROMPT, R.SESSION_SET_MODE, AGENT_SCHEDULE,
             CONV_DELETE,
             R.TASK_PAUSE, R.TASK_RESET}
_UNDECIDED = R.ALL_TYPES - ALWAYS_SAFE - ALWAYS_UNSAFE - _BRANCHED
assert not _UNDECIDED, f"unclassified Requests: {sorted(_UNDECIDED)}"

#: What a command must declare a gate for — ``ALWAYS_UNSAFE`` plus the branched
#: types that never actually answer SAFE.
#:
#: Living here rather than in the test that reads it, because it is a statement
#: about policy and only policy can keep it true. The test derived it as
#: ``ALWAYS_UNSAFE`` plus three hand-listed branches, which meant a Request
#: that is unconditionally unsafe but *spelled* as a branch was invisible to
#: it — precisely the case the derivation existed to catch. ``task.reset`` was
#: that case: unsafe for every argument, missing from the list, so ``/tasks``
#: shipped with no gate and its resets took the mid-run path.
#:
#: ``proc.run``/``net.http``/``secret.reveal`` do have safe arguments, and are
#: here anyway: running a shell, reaching the network and handing over a
#: credential are consequential however they are spelled. ``config.write`` is
#: deliberately absent, since a plugin persisting its own declared setting is
#: safe and gating it would demand declarations from commands that only ever
#: write their own keys.
#: ``session.set_mode`` is here for the same reason as those three: it has a
#: safe argument (``lockdown``, which can only narrow) and is consequential
#: however it is spelled, because the widening direction changes how every
#: later Request in the conversation is answered.
CONSEQUENTIAL = ALWAYS_UNSAFE | {
    PROC_RUN, R.PROC_START, NET_HTTP, R.SECRET_REVEAL, R.TASK_RESET,
    R.SESSION_SET_MODE,
}


# ──────────────────────────────────────────────────────────────────────
# Database writes.
# ──────────────────────────────────────────────────────────────────────
#
# Writing rows is ordinary and stays ordinary — including rows in the kernel's
# own tables, because data cannot change how the kernel works and only
# structure can. ``sandbox/users.py`` is where that argument lives and where
# schema changes are refused outright; this function asks the narrower
# question the policy layer is actually for: *should a person be told first.*
#
# For one statement, yes. A DELETE against a kernel table is the only row write
# that cannot be undone by writing again — an UPDATE with a bad WHERE clause
# leaves the rows there to fix, and a DELETE with the same bad WHERE clause
# leaves a person asking where their conversations went. Nothing else about it
# is dangerous, which is exactly why asking is the right response rather than
# refusing: the act is legitimate, it is just irreversible, and irreversible is
# what a dialog is for.
#
# A plugin's own tables are not included. Losing a plugin's cache is a plugin's
# problem, and a dialog for every row it tidies up is how people learn to stop
# reading dialogs.

def _classify_db_write(args: dict) -> Decision:
    """Decide about one database write. See the section comment above."""
    from .users import is_kernel_delete

    sql = args.get("sql") or ""
    if (table := is_kernel_delete(sql)):
        return Decision(UNSAFE, f"delete rows from {table}",
                        say="Deleted rows are not recoverable.")
    return Decision(SAFE, "write rows")


# ──────────────────────────────────────────────────────────────────────
# Egress.
# ──────────────────────────────────────────────────────────────────────
#
# ``net.http`` is the single control that makes generous filesystem and
# database reads safe: a plugin that reads everything still cannot send
# anything anywhere. So this branch is the one place where relaxing anything
# has to be argued for rather than assumed, and the argument is narrow.
#
# What relaxes it is a **host allowlist the user maintains** — the config
# setting ``net_allowed_hosts`` — and deliberately not a declaration on the
# plugin, which would make contained code the authority on its own reach (the
# ``isolation`` bug one level down; see ``sandbox/isolation.py``). A person
# deciding what code may reach is a different act.
#
# The host is the unit rather than the URL because the host is what a person
# can actually decide about. Nobody can usefully answer "may this plugin fetch
# /v1/web/search?q=…" every time; "may this app talk to api.search.brave.com"
# is a question with a stable answer. Path and query are therefore *not*
# matched — inside an allowed host the plugin may ask anything, which is honest
# about what the grant means.
#
# Recognizers exist for the same reason the shell has them: somewhere for a
# remembered or structural allowance to live later, visible in the policy
# rather than inside the plugin it authorizes. This one ships empty and the
# allowlist check is the only built-in rule — a remembered host is written
# straight into ``net_allowed_hosts`` by the dialog (``sandbox/options.py``),
# so there is nothing for a recognizer here to do yet.
_NET_RECOGNIZERS: list = []


def _allowed_hosts() -> set:
    """Hosts the user has said this app may talk to.

    Read live from the kernel's config on every call rather than cached, so a
    ``/config`` edit takes effect on the next Request instead of at the next
    restart. Removing a host has to be as immediate as adding one — a stale
    allowlist that still permits what a person just revoked is the failure this
    cannot afford.

    Absent kernel (tests, a bare container) means an empty allowlist, which
    refuses everything. Failing closed is the only safe direction here.
    """
    return {host.lower().lstrip(".")
            for host in kernel_list("net_allowed_hosts")}


def request_host(url) -> str:
    """The host a URL names, lowercased, or "" if it names none.

    Shared by the classifier and the dialog so the host a person is asked
    about is the host that was matched. Anything unparseable comes back empty,
    which no allowlist entry can equal — so a malformed URL is asked about
    rather than slipping through on a comparison that failed open.
    """
    from urllib.parse import urlsplit

    try:
        return (urlsplit(str(url or "")).hostname or "").lower()
    except ValueError:
        return ""


def _host_allowed(host: str, allowed: set) -> bool:
    """Whether ``host`` is covered, exactly or as a subdomain.

    ``example.com`` in the allowlist covers ``api.example.com``, because a
    person naming a service means the service and not one hostname of it. It
    does **not** cover ``notexample.com`` — the match is on a dot boundary, or
    a suffix comparison would hand every attacker-registered lookalike the
    grant.
    """
    if not host or not allowed:
        return False
    return any(host == entry or host.endswith("." + entry)
               for entry in allowed)


def _classify_net(args: dict) -> Decision:
    """Decide about one outbound request. See the section comment above."""
    url = args.get("url", "")
    host = request_host(url)
    # A GET with data in the query string is exfiltration exactly as much as a
    # POST body is, so the verb is not consulted — only where it is going.
    if _host_allowed(host, _allowed_hosts()):
        return Decision(SAFE, f"{host} is an allowed host")
    for recognize in _NET_RECOGNIZERS:
        try:
            if (why := recognize(url, args)):
                return Decision(SAFE, why)
        except Exception:
            # A recognizer that raises abstains. It can only ever widen, so
            # failing it closed costs a dialog and nothing else.
            continue
    # The host is named separately from the URL because it is the thing the
    # answer is really about — and because a URL can be long enough to push
    # the decision off the end of a dialog.
    where = f" ({host})" if host else ""
    return Decision(UNSAFE, f"outbound request to {url}{where}",
                    say=(f"{host} is not on your allowed-hosts list."
                         if host else "This reaches a host outside your "
                         "allowed-hosts list."))


#: How much of a value the dialog will show before giving up on it. A person
#: skimming a question does not read a hundred-entry list, and a dialog that
#: scrolls is one that gets answered without being read.
_VALUE_CHARS = 160


def describe_setting_change(key: str, value) -> str:
    """What a ``config.write`` is actually asking for, as a person reads it.

    The key alone is not a question anybody can answer. "Change setting
    net_allowed_hosts" tells you a plugin wants to touch the egress allowlist
    and nothing about *what it wants added* — which is the entire decision.
    Same for a model endpoint, a data directory, a retention period.

    Values are redacted first. A setting named ``secret_*`` must not become
    readable by asking permission to write it, and the dialog is the one place
    that would otherwise print it in full.

    Long values are summarised rather than truncated mid-token, because half a
    hostname is worse than a count.
    """
    if value is None:
        return f"clear setting {key}"
    shown = redact(key, value, guess=True)
    if isinstance(shown, (list, tuple, set)):
        items = [str(item) for item in shown]
        joined = ", ".join(items)
        if len(joined) > _VALUE_CHARS:
            joined = _entries(len(items))
        return f"set {key} to [{joined}]"
    if isinstance(shown, dict):
        # The keys, when they fit. This branch used to collapse to a bare
        # count *whatever the size*, so a one-key write read "set
        # scheduled_jobs (1 entries)" - a number standing in for a value that
        # would have fitted, and ungrammatical besides.
        joined = ", ".join(str(name) for name in shown)
        if not joined:
            return f"set {key} to nothing"
        if len(joined) > _VALUE_CHARS:
            joined = _entries(len(shown))
        return f"set {key} ({joined})"
    if isinstance(shown, bool) or isinstance(shown, (int, float)):
        return f"set {key} to {shown}"
    text = str(shown)
    if len(text) > _VALUE_CHARS:
        # ASCII: this reaches the REPL's cp1252 console, where a unicode
        # ellipsis raises rather than renders.
        text = text[:_VALUE_CHARS] + "..."
    return f"set {key} to {text!r}"


def _entries(count: int) -> str:
    """``N entries``, pluralised. A count is a fallback for a value too long
    to show, and one that cannot count to one reads as a bug."""
    return f"{count} entry" if count == 1 else f"{count} entries"


# ──────────────────────────────────────────────────────────────────────
# Scripts.
# ──────────────────────────────────────────────────────────────────────
#
# A script is the answer to the shell's problem rather than another instance of
# it: SDK code has nothing to interpret, so every effect inside arrives here
# individually with the script still in the chain, and running one widens
# nothing. Same argument as ``tool.call``; see CLAUDE.md, "Scripts are what the
# classifier's death left missing".
#
# Two things are checked, and both are read off the *destination* — the same
# shape as the filesystem branches, which ask where a write is aimed rather
# than what it contains.

def _script_report(path):
    """Validate a script, or None if it could not be read.

    The kernel re-derives this rather than accepting a verdict passed in the
    Request. A caller supplying its own report — or a digest standing in for
    one — would be the code being contained acting as the authority on its own
    containment, which is the bug ``sandbox/isolation.py`` exists to prevent.

    It is a pure AST walk over one file and never imports what it reads, so the
    cost is a parse. Worth watching if a loop ever runs scripts in bulk.
    """
    from .validator import validate_file

    try:
        return validate_file(path)
    except (OSError, ValueError):
        return None


def _classify_script(args: dict) -> Decision:
    """Decide about running one script.

    The path is resolved here by the *same* resolver the handler uses, and that
    sharing is load-bearing rather than tidy. Classifying the raw argument while
    the handler resolved it meant a correctly-named bare script drew a dialog
    and then ran fine — the "asked about something that should have been free"
    complaint, one layer below the one that motivated scripts existing at all.
    """
    from .isolation import is_script, resolve_script

    path = args.get("path")
    if not path:
        return Decision(SAFE, "script launch preflight will require a path")
    path = resolve_script(path) or path
    if not is_script(path):
        # Not a refusal of *this* file so much as of the shape of the ask: the
        # containment story rests entirely on the file living somewhere the
        # kernel subprocesses unconditionally. Anywhere else and the honest
        # answer is that nobody knows what this is.
        return Decision(SAFE, f"script launch preflight will reject {path} "
                             "outside scripts/")

    report = _script_report(path)
    if report is None:
        return Decision(SAFE, f"script launch preflight will report that "
                             f"{path} cannot be read")
    if not report.ok:
        return Decision(SAFE, f"script launch preflight will report validation "
                             f"errors in {Path(path).name}")
    if report.unmediated:
        # The one case that is asked about. An installed package importing a
        # foreign library is subprocessed and *not* asked, because somebody
        # approved it once at ``plugin.install``; a script was never approved
        # by anyone, and a library the validator cannot see inside is the only
        # part of a script whose effects do not come back through this
        # function. Naming it is most of the value of the dialog.
        libraries = ", ".join(sorted(report.unmediated))
        return Decision(UNSAFE,
                        f"run {Path(path).name}, which imports {libraries} - "
                        f"that library's own actions are not mediated",
                        say=f"It imports {libraries}, whose own actions the "
                            f"kernel cannot see or stop.")
    return Decision(SAFE, f"run the script {Path(path).name} (contained)")


def _setting_owners(key: str) -> set:
    """Which plugins declared a config setting, per the setting registry.

    The registry and not the guest, deliberately: ownership has to be a fact
    the caller cannot assert about itself, or the exemption it unlocks is one
    any plugin can claim by saying the right thing.
    """
    if not key:
        return set()
    try:
        from plugins.plugin_discovery import get_setting_plugin_names

        return set(get_setting_plugin_names(key))
    except Exception:
        return set()


def _callers(chain: Chain) -> set:
    """Every plugin identity in this chain, including the resident root.

    Resident roots are assigned by the kernel adapter rather than supplied by
    guest code, so ``frontend:telegram`` is an authenticated identity and
    belongs in the set. The root is added rather than relied upon alone
    because a box acting on its own initiative roots its chain at
    ``service:timekeeper`` while a box adopting a caller's chain appears as a
    *link* on it — see ``PersistentBox._identity`` for why that link is the
    registered name and not the file stem it used to be.
    """
    callers = set(chain.links)
    if chain.root.startswith(("service:", "frontend:")):
        callers.add(chain.root.split(":", 1)[1])
    return callers


def _owns_setting(chain: Chain, key: str) -> bool:
    """Whether this chain contains the plugin that declared ``key``."""
    return bool(_setting_owners(key) & _callers(chain))


# ──────────────────────────────────────────────────────────────────────
# The eight mechanisms.
# ──────────────────────────────────────────────────────────────────────
#
# Of the ~110 Request types, the great majority are a flat lookup: a set
# membership in ALWAYS_SAFE or ALWAYS_UNSAFE. Only a handful are decided from
# their arguments, and every one of those uses one of **eight** mechanisms.
# They are the working vocabulary of this file, and writing them down is the
# difference between adding a ninth deliberately and growing an ad-hoc branch.
#
#   1. DESTINATION  — where is this aimed?
#      ``fs.write``, ``fs.write_bytes``, ``fs.move``, ``fs.delete`` against
#      ``_scratch_roots()``. The path decides; the content never does.
#
#   2. ALLOWLIST    — is this on a list a *person* maintains?
#      ``net.http`` against ``net_allowed_hosts``, matched on a dot boundary.
#      Deliberately config and not a plugin declaration — see the egress
#      section for why that distinction is the whole of it.
#
#   3. OWNERSHIP    — did the asker declare this thing?
#      ``secret.reveal`` and ``config.write``, resolved through the setting
#      registry (``_owns_setting``) so it is a fact the caller cannot assert
#      about itself.
#
#   4. SHAPE        — what does the payload actually say?
#      ``db.write`` via ``is_kernel_delete``. Sound here only because
#      ``conn.execute`` runs exactly one statement; see ``sandbox/users.py``
#      for why the same trick was wrong for the shell.
#
#   5. POLARITY     — does this widen or narrow?
#      ``task.pause``: pausing is safe, unpausing is not. The rule the module
#      docstring states, applied where a family had no other obvious answer.
#
#   6. ATTENDANCE   — is a person there?
#      ``ui.ask`` via ``attended_now``. About the chain's *root*, never about
#      which plugin family is asking.
#
#   7. PROVENANCE   — who caused this, exactly?
#      ``config.write`` via ``chain.typed_command``, and the ``chain.approved``
#      grant short-circuit at the top of ``classify``.
#
#   8. RECOGNIZER   — does a pluggable predicate vouch for it?
#      ``shell._SHELL_RECOGNIZERS`` holds two — a structural read-only check
#      and a *remembered* one reading ``shell_allowed_prefixes``.
#      ``_NET_RECOGNIZERS`` is empty, since egress is served by its allowlist
#      directly. This is the only one of the eight that is an extension point
#      rather than a rule, and it can only ever widen — see
#      ``sandbox/shell.py`` and ``docs/PERMISSIONS_MAP.md``.
#
# Three of these (1, 2, 8) can only ever widen, and abstain by failing closed.
# Three (3, 6, 7) read facts the kernel owns and guest code cannot state about
# itself. That split is not decoration: a mechanism a plugin could influence
# would be authorization by declaration, which is the bug ``isolation.py``
# exists to prevent, one level up.


def classify(request: Request, chain: Chain) -> Decision:
    """Decide whether a Request may execute without asking the user.

    This is the complete authorization surface for sandboxed code. The
    argument-conditional branches below use one of eight mechanisms, catalogued
    in the section comment above.
    """
    # Runaway nesting is stupidity, not malice, and it is caught here rather
    # than by exhausting the machine. Both checks read the chain, which is the
    # second job it does: it is the call stack, so it is also the cycle
    # detector.
    if chain.depth > MAX_DEPTH:
        return Decision(UNSAFE, f"call chain deeper than {MAX_DEPTH}",
                        say="This is nested unusually deep, which is "
                            "normally a runaway rather than intent.")
    if chain.cyclic:
        return Decision(UNSAFE, f"call cycle: {chain.render()}",
                        say="Something here is calling itself.")
    kind = request.type
    args = request.args

    # An approval covers what the command declared and nothing else. Anything
    # outside the grant falls through to the branches below and is asked about
    # on its own — so a command that reaches past its manifest gets caught
    # rather than riding in on the one "yes" the user already gave.
    if chain.approved and kind in chain.approved:
        return Decision(SAFE, "approved command")

    # ── egress: checked regardless of verb ────────────────────────
    if kind == NET_HTTP:
        return _classify_net(args)

    # ── database writes: rows are ordinary, losing them is not ────
    if kind == R.DB_WRITE:
        return _classify_db_write(args)

    # ── the shell ─────────────────────────────────────────────────
    # Its own module, because working out *what commands a line runs* is a
    # subject rather than a branch — a lexer, a decomposition and two
    # recognizers. The verdict comes back as an ordinary Decision.
    if kind in (PROC_RUN, R.PROC_START):
        from .shell import classify_shell

        return classify_shell(kind, args)

    # ── scripts: the shell's job, done where it can be answered ───
    if kind == R.SCRIPT_RUN:
        return _classify_script(args)

    # ── filesystem writes depend entirely on where ────────────────
    # Text and bytes are the same act with a different encoding, so they get
    # the same answer — anything else would make the encoding a way around
    # the rule.
    if kind in (FS_WRITE, R.FS_WRITE_BYTES):
        if _freely_writable(args.get("path")):
            return Decision(SAFE, _write_reason(args.get("path")))
        return Decision(UNSAFE, f"write to {args.get('path')}",
                        say="That is outside the folders you allow writes in.")

    if kind == FS_MOVE:
        # Both ends, because a move is a write to one place and a delete from
        # the other. Dragging a file *out* of an allowed folder is exactly as
        # much a change to somewhere unlisted as writing there would be.
        if (_freely_writable(args.get("src"))
                and _freely_writable(args.get("dst"))):
            return Decision(SAFE, _write_reason(args.get("dst"), verb="move"))
        return Decision(UNSAFE,
                        f"move {args.get('src')} to {args.get('dst')}",
                        say="One end is outside the folders you allow writes "
                            "in.")

    if kind == FS_DELETE:
        if _freely_writable(args.get("path")):
            return Decision(SAFE, _write_reason(args.get("path"),
                                                verb="delete"))
        return Decision(UNSAFE, f"delete {args.get('path')}",
                        say="That is outside the folders you allow writes in.")

    if kind == FS_TEMP:
        return Decision(SAFE, "scratch space")

    # ── plaintext is the one thing always worth asking about ──
    #
    # With one exception, and it is the same one ``config.write`` makes below:
    # a plugin reading back the credential *it declared* is not asked, because
    # configuring that setting was the consent. Anyone else is asked, which is
    # the whole point — the exemption is ownership, not need.
    #
    # This was documented as the rule and not implemented as one, and the gap
    # had teeth. A frontend needs its own token during ``start()``, before any
    # frontend is up to draw a dialog; an unconditional ask there is a question
    # nobody can answer, at boot, every boot. Ownership comes from the setting
    # registry rather than from anything the guest says about itself.
    if kind == R.SECRET_REVEAL:
        name = args.get("name") or ""
        if _owns_setting(chain, name):
            owner = sorted(_setting_owners(name) & _callers(chain))[0]
            return Decision(SAFE, f"{owner} reads its own {name}")
        return Decision(UNSAFE, f"plaintext of {name or 'a secret'}",
                        say="It is not the plugin this credential was "
                            "configured for.")

    # ── secrets are readable as handles, so reading them is safe ──
    if kind in (CONFIG_READ, ENV_READ):
        name = args.get("key") or args.get("name") or ""
        if is_secret(name):
            return Decision(SAFE, f"{name} (returned as a handle)")
        return Decision(SAFE, "read setting")

    # A resident plugin may persist its own declared state without asking.
    # The setting registry, not the guest's requested scope, establishes
    # ownership; every other config write still requires approval.
    if kind == R.CONFIG_WRITE:
        # A command the person just typed is its own consent: /config exists to
        # change settings, and a dialog confirming what someone asked for one
        # keystroke ago is friction with no decision in it. Scoped to the
        # command's own code, so a command that delegates to a tool or service
        # is asked about that callee's write as usual. The root is assigned by
        # the kernel from the dispatch path (``bridge._root_for``), never
        # claimed by guest code.
        #
        # It stays audible either way: every write announces itself to the chat
        # by key name, so a change nobody approved is still a change nobody
        # missed.
        if chain.typed_command:
            return Decision(SAFE, "a command the user typed")
        key = args.get("key") or ""
        if args.get("scope") == "plugin" and _owns_setting(chain, key):
            owner = sorted(_setting_owners(key) & _callers(chain))[0]
            return Decision(SAFE, f"{owner} persists its own {key}")
        return Decision(UNSAFE, f"config.write: {describe_setting_change(key, args.get('value'))}")

    # ── a slash command is the user's own surface ─────────────────
    # Unlike a tool, a command is not scoped by the agent's tool catalogue and
    # is not written to be called by other code: it is what the *person* types,
    # and the set includes things like package installation and config editing.
    # Running one on someone's behalf is worth a sentence, and the dialog can
    # name it exactly. If the completed arguments also declare a gated action,
    # reaching the handler carries this exact approval into that command's
    # scoped nested-Request grant; ordinary commands gain no such authority.
    if kind == R.COMMAND_CALL:
        return Decision(UNSAFE, f"run the command /{args.get('name', '')}")

    # ── asking a human is only possible when one is there ─────────
    if kind == UI_ASK:
        # ``attended_now`` and not ``chain.attended``: a tool the agent called
        # mid-turn roots at the session key, so the bare property reads False
        # with the person sitting right there — which made asking a question
        # the one thing an interactive tool could not do.
        if not attended_now(chain):
            return Decision(UNSAFE,
                            "nobody is present to answer this question")
        return Decision(SAFE, "asking the user")

    # ── seeking authorization is not gathering information ────────
    #
    # ``ui.ask`` collects a value the plugin needs; ``ui.approve`` asks for
    # permission. Only the second is a policy decision, and it is *this*
    # layer's decision — so it is unconditionally unsafe and the approver
    # handles it like every other one. That is the whole of the consolidation:
    # the Request is the question, so the gate is the asker, and the handler
    # has nothing left to do but report that the answer was yes.
    #
    # It used to be ALWAYS_SAFE, executing a handler that reached a second
    # approval doorway (``context.approve_command``) with its own hook call,
    # its own reading of the trusted list, and no attendance check at all.
    # Routing it here retires that
    # doorway entirely rather than teaching it to agree.
    #
    # The justification becomes the reason, which is exactly the slot it
    # wants: the dialog renders it under "Why it needs asking".
    if kind == R.UI_APPROVE:
        justification = str(args.get("justification") or "")
        return Decision(UNSAFE,
                        justification or "sandboxed code asks before acting",
                        say=justification)

    # ── prompt overlays: mechanism 3 (ownership), aimed at a session ──
    #
    # This was ALWAYS_UNSAFE, which sounds right and is not, because the
    # capability it was guarding is already available with no dialog at all:
    # any loaded plugin puts arbitrary text into every prompt by declaring
    # ``agent_prompt``, and nobody is asked. So refusing the same text through
    # this Request bought no safety — it only made the *targeted*, removable,
    # per-session spelling the expensive one, which is the spelling a hook
    # should be using.
    #
    # What is genuinely new here, and stays UNSAFE, is the ``key`` argument:
    # naming a session other than your own means writing into a prompt built
    # for somebody else, possibly another user. That is the ownership question
    # ``config.write`` and ``secret.reveal`` ask about settings, one subject
    # over, and it is decided the same way — against the chain, which the
    # guest cannot misstate.
    #
    # Note the overlay persists into the state marker, so a slot outlives a
    # restart until its writer refreshes it. A stale line of guidance is not
    # a permission, and the next ``turn_start`` overwrites it.
    if kind == SESSION_ADD_PROMPT:
        target = str(args.get("key") or "")
        if not target or target == chain_session(chain):
            return Decision(SAFE, "adds prompt text to its own session")
        return Decision(UNSAFE, f"inject prompt text into session {target}",
                        say="This writes into a prompt built for another "
                            "session, which may belong to another user.")

    # ── unattended work gets less benefit of the doubt ────────────
    if kind in (SESSION_ADD_TOOL, AGENT_SCHEDULE, CONV_DELETE):
        return Decision(UNSAFE, f"{kind} ({chain.render()})")

    # ── the standing answer itself ────────────────────────────────
    #
    # Mechanisms 5 and 7 together, and both halves are load-bearing.
    #
    # **Polarity.** ``lockdown`` is the tightest of the three, so arriving at
    # it widens nothing whatever the conversation was in a moment ago — which
    # is decidable without knowing the current mode, where "is this looser
    # than what we have?" would not be. Every other value could widen, so
    # every other value is asked about.
    #
    # **Provenance**, and this is what stops lockdown being a trap. The mode
    # is enforced at the approver; the one act that leaves it must therefore
    # never reach the approver, or ``/mode ask`` would be auto-refused by the
    # very thing it exists to lift and the only way out would be restarting
    # the app. A command the person typed is its own consent, exactly as it is
    # for ``config.write`` above — and scoped the same way, so a *tool* that
    # reaches ``session.set_mode`` is judged on its own and gets a dialog.
    if kind == R.SESSION_SET_MODE:
        if chain.typed_command:
            return Decision(SAFE, "a command the user typed")
        mode = _security_mode(args.get("mode"))
        if _tightens(mode):
            return Decision(SAFE, "lockdown only narrows what may run")
        return Decision(
            UNSAFE, f"switch this conversation to {mode} mode",
            say="This changes how every later request in this conversation "
                "is answered, not just this one.")

    if kind == R.TASK_PAUSE:
        if args.get("paused", True):
            return Decision(SAFE, "pausing narrows scheduled work")
        return Decision(UNSAFE, "unpausing resumes scheduled work",
                        say="It will start picking up work again.")

    if kind == R.TASK_RESET:
        return Decision(
            UNSAFE,
            "resetting task state makes pipeline work eligible to run again",
            say="Everything it has already done becomes work to do again.",
        )

    # In ALWAYS_UNSAFE *and* branched, like CONV_DELETE, because set membership
    # alone answers "may this happen" and says nothing a person can act on:
    # the fallback below renders "session.compact changes what the system can
    # do", which is true of everything in the set. The one fact that decides
    # this dialog is that it cannot be undone, and the ``say`` half is where a
    # fact for a *person* goes — the action line above it already names the act.
    if kind == R.SESSION_COMPACT:
        return Decision(
            UNSAFE, "compacting drops the earlier turns from view for good",
            say="The full transcript stays in the database, but the "
                "conversation cannot be restored to reading in full.",
        )

    if kind in ALWAYS_UNSAFE:
        return Decision(UNSAFE, f"{kind} changes what the system can do")

    if kind in ALWAYS_SAFE:
        return Decision(SAFE, kind)

    # Anything not classified is refused. A new Request type is unsafe until
    # somebody decides otherwise, which is the right direction to fail.
    return Decision(UNSAFE, f"unclassified request: {kind}")
