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

import shlex
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
    Falls back to the system temp directory when the kernel is absent, which
    is the case in tests and in a bare container.

    The ``workspace`` tree is here for a different reason than the rest, and
    it is the point of the whole boundary — see :func:`_authoring_root`.
    """
    import tempfile

    roots = [Path(tempfile.gettempdir())]
    try:
        from paths import ATTACHMENT_CACHE
        roots.append(Path(ATTACHMENT_CACHE))
    except Exception:
        pass
    if (authoring := _authoring_root()) is not None:
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
    try:
        from runtime.context import kernel_config

        raw = (kernel_config() or {}).get("fs_writable_dirs") or []
    except Exception:
        return []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    dirs = []
    for entry in raw:
        if (text := str(entry).strip()):
            try:
                dirs.append(Path(text).expanduser())
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
    R.SESSION_ADD_TOOL, R.SESSION_ADD_PROMPT,
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
    # Ending the process. Unconditional, including the restart variant: coming
    # back up is not a mitigation, since everything in flight still dies.
    R.APP_STOP,
}

# Requests that narrow capability, or only ever affect this execution.
ALWAYS_SAFE = {
    R.SELF_RESPOND, R.FS_TEMP, R.FS_READ, R.FS_READ_BYTES,
    R.FS_LIST, R.FS_SEARCH,
    R.DB_QUERY, R.DB_DEFINE,
    R.CONV_READ, R.CONV_LIST, R.CONV_CREATE, R.CONV_APPEND,
    R.CONV_SET_TITLE, R.CONV_SET_CATEGORY, R.CONV_SET_NOTIFICATION_MODE,
    R.CONV_LOAD, R.CONV_CLEAR,
    R.SESSION_GET, R.SESSION_LIST, R.SESSION_PUSH, R.SESSION_CANCEL,
    R.SESSION_STATE_GET, R.SESSION_STATE_SET, R.SESSION_REMOVE_TOOL,
    R.SESSION_REMOVE_PROMPT,
    # UI_APPROVE is *not* here. Asking permission is a policy decision, so it
    # is always unsafe and the approver asks it — see the branch below.
    R.UI_RENDER,
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
    # The console is scoped harder still: not merely "a frontend", but the one
    # frontend that claimed it. Reading takes only what a person already typed
    # at this machine's own keyboard, and writing puts text on the screen in
    # front of them — neither reaches past the console, and gating them would
    # mean asking permission to show the prompt that asks permission.
    R.CONSOLE_READ, R.CONSOLE_WRITE,
    R.TASK_ENQUEUE, R.TASK_STATUS, R.TASK_OUTPUT,
    R.TASK_LIST, R.TASK_GRAPH, R.TASK_TRIGGER,
    # The three ways of speaking about a process this system already started.
    # Listing and asking after one read a registry the kernel owns, and it
    # holds nothing that was not approved at ``proc.start``. Stopping is here
    # for the same reason SESSION_REMOVE_TOOL is: it narrows. A dev server the
    # agent cannot kill without a dialog is a dev server the agent will not
    # start, and the alternative to stopping it is leaving it running.
    R.PROC_STATUS, R.PROC_LIST, R.PROC_STOP,
    R.FILE_REGISTER, R.FILE_LIST,
    R.PARSE_FILE, R.PARSE_MODALITY,
    R.LEDGER_RECORD, R.LEDGER_READ,
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
             UI_ASK, R.UI_APPROVE, SESSION_ADD_TOOL,
             SESSION_ADD_PROMPT, AGENT_SCHEDULE, CONV_DELETE,
             R.TASK_PAUSE, R.TASK_RESET}
_UNDECIDED = R.ALL_TYPES - ALWAYS_SAFE - ALWAYS_UNSAFE - _BRANCHED
assert not _UNDECIDED, f"unclassified Requests: {sorted(_UNDECIDED)}"


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
        return Decision(UNSAFE, f"delete rows from {table}")
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
    try:
        from runtime.context import kernel_config

        raw = (kernel_config() or {}).get("net_allowed_hosts") or []
    except Exception:
        return set()
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    return {str(host).strip().lower().lstrip(".") for host in raw
            if str(host).strip()}


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
    return Decision(UNSAFE, f"outbound request to {url}{where}")


# ──────────────────────────────────────────────────────────────────────
# The shell.
# ──────────────────────────────────────────────────────────────────────
#
# Everything is asked. Why there is no classifier here and why there must not
# be one again is in CLAUDE.md, "And it is where the classifier died".
#
# It is meant to get less onerous, and this is where better goes. A *recognizer*
# takes the rendered command line and returns a reason to allow it, or None to
# stay out of the way. Two kinds are expected:
#
#   - structural — "every segment of this pipeline is a read-only command",
#     the classifier the store tool used to carry, rebuilt where the policy
#     can see it rather than inside the plugin it authorizes
#   - remembered — "the user already approved exactly this, in this session /
#     for this chain root", which needs somewhere to persist a decision and a
#     scope to persist it against, and is the more useful of the two
#
# Adding to it is a deliberate widening of the authorization surface, so it
# should be as visible as this comment makes it. One recognizer lives here; it
# is the structural kind, and it is built below.


# ──────────────────────────────────────────────────────────────────────
# The read-only recognizer.
# ──────────────────────────────────────────────────────────────────────
#
# This is the classifier's second attempt, and the difference is entirely in
# what it refuses to attempt. The one that died tried to decide *every*
# command: split the line at unquoted ``&&``/``||``/``;``/``|``, match each
# segment against a whitelist of program names, send redirection and
# substitution to approval. Deciding an arbitrary command line is Rice's
# theorem, and it failed in the invisible direction — a wrong "unsafe" gets
# reported, a wrong "safe" does not.
#
# Deciding *a few* commands and abstaining on everything else is not the same
# problem, and it is trivial. Three rules keep it that way:
#
#   1. **Abstain, never deny.** A recognizer returns a reason or ``None``; it
#      has no authority to say "unsafe" because nothing needs it to. A bug
#      here costs a dialog. That is the shape ``_SHELL_RECOGNIZERS`` already
#      had, and it is what makes the rest of this safe to write at all.
#
#   2. **The unit is (program, subcommand).** ``git`` is not read-only;
#      ``git status`` is. ``find`` is read-only right up until ``-exec``.
#      A whitelist keyed on the program name is wrong at the root, which is
#      most of why the first one could not be fixed.
#
#   3. **No parsing.** Any shell at all, any metacharacter anywhere, and this
#      abstains without looking further. It reads the *structured* argv and
#      not the rendered line, because the rendered line is precisely the thing
#      that cannot be reasoned about — quoting, ``$(...)``, backticks, aliases
#      and ``eval`` all beat a parser, so there is no parser.
#
# What it buys is the high-frequency, zero-information prompts: a person
# clicking through ``git status`` forty times a day is being trained not to
# read dialogs, and that training is what the one dialog that mattered will
# run into. What it deliberately never covers is anything that runs arbitrary
# code by design — ``npm install``, ``pip install``, ``pytest``, ``make``,
# ``python script.py``. Those are the minority by count and the majority by
# consequence, and they stay asked forever.
#
# **Sound against carelessness, not against a hostile repository.** Even
# ``git diff`` can invoke an external diff driver named by a repo's
# ``.gitattributes``. That is the documented threat model (see the security
# contract), and it is the line this sits on.

#: Argv elements containing any of these abstain. With ``shell=None`` the
#: handler executes the list as given, so none of them can actually *do*
#: anything — but an argv carrying one was written by somebody who expected a
#: shell, and abstaining on that costs a dialog and settles it.
_SHELL_METACHARACTERS = frozenset("|&;<>$`()\n\r")

#: ``(program, subcommand) -> (allowed_flags, positionals_ok)``. A subcommand
#: of ``""`` means the program takes none. An unknown flag abstains, which is
#: what keeps this a whitelist rather than a deny list.
#:
#: ``positionals_ok`` exists because a bare word means opposite things to
#: neighbouring verbs: to ``git log`` it is a ref or a path to read, to
#: ``git branch`` it is a branch to *create*. Allowing free positionals
#: everywhere would have let ``git remote add origin <url>`` through, which is
#: the concrete bug that makes the pair the right unit.
#:
#: Deliberately absent, each for a reason worth keeping written down:
#:   - ``pytest --collect-only`` — collection *imports* every test module, so
#:     module-level code runs. It looks inert and is not.
#:   - ``find`` — read-only until ``-exec``, and ``-exec`` is a flag.
#:   - ``git config`` — sets exactly as readily as it gets.
#:   - ``cat``/``ls`` — ``sdk.fs.read`` and ``sdk.fs.list`` do these mediated
#:     and better. A dialog is the right nudge toward the SDK.
_READ_ONLY_COMMANDS = {
    ("git", ""): (frozenset({"--version"}), False),
    ("git", "status"): (frozenset({"-s", "--short", "-b", "--branch",
                                   "--porcelain", "--long"}), False),
    ("git", "log"): (frozenset({"--oneline", "--graph", "--decorate", "--stat",
                                "--pretty", "--format", "--author", "--since",
                                "--until", "--max-count", "-n", "--no-merges",
                                "--name-only", "--name-status", "--reverse"}),
                     True),
    ("git", "diff"): (frozenset({"--stat", "--cached", "--staged", "--numstat",
                                 "--shortstat", "--name-only",
                                 "--name-status"}), True),
    ("git", "show"): (frozenset({"--stat", "--oneline", "--pretty", "--format",
                                 "--name-only"}), True),
    ("git", "branch"): (frozenset({"-a", "--all", "-v", "-vv", "--list",
                                   "--show-current"}), False),
    ("git", "remote"): (frozenset({"-v", "--verbose"}), False),
    ("git", "rev-parse"): (frozenset({"--abbrev-ref", "--short",
                                      "--show-toplevel", "--git-dir",
                                      "--is-inside-work-tree"}), True),
    # Version and identity queries: nothing to configure, nothing to write.
    ("python", ""): (frozenset({"--version", "-V"}), False),
    ("python3", ""): (frozenset({"--version", "-V"}), False),
    ("node", ""): (frozenset({"--version", "-v"}), False),
    ("npm", ""): (frozenset({"--version", "-v"}), False),
    ("pip", ""): (frozenset({"--version", "-V"}), False),
    ("pwd", ""): (frozenset(), False),
    ("whoami", ""): (frozenset(), False),
    ("hostname", ""): (frozenset(), False),
}


#: Characters a shell has nothing to say about. A **whitelist**, because the
#: interesting question is not "which characters are dangerous" — that list is
#: never finished — but "which are provably inert", which is short and closed.
#: Letters, digits, space, and the punctuation that appears in program names,
#: flags, versions and POSIX paths.
#:
#: Everything else is out, including several that look harmless: ``~`` (tilde
#: expansion), ``*`` and ``?`` (globs), quotes and ``\`` (both change how the
#: line splits), ``%`` and ``^`` (cmd.exe), ``!`` (history), ``#`` (comment).
_SHELL_INERT = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 -_./:@+,=")


def _exact_argv(args: dict):
    """The argv the handler will execute, or ``None`` if it cannot be known.

    Mirrors ``handlers.fs_net._invocation`` deliberately: a recognizer that
    reasons about a different command than the one that runs is worse than no
    recognizer.

    ``shell=None`` runs the argv as given, so that case is exact. A *named*
    shell (``default``, ``cmd``, ``powershell``) hands the line to something
    that parses it, and what a parser will make of an arbitrary string is the
    question this module refuses to answer — **unless the string contains
    nothing a parser could act on.** ``git pull`` under ``sh -c`` runs exactly
    ``git`` with the argument ``pull``; there is no expansion, no splitting
    ambiguity and no substitution to be wrong about, so whitespace splitting is
    what the shell itself will do.

    That carve-out is not a nicety. ``tool_run_command`` defaults ``shell`` to
    ``"default"``, so without it *every* command in practice came back ``None``
    — which quietly disabled both recognizers. ``git status`` was never
    auto-allowed and no command was ever offered a remembered grant, while the
    tests passed because they construct Requests with no shell at all.
    """
    argv = args.get("argv")
    if args.get("shell") is not None:
        line = argv if isinstance(argv, str) else " ".join(
            str(part) for part in (argv or []))
        if not line.strip() or set(line) - _SHELL_INERT:
            return None
        return line.split()
    if isinstance(argv, str):
        try:
            argv = shlex.split(argv)
        except ValueError:
            return None
    return [str(part) for part in (argv or [])]


def _read_only_command(shown: str, args: dict):
    """Allow a known read-only invocation, or abstain.

    ``shown`` is ignored on purpose: the rendered line is for the person and
    the ledger, and reading it back to make a decision is the mistake the
    first classifier was built on.
    """
    argv = _exact_argv(args)
    if not argv:
        return None
    if any(_SHELL_METACHARACTERS & set(part) for part in argv):
        return None

    program = argv[0]
    # A bare name only. ``/tmp/somewhere/git`` is not the ``git`` this list is
    # talking about, and resolving which one it is means trusting PATH.
    if "/" in program or "\\" in program:
        return None
    program = program.lower().removesuffix(".exe")

    rest = argv[1:]
    subcommand = ""
    if rest and not rest[0].startswith("-"):
        subcommand, rest = rest[0].lower(), rest[1:]

    entry = _READ_ONLY_COMMANDS.get((program, subcommand))
    if entry is None:
        return None
    flags, positionals_ok = entry
    for part in rest:
        if part == "--" and positionals_ok:
            continue                      # ends flag parsing; changes nothing
        if part.startswith("-"):
            # ``--flag=value`` is one token; the value cannot turn a read into
            # a write, so only the name is checked.
            if part.split("=", 1)[0] not in flags:
                return None
        elif not positionals_ok:
            return None
    named = f"{program} {subcommand}".strip()
    return f"{named} only reads"


def command_prefix(argv) -> str:
    """The unit a shell grant is written down as, or "" for none.

    **Shared by the recognizer that reads the list and the dialog option that
    writes to it**, for the reason ``render_command`` is shared by the dialog
    and the ledger: a grant stored in one vocabulary and matched in another is
    a grant nobody can reason about.

    ``(program, subcommand)``, and a raw string prefix deliberately not —
    ``"git push"`` also prefixes ``"git push && rm -rf /"``. Everything the
    read-only recognizer refuses to look at is refused here too: a named shell
    (the caller passes :func:`_exact_argv`, which is already ``None`` there),
    any metacharacter anywhere, a program that is not a bare name.

    One rule of its own: **a second word that names a file is not a verb.**
    ``python train.py`` reduces to ``python``, not to ``python train.py``,
    because the latter reads like a grant for one known script while actually
    granting whatever that file says tomorrow. Reducing keeps the label honest
    about what is being handed over — a person shown "Always allow: python"
    knows exactly how much that is, and can decline.
    """
    argv = list(argv or [])
    if not argv:
        return ""
    if any(_SHELL_METACHARACTERS & set(part) for part in argv):
        return ""
    program = argv[0]
    if "/" in program or "\\" in program:
        return ""
    program = program.lower().removesuffix(".exe")
    if not program:
        return ""
    rest = argv[1:]
    if rest and not rest[0].startswith("-"):
        word = rest[0]
        if "/" not in word and "\\" not in word and not Path(word).suffix:
            return f"{program} {word.lower()}"
    return program


def _allowed_prefixes() -> set:
    """Command prefixes the user has said may run without being asked.

    Read live on every call, like :func:`_allowed_hosts` — revoking has to be
    as immediate as granting, and the user revokes with ``/config``.
    """
    try:
        from runtime.context import kernel_config

        raw = (kernel_config() or {}).get("shell_allowed_prefixes") or []
    except Exception:
        return set()
    if isinstance(raw, str):
        raw = raw.split(",")
    return {" ".join(str(entry).split()).casefold()
            for entry in raw if str(entry).strip()}


def _remembered_prefix(shown: str, args: dict):
    """Allow a command whose prefix the user already answered "always" to.

    The *remembered* recognizer this section has been describing since it
    shipped empty: a decision persisted where the policy can see it, rather
    than inside the plugin it authorizes.

    It re-derives the prefix from the structured argv instead of matching
    ``shown``, which is the entire soundness argument — see
    :func:`command_prefix`.
    """
    prefix = command_prefix(_exact_argv(args))
    if prefix and prefix.casefold() in _allowed_prefixes():
        return f"you allowed `{prefix}`"
    return None


_SHELL_RECOGNIZERS: list = [_read_only_command, _remembered_prefix]


def render_command(args: dict) -> str:
    """The command line a shell Request is asking for, as a person reads it.

    One function because three callers need to agree: the dialog, the ledger
    row, and any future recognizer. A list and the string it would join to
    must describe the same act, or the record and the question drift apart.
    """
    argv = args.get("argv")
    if isinstance(argv, str):
        rendered = argv
    else:
        rendered = " ".join(str(part) for part in (argv or []))
    shell = args.get("shell")
    return f"{rendered} [{shell}]" if shell and shell != "default" else rendered


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
            joined = f"{len(items)} entries"
        return f"set {key} to [{joined}]"
    if isinstance(shown, dict):
        return f"set {key} ({len(shown)} entries)"
    if isinstance(shown, bool) or isinstance(shown, (int, float)):
        return f"set {key} to {shown}"
    text = str(shown)
    if len(text) > _VALUE_CHARS:
        text = text[:_VALUE_CHARS] + "…"
    return f"set {key} to {text!r}"


def _classify_shell(kind: str, args: dict) -> Decision:
    """Decide about one shell Request. See the section comment above."""
    shown = render_command(args)
    for recognize in _SHELL_RECOGNIZERS:
        try:
            if (why := recognize(shown, args)):
                return Decision(SAFE, why)
        except Exception:
            # A recognizer that raises abstains. It can only ever widen, so
            # failing it closed costs a dialog and nothing else.
            continue
    verb = "start" if kind == R.PROC_START else "run"
    where = args.get("cwd")
    return Decision(UNSAFE,
                    f"{verb} shell command: {shown[:200]}"
                    + (f" (in {where})" if where else ""))


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
        return Decision(UNSAFE, "no script named")
    path = resolve_script(path) or path
    if not is_script(path):
        # Not a refusal of *this* file so much as of the shape of the ask: the
        # containment story rests entirely on the file living somewhere the
        # kernel subprocesses unconditionally. Anywhere else and the honest
        # answer is that nobody knows what this is.
        return Decision(UNSAFE, f"{path} is not in a scripts/ directory")

    report = _script_report(path)
    if report is None:
        return Decision(UNSAFE, f"could not read {path}")
    if not report.ok:
        # It would be refused by the loader moments later anyway. Saying so
        # here means the user is not asked to approve something that cannot
        # run, which is the worst possible thing to put in a dialog.
        return Decision(UNSAFE, f"{Path(path).name} does not pass validation")
    if report.unmediated:
        # The one case that is asked about. An installed package importing a
        # foreign library is subprocessed and *not* asked, because somebody
        # approved it once at ``plugin.install``; a script was never approved
        # by anyone, and a library the validator cannot see inside is the only
        # part of a script whose effects do not come back through this
        # function. Naming it is most of the value of the dialog.
        libraries = ", ".join(sorted(report.unmediated))
        return Decision(UNSAFE,
                        f"run {Path(path).name}, which imports {libraries} — "
                        f"that library's own actions are not mediated")
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
    belongs in the set. The box link is commonly the source-file stem
    (``service_timekeeper``), which is why the root is added rather than
    relied upon alone.
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
# Of the ~109 Request types, the great majority are a flat lookup: a set
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
#      ``_SHELL_RECOGNIZERS`` holds two — a structural read-only check and a
#      *remembered* one reading ``shell_allowed_prefixes``. ``_NET_RECOGNIZERS``
#      is empty, since egress is served by its allowlist directly. This is the
#      only one of the eight that is an extension point rather than a rule, and
#      it can only ever widen — see ``docs/PERMISSIONS_MAP.md``.
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
        return Decision(UNSAFE, f"call chain deeper than {MAX_DEPTH}")
    if chain.cyclic:
        return Decision(UNSAFE, f"call cycle: {chain.render()}")
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
    if kind in (PROC_RUN, R.PROC_START):
        return _classify_shell(kind, args)

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
        return Decision(UNSAFE, f"write to {args.get('path')}")

    if kind == FS_MOVE:
        # Both ends, because a move is a write to one place and a delete from
        # the other. Dragging a file *out* of an allowed folder is exactly as
        # much a change to somewhere unlisted as writing there would be.
        if (_freely_writable(args.get("src"))
                and _freely_writable(args.get("dst"))):
            return Decision(SAFE, _write_reason(args.get("dst"), verb="move"))
        return Decision(UNSAFE,
                        f"move {args.get('src')} to {args.get('dst')}")

    if kind == FS_DELETE:
        if _freely_writable(args.get("path")):
            return Decision(SAFE, _write_reason(args.get("path"),
                                                verb="delete"))
        return Decision(UNSAFE, f"delete {args.get('path')}")

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
        return Decision(UNSAFE, f"plaintext of {name or 'a secret'}")

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
    # name it exactly. (Commands that carry ``require_approval`` never reach
    # here at all — the handler refuses them, because the answer they need is
    # one only the state machine can obtain.)
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
    # its own reading of ``skip_permissions`` — by tool name rather than by
    # chain — and no attendance check at all. Routing it here retires that
    # doorway entirely rather than teaching it to agree.
    #
    # The justification becomes the reason, which is exactly the slot it
    # wants: the dialog renders it under "Why it needs asking".
    if kind == R.UI_APPROVE:
        return Decision(UNSAFE, str(args.get("justification") or "")
                        or "sandboxed code asks before acting")

    # ── unattended work gets less benefit of the doubt ────────────
    if kind in (SESSION_ADD_TOOL, SESSION_ADD_PROMPT, AGENT_SCHEDULE,
                CONV_DELETE):
        return Decision(UNSAFE, f"{kind} ({chain.render()})")

    if kind == R.TASK_PAUSE:
        if args.get("paused", True):
            return Decision(SAFE, "pausing narrows scheduled work")
        return Decision(UNSAFE, "unpausing resumes scheduled work")

    if kind == R.TASK_RESET:
        return Decision(
            UNSAFE,
            "resetting task state makes pipeline work eligible to run again",
        )

    if kind in ALWAYS_UNSAFE:
        return Decision(UNSAFE, f"{kind} changes what the system can do")

    if kind in ALWAYS_SAFE:
        return Decision(SAFE, kind)

    # Anything not classified is refused. A new Request type is unsafe until
    # somebody decides otherwise, which is the right direction to fail.
    return Decision(UNSAFE, f"unclassified request: {kind}")
