"""The SDK — what sandboxed code imports.

Two kinds of thing live here, and the boundary between them is the whole
design:

- **Requests** (``sdk.fs``, ``sdk.db``, ``sdk.net``, …) yield to the kernel.
  They block, they are classified, they may be refused, and they land in the
  ledger. Each namespace is one Request family, so ``sdk.fs.read`` is the
  ``fs.read`` Request and the catalogue reads as a table of contents for the
  SDK.
- **Helpers** (``sdk.text``, ``sdk.md``) are plain functions running inside
  the sandbox. They cost nothing, need no approval, and never reach the
  kernel.

The test for which is which: *does it touch disk, network, clock, or process?*
If no, it belongs in a helper. If yes, it is a Request.

Plugin code looks like ordinary synchronous Python. The suspend/resume loop is
real, but it lives on the kernel side — the author never writes ``yield``, and
helper functions can make Requests freely without becoming generators.

**Requests return their value and raise when they fail.** That is Python's
answer to an operation that can fail, and it keeps plugin code to the shape
it would have had without a sandbox at all::

    def run(self, sdk, path):
        # Count the words in a file.
        return len(sdk.fs.read(path).split())

No result object to unwrap, no branch to write. If the read fails, the runner
turns the exception into a failed result carrying the reason — which is what
the caller wanted anyway.

Handling a failure is an ordinary ``try``, and refusals have their own class
so "the user said no" can be caught without also swallowing "the disk is
full"::

    try:
        page = sdk.net.http(url)
    except sdk.Denied:
        return "I need permission to fetch that."

Returning is just as plain: return any value and the runner wraps it. Reach
for ``sdk.ok(...)`` only to attach ``llm_summary`` or attachments, and
``sdk.fail(...)`` only to fail without raising.
"""

from __future__ import annotations

import base64
import difflib
import json as json_module
# ``ntpath``/``posixpath`` rather than ``os.path``, which is one of these two
# under a name that also drags in ``os``. The guest ships stdlib-only and
# environment-free (pinned by tests/test_sandbox_guest_boundary.py), and these
# two modules are pure string arithmetic — no cwd, no stat, no environment.
# ``os.path`` would have imported the very module the boundary exists to keep
# out, to get functions these already provide.
import ntpath
import posixpath
import sys

from .channel import Terminated
from .requests import Denied, RequestFailed
from .requests import (AGENT_COLLECT, AGENT_COMPLETE, AGENT_SCHEDULE,
                       AGENT_SPAWN, AGENT_STOP, APP_STOP,
                       COMMAND_CALL, COMMAND_LIST, CONFIG_READ, CONFIG_WRITE,
                       CONV_APPEND, CONV_CLEAR, CONV_CREATE, CONV_DELETE, CONV_LIST,
                       CONV_LOAD, CONV_NEW, CONV_READ, CONV_SET_CATEGORY,
                       CONV_SET_NOTIFICATION_MODE, CONV_SET_TITLE,
                       CRON_CREATE, CRON_ENABLE, CRON_GET, CRON_LIST,
                       CONSOLE_READ, CONSOLE_WRITE,
                       HTTP_CLOSE, HTTP_DRAIN, HTTP_PUSH, HTTP_RESPOND,
                       CRON_REMOVE, CRON_UPDATE, DB_DEFINE, DB_QUERY, DB_WRITE,
                       ENV_READ, EVENT_EMIT, EVENT_REQUEST, FILE_LIST,
                       FILE_REGISTER, FRONTEND_ACT, FRONTEND_ATTEND,
                       FRONTEND_BIND, FRONTEND_CANCEL, FRONTEND_COLLECT,
                       FRONTEND_PENDING, FRONTEND_RESOLVE, FRONTEND_SUBMIT,
                       FS_DELETE, FS_LIST, FS_MKDIR, FS_MOVE, FS_READ,
                       FS_READ_BYTES,
                       FS_SEARCH, FS_STAT, FS_TEMP, FS_WRITE, FS_WRITE_BYTES,
                       LEDGER_READ,
                       LEDGER_RECORD, NET_HTTP, NOTIFICATION_LIST,
                       NOTIFICATION_MARK_READ, PARSE_FILE, PARSE_MODALITY,
                       LLM_DELTA, LLM_LIST, LLM_LOAD, LLM_PROCEED,
                       LLM_UNLOAD, PATH_GET,
                       PLUGIN_DESCRIBE, PLUGIN_INSTALL, PLUGIN_LIST,
                       PLUGIN_REGISTER, PLUGIN_RELOAD, PLUGIN_UNREGISTER,
                       PLUGIN_UNINSTALL, PLUGIN_UPDATE, PLUGIN_VALIDATE,
                       PROC_LIST, PROC_RUN, PROC_START, PROC_STATUS,
                       PROC_STOP,
                       SCRIPT_COLLECT, SCRIPT_RUN, SCRIPT_STOP,
                       SECRET_REVEAL, SELF_BUDGET, SELF_RESPOND,
                       SERVICE_CALL, SERVICE_LIST, SERVICE_LOAD,
                       SERVICE_UNLOAD, SESSION_ADD_ATTACHMENT,
                       SESSION_ADD_PROMPT,
                       SESSION_ADD_TOOL, SESSION_CANCEL, SESSION_COMPACT,
                       SESSION_GET,
                       SESSION_LIST, SESSION_PUSH, SESSION_REMOVE_PROMPT,
                       SESSION_REMOVE_TOOL, SESSION_SET_MODE,
                       SESSION_STATE_GET,
                       SESSION_STATE_SET, TASK_ENQUEUE, TASK_GRAPH, TASK_LIST,
                       TASK_OUTPUT, TASK_PAUSE, TASK_RESET, TASK_STATUS,
                       TASK_TRIGGER, TOOL_CALL, TOOL_LIST, UI_APPROVE, UI_ASK,
                       UI_PROGRESS, UI_RENDER, USER_LIST, USER_READ,
                       USER_WRITE, Request,
                       Result)


class _Namespace:
    """Base for Request-making SDK namespaces."""

    def __init__(self, sdk: "SDK"):
        self._sdk = sdk

    def __getattr__(self, name: str):
        """Name the closest method rather than failing blank.

        A miss here is almost always a guess at a surface the author has not
        read — ``sdk.fs.readfile`` for ``sdk.fs.read``. The validator already
        treats a near-miss as a teaching opportunity, suggesting Request types
        the same way; this is that idea applied to the SDK itself, and it is
        worth more here because it fires at the moment of the mistake.
        """
        if name.startswith("_"):
            raise AttributeError(name)     # dunder probing, not a typo
        # Every namespace class is its attribute name with an underscore and
        # some capitals: _FS -> fs, _Conv -> conv, _LLM -> llm.
        space = type(self).__name__.lstrip("_").lower()
        methods = sorted(m for m in dir(type(self)) if not m.startswith("_"))
        close = difflib.get_close_matches(name, methods, 1, 0.6)
        hint = f" Did you mean sdk.{space}.{close[0]}?" if close else ""
        raise AttributeError(f"sdk.{space} has no {name!r}.{hint} "
                             f"{space} has: {', '.join(methods)}")

    def _ask(self, kind: str, **args):
        """Build a Request, send it, and return what it produced.

        Raises :class:`Denied` when the kernel refused and
        :class:`RequestFailed` when it broke, so callers write straight-line
        code and handle the exceptional case only when they have something to
        do about it.
        """
        result = self._sdk._send(Request(kind, args))
        if result.ok:
            return result.data
        raise (Denied if result.denied else RequestFailed)(result, kind)


class _FS(_Namespace):
    """Filesystem Requests."""

    def read(self, path):
        """Read a file as text."""
        return self._ask(FS_READ, path=str(path))

    def write(self, path, data: str, mode: str = "overwrite"):
        """Create, overwrite, or append. ``mode="append"`` to add.

        Missing parent folders are created, so writing ``out/report.json``
        into an empty workspace works without making ``out/`` first.
        """
        return self._ask(FS_WRITE, path=str(path), data=data, mode=mode)

    def read_bytes(self, path, offset: int = 0, length: int = 0) -> bytes:
        """Read a file as raw bytes, or one window of it.

        Use this for anything that is not text — an image, audio, a PDF.
        ``read`` decodes as UTF-8 with replacement, which silently mangles
        binary content rather than failing.

        One answer has to fit in one wire message, and base64 costs a third on
        top, so a whole-file read is capped well below the file sizes a
        frontend deals in. ``offset``/``length`` are the way past that: ask for
        successive windows and join them. Reading past the end returns fewer
        bytes than asked for, and an ``offset`` at or past the end returns
        ``b""`` — so the loop terminates on a short read, not on a size you had
        to learn first.
        """
        args = {"path": str(path)}
        # Sent only when asked for, so the plain call is byte-identical on the
        # wire to what it always was.
        if offset:
            args["offset"] = int(offset)
        if length:
            args["length"] = int(length)
        return base64.b64decode(self._ask(FS_READ_BYTES, **args) or "")

    def iter_bytes(self, path, chunk_size: int = 4 * 1024 * 1024,
                   offset: int = 0, limit=None):
        """Yield a binary file in wire-sized windows.

        ``offset`` is where reading starts. ``limit`` caps the total bytes
        yielded from there; ``None`` reads to EOF. Each window is an ordinary
        ``fs.read_bytes`` Request.
        """
        chunk_size = int(chunk_size)
        offset = int(offset)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if offset < 0:
            raise ValueError("offset must not be negative")
        remaining = None if limit is None else int(limit)
        if remaining is not None and remaining < 0:
            raise ValueError("limit must not be negative")

        while remaining is None or remaining:
            length = (chunk_size if remaining is None
                      else min(chunk_size, remaining))
            chunk = self.read_bytes(path, offset=offset, length=length)
            if not chunk:
                break
            yield chunk
            offset += len(chunk)
            if remaining is not None:
                remaining -= len(chunk)
            if len(chunk) < length:
                break

    def stat(self, path):
        """Return metadata for one path; raise when it is missing."""
        return self._ask(FS_STAT, path=str(path))

    def exists(self, path) -> bool:
        """Whether a readable path exists."""
        return self._ask(
            FS_STAT, path=str(path), missing_ok=True) is not None

    def write_bytes(self, path, data, mode: str = "overwrite"):
        """Write raw bytes. ``mode="append"`` to add.

        A ``str`` is encoded as UTF-8 rather than refused — the mistake is
        harmless and the alternative is a TypeError from deep inside the SDK.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        return self._ask(FS_WRITE_BYTES, path=str(path),
                         data=base64.b64encode(bytes(data)).decode("ascii"),
                         mode=mode)

    def list(self, path, pattern: str = "*", details: bool = False,
             recursive=None, files_only=None, sort=None, limit=None):
        """List a directory, optionally with entry metadata.

        ``details`` adds ``is_dir``, ``size`` and ``mtime`` (``st_mtime_ns``,
        an int — compare with ``!=``, since a file restored to an older
        version has also changed).

        Pointing this at a **file** returns that one entry, which is how you
        ask "has this changed?" without building a glob out of a filename.

        **Not an existence test as written.** A missing path is a failed
        Request. Use :meth:`exists` when absence is the expected answer, or
        :meth:`stat` when you need one path's metadata.

        Passing any of ``recursive`` / ``files_only`` / ``sort`` / ``limit``
        switches on the walking listing and changes the answer's shape to
        ``{"root", "entries", "truncated", "scan_truncated"}``. Use it for
        anything tree-shaped: it prunes ``.git``, ``node_modules`` and friends,
        never follows a symlink, and caps the enumeration, none of which a
        plain glob does. ``sort="mtime"`` is newest-first.
        """
        args = {"path": str(path), "pattern": pattern, "details": details}
        # Sent only when asked for: the handler decides which shape to answer
        # in by whether these keys are *present*, so a default value here
        # would silently move every existing caller onto the new shape.
        for key, value in (("recursive", recursive), ("files_only", files_only),
                           ("sort", sort), ("limit", limit)):
            if value is not None:
                args[key] = value
        return self._ask(FS_LIST, **args)

    def search(self, pattern: str, root=".", glob: str = "**/*",
               regex=None, case_insensitive=None, multiline=None,
               mode=None, context_lines=None, limit=None):
        """Search file contents beneath a root.

        Plain, this is a substring scan returning ``{path, line, text}`` hits.
        Passing any of the arguments below switches on the full search and
        answers with ``{"root", "mode", "results", "truncated",
        "scan_truncated", "skipped_binary", "skipped_large", "backend"}``:

        - ``regex`` — read ``pattern`` as a Python ``re`` pattern.
        - ``case_insensitive``, ``multiline`` (``.`` matches newlines too).
        - ``mode`` — ``"content"`` (``rel:lineno: text`` lines, the default),
          ``"files"`` (matching paths), or ``"count"`` (``[path, n]`` pairs).
        - ``context_lines`` — lines either side of each hit, content mode, max 10.
        - ``limit`` — result cap; default 100, max 500.

        The full search prunes junk directories, skips binary and oversized
        files, and uses ripgrep when it is installed. ``glob`` filters which
        files are searched: ``"*.py"`` is top level only, ``"**/*.py"`` any depth.
        """
        args = {"pattern": pattern, "root": str(root), "glob": glob}
        for key, value in (("regex", regex),
                           ("case_insensitive", case_insensitive),
                           ("multiline", multiline), ("mode", mode),
                           ("context_lines", context_lines), ("limit", limit)):
            if value is not None:
                args[key] = value
        return self._ask(FS_SEARCH, **args)

    def delete(self, path):
        """Remove a file or a tree."""
        return self._ask(FS_DELETE, path=str(path))

    def move(self, src, dst, copy: bool = False):
        """Move or copy one path to another."""
        return self._ask(FS_MOVE, src=str(src), dst=str(dst), copy=copy)

    def mkdir(self, path, exist_ok: bool = True):
        """Create a directory and any missing parents.

        **You rarely need this.** :meth:`write` and :meth:`write_bytes` already
        create the folders above the file, so ``write("out/report.json", ...)``
        makes ``out/`` on its own. Reach for this only when a directory has to
        exist while still empty.

        ``exist_ok=False`` to fail when it is already there.
        """
        return self._ask(FS_MKDIR, path=str(path), exist_ok=bool(exist_ok))

    def temp(self, directory: bool = False, suffix: str = ""):
        """Scratch space you may always have."""
        return self._ask(FS_TEMP, directory=directory, suffix=suffix)


class _DB(_Namespace):
    """Database Requests.

    Reads are broad; what is narrowed is *whose* rows. User-scoped tables are
    reached through their ``my_`` name — ``my_conversations`` rather than
    ``conversations``.
    """

    def query(self, sql: str, params=None, max_rows: int = 0):
        """Read rows.

        Answers a list of dicts, capped by the kernel (``max_rows`` may only
        lower that cap). Getting exactly the cap back means there was more.
        """
        return self._ask(DB_QUERY, sql=sql, params=list(params or []),
                         max_rows=int(max_rows or 0))

    def write(self, sql: str, params=None):
        """Insert, update or delete."""
        return self._ask(DB_WRITE, sql=sql, params=list(params or []))

    def define(self, ddl: str):
        """Create a table this plugin owns."""
        return self._ask(DB_DEFINE, ddl=ddl)


class _Conv(_Namespace):
    """Conversation Requests."""

    def create(
        self,
        title: str = "",
        *,
        category=None,
        activate: bool = False,
    ):
        """Create a current-user conversation and optionally activate it."""
        return self._ask(
            CONV_CREATE, title=title, category=category, activate=activate)

    def read(self, conversation_id, details: bool = False, *,
             limit: int | None = None, before_id=None, since_id=None,
             max_bytes: int | None = None):
        """One page of a conversation, newest by default, plus its metadata.

        Answers ``{conversation, messages, has_more, oldest_id, newest_id}``,
        and with ``details`` also ``agent_profile`` and ``notification_mode``.
        Messages always arrive oldest-first, whichever way you paged to them.

        **It is a page, not the conversation.** A transcript grows forever —
        compaction shrinks what the model sees and deletes nothing — so a call
        that answered with all of it was a call that eventually could not be
        answered at all. Read ``has_more`` and page, or accept that you are
        looking at the recent end.

        - ``before_id`` walks backwards: the newest rows *older* than that id.
          This is what a scrollback asks for as somebody scrolls up.
        - ``since_id`` walks forwards: the oldest rows *newer* than that id.
          ``since_id=0`` is therefore how to ask for the very start of a
          conversation, which is what a titler or a summariser wants.
        - ``limit=0`` asks for no messages at all — for when you came only for
          the conversation's own row and would otherwise pull a transcript to
          read a title.

        Kernel bookkeeping never comes back: the state machine's markers are
        filtered out kernel-side. Compaction markers do, because those say
        something about the conversation rather than about the kernel.
        """
        payload = {"id": conversation_id, "details": details}
        if limit is not None:
            payload["limit"] = limit
        if before_id is not None:
            payload["before_id"] = before_id
        if since_id is not None:
            payload["since_id"] = since_id
        if max_bytes is not None:
            payload["max_bytes"] = max_bytes
        return self._ask(CONV_READ, **payload)

    def list(
        self,
        *,
        category=None,
        limit: int = 50,
        offset: int = 0,
        details: bool = False,
    ):
        """Current-user conversations, optionally with category metadata.

        ``category``: ``None`` for every conversation, ``""`` for the Main
        bucket, or a name. ``offset`` pages: ``details`` answers ``has_more``,
        so a caller walks until that is false rather than guessing a total.
        """
        return self._ask(
            CONV_LIST, category=category, limit=limit, offset=offset,
            details=details)

    def append(self, conversation_id, role: str, content: str):
        """Add a message."""
        return self._ask(CONV_APPEND, id=conversation_id, role=role,
                         content=content)

    def set_title(self, conversation_id, title: str):
        """Retitle."""
        return self._ask(CONV_SET_TITLE, id=conversation_id, title=title)

    def set_category(self, conversation_id, category: str):
        """Categorize."""
        return self._ask(CONV_SET_CATEGORY, id=conversation_id,
                         category=category)

    def set_notification_mode(self, conversation_id, mode: str):
        """Change background notification behavior."""
        return self._ask(
            CONV_SET_NOTIFICATION_MODE, id=conversation_id, mode=mode)

    def load(self, conversation_id):
        """Load a conversation and its saved state into this session."""
        return self._ask(CONV_LOAD, id=conversation_id)

    def new(self):
        """Start a fresh conversation — the counterpart to ``load``.

        This writes nothing. A conversation is created by the first message
        sent into it, so calling this twice with nothing said in between costs
        nothing and leaves nothing behind. The conversation you were in stays
        where it is and can be loaded again.
        """
        return self._ask(CONV_NEW)

    def clear(self, conversation_id=None):
        """Clear messages and reload the active conversation."""
        return self._ask(CONV_CLEAR, id=conversation_id)

    def delete(self, conversation_id):
        """Delete a conversation and its messages."""
        return self._ask(CONV_DELETE, id=conversation_id)


class _Session(_Namespace):
    """Session Requests. Widening is unsafe, narrowing is safe."""

    def get(self, key: str = "", details: bool = False):
        """Describe a session, optionally including its debug snapshot."""
        return self._ask(SESSION_GET, key=key, details=details)

    def list(self):
        """Every live session key."""
        return self._ask(SESSION_LIST)

    def push(self, message: str, key: str = "", *, title: str = "",
             notify: bool = False, level: str = "info"):
        """Send the user a message out of band.

        ``notify=True`` raises it as a *notification* instead: the system
        telling the user something, rather than something said in the
        conversation. A frontend with somewhere to put those — a panel, a
        badge, a toast — draws it there; one without shows it in the chat
        exactly as a plain push would, so nothing is lost by asking.

        Use it for work the user did not just ask for and is not watching: a
        background write finishing, something that needs their attention later.
        A plain push is right when you are speaking *into* the conversation.

        ``level`` is ``info`` / ``success`` / ``warning`` / ``error`` and only
        styles the result. ``title`` is the header; with ``notify`` it is what
        the panel shows collapsed, so make it say what happened.

        You cannot state who sent it — the kernel stamps that from the
        provenance chain, so attribution is something a reader can trust.
        """
        return self._ask(SESSION_PUSH, message=message, key=key, title=title,
                         notify=notify, level=level)

    def state_get(self, namespace: str = "sandbox", key: str = ""):
        """Read per-session scratch state."""
        return self._ask(SESSION_STATE_GET, namespace=namespace, key=key)

    def state_set(self, value, namespace: str = "sandbox",
                  key: str = ""):
        """Write per-session scratch state."""
        return self._ask(SESSION_STATE_SET, value=value, namespace=namespace,
                         key=key)

    def cancel(self, key: str = ""):
        """Cancel the turn running on a session."""
        return self._ask(SESSION_CANCEL, key=key)

    def compact(self):
        """Summarize this session's history and shrink what the model sees.

        The kernel does this on its own when the context gets tight; asking
        does the same thing now. Answers with a report — ``messages_before`` /
        ``messages_after``, ``chars_before`` / ``chars_after`` /
        ``chars_saved``, ``summary_chars`` — and fails with a plain reason when
        there is nothing to compact, the compactor is not installed, or the
        agent is mid-turn.

        Takes no session key on purpose: it acts on the session you are
        serving, so there is no argument to point at somebody else's.
        """
        return self._ask(SESSION_COMPACT)

    def add_tool(self, tool: str, key: str = ""):
        """Widen the agent's scope."""
        return self._ask(SESSION_ADD_TOOL, tool=tool, key=key)

    def remove_tool(self, tool: str, key: str = ""):
        """Narrow the agent's scope."""
        return self._ask(SESSION_REMOVE_TOOL, tool=tool, key=key)

    def add_prompt(self, text: str, key: str = "", slot: str = ""):
        """Inject system prompt text, and answer with the slot to remove it by.

        ``key`` is the session; ``slot`` is the named overlay within it, and
        defaults to your plugin. Writing the same slot again replaces what was
        there, so a hook that refreshes guidance every turn wants one stable
        slot rather than a new one each time.
        """
        return self._ask(SESSION_ADD_PROMPT, text=text, key=key, slot=slot)

    def remove_prompt(self, handle, key: str = ""):
        """Withdraw injected prompt text, by the slot ``add_prompt`` returned."""
        return self._ask(SESSION_REMOVE_PROMPT, handle=handle, key=key)

    def add_attachment(self, path, key: str = ""):
        """Stage a file for this session's next model call.

        The kernel opens the path, parses it, and puts it in front of the
        model — so this is how a tool shows the *model* an image, audio or
        video file, as against ``sdk.ok(attachments=...)``, which shows a file
        to the *user*. Different destinations, and this is the one that ends
        with the model actually looking.

        You do not need to know whether the model can see the modality. If it
        cannot, the kernel substitutes the file's parsed text, and failing that
        a line naming where the file is. Staging is always the right call.
        """
        return self._ask(SESSION_ADD_ATTACHMENT, path=str(path), key=key)

    def set_mode(self, mode: str, key: str = "", scope: str = "conversation"):
        """Set how this conversation answers approval dialogs.

        ``mode`` is ``"lockdown"`` (refuse anything that would be asked about),
        ``"ask"`` (the default), or ``"yolo"`` (approve it). ``scope`` is
        ``"conversation"``, which lasts until the conversation changes, or
        ``"turn"``, which is dropped when the agent turn ends.

        Tightening to ``"lockdown"`` is safe and never asks. Anything else is
        a widening, so it raises an approval dialog unless the user typed the
        command that is doing it — which is also what stops lockdown being a
        trap, since ``/mode ask`` is that exact case.
        """
        return self._ask(SESSION_SET_MODE, mode=mode, key=key, scope=scope)


class _UI(_Namespace):
    """Talking to the person."""

    def ask(self, prompt: str, title: str = "Question", type: str = "string",
            choices=None, timeout: float = 300.0, required: bool = True,
            default=None):
        """Ask a question and wait. Refused when nobody is present.

        ``type`` is the shape of the answer — ``"string"``, ``"integer"``,
        ``"number"``, ``"boolean"``, ``"array"`` or ``"object"`` — and decides
        how the question is presented and how the reply is parsed.
        ``choices`` limits the answer to a list of options; ``required=False``
        lets the person skip, in which case ``default`` comes back.

        Raises ``sdk.Denied`` when the person cancels, and fails when they
        never answer — a cancelled question and an unanswered one are
        different events and a plugin usually wants to treat them differently.
        """
        return self._ask(UI_ASK, prompt=prompt, title=title, type=type,
                         choices=list(choices or []) or None, timeout=timeout,
                         required=required, default=default)

    def approve(self, action: str, justification: str = ""):
        """Ask the user to approve a described action."""
        return self._ask(UI_APPROVE, action=action,
                         justification=justification)

    def render(self, paths, caption: str = ""):
        """Show files to the user in chat."""
        return self._ask(UI_RENDER, paths=[str(p) for p in paths],
                         caption=caption)

    def progress(self, message: str):
        """Say what a slash command is doing, while it does it.

        Addressed to the *call* the person is already watching, not to the
        conversation — a frontend showing "⋯ /packages" updates that line in
        place. Use it for work long enough to be worth narrating: a bulk task
        reset, a service load that imports torch, an install.

        **Not ``sdk.session.push``.** That destination is the chat, and progress
        from a command run out of a settings screen does not belong in the
        transcript of the conversation. This is the difference.

        **Silent when nothing is watching.** Called from an agent-invoked tool,
        a task or a service — anywhere no slash command is running — it does
        nothing and answers False. Narrating nowhere is the right failure; the
        alternative is falling back to the chat, which is what this exists to
        stop. So a shared helper may call it unconditionally.
        """
        return self._ask(UI_PROGRESS, message=str(message))


class _Config(_Namespace):
    """Settings. Credentials come back as handles, never plaintext."""

    def read(
        self,
        key: str = "",
        *,
        present: bool = False,
        keys: bool = False,
        details: bool = False,
    ):
        """Read a setting, test presence, list keys, or inspect descriptors."""
        return self._ask(
            CONFIG_READ, key=key or None, present=present, keys=keys,
            details=details)

    def write(
        self,
        key: str,
        value,
        *,
        merge: bool = False,
        scope: str = "",
    ):
        """Change a setting.

        ``merge`` updates a mapping without returning its existing contents to
        the guest. ``scope="plugin"`` explicitly persists plugin-owned data.
        """
        return self._ask(
            CONFIG_WRITE, key=key, value=value, merge=merge,
            scope=scope or None)


class _Paths(_Namespace):
    """Kernel-owned application locations, and two facts about the host.

    ``project``, ``data``, ``bundled``, ``installed``, ``workspace``,
    ``scripts``, ``python``, ``platform`` —
    directories. ``python`` is the interpreter running the app, which is what
    ``pip`` should be invoked through so it installs into *this* environment;
    ``platform`` is ``sys.platform``. Both are here because the validator
    refuses ``sys`` and these are the only things behind it a plugin has a
    real claim on.
    """

    def get(self, name: str):
        """Resolve a named application location."""
        return self._ask(PATH_GET, name=name)


class _Users(_Namespace):
    """Users. ``password_hash`` is never returned."""

    def read(self, user_id=None):
        """One user; defaults to the current one."""
        return self._ask(USER_READ, id=user_id)

    def list(self):
        """Every user."""
        return self._ask(USER_LIST)

    def write(self, user_id=None, **fields):
        """Update a user's config blob or type."""
        return self._ask(USER_WRITE, id=user_id, **fields)


class _Plugins(_Namespace):
    """Introspection over what is registered."""

    def list(
        self,
        source: str = "registered",
        category: str = "",
        role: str = "",
        details: bool = False,
        name: str = "",
    ):
        """List plugins, optionally narrowed by a kernel-defined role.

        ``source="info"`` (store) and ``source="installed_info"`` (the
        installed tree) narrow all the way to one package named by ``name``,
        answering with a single dict carrying its description and
        dependencies rather than a list.
        """
        return self._ask(
            PLUGIN_LIST, source=source, category=category or None,
            role=role or None, details=details, name=name or None)

    def describe(self, name: str):
        """Metadata for one plugin."""
        return self._ask(PLUGIN_DESCRIBE, name=name)

    def validate(self, path: str):
        """Check a plugin source file against the sandbox contract.

        The same validator the loader runs, so its verdict is the real one:
        ``ok`` means the file will load, ``disclaimed`` means it will load
        with a warning, and ``unmediated`` names the imports that will put it
        in a subprocess. ``findings`` carries ``level``, ``line``, ``message``
        and ``fix`` per problem. Nothing is imported or executed.
        """
        return self._ask(PLUGIN_VALIDATE, path=str(path))

    def register(self, path: str):
        """Load a recognized plugin source file into the live runtime."""
        return self._ask(PLUGIN_REGISTER, path=str(path))

    def unregister(
        self,
        *,
        path: str = "",
        name: str = "",
        family: str = "",
    ):
        """Unload a plugin by source path or unambiguous registered identity."""
        return self._ask(
            PLUGIN_UNREGISTER,
            path=str(path) if path else "",
            name=name,
            family=family,
        )

    def reload(
        self,
        *,
        path: str = "",
        name: str = "",
        family: str = "",
    ):
        """Reload a plugin by source path or unambiguous registered identity."""
        return self._ask(
            PLUGIN_RELOAD,
            path=str(path) if path else "",
            name=name,
            family=family,
        )

    def install(self, package_id: str):
        """Install a package or bundle from the kernel store."""
        return self._ask(PLUGIN_INSTALL, package_id=package_id)

    def uninstall(self, package_id: str):
        """Uninstall an installed package, helper, or bundle."""
        return self._ask(PLUGIN_UNINSTALL, package_id=package_id)

    def update(self):
        """Update installed packages from the kernel store."""
        return self._ask(PLUGIN_UPDATE)


class _Services(_Namespace):
    """Calling into loaded services."""

    def list(self, details: bool = False):
        """Loaded services, optionally with lifecycle and setting metadata."""
        return self._ask(SERVICE_LIST, details=details)

    def call(self, service: str, method: str, /, *args, **kwargs):
        """Invoke an exported method. Simple data comes back, never objects.

        ``service`` and ``method`` are **positional-only**, and that is a fix
        rather than a style. They were ordinary parameters named ``name`` and
        ``method``, so every export taking its own ``name`` or ``method``
        argument was unreachable by keyword — ``call("timekeeper", "get_job",
        name="x")`` raised *"got multiple values for argument 'name'"*, naming
        the caller's own argument and blaming the wrong thing entirely. The
        two positions belong to the call, not to the callee, so nothing here
        should occupy a name the callee might want.

        Positional arguments pass through for the same reason. Only ``kwargs``
        crossed, so ``call("timekeeper", "cron_to_text", cron)`` was a
        ``TypeError`` at the SDK — which is what ``/schedule`` had been
        swallowing into a fallback, printing raw cron where it meant to print
        English.
        """
        return self._ask(SERVICE_CALL, name=service, method=method,
                         args=list(args), kwargs=kwargs)

    def load(self, name: str):
        """Load a user-managed service."""
        return self._ask(SERVICE_LOAD, name=name)

    def unload(self, name: str):
        """Unload a user-managed service."""
        return self._ask(SERVICE_UNLOAD, name=name)


class _Tools(_Namespace):
    """Calling other tools."""

    def list(self, details: bool = False):
        """Tools the current scope exposes, optionally with schemas/settings."""
        return self._ask(TOOL_LIST, details=details)

    def call(
        self,
        name: str,
        *,
        _result: bool = False,
        _user_initiated: bool = False,
        **kwargs,
    ):
        """Call another tool.

        ``_result`` preserves the complete result envelope for presentation.
        ``_user_initiated`` is honored only for command-originated calls.
        """
        return self._ask(
            TOOL_CALL, name=name, kwargs=kwargs, result=_result,
            user_initiated=_user_initiated)


class _Commands(_Namespace):
    """Running slash commands."""

    def list(self, details: bool = False, visible: bool = False):
        """Registered commands, optionally with metadata and session filtering."""
        return self._ask(COMMAND_LIST, details=details, visible=visible)

    def run(self, name: str, **args):
        """Run a slash command in one shot."""
        return self._ask(COMMAND_CALL, name=name, args=args)


class _Agent(_Namespace):
    """The model, and other agents."""

    def complete(
        self,
        prompt: str = "",
        messages=None,
        session_key: str | None = None,
        profile: str = "",
    ):
        """A model call. Keys and sockets stay kernel-side.

        ``profile`` names a configured LLM profile — a *name*, never a model
        object, because a box could not hold one anyway. Give it when the
        choice of model is the plugin's own (a cheap model for a background
        chore); leave it empty to drive with whatever the session drives with,
        or the default profile when there is no session.
        """
        return self._ask(AGENT_COMPLETE, prompt=prompt,
                         messages=list(messages or []),
                         session_key=session_key or None,
                         profile=profile)

    def spawn(
        self,
        prompt: str,
        *,
        title: str = "Subagent",
        attachments=None,
        wait: bool = True,
        timeout_seconds: int | None = None,
        profile: str | None = None,
    ):
        """Run a subagent now, in its own conversation.

        The prompt must be complete and self-contained: nobody will answer a
        follow-up question, and a child can use no tool that needs approval.

        ``profile`` names an agent profile from the user's config, which is how
        a child is given a *narrower* set of tools than the caller has — a
        curator that may write notes and nothing else. Naming none inherits the
        caller's own profile; naming one that does not exist fails rather than
        quietly running the child unrestricted.

        ``wait=True`` returns the finished report::

            report = sdk.agent.spawn("Summarise docs/SDK.md")
            sdk.log(report["text"])

        ``wait=False`` returns a handle immediately so several can run at
        once. Collect them when you need the answers::

            ids = [sdk.agent.spawn(p, wait=False)["id"] for p in prompts]
            for report in sdk.agent.collect(ids):
                sdk.log(report["title"], report["text"])

        Either way the report is a dict with ``id``, ``conversation_id``,
        ``title``, ``state``, ``ok``, ``text``, ``error`` and ``profile``.
        ``state`` is ``running``, ``done``, ``failed`` or ``cancelled`` — and
        ``cancelled`` means the child hit its deadline and produced nothing, so
        there is never anything to report on its behalf.
        """
        return self._ask(AGENT_SPAWN, prompt=prompt, title=title,
                         attachments=list(attachments or []), wait=wait,
                         timeout_seconds=timeout_seconds, profile=profile)

    def collect(self, ids=None, timeout: float | None = None):
        """Wait for subagents and take their reports.

        ``ids=None`` takes every child this session started and has not
        collected yet. ``timeout=0`` polls without waiting — children still
        running come back with ``state == "running"`` and stay uncollected, so
        a later call still gets them. ``timeout=None`` waits until each
        child's own deadline, which is the usual thing to want.

        Each report is delivered once. Inside an agent turn, whatever you do
        not collect is collected for you before the turn ends.
        """
        return self._ask(AGENT_COLLECT, ids=ids, timeout=timeout)

    def stop(self, id: str):
        """Cancel a running subagent. Narrows, so it is the safe direction."""
        return self._ask(AGENT_STOP, id=id)

    def schedule(self, prompt: str, cron: str, *, title: str = "Scheduled subagent",
                 attachments=None, one_time: bool = False, name: str = ""):
        """Run a subagent later, on a schedule. Unattended, so always checked."""
        return self._ask(AGENT_SCHEDULE, prompt=prompt, cron=cron, title=title,
                         attachments=list(attachments or []),
                         one_time=one_time, name=name)


class _LLM(_Namespace):
    """The model authority, and the call in flight.

    Two groups that share a namespace because they share a subject.

    ``proceed`` and ``delta`` are scoped to a call the kernel already decided
    to place, and neither means "make a model call". ``proceed`` is for an
    escort standing at the ``llm_call`` doorway; ``delta`` is for the backend
    actually placing it. Outside those, there is no call and the Request is
    refused.

    ``list``/``load``/``unload`` are about the *registry* rather than any one
    call: which profiles are configured, which backends could serve them, and
    which are open. Profiles stopped being services when ``service_llm.py``
    was deleted, so asking ``sdk.services`` about them — which is what ``/llm``
    did — reports every profile missing and unloaded while conversations using
    those same profiles work fine.
    """

    def list(self, *, providers=False, models=None, params=None, info=None,
             key=None, provider=None, endpoint=None, backend=None,
             live: bool = False) -> dict:
        """Configured profiles, installed backends, and the default.

        Answers ``{"profiles": [...], "backends": [...], "aliases": {...},
        "default": str}``. A profile row carries ``model_name``, ``class``,
        ``endpoint``, ``context_size``, ``params``, ``loaded`` and
        ``sandboxed``; a backend row carries ``name`` and ``display_name``.
        ``params`` is the extra provider kwargs that profile sends on every
        call — reasoning effort and whatever else it configures — as resolved
        rather than as configured: declined params (a ``null`` in the profile)
        are gone, and the names are the ones that go on the wire. Nothing is
        filled *in*; the kernel names no provider parameter, so what a profile
        sends is what somebody configured. ``aliases`` maps a retired
        backend name to the one that replaced it, which is what a stored
        ``llm_service_class`` may still be. Each profile row also carries
        ``param_status`` — ``{param: [supported, note]}`` for the params that
        profile sends, and ``{}`` for a profile whose box is closed, since
        nothing opens one merely to answer this.

        The four optional arguments are the setup questions, narrowing in
        order. Pass ``providers=True`` for the provider list, or
        ``providers=<name>`` for that one provider with its endpoint resolved;
        ``models=<url>``
        (with ``key`` and ``provider``) for what one endpoint serves; or
        ``params=<model name>`` (with ``endpoint``) for what one model takes;
        or ``info=<model name>`` for facts about it, currently its context
        window.
        Each adds a key of the same name to the answer. ``backend`` limits
        any setup question to the selected backend; omit it only when an
        aggregate answer is wanted. ``live`` additionally
        lets ``models`` ask the endpoint rather than answering from what the
        backend knows offline. That is egress, so it is off by default and
        must **never** be set from a command's ``form``: approval is evaluated
        on completed form arguments, so a form runs ungranted and the dialog
        it raises deadlocks against the lock it is holding.

        Every one of them can answer ``[]``, and that is ordinary rather than
        a failure — no backend is obliged to introspect, and a caller handed
        nothing should let the value be typed instead. ``models`` may make a
        live call to the endpoint, so ask it once, at the point an endpoint
        has just been given.
        """
        args = {}
        if backend:
            args["backend"] = backend
        if providers:
            # Passed through as given: ``True`` asks for the menu, a name asks
            # for that one provider with its endpoint resolved. Coercing to
            # ``True`` here threw the name away and answered the menu, so the
            # caller got no endpoint and could not tell why.
            args["providers"] = providers
        if models is not None:
            args["models"] = models
        if params is not None:
            args["params"] = params
        if info is not None:
            args["info"] = info
        if key is not None:
            args["key"] = key
        if provider is not None:
            args["provider"] = provider
        if endpoint is not None:
            args["endpoint"] = endpoint
        if live:
            args["live"] = True
        return self._ask(LLM_LIST, **args)

    def load(self, name: str) -> bool:
        """Open one profile's box pool."""
        return self._ask(LLM_LOAD, name=name)

    def unload(self, name: str) -> bool:
        """Close one profile's box pool."""
        return self._ask(LLM_UNLOAD, name=name)

    def delta(self, text: str) -> None:
        """Push one fragment of assistant text as it arrives.

        Only meaningful inside a backend's ``chat`` when ``request.stream``
        was set. One-way and unanswered, so streaming costs a frame per chunk
        rather than a round trip per chunk.

        There is deliberately nothing to check here. Whether the user wants
        this stream to continue is the kernel's decision, not the backend's.

        Note what that decision costs, because it is not the usual one. A
        cancelled execution normally unwinds at its *next* Request, which
        answers ``Terminated`` — but a streaming loop's only Request is this
        one, and a notice is never answered. So there is nothing to refuse and
        nothing to raise, and the kernel ends such a call by ending the box
        (``PersistentBox.interrupt``). Do not add a check here hoping to be
        told: the boundary cannot tell you.
        """
        if not text:
            return
        self._sdk._notify(Request(LLM_DELTA, {
            "token": self._sdk._delta_token, "text": text}))

    def proceed(self, request=None):
        """Place the call, optionally rewritten, and return the response.

        Call it more than once to retry: each is a fresh trip to the model.
        Not calling it at all is allowed too — return a response you built
        yourself and the model is never troubled.
        """
        from .hooks import ModelRequest, ModelResponse

        payload = None
        if request is not None:
            payload = {k: getattr(request, k)
                       for k in ModelRequest.__dataclass_fields__}
        answer = self._ask(LLM_PROCEED, token=self._sdk._hook_token,
                           request=payload)
        allowed = set(ModelResponse.__dataclass_fields__)
        return ModelResponse(**{k: v for k, v in dict(answer or {}).items()
                                if k in allowed})


class _Frontend(_Namespace):
    """Carrying what a person did into the state machine.

    Only meaningful inside a loaded frontend. Every call resolves to *this*
    frontend's own adapter, so a frontend cannot submit on another's behalf,
    and code that is not a frontend reaches no adapter and is refused.

    This is the inbound half of a frontend. The outbound half — showing things
    to a person — is not a Request at all: the kernel calls ``render`` on you.
    """

    def _token(self) -> str:
        """The handle on this frontend's adapter, set when its box opened."""
        return getattr(self._sdk, "_frontend_token", "")

    def submit_text(self, session_key: str, text: str):
        """Hand over a line someone typed. The usual one."""
        return self._ask(FRONTEND_SUBMIT, token=self._token(),
                         session_key=session_key, input_kind="text", text=text)

    def submit_attachment(self, session_key: str, path: str,
                          extension: str = "", file_name: str = "",
                          caption: str = "", is_photo: bool = False,
                          ingest: bool = False):
        """Hand over a file someone sent.

        ``ingest`` is for a file that arrived over your *transport* rather than
        off the user's disk: the kernel moves it into the attachment cache,
        which is a watched directory, so the pipeline indexes it like any other
        incoming file. Point it at scratch space you got from ``sdk.fs.temp``
        and let your transport write there — the bytes never have to cross the
        boundary, and the temp file is removed once it has been taken.

        ``file_name`` is the name the person's own machine used, which is worth
        keeping when the transport handed you an opaque path.
        """
        return self._ask(FRONTEND_SUBMIT, token=self._token(),
                         session_key=session_key, input_kind="attachment",
                         path=str(path), extension=extension,
                         file_name=file_name, caption=caption,
                         is_photo=bool(is_photo), ingest=bool(ingest))

    def submit_attachments(self, session_key: str, files, caption: str = "",
                           ingest: bool = False):
        """Hand over several files someone sent as **one** message.

        ``files`` is a list of the same fields ``submit_attachment`` takes
        inline — ``{"path": …, "file_name": …, "extension": …,
        "is_photo": …, "caption": …}`` — or just the paths, when there is
        nothing else to say about them. ``caption`` and ``ingest`` are the
        message's, applying to every file that does not state its own. The
        message's caption rides on the first file, since it is the line the
        person typed rather than a label on each.

        Submitting them one at a time does not work and cannot be made to: the
        first hands the turn to the agent, and the rest arrive at a session
        that is already busy. This is one action, so the model sees all of
        them in the same call — which is what a person means by attaching
        three files.
        """
        return self._ask(FRONTEND_SUBMIT, token=self._token(),
                         session_key=session_key, input_kind="attachment",
                         files=[dict(f) if isinstance(f, dict)
                                else {"path": str(f)} for f in files or []],
                         caption=caption, ingest=bool(ingest))

    def submit_action(self, session_key: str, action_type: str, payload=None):
        """Hand over a typed action — a button press, a menu choice."""
        return self._ask(FRONTEND_SUBMIT, token=self._token(),
                         session_key=session_key, input_kind="action",
                         action_type=action_type, payload=payload)

    def cancel(self, session_key: str):
        """Stop whatever that session is doing."""
        return self._ask(FRONTEND_CANCEL, token=self._token(),
                         session_key=session_key)

    def bind(self, session_key: str, external_id=None, user_type: str = "user",
             config=None):
        """Say whose data this session is. Returns the user id.

        With no ``external_id`` the session takes this frontend's declared
        default user. With one, it is upgraded to that identity's own user —
        what a ``per_user`` frontend does on login. Authenticating is your
        job; the kernel stores what you give it and asks nothing.
        """
        return self._ask(FRONTEND_BIND, token=self._token(),
                         session_key=session_key,
                         external_id=(None if external_id is None
                                      else str(external_id)),
                         user_type=user_type, config=config)

    def attended(self, session_key: str, present: bool = True):
        """Say whether a person is actually watching this session.

        The kernel only reads attendance; a frontend owns the policy. Say it
        on connect and disconnect and background-safety gating follows.
        """
        return self._ask(FRONTEND_ATTEND, token=self._token(),
                         session_key=session_key, present=bool(present))

    def pending_input(self, session_key: str, details: bool = False):
        """What this session is blocked on, or None.

        Ask rather than remember. You are told a question exists — you were
        handed one to render — but not when it stops existing: another frontend
        can answer it, or it can time out. Acting on a stale record means
        swallowing the next thing a person types as a yes or no.

        Plain, this is the approval's *id*, which is all it takes to answer one.
        With ``details`` it is the question itself, tagged with the render kind
        that carried it — ``{"kind": "approval"|"form_field", "payload": {...}}``
        — or None. Renders are events and are not re-sent, so ``details`` is how
        a frontend that reconnected gets back to a question it never saw: the
        payloads are the same ones the render made, so the dialog it draws is
        the real one rather than a reconstruction.

        Answered from the session's own phase stack when this frontend has no
        record of one, so a restart — or a frontend loaded after its session was
        restored — does not report a blocked session as an idle one.
        """
        return self._ask(FRONTEND_PENDING, token=self._token(),
                         session_key=session_key, details=bool(details))

    #: The name this had when it only ever spoke about approvals. ``details``
    #: made that untrue — a suspended *form* is the same fact about a session,
    #: and a frontend that restores one but not the other still strands people
    #: — but a frontend in the wild is calling this, so the old spelling stays.
    pending_approval = pending_input

    def resolve(self, session_key: str, value, request_id: str = ""):
        """Answer the approval a ``render`` of kind ``approval`` showed.

        With no ``request_id`` the session's next pending request is answered,
        which is what a transport with one message at a time wants.
        """
        return self._ask(FRONTEND_RESOLVE, token=self._token(),
                         session_key=session_key, value=value,
                         request_id=request_id)

    def act(self, session_key: str, request_type: str, args=None) -> str:
        """Run one Request *as* one of your sessions. Returns a handle.

        Your own box is rooted ``frontend:<name>``, which names no session, so
        it is unattended and anything unsafe it asks for is refused rather than
        asked — there is nobody a dialog could be drawn for. This roots the
        Request at a session you own instead, so attendance is resolved through
        :meth:`attended`, which you declared. What that buys is a real dialog,
        rendered back to you as an ``approval``, for the things a person should
        be answering.

        Nothing is exempted. The Request is classified exactly as it would be
        anywhere else; only who is asking changes, and only to a session you
        already own and have said somebody is watching.

        It does not wait, because it must not: your box serves one call at a
        time, and an approval has to render back into it to be seen. Collect
        the answer with :meth:`collect`, which is what your next poll is for.
        """
        return self._ask(FRONTEND_ACT, token=self._token(),
                         session_key=session_key, request_type=request_type,
                         args=dict(args or {}))

    def collect(self, handle: str):
        """The result of an :meth:`act`, or ``None`` while it is still running.

        Delivered once — the second call answers ``None`` — and dropped if
        nobody comes back for it. A ``Result`` comes back as a plain dict
        (``ok``, ``data``, ``error``, ``code``), never raised, because a
        refusal is an ordinary answer to forward to whoever asked.
        """
        return self._ask(FRONTEND_COLLECT, token=self._token(),
                         handle=str(handle))


class _Console(_Namespace):
    """The machine's console, if this frontend claimed it.

    Declare ``uses_console = True`` and the kernel lends it to you — to exactly
    one frontend, because two readers would split a person's keystrokes between
    them. Everything else reaches nothing here.

    The kernel does the reading, on its own thread. That is what makes this
    usable from a poll loop at all: there is nothing to block on, and a
    subprocess box never opens stdin, so a console frontend can be isolated.
    """

    def read_line(self):
        """The next line someone typed, or None if none has arrived yet.

        Never blocks. Raises once the console is closed and drained — on a
        piped stdin that is end of input, and letting it propagate out of
        ``poll`` is how a frontend stops itself when there is no more to read.
        """
        return self._ask(CONSOLE_READ, token=getattr(
            self._sdk, "_frontend_token", ""))

    def write(self, text: str, end: str = "\n"):
        """Put a line on the console."""
        return self._ask(CONSOLE_WRITE, token=getattr(
            self._sdk, "_frontend_token", ""), text=str(text), end=end)


class _Http(_Namespace):
    """A listening port, if this frontend claimed one.

    Declare ``serves_http = <port>`` and the kernel binds it on loopback and
    lends it to you — to exactly one frontend, because two cannot bind one
    port. Everything else reaches nothing here.

    The kernel does the accepting and parsing, on its own threads. That is what
    makes this usable from a poll loop: ``drain`` never blocks, and a child
    process never opens a socket, so an HTTP frontend can be isolated. It is
    also why ``socket`` stays refused — this is the mediated route, not an
    exception to the rule.

    The shape is inverted from what a web framework teaches. You are not handed
    a request and asked for a response; you collect what arrived, answer by id,
    and get on with the poll. A reply may outlive the drain that produced it::

        for request in sdk.http.drain():
            sdk.http.respond(request["id"], stream=True)   # an SSE stream
            ...                                            # later, elsewhere
            sdk.http.push(request["id"], json.dumps(event))
            sdk.http.close(request["id"])
    """

    def _token(self) -> str:
        """The handle on this frontend's claim, set when its box opened."""
        return getattr(self._sdk, "_frontend_token", "")

    def drain(self, limit: int = 0):
        """Every request that has arrived since the last call.

        Never blocks; answers ``[]`` when nothing has come in. Each item is
        ``{id, method, path, query, headers, body}`` — the connection itself
        stays kernel-side, so holding an id is enough to answer and only enough
        to answer.
        """
        return self._ask(HTTP_DRAIN, token=self._token(), limit=int(limit))

    def respond(self, request_id: str, status: int = 200, headers=None,
                body="", stream: bool = False):
        """Answer a request, or open it as an event stream.

        With ``stream=True`` the reply stays open for ``push`` and the SSE
        headers are written for you; ``body`` is then an optional first frame.
        Without it this is an ordinary one-shot response and the request is
        finished.

        ``body`` may be ``str`` or ``bytes``. Hand it the bytes from
        ``sdk.fs.read_bytes`` to serve an image or a font — encoding those as
        text mangles them, and nothing downstream would tell you.
        """
        return self._ask(HTTP_RESPOND, token=self._token(),
                         request_id=str(request_id), status=int(status),
                         headers=headers or {},
                         body=(body if isinstance(body, (str, bytes, bytearray))
                               else str(body)),
                         stream=bool(stream))

    def push(self, request_id: str, data: str, event: str = "",
             ident: str = ""):
        """Write one frame to an open stream.

        Fails once the client has gone, which is the ordinary end of a stream
        rather than a fault — but it is told to you, because a frontend that
        never hears it goes on rendering a whole turn into a closed socket.

        ``ident`` becomes the frame's ``id:``, which a browser hands straight
        back as the ``Last-Event-ID`` header when ``EventSource`` reconnects.
        Number your frames and a page refresh resumes where it left off with
        no client code at all; leave it out and whatever was said during the
        reload is gone.
        """
        return self._ask(HTTP_PUSH, token=self._token(),
                         request_id=str(request_id), data=str(data),
                         event=str(event), ident=str(ident))

    def close(self, request_id: str):
        """End a reply."""
        return self._ask(HTTP_CLOSE, token=self._token(),
                         request_id=str(request_id))


class _Cron(_Namespace):
    """Scheduled jobs."""

    def list(self):
        """Every job."""
        return self._ask(CRON_LIST)

    def get(self, name: str):
        """One job."""
        return self._ask(CRON_GET, name=name)

    def create(self, name: str, job: dict):
        """Add a job."""
        return self._ask(CRON_CREATE, name=name, job=job)

    def update(self, name: str, patch: dict):
        """Change a job."""
        return self._ask(CRON_UPDATE, name=name, patch=patch)

    def remove(self, name: str):
        """Delete a job."""
        return self._ask(CRON_REMOVE, name=name)

    def enable(self, name: str, enabled: bool = True):
        """Enable or disable. Disabling narrows, so it is the safe direction."""
        return self._ask(CRON_ENABLE, name=name, enabled=enabled)


class _Events(_Namespace):
    """The bus."""

    def emit(self, channel: str, payload=None):
        """Publish."""
        return self._ask(EVENT_EMIT, channel=channel, payload=payload)

    def request(self, channel: str, payload=None,
                timeout: float = 120.0):
        """Publish and wait for one answer."""
        return self._ask(EVENT_REQUEST, channel=channel, payload=payload,
                         timeout=timeout)


class _Tasks(_Namespace):
    """Pipeline work."""

    def enqueue(self, name: str, paths):
        """Queue work."""
        return self._ask(TASK_ENQUEUE, name=name,
                         paths=[str(p) for p in paths])

    def status(self, name: str, path):
        """Where a task stands for a path."""
        return self._ask(TASK_STATUS, name=name, path=str(path))

    def output(self, name: str, path=None):
        """Read a task's output table."""
        return self._ask(TASK_OUTPUT, name=name,
                         path=str(path) if path else None)

    def list(self, details: bool = False):
        """Registered tasks, optionally with status and setting metadata."""
        return self._ask(TASK_LIST, details=details)

    def graph(self):
        """Render the dependency pipeline."""
        return self._ask(TASK_GRAPH)

    def pause(self, name: str, paused: bool = True):
        """Pause or unpause a task."""
        return self._ask(TASK_PAUSE, name=name, paused=paused)

    def reset(self, name: str, failed_only: bool = False):
        """Reset path-task rows, optionally only failed rows."""
        return self._ask(
            TASK_RESET, name=name, failed_only=failed_only)

    def trigger(self, name: str, payload=None):
        """Create a manual run for an event-driven task."""
        return self._ask(TASK_TRIGGER, name=name, payload=payload or {})


class _Files(_Namespace):
    """The watched-file table the pipeline runs on."""

    def register(self, path, **meta):
        """Add a path to the watched-file table."""
        return self._ask(FILE_REGISTER, path=str(path), meta=meta)

    def list(self, modality: str = ""):
        """Query the watched-file table."""
        return self._ask(FILE_LIST, modality=modality or None)


class _Parse(_Namespace):
    """Parsing a file, from whichever side can actually hold the answer.

    Two mechanisms behind one method, and which one runs is decided by what
    the plugin *declared* rather than by what it calls.

    Declare ``parse_modalities = ["image"]`` and the kernel resolves that
    against its registry, imports those parser files into **this box** before
    anything runs, and :meth:`file` calls one directly. The result never
    crosses a boundary because it never had to — which is the only way to get
    a PIL image, a DataFrame or an open container, all of which are live
    objects a wire cannot carry.

    Declare nothing and :meth:`file` is an ordinary Request: the kernel routes
    to the parser, runs it wherever that parser belongs, and answers with what
    fits on a wire — text, or the child paths a container yielded. This is the
    cheaper path and the right default, because the parser's dependencies stay
    out of your box.

    So an undeclared heavy modality is refused, and the refusal names the
    declaration that would fix it. That is the rule the boundary always had;
    what changed is that there is now something to do about it.
    """

    def file(self, path, modality: str = "text", detail: bool = False,
             config: dict = None):
        """Parse a file. Local parser if one was provisioned, else the kernel.

        Answers the parser's ``output``: a string for text, a list of child
        paths for a container, and for a provisioned heavy modality whatever
        that parser produces — live objects that are only valid in this box.

        ``detail=True`` answers ``{"output", "also_contains"}`` instead. That
        second key is how one parse tells the pipeline there is another route
        out of the same file — a PDF reporting ``["image"]`` is what gets it
        re-enqueued into the OCR and image-embedding tasks. It is an argument
        rather than a second Request because the kernel already puts it on the
        Result; the plain call simply unwraps to the payload, which is what
        almost every caller wants.
        """
        from . import parsing as guest_parsing

        parser = guest_parsing.local_parser(guest_parsing.suffix(str(path)),
                                            modality)
        if parser is None:
            if not detail:
                return self._ask(PARSE_FILE, path=str(path), modality=modality)
            answer = self._sdk._send(Request(
                PARSE_FILE, {"path": str(path), "modality": modality}))
            if not answer.ok:
                raise (Denied if answer.denied else RequestFailed)(
                    answer, PARSE_FILE)
            return {"output": answer.data,
                    "also_contains": list(answer.also_contains)}

        # Called with this sdk, exactly as the kernel calls it with its own
        # stand-in. The parser cannot tell which it has, which is what lets one
        # file serve both callers — including the empty dict: parsers read
        # their tuning with ``config.get(...)`` and ``parsing.parse`` does
        # ``config or {}``, so passing None here would crash every parser that
        # has a knob rather than every parser that does not.
        parsed = parser(self._sdk, str(path), dict(config or {}))
        if not getattr(parsed, "success", True):
            raise RequestFailed(Result.failure(
                str(getattr(parsed, "error", "") or "parse failed")))
        output = getattr(parsed, "output", None)
        if not detail:
            return output
        return {"output": output,
                "also_contains": list(getattr(parsed, "also_contains", None)
                                      or [])}

    def modality(self, extension: str, detail: bool = False):
        """Resolve an extension's modality.

        ``detail=True`` answers ``{"modality", "known", "generic"}`` rather
        than the bare string, which is what a caller deciding *whether to
        parse at all* needs:

            info = sdk.parse.modality(sdk.path.suffix(path), detail=True)
            if info["known"] and not info["generic"]:
                text = sdk.parse.file(path)     # the parser owns this format
            else:
                text = sdk.fs.read(path)        # the bytes are the content

        ``generic`` is true only for the kernel's text parser. Without it a
        caller cannot tell ``.py`` from ``.gdoc`` -- both register as "text",
        but one *is* its bytes and the other is a pointer to a document that
        has to be fetched. Routing on the modality alone hands the agent a
        JSON stub and calls it the file.
        """
        return self._ask(PARSE_MODALITY, extension=extension, detail=detail)


class _Ledger(_Namespace):
    """The flight recorder."""

    def record(self, action: str, ok: bool = True, data=None):
        """Note something that is not itself a Request."""
        return self._ask(LEDGER_RECORD, action=action, ok=ok, data=data)

    def read(self, limit: int = 50, *, conversation_id: int | None = None,
             origin: str = "", session_key: str = "",
             action_types=None, since_id: int | None = None):
        """Read recent rows, newest first. Query it targeted, never linearly.

        Every argument narrows in SQL, which is what makes "targeted" possible
        rather than merely advised — the ledger is write-optimized filler by
        volume, so an unfiltered read scans the whole flight recorder.

            # what this conversation touched on disk
            sdk.ledger.read(conversation_id=7,
                            action_types=["fs.write", "fs.delete"])

            # and, later, only what has happened since
            sdk.ledger.read(conversation_id=7, since_id=rows[0]["id"])

        Naming a conversation somebody else owns is refused.
        """
        return self._ask(LEDGER_READ, limit=limit,
                         conversation_id=conversation_id, origin=origin,
                         session_key=session_key,
                         action_types=list(action_types or []),
                         since_id=since_id)


class _Notifications(_Namespace):
    """What the system has told the user.

    Raising one is ``sdk.session.push(..., notify=True)`` — it lives with the
    other ways of reaching a person, because that is what it is. This namespace
    is the other direction: reading back what was raised, for a surface that
    draws them in a panel.

    A panel needs this because the bus only ever answers "what happened since
    you connected". Everything from before that — which is most of it, for a
    client that was closed while the work ran — is only here.
    """

    def list(self, limit: int = 50, *, since_id: int | None = None,
             unread_only: bool = False):
        """Read notifications, newest first. Only ever your own user's.

        ``since_id`` is the incremental form: a client holding rows up to N
        asks for what followed, which is what a reconnect wants rather than the
        whole history again.

            recent = sdk.notifications.list(limit=20)
            fresh  = sdk.notifications.list(since_id=recent[0]["id"])
        """
        return self._ask(NOTIFICATION_LIST, limit=limit, since_id=since_id,
                         unread_only=unread_only)

    def mark_read(self, ids=None, *, before_id: int | None = None):
        """Settle notifications, by id or everything up to one.

        ``before_id`` is the "mark all read" spelling. Returns how many rows
        actually changed, so calling it twice is idempotent rather than
        double-counted. Naming another user's rows changes nothing.
        """
        return self._ask(NOTIFICATION_MARK_READ, ids=list(ids or []),
                         before_id=before_id)


class _Net(_Namespace):
    """Network Requests — always classified, never auto-safe."""

    def http(self, url: str, method: str = "GET", headers: dict | None = None,
             body=None, *, params=None, json=None, to_file: str = "",
             max_bytes: int = 0):
        """Perform an outbound HTTP request.

        Secret handles may appear anywhere in the url, headers, or body; the
        kernel substitutes the real values on the way out, so the sandbox uses
        a credential it never held. Redirects are returned as 3xx replies and
        are never followed automatically: call ``http`` again with the
        response's ``location`` header so the new host gets its own policy
        decision.

        Answers ``{status, body, headers, truncated}``. ``body`` is decoded
        text and is capped — ``truncated`` says when the cap bit, and a
        truncated body is a reason to reach for ``to_file`` rather than to
        parse what came back.

        **``to_file`` downloads.** Name a path and the reply is streamed
        straight to it instead of crossing the wire, which is what makes an
        image, a PDF, an archive or anything large fetchable at all — the wire
        refuses every one of them. The answer then carries ``path``, ``bytes``,
        ``content_type`` and ``final_url``, with ``body`` empty; a non-2xx
        status answers in the same shape with ``path`` empty and the server's
        explanation in ``body``, so one branch on ``status`` covers both.

        Two things to know about it. Writing is a **second capability**: the
        kernel asks about the destination as well as the host, so a path
        outside the folders you may write to is refused even when the host is
        allowed. And redirects behave differently here — hops *within the same
        host* are followed for you (the host was already decided), while a hop
        to a different host comes back as a 3xx to re-call, because that host
        has not been asked about.

        ``max_bytes`` lowers the user's own download ceiling for this one
        call. It cannot raise it.
        """
        if body is not None and json is not None:
            raise ValueError("body and json are mutually exclusive")
        args = {"url": url, "method": method,
                "headers": headers or {}, "body": body}
        if params is not None:
            args["params"] = params
        if json is not None:
            args["json"] = json
        if to_file:
            args["to_file"] = str(to_file)
        if max_bytes:
            args["max_bytes"] = int(max_bytes)
        return self._ask(NET_HTTP, **args)

    def http_json(self, url: str, method: str = "GET",
                  headers: dict | None = None, body=None, *, params=None,
                  json=None):
        """Perform an HTTP request and decode its text body as JSON.

        No ``to_file``: a download has no body to decode, and asking for both
        is asking for two different answers to one call.
        """
        answer = self.http(url, method=method, headers=headers, body=body,
                           params=params, json=json) or {}
        raw = answer.get("body", "")
        if raw == "":
            decoded = None
        else:
            try:
                decoded = json_module.loads(raw)
            except (TypeError, ValueError) as exc:
                status = answer.get("status", 0)
                raise ValueError(
                    f"HTTP {status} response is not valid JSON: {exc}") from exc
        return {**answer, "body": decoded}


class _Proc(_Namespace):
    """Running commands.

    Two shapes, because running a command and *keeping* one are different
    acts. :meth:`run` blocks and hands back what the command printed;
    :meth:`start` hands back a handle to something still running, which
    :meth:`status`, :meth:`stop` and :meth:`list` then speak about.

    ``shell`` is the difference between an argv and a *command line*.
    Left at ``None`` the argv is executed directly, which is what you want
    when you built the list yourself — no quoting, no metacharacters, no
    surprises. Pass ``"default"`` to hand the string to the platform shell
    (``cmd.exe`` on Windows, ``/bin/sh`` elsewhere) when you need pipes,
    redirection or ``&&``; ``"powershell"`` picks that one explicitly. The
    kernel builds the invocation, because getting it wrong on Windows
    mangles every embedded quote.
    """

    def run(self, argv, timeout: float = 120.0, cwd=None, shell=None):
        """Run a command to completion."""
        return self._ask(PROC_RUN, argv=argv, timeout=timeout,
                         cwd=str(cwd) if cwd else None, shell=shell)

    def start(self, argv, cwd=None, shell=None, label: str = ""):
        """Start a command and leave it running.

        Answers ``{id, pid, log, command}``. Output is teed to ``log``, a
        file readable with ``sdk.fs.read`` — a live process cannot stream
        across the boundary, so the log is how you watch one.
        """
        return self._ask(PROC_START, argv=argv,
                         cwd=str(cwd) if cwd else None, shell=shell,
                         label=label)

    def status(self, id: int, tail: int = 4000):
        """Ask after a started process, with the tail of its output."""
        return self._ask(PROC_STATUS, id=id, tail=tail)

    def stop(self, id: int):
        """End a started process and forget it."""
        return self._ask(PROC_STOP, id=id)

    def list(self):
        """Every process this system started and still remembers."""
        return self._ask(PROC_LIST)


class _Scripts(_Namespace):
    """Running SDK code that is not a plugin.

    A script is a file under ``<tree>/scripts/`` with functions that take
    ``sdk`` — no base class, no declarations, nothing that registers it. It is
    what to reach for instead of ``sdk.proc.run`` whenever the work can be
    written in Python: a shell command is an OS process outside the boundary
    and is asked about every single time, while a script is contained, so
    running one costs no dialog at all.

    The exception is a script importing a library the validator cannot see
    inside. That is asked about, and the library is named — it is the one part
    of a script whose effects do not come back as Requests.

    A script gets 60s by default and may declare more at module scope
    (``timeout = 600``), the same way it declares ``box``. That measures
    *running* time — waiting on the kernel is not charged — and the kernel
    clamps it at 600s, with a separate 600s wall-clock ceiling bounding the run
    however it spends the time.
    """

    def run(self, path: str, entry: str = "main", *, wait: bool = True,
            **args):
        """Run a script and hand back what its entry function returned.

        ``entry`` names the function; everything else is passed to it as
        keyword arguments, so ``sdk.scripts.run(p, total=3)`` calls
        ``main(sdk, total=3)``.

        ``wait=False`` answers as soon as it has *begun*, with
        ``{"script", "id", "started"}`` — the ``id`` is the handle
        :meth:`collect` and :meth:`stop` take. That is how several run at
        once::

            ids = [sdk.scripts.run(p, "analyse", wait=False, doc=d)["id"]
                   for d in documents]
            for report in sdk.scripts.collect(ids):
                sdk.log(report["state"], report["data"])

        Each one is a box of its own, so they genuinely run in parallel — and
        unlike a subagent, nothing here involves a model. Reach for this when
        the work is code and for ``sdk.agent.spawn`` when it is judgement.
        """
        return self._ask(SCRIPT_RUN, path=str(path), entry=entry,
                         args=args, wait=bool(wait))

    def collect(self, ids=None, timeout: float | None = None):
        """Wait for detached scripts and take their results.

        ``ids=None`` takes every script this caller started with
        ``wait=False`` and has not collected yet. ``timeout=0`` polls without
        waiting — anything still going comes back with ``state ==
        "running"`` and stays uncollected, so a later call still gets it.
        ``timeout=None`` waits until each one's own deadline.

        Each report is a dict with ``id``, ``script``, ``state``, ``ok``,
        ``data`` and ``error``, where ``state`` is ``running``, ``done``,
        ``failed`` or ``cancelled``. **Delivered once**: a finished result is
        held until somebody takes it, and taken only once, so two collectors
        cannot both act on the same answer.
        """
        return self._ask(SCRIPT_COLLECT, ids=ids, timeout=timeout)

    def stop(self, id: str):
        """Cancel a detached script. Narrows, so it is the safe direction."""
        return self._ask(SCRIPT_STOP, id=id)


class _App(_Namespace):
    """The application itself."""

    def stop(self, restart: bool = False):
        """Shut the application down, optionally starting it again.

        Returns the message to show the user. The kernel defers the actual
        stop briefly so the answer reaches the frontend first — otherwise the
        process ends before anyone is told why.
        """
        return self._ask(APP_STOP, restart=bool(restart))


class _Env(_Namespace):
    """The environment. Credentials come back as handles."""

    def read(self, name: str):
        """Read a variable."""
        return self._ask(ENV_READ, name=name)


class _Secrets(_Namespace):
    """Credentials.

    Prefer the handle. ``sdk.config.read`` and ``sdk.env.read`` give you one
    for anything credential-shaped, and passing it to ``sdk.net.http`` works
    without your code ever holding the value.

    ``reveal`` is for the case handles cannot cover: driving a library that
    performs its own network I/O, so there is no Request for the kernel to
    substitute into. It always asks the user, naming the secret and what asked
    for it.
    """

    def reveal(self, name: str):
        """The plaintext of a secret. Always asks."""
        return self._ask(SECRET_REVEAL, name=name)


# ──────────────────────────────────────────────────────────────────────
# Helpers: no Request, no cost, no ledger row.
# ──────────────────────────────────────────────────────────────────────

class _Path:
    """Pure path helpers — string arithmetic, never the filesystem.

    Guest code cannot import ``pathlib`` or ``os.path``: both reach the
    environment, and the validator refuses them. But manipulating a path is
    *computation* — joining, splitting, taking a parent — and by this SDK's own
    test (does it touch disk, network, clock, or process?) that makes it a
    helper, not a Request. Without these, a plugin doing anything path-shaped
    had to concatenate strings by hand and get separators wrong.

    Two things it deliberately does not do, both because they would reach the
    environment and stop being helpers:

    - **No symlink resolution.** ``normalize`` is textual, so two names for
      one file through a link normalize differently. For the caller that
      matters — read-before-edit tracking — that fails toward "not read yet",
      the strict direction.
    - **No current directory.** A relative path with no ``base`` stays
      relative rather than being resolved against a cwd, which inside a box
      is ``sandbox/`` and means nothing to the plugin anyway. Pass the base
      you mean, usually ``sdk.paths.get("project")``.
    """

    @staticmethod
    def _os():
        """The path flavour this platform actually uses."""
        return ntpath if sys.platform == "win32" else posixpath

    @staticmethod
    def join(*parts) -> str:
        """Join path segments with the platform separator."""
        usable = [str(p) for p in parts if p not in (None, "")]
        return _Path._os().join(*usable) if usable else ""

    @staticmethod
    def parent(path) -> str:
        """The containing directory."""
        return _Path._os().dirname(str(path))

    @staticmethod
    def name(path) -> str:
        """The final component, extension included."""
        return _Path._os().basename(str(path))

    @staticmethod
    def stem(path) -> str:
        """The final component without its extension."""
        flavour = _Path._os()
        return flavour.splitext(flavour.basename(str(path)))[0]

    @staticmethod
    def suffix(path) -> str:
        """The extension, dot included, or ``""``."""
        return _Path._os().splitext(str(path))[1]

    @staticmethod
    def is_absolute(path) -> bool:
        """Whether this path stands on its own."""
        return _Path._os().isabs(str(path))

    @staticmethod
    def absolute(path, base="") -> str:
        """Resolve against ``base`` and collapse ``..`` textually.

        A relative path with no base is returned normalized but still
        relative — see the class docstring on why the cwd is not consulted.
        """
        flavour = _Path._os()
        raw = str(path)
        if base and not flavour.isabs(raw):
            raw = flavour.join(str(base), raw)
        return flavour.normpath(raw)

    @staticmethod
    def normalize(path) -> str:
        """A canonical key for comparing two paths.

        Case-folded where the platform is case-insensitive, so a plugin does
        not treat ``C:\\A.py`` and ``c:\\a.py`` as different files.

        ``normcase`` alone does not deliver that promise: it is the identity
        function on every POSIX platform, macOS included — where APFS is
        case-insensitive by default. So the fold has to be asked for by
        platform rather than borrowed from the path flavour, or a Mac keys two
        spellings of one file differently and ``file_reads`` reports a file the
        agent just read as never read.
        """
        canonical = _Path._os().normcase(_Path.absolute(path))
        return canonical.lower() if sys.platform == "darwin" else canonical

    @staticmethod
    def as_posix(path) -> str:
        """Render a platform path with forward-slash separators."""
        flavour = _Path._os()
        return str(path).replace(flavour.sep, "/")

    @staticmethod
    def relative(path, start) -> str:
        """Render ``path`` relative to ``start`` without touching disk."""
        return _Path._os().relpath(str(path), str(start))

    @staticmethod
    def with_suffix(path, suffix) -> str:
        """Replace the final suffix using ``pathlib``-style validation."""
        flavour = _Path._os()
        raw_path = str(path)
        final_name = flavour.basename(raw_path)
        if not final_name or final_name in (flavour.curdir, flavour.pardir):
            raise ValueError(f"{raw_path!r} has no final name")
        suffix = str(suffix)
        separators = tuple(
            item for item in (flavour.sep, flavour.altsep) if item)
        if suffix and (not suffix.startswith(".")
                       or any(sep in suffix for sep in separators)):
            raise ValueError("suffix must be empty or start with '.'")
        root, _ = flavour.splitext(raw_path)
        return root + suffix

    @staticmethod
    def within(path, root) -> bool:
        """Whether ``path`` is ``root`` or sits underneath it.

        Compared on normalized strings with a separator guard, so ``/data``
        does not appear to contain ``/database``.
        """
        flavour = _Path._os()
        target, base = _Path.normalize(path), _Path.normalize(root)
        return (target == base
                or target.startswith(base.rstrip(flavour.sep) + flavour.sep))


class _Text:
    """Pure text helpers."""

    @staticmethod
    def truncate(text: str, limit: int, suffix: str = "...") -> str:
        """Shorten text to limit characters."""
        text = text or ""
        if len(text) <= limit:
            return text
        return text[:max(0, limit - len(suffix))] + suffix

    @staticmethod
    def value(value) -> str:
        """Render a configuration value without Python repr artifacts."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list):
            return "(none)" if not value else ", ".join(
                str(item) for item in value)
        return str(value)


class _Markdown:
    """Presentation helpers, mirroring the kernel's markdown-on-the-wire
    convention so sandboxed output renders identically."""

    @staticmethod
    def table(headers, rows, *, leading_blank: bool = True) -> str:
        """Render a GitHub-flavored markdown table."""
        def cell(value):
            return str("" if value is None else value).replace(
                "\n", " ").replace("|", "\\|")

        head = "| " + " | ".join(cell(h) for h in headers) + " |"
        rule = "|" + "|".join(" --- " for _ in headers) + "|"
        body = [
            "| " + " | ".join(cell(c) for c in row) + " |"
            for row in rows
        ]
        table = "\n".join([head, rule, *body])
        return "\n" + table if leading_blank else table

    @staticmethod
    def card(title: str, pairs) -> str:
        """Render a detail card as a two-column table."""
        return _Markdown.table(
            [title, ""], pairs, leading_blank=False)

    @staticmethod
    def align_tables(text: str) -> str:
        """Pad markdown tables into monospace columns; leave the rest alone.

        Half of what ``plain`` does, on its own because the two halves are
        wanted separately. A surface that renders code fences natively — a
        chat client showing a ``<pre>`` block — wants the alignment without
        the fence-stripping, and inlining the padding loop to get it is how
        two copies of this algorithm start.
        """
        import re

        lines = (text or "").split("\n")
        row = re.compile(r"^\s*\|.*\|\s*$")
        separator = re.compile(r"^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$")

        def cells(line):
            """Split one table row, honouring escaped pipes."""
            parts = re.split(r"(?<!\\)\|", line.strip().strip("|"))
            return [p.strip().replace("\\|", "|") for p in parts]

        out, i = [], 0
        while i < len(lines):
            if (row.match(lines[i]) and i + 1 < len(lines)
                    and separator.match(lines[i + 1])):
                block = [lines[i]]
                j = i + 2
                while j < len(lines) and row.match(lines[j]):
                    block.append(lines[j])
                    j += 1
                rows = [cells(line) for line in block]
                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]
                sizes = [max(len(r[c]) for r in rows) for c in range(width)]

                def fmt(cs):
                    """One padded line."""
                    return "  ".join(v.ljust(w)
                                     for v, w in zip(cs, sizes)).rstrip()

                out.append(fmt(rows[0]))
                out.append("  ".join("-" * w for w in sizes))
                out.extend(fmt(r) for r in rows[1:])
                i = j
            else:
                out.append(lines[i])
                i += 1
        return "\n".join(out)

    @staticmethod
    def quote(text: str) -> str:
        """Render text as a markdown blockquote."""
        return "\n".join(
            f"> {line}" if line.strip() else ">"
            for line in (text or "").splitlines()
        )

    @staticmethod
    def tools(tools) -> str:
        """Render structured tool metadata in the standard command table."""
        if not tools:
            return "No tools registered."
        rows = []
        for tool in tools:
            params = tool.get("parameters") or {}
            required = set(params.get("required") or [])
            fields = ", ".join(
                f"{name}{'*' if name in required else ''}"
                for name in (params.get("properties") or {})
            )
            desc = _Text.truncate(
                (tool.get("description") or "").split("\n")[0], 100)
            services = tool.get("requires_services") or []
            if services:
                desc += f" (needs: {', '.join(services)})"
            rows.append((tool["name"], fields, desc))
        return "Tools:\n\n" + _Markdown.table(
            ["Tool", "Args", "Description"], rows, leading_blank=False)

    @staticmethod
    def tool_result(result) -> str:
        """Render a complete structured tool-result envelope."""
        import json

        if not result.get("success", True):
            return (
                "Failed: "
                + (result.get("error") or result.get("llm_summary")
                   or "(no details)")
            )
        data = result.get("data")
        summary = result.get("llm_summary") or ""
        if isinstance(data, dict) and "columns" in data and "rows" in data:
            rows = data["rows"]
            if not rows:
                return "(no results)"
            table = _Markdown.table(
                data["columns"],
                [
                    [_Text.truncate(str(value), 60) for value in row]
                    for row in rows
                ],
                leading_blank=False,
            )
            if data.get("truncated"):
                table += "\n... (results capped at 100 rows)"
            return table
        if data is None:
            return summary or "(no output)"
        if summary:
            text = f"Done: {summary.strip()}"
            final = data.get("final_text") if isinstance(data, dict) else None
            return f"{text}\n\n{str(final).strip()}" if final else text
        try:
            return json.dumps(data, indent=2, default=str)
        except Exception:
            return str(data)

    @staticmethod
    def tasks(tasks) -> str:
        """Render structured task metadata in the standard status sections."""
        if not tasks:
            return "No tasks registered."
        empty = {
            "PENDING": 0,
            "PROCESSING": 0,
            "DONE": 0,
            "FAILED": 0,
        }
        normalized = [
            {
                **task,
                "trigger": task.get("trigger", "path"),
                "counts": {**empty, **(task.get("counts") or {})},
                "paused": bool(task.get("paused")),
            }
            for task in tasks
        ]
        normalized.sort(key=lambda task: task["name"])
        sections = [
            (
                "Path-driven tasks",
                [task for task in normalized
                 if task["trigger"] == "path"],
            ),
            (
                "Event-driven tasks",
                [task for task in normalized
                 if task["trigger"] == "event"],
            ),
        ]
        other = [
            task for task in normalized
            if task["trigger"] not in {"path", "event"}
        ]
        if other:
            sections.append(("Other tasks", other))

        lines = ["Tasks:"]
        for title, section in sections:
            lines += ["", f"**{title}**", ""]
            if not section:
                lines.append("(none)")
                continue
            rows = []
            for task in section:
                details = []
                if task["paused"]:
                    details.append("paused")
                channels = task.get("trigger_channels") or []
                if channels:
                    details.append(f"listens on: {', '.join(channels)}")
                services = task.get("requires_services") or []
                if services:
                    details.append(f"needs: {services}")
                # ``schedule_count`` is what ``task.list`` emits; this read
                # ``schedules``, which nothing has ever put on the wire, so the
                # branch contributed nothing for as long as it existed.
                if scheduled := task.get("schedule_count"):
                    details.append(f"{scheduled} scheduled job(s)")
                counts = task["counts"]
                rows.append((
                    task["name"],
                    counts["PENDING"],
                    counts["PROCESSING"],
                    counts["DONE"],
                    counts["FAILED"],
                    "; ".join(details),
                ))
            lines.append(_Markdown.table(
                ["Task", "Pending", "Running", "Done", "Failed", "Notes"],
                rows,
                leading_blank=False,
            ))
        return "\n".join(lines)


class _Forms:
    """Pure helpers for describing command forms."""

    @staticmethod
    def from_schema(schema, *, prompt_optional: bool = False):
        """Convert a JSON object schema into serializable form steps."""
        from .forms import FormStep

        props = (schema or {}).get("properties", {})
        required = set((schema or {}).get("required", []))
        return [
            FormStep(
                name,
                _Forms._prompt(name, info),
                name in required,
                info.get("type", "string"),
                info.get("enum"),
                default=info.get("default"),
                prompt_when_missing=(
                    prompt_optional and name not in required),
            )
            for name, info in props.items()
        ]

    @staticmethod
    def _prompt(name, info):
        label = str(name or "value").replace("_", " ")
        desc = str((info or {}).get("description") or "").strip()
        choose = (info or {}).get("enum") or (
            info or {}).get("type") == "boolean"
        if choose:
            prompt = f"Choose {label}."
        else:
            article = (
                label if label.startswith(("a ", "an ", "the "))
                else f"{'an' if label[:1].lower() in 'aeiou' else 'a'} {label}"
            )
            prompt = f"Enter {article}."
        return f"{prompt}\n{desc}" if desc else prompt

    @staticmethod
    def setting_actions(settings, prefix: str = "edit_setting:"):
        """Return action values and labels for editable setting metadata."""
        settings = settings or []
        return (
            [f"{prefix}{setting['key']}" for setting in settings],
            [f"Edit {setting['title']}" for setting in settings],
        )

    @staticmethod
    def setting_for_action(
        settings,
        action,
        prefix: str = "edit_setting:",
    ):
        """Resolve an encoded setting action to its declared metadata."""
        if not isinstance(action, str) or not action.startswith(prefix):
            return None
        key = action[len(prefix):]
        return next(
            (setting for setting in (settings or [])
             if setting["key"] == key),
            None,
        )

    @staticmethod
    def setting_value_step(setting):
        """Build the standard typed value step for a setting."""
        from .forms import FormStep

        type_ = _Forms._setting_type(setting)
        if type_ == "path_list":
            prompt = (
                "Enter one folder path per line. / and \\ are both accepted; "
                "each folder must already exist. Example:\n\n"
                "C:\\Users\\you\\Notes\nD:\\Archive"
            )
        elif type_ == "path":
            prompt = (
                "Enter a path. / and \\ are both accepted; the parent folder "
                "must exist."
            )
        elif type_ == "array":
            prompt = (
                "Enter a list of items, one on each line, like so:\n\n"
                "item 1\nitem 2"
            )
        else:
            prompt = "Enter the new value."
        return FormStep("value", prompt, True, type_)

    @staticmethod
    def _setting_type(setting):
        info = setting.get("info") or {}
        type_ = info.get("type")
        if type_ in {"path", "path_list"}:
            return type_
        if type_ == "json_list":
            return "array"
        if type_ == "json_dict":
            return "object"
        if type_ in {"bool", "boolean"}:
            return "boolean"
        if type_ == "slider":
            return "number" if info.get("is_float") else "integer"
        default = setting.get("default")
        if isinstance(default, list):
            return "array"
        if isinstance(default, dict):
            return "object"
        return "string"

    @staticmethod
    def plain(text: str) -> str:
        """Markdown rendered for a monospace surface: a terminal.

        Tables become padded columns and code-fence markers are dropped, since
        the content inside already reads as plain text. Every other line passes
        through untouched, so one message body works on rich and plain surfaces
        alike — which is the whole point of markdown being the wire format.

        Mirrors the kernel's own ``render_plain``. It lives here because a
        sandboxed frontend cannot import kernel helpers, and because it is
        pure: no Request, no cost. The table half is ``md.align_tables``,
        which some surfaces want without the fence-stripping.
        """
        import re

        return "\n".join(
            line for line in _Markdown.align_tables(text).split("\n")
            if not re.fullmatch(r"\s*```\w*\s*", line))


# ``plain`` predates the forms namespace; keep it on the markdown surface.
_Markdown.plain = staticmethod(_Forms.plain)


class SDK:
    """The handle sandboxed code is given.

    Bound to one execution. Holds no kernel objects — only the channel it
    sends Requests down — so the same code runs unchanged in-process or in a
    subprocess.
    """

    #: Raised when a Request is refused by policy or by the user.
    Denied = Denied
    #: Raised when a Request breaks. ``Denied`` is a subclass of it.
    Failed = RequestFailed

    def __getattr__(self, name: str):
        """Name the closest namespace. See :meth:`_Namespace.__getattr__`."""
        if name.startswith("_"):
            raise AttributeError(name)
        spaces = sorted(k for k in vars(self) if not k.startswith("_"))
        close = difflib.get_close_matches(name, spaces, 1, 0.6)
        hint = f" Did you mean sdk.{close[0]}?" if close else ""
        raise AttributeError(f"sdk has no {name!r}.{hint} "
                             f"The namespaces are: {', '.join(spaces)}")

    def __init__(self, channel):
        self._channel = channel
        self.fs = _FS(self)
        self.db = _DB(self)
        self.conv = _Conv(self)
        self.session = _Session(self)
        self.ui = _UI(self)
        self.config = _Config(self)
        self.paths = _Paths(self)
        self.users = _Users(self)
        self.plugins = _Plugins(self)
        self.services = _Services(self)
        self.tools = _Tools(self)
        self.commands = _Commands(self)
        self.agent = _Agent(self)
        self.llm = _LLM(self)
        # Set by BasePlugin.__hook__ for the duration of one doorway visit, so
        # ``llm.proceed`` can name the call it is meant to place without the
        # author having to carry a token around.
        self._hook_token = ""
        # Set by BaseLLMBackend.__chat__ for the duration of one call, the same
        # shape as the hook token: it reaches the delta sink for *this* call
        # and nothing else, and is cleared however the call ends.
        self._delta_token = ""
        self.frontend = _Frontend(self)
        self.console = _Console(self)
        self.http = _Http(self)
        # Set once by BaseFrontend.__bind__ when this box opens, and it stays
        # for the box's life — unlike the hook token, a frontend is not visiting
        # a doorway, it *is* resident. The kernel parks the matching adapter and
        # drops it at stop, so a token that outlived its frontend reaches
        # nothing.
        self._frontend_token = ""
        self.cron = _Cron(self)
        self.events = _Events(self)
        self.tasks = _Tasks(self)
        self.files = _Files(self)
        self.parse = _Parse(self)
        self.ledger = _Ledger(self)
        self.notifications = _Notifications(self)
        self.net = _Net(self)
        self.proc = _Proc(self)
        self.scripts = _Scripts(self)
        self.env = _Env(self)
        self.app = _App(self)
        self.secrets = _Secrets(self)
        self.path = _Path()
        self.text = _Text()
        self.md = _Markdown()
        self.forms = _Forms()

    # ── the channel ────────────────────────────────────────────────

    def _send(self, request: Request):
        """Send a Request and block until the kernel answers."""
        return self._channel.send(request)

    def _notify(self, request: Request) -> None:
        """Send a Request without waiting for an answer.

        Falls back to ``send`` for a channel that predates the one-way path —
        a test double, most often — since discarding an answer is always
        possible and refusing to run is not.
        """
        notify = getattr(self._channel, "notify", None)
        if notify is None:
            self._channel.send(request)
            return
        notify(request)

    def log(self, message: str, level: str = "info") -> None:
        """Write to the kernel's log sink.

        The deliberate edge case: logging does reach disk, but the SDK routes
        it so the author never writes a Request for it. Reuse this pattern
        wherever a Request would be too noisy to write by hand.
        """
        self._channel.log(level, str(message))

    # ── returning ──────────────────────────────────────────────────

    def ok(self, data=None, *, llm_summary: str = "", attachments=None,
           also_contains=None, discovered_paths=None, per_path=None):
        """Succeed with a value.

        ``llm_summary`` is what the model is told when the raw data is the
        wrong thing to show it; ``attachments`` are files to put in front of
        the user. The last three are for tasks: nested content found while
        parsing, new files the pipeline should register, and — for a task
        handed a *batch* — one outcome per path.

        ``per_path`` is how a batch task says that file three failed and the
        rest are fine. Each entry is a dict of the same shape as the whole
        result: ``{"ok", "error", "data", "also_contains", "discovered_paths"}``,
        in the order the paths arrived. Give it and the outer ``data`` is
        ignored; omit it and the one outcome applies to every path.
        """
        return Result(data=data, llm_summary=llm_summary,
                      attachment_paths=list(attachments or []),
                      also_contains=list(also_contains or []),
                      discovered_paths=list(discovered_paths or []),
                      per_path=list(per_path or []))

    def fail(self, error: str, retryable: bool = False):
        """Fail with a reason."""
        return Result.failure(error, retryable=retryable)

    # ── trying again ───────────────────────────────────────────────

    def retry(self, fn, attempts: int = 3, backoff: float = 0.5, on=None):
        """Run ``fn()``, retrying only what the kernel said was worth retrying.

        ``Result.retryable`` is set by the handler that failed — the one place
        that actually knows whether trying again could plausibly work. A read
        that hit a locked file, an HTTP call that timed out and a box that died
        all carry it; a malformed query and a refusal do not. This spends that
        signal, so the common loop is one line::

            page = sdk.retry(lambda: sdk.net.http(url))

        Waits ``backoff``, then double, between attempts. Pass ``on=`` a
        predicate over the exception to decide for yourself — ``on=lambda e:
        e.code == "not_found"`` to retry something the kernel does not consider
        transient, or ``on=lambda e: False`` to disable retrying entirely
        without changing the call.

        **A refusal is never retried.** Policy is not a transient condition,
        and asking three times is how one dialog becomes three. ``Denied``
        propagates on the first attempt whatever ``on`` says, so a caller can
        still tell "you said no" from "the disk was busy"::

            try:
                rows = sdk.retry(lambda: sdk.db.query(sql))
            except sdk.Denied:
                return "I need permission to read that."
            except sdk.Failed as exc:
                return f"Gave up after {exc.error}"

        The backoff sleeps, and sleeping is *running* time — no Request is in
        flight for the kernel to discount, so it is charged against the
        deadline like any other computation. At the defaults that is 1.5s,
        which is why a long loop that retries wants to check :meth:`budget`.
        """
        import time

        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        for attempt in range(attempts):
            try:
                return fn()
            except Denied:
                # Ordering is load-bearing: ``Denied`` subclasses ``Failed``,
                # so catching the general case first would swallow every
                # refusal into the retry loop.
                raise
            except RequestFailed as exc:
                if attempt == attempts - 1:
                    raise
                if not (on(exc) if on is not None else exc.retryable):
                    raise
                self.log(f"retrying after {exc.error} "
                         f"({attempt + 1}/{attempts})", "debug")
                time.sleep(backoff * 2 ** attempt)
        # Unreachable: the loop either returns or raises on the last attempt.
        raise RuntimeError("retry fell through")

    # ── how long is left ───────────────────────────────────────────

    def budget(self) -> dict:
        """How much of this execution's deadline is left.

        Answers ``{"running", "wall", "deadline", "ceiling"}`` — seconds of
        running time remaining, seconds of wall clock remaining, the clamped
        deadline this execution was actually given, and the wall ceiling that
        bounds it however it spends the time. ``running`` is ``None`` when
        nothing is enforcing a deadline right now.

        The two numbers differ, and which one bites depends on what the code
        spends its time doing. **Running** time is what a declared ``timeout``
        measures: elapsed minus whatever the kernel spent owing an answer, so
        four minutes inside ``sdk.proc.run`` costs almost nothing against it.
        **Wall** clock is the backstop, and it is not declarable — it is what
        stops a runaway hiding inside long waits.

        This exists so long work can stop *itself*. Without it the watchdog is
        the only thing that ends a run that has gone on too long, and it ends
        it by killing the box — so a loop three-quarters of the way through a
        corpus returns nothing at all. Checking lets it hand back what it has::

            done = []
            for doc in documents:
                if sdk.budget()["running"] < 20:
                    break
                done.append(analyse(sdk, doc))
            return sdk.ok({"done": done, "resume_at": len(done)},
                          llm_summary=f"{len(done)}/{len(documents)} analysed")

        Free to call in a loop: it is read-only, so it neither writes a ledger
        row nor counts as a change.
        """
        # Raising on failure like every other Request. ``respond`` next door
        # reaches ``_send`` bare because it is about to unwind anyway; this one
        # answers a caller who will use the number.
        result = self._send(Request(SELF_BUDGET, {}))
        if result.ok:
            return result.data
        raise (Denied if result.denied else RequestFailed)(result, SELF_BUDGET)

    def respond(self, value) -> None:
        """Return a result and terminate.

        Asking to end has to actually end it, so this never returns — the
        runner catches the unwind and takes the carried value as the result.
        Invalid for persistent containers, which yield instead.
        """
        self._send(Request(SELF_RESPOND, {"value": value}))
        raise Terminated(value)
