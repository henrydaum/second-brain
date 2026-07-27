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

from .channel import Terminated
from .requests import Denied, RequestFailed
from .requests import (AGENT_COMPLETE, AGENT_SCHEDULE, AGENT_SPAWN,
                       COMMAND_CALL, COMMAND_LIST, CONFIG_READ, CONFIG_WRITE,
                       CONV_APPEND, CONV_CLEAR, CONV_CREATE, CONV_DELETE, CONV_LIST,
                       CONV_LOAD, CONV_READ, CONV_SET_CATEGORY,
                       CONV_SET_NOTIFICATION_MODE, CONV_SET_TITLE,
                       CRON_CREATE, CRON_ENABLE, CRON_GET, CRON_LIST,
                       CONSOLE_READ, CONSOLE_WRITE,
                       CRON_REMOVE, CRON_UPDATE, DB_DEFINE, DB_QUERY, DB_WRITE,
                       ENV_READ, EVENT_EMIT, EVENT_REQUEST, FILE_LIST,
                       FILE_REGISTER, FRONTEND_ATTEND, FRONTEND_BIND,
                       FRONTEND_CANCEL, FRONTEND_PENDING, FRONTEND_RESOLVE,
                       FRONTEND_SUBMIT,
                       FS_DELETE, FS_LIST, FS_MOVE, FS_READ, FS_READ_BYTES,
                       FS_SEARCH, FS_TEMP, FS_WRITE, FS_WRITE_BYTES,
                       LEDGER_READ,
                       LEDGER_RECORD, NET_HTTP, PARSE_FILE, PARSE_MODALITY,
                       MODEL_DELTA, MODEL_PROCEED, PATH_GET,
                       PLUGIN_DESCRIBE, PLUGIN_INSTALL, PLUGIN_LIST,
                       PLUGIN_UNINSTALL, PLUGIN_UPDATE, PROC_RUN,
                       SECRET_REVEAL, SELF_RESPOND,
                       SERVICE_CALL, SERVICE_LIST, SERVICE_LOAD,
                       SERVICE_UNLOAD, SESSION_ADD_PROMPT,
                       SESSION_ADD_TOOL, SESSION_CANCEL, SESSION_GET,
                       SESSION_LIST, SESSION_PUSH, SESSION_REMOVE_PROMPT,
                       SESSION_REMOVE_TOOL, SESSION_STATE_GET,
                       SESSION_STATE_SET, TASK_ENQUEUE, TASK_GRAPH, TASK_LIST,
                       TASK_OUTPUT, TASK_PAUSE, TASK_RESET, TASK_STATUS,
                       TASK_TRIGGER, TOOL_CALL, TOOL_LIST, UI_APPROVE, UI_ASK,
                       UI_RENDER, USER_LIST, USER_READ, USER_WRITE, Request,
                       Result)


class _Namespace:
    """Base for Request-making SDK namespaces."""

    def __init__(self, sdk: "SDK"):
        self._sdk = sdk

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
        """Create, overwrite, or append. ``mode="append"`` to add."""
        return self._ask(FS_WRITE, path=str(path), data=data, mode=mode)

    def read_bytes(self, path) -> bytes:
        """Read a file as raw bytes.

        Use this for anything that is not text — an image, audio, a PDF.
        ``read`` decodes as UTF-8 with replacement, which silently mangles
        binary content rather than failing.
        """
        return base64.b64decode(self._ask(FS_READ_BYTES, path=str(path)) or "")

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

    def list(self, path, pattern: str = "*", details: bool = False):
        """List a directory, optionally with entry type metadata."""
        return self._ask(
            FS_LIST, path=str(path), pattern=pattern, details=details)

    def search(self, pattern: str, root=".", glob: str = "**/*"):
        """Search file contents beneath a root."""
        return self._ask(FS_SEARCH, pattern=pattern, root=str(root), glob=glob)

    def delete(self, path):
        """Remove a file or a tree."""
        return self._ask(FS_DELETE, path=str(path))

    def move(self, src, dst, copy: bool = False):
        """Move or copy one path to another."""
        return self._ask(FS_MOVE, src=str(src), dst=str(dst), copy=copy)

    def temp(self, directory: bool = False, suffix: str = ""):
        """Scratch space you may always have."""
        return self._ask(FS_TEMP, directory=directory, suffix=suffix)


class _DB(_Namespace):
    """Database Requests.

    Reads are broad; what is narrowed is *whose* rows. User-scoped tables are
    reached through their ``my_`` name — ``my_conversations`` rather than
    ``conversations``.
    """

    def query(self, sql: str, params=None):
        """Read rows."""
        return self._ask(DB_QUERY, sql=sql, params=list(params or []))

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

    def read(self, conversation_id, details: bool = False):
        """Messages and metadata, optionally with restored-state details."""
        return self._ask(
            CONV_READ, id=conversation_id, details=details)

    def list(
        self,
        *,
        category=None,
        limit: int = 50,
        details: bool = False,
    ):
        """Current-user conversations, optionally with category metadata."""
        return self._ask(
            CONV_LIST, category=category, limit=limit, details=details)

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

    def push(self, message: str, key: str = ""):
        """Send the user a message out of band."""
        return self._ask(SESSION_PUSH, message=message, key=key)

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

    def add_tool(self, tool: str, key: str = ""):
        """Widen the agent's scope."""
        return self._ask(SESSION_ADD_TOOL, tool=tool, key=key)

    def remove_tool(self, tool: str, key: str = ""):
        """Narrow the agent's scope."""
        return self._ask(SESSION_REMOVE_TOOL, tool=tool, key=key)

    def add_prompt(self, text: str, key: str = ""):
        """Inject system prompt text."""
        return self._ask(SESSION_ADD_PROMPT, text=text, key=key)

    def remove_prompt(self, handle, key: str = ""):
        """Withdraw injected prompt text."""
        return self._ask(SESSION_REMOVE_PROMPT, handle=handle, key=key)


class _UI(_Namespace):
    """Talking to the person."""

    def ask(self, prompt: str, title: str = "Question", type: str = "text",
            choices=None, timeout: float = 300.0):
        """Ask a question and wait. Refused when nobody is present."""
        return self._ask(UI_ASK, prompt=prompt, title=title, type=type,
                         choices=list(choices or []), timeout=timeout)

    def approve(self, action: str, justification: str = ""):
        """Ask the user to approve a described action."""
        return self._ask(UI_APPROVE, action=action,
                         justification=justification)

    def render(self, paths, caption: str = ""):
        """Show files to the user in chat."""
        return self._ask(UI_RENDER, paths=[str(p) for p in paths],
                         caption=caption)


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
    """Kernel-owned application locations."""

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
    ):
        """List plugins, optionally narrowed by a kernel-defined role."""
        return self._ask(
            PLUGIN_LIST, source=source, category=category or None,
            role=role or None, details=details)

    def describe(self, name: str):
        """Metadata for one plugin."""
        return self._ask(PLUGIN_DESCRIBE, name=name)

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

    def call(self, name: str, method: str, **kwargs):
        """Invoke an exported method. Simple data comes back, never objects."""
        return self._ask(SERVICE_CALL, name=name, method=method,
                         kwargs=kwargs)

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
    ):
        """A model call. Keys and sockets stay kernel-side."""
        return self._ask(AGENT_COMPLETE, prompt=prompt,
                         messages=list(messages or []),
                         session_key=session_key or None)

    def spawn(self, prompt: str, wait: bool = True):
        """Run a subagent now."""
        return self._ask(AGENT_SPAWN, prompt=prompt, wait=wait)

    def schedule(self, prompt: str, cron: str):
        """Run a subagent later. Unattended, so always checked."""
        return self._ask(AGENT_SCHEDULE, prompt=prompt, cron=cron)


class _Model(_Namespace):
    """The call in flight.

    Both members are scoped to a call the kernel already decided to place, and
    neither means "make a model call". ``proceed`` is for an escort standing at
    the ``model_call`` doorway; ``delta`` is for the backend actually placing
    it. Outside those, there is no call and the Request is refused.
    """

    def delta(self, text: str) -> None:
        """Push one fragment of assistant text as it arrives.

        Only meaningful inside a backend's ``chat`` when ``request.stream``
        was set. One-way and unanswered, so streaming costs a frame per chunk
        rather than a round trip per chunk.

        There is deliberately nothing to check here. Whether the user wants
        this stream to continue is the kernel's decision, not the backend's:
        if they cancel, this execution is cancelled and the next Request
        raises ``Terminated``.
        """
        if not text:
            return
        self._sdk._notify(Request(MODEL_DELTA, {
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
        answer = self._ask(MODEL_PROCEED, token=self._sdk._hook_token,
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
                          extension: str = ""):
        """Hand over a file someone sent."""
        return self._ask(FRONTEND_SUBMIT, token=self._token(),
                         session_key=session_key, input_kind="attachment",
                         path=str(path), extension=extension)

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

    def pending_approval(self, session_key: str):
        """The id of the approval this session is waiting on, or None.

        Ask rather than remember. You are told an approval exists — you were
        handed one to render — but not when it stops existing: another frontend
        can answer it, or it can time out. Acting on a stale record means
        swallowing the next thing a person types as a yes or no.
        """
        return self._ask(FRONTEND_PENDING, token=self._token(),
                         session_key=session_key)

    def resolve(self, session_key: str, value, request_id: str = ""):
        """Answer the approval a ``render`` of kind ``approval`` showed.

        With no ``request_id`` the session's next pending request is answered,
        which is what a transport with one message at a time wants.
        """
        return self._ask(FRONTEND_RESOLVE, token=self._token(),
                         session_key=session_key, value=value,
                         request_id=request_id)


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
    """The parser registry."""

    def file(self, path, modality: str = "text"):
        """Parse a file to text."""
        return self._ask(PARSE_FILE, path=str(path), modality=modality)

    def modality(self, extension: str):
        """Resolve an extension's modality."""
        return self._ask(PARSE_MODALITY, extension=extension)


class _Ledger(_Namespace):
    """The flight recorder."""

    def record(self, action: str, ok: bool = True, data=None):
        """Note something that is not itself a Request."""
        return self._ask(LEDGER_RECORD, action=action, ok=ok, data=data)

    def read(self, limit: int = 50):
        """Read recent rows. Query it targeted, never linearly."""
        return self._ask(LEDGER_READ, limit=limit)


class _Net(_Namespace):
    """Network Requests — always classified, never auto-safe."""

    def http(self, url: str, method: str = "GET", headers: dict | None = None,
             body=None):
        """Perform an outbound HTTP request.

        Secret handles may appear anywhere in the url, headers, or body; the
        kernel substitutes the real values on the way out, so the sandbox uses
        a credential it never held.
        """
        return self._ask(NET_HTTP, url=url, method=method,
                         headers=headers or {}, body=body)


class _Proc(_Namespace):
    """Running commands."""

    def run(self, argv, timeout: float = 120.0, cwd=None):
        """Run a command to completion."""
        return self._ask(PROC_RUN, argv=argv, timeout=timeout,
                         cwd=str(cwd) if cwd else None)


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
    def cosine(a, b) -> float:
        """Cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

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
                details.extend(task.get("schedules") or [])
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
        pure: no Request, no cost.
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

        return "\n".join(line for line in out
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
        self.model = _Model(self)
        # Set by BasePlugin.__hook__ for the duration of one doorway visit, so
        # ``model.proceed`` can name the call it is meant to place without the
        # author having to carry a token around.
        self._hook_token = ""
        # Set by BaseLLMBackend.__chat__ for the duration of one call, the same
        # shape as the hook token: it reaches the delta sink for *this* call
        # and nothing else, and is cleared however the call ends.
        self._delta_token = ""
        self.frontend = _Frontend(self)
        self.console = _Console(self)
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
        self.net = _Net(self)
        self.proc = _Proc(self)
        self.env = _Env(self)
        self.secrets = _Secrets(self)
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
           also_contains=None, discovered_paths=None):
        """Succeed with a value.

        ``llm_summary`` is what the model is told when the raw data is the
        wrong thing to show it; ``attachments`` are files to put in front of
        the user. The last two are for tasks: nested content found while
        parsing, and new files the pipeline should register.
        """
        return Result(data=data, llm_summary=llm_summary,
                      attachment_paths=list(attachments or []),
                      also_contains=list(also_contains or []),
                      discovered_paths=list(discovered_paths or []))

    def fail(self, error: str, retryable: bool = False):
        """Fail with a reason."""
        return Result.failure(error, retryable=retryable)

    def respond(self, value) -> None:
        """Return a result and terminate.

        Asking to end has to actually end it, so this never returns — the
        runner catches the unwind and takes the carried value as the result.
        Invalid for persistent containers, which yield instead.
        """
        self._send(Request(SELF_RESPOND, {"value": value}))
        raise Terminated(value)
