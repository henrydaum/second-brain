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

from .channel import Terminated
from .requests import Denied, RequestFailed
from .requests import (AGENT_COMPLETE, AGENT_SCHEDULE, AGENT_SPAWN,
                       COMMAND_CALL, COMMAND_LIST, CONFIG_READ, CONFIG_WRITE,
                       CONV_APPEND, CONV_CREATE, CONV_DELETE, CONV_LIST,
                       CONV_READ, CONV_SET_CATEGORY, CONV_SET_TITLE,
                       CRON_CREATE, CRON_ENABLE, CRON_GET, CRON_LIST,
                       CRON_REMOVE, CRON_UPDATE, DB_DEFINE, DB_QUERY, DB_WRITE,
                       ENV_READ, EVENT_EMIT, EVENT_REQUEST, FILE_LIST,
                       FILE_REGISTER, FS_DELETE, FS_LIST, FS_MOVE, FS_READ,
                       FS_SEARCH, FS_TEMP, FS_WRITE, LEDGER_READ,
                       LEDGER_RECORD, NET_HTTP, PARSE_FILE, PARSE_MODALITY,
                       PLUGIN_DESCRIBE, PLUGIN_LIST, PROC_RUN,
                       SECRET_REVEAL, SELF_RESPOND,
                       SERVICE_CALL, SERVICE_LIST, SESSION_ADD_PROMPT,
                       SESSION_ADD_TOOL, SESSION_CANCEL, SESSION_GET,
                       SESSION_LIST, SESSION_PUSH, SESSION_REMOVE_PROMPT,
                       SESSION_REMOVE_TOOL, SESSION_STATE_GET,
                       SESSION_STATE_SET, TASK_ENQUEUE, TASK_OUTPUT,
                       TASK_STATUS, TOOL_CALL, TOOL_LIST, UI_APPROVE, UI_ASK,
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

    def list(self, path, pattern: str = "*"):
        """List a directory, optionally filtered by glob pattern."""
        return self._ask(FS_LIST, path=str(path), pattern=pattern)

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

    def create(self, title: str = ""):
        """Start a conversation."""
        return self._ask(CONV_CREATE, title=title)

    def read(self, conversation_id):
        """Messages and metadata."""
        return self._ask(CONV_READ, id=conversation_id)

    def list(self):
        """Conversations belonging to the current user."""
        return self._ask(CONV_LIST)

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

    def delete(self, conversation_id):
        """Delete a conversation and its messages."""
        return self._ask(CONV_DELETE, id=conversation_id)


class _Session(_Namespace):
    """Session Requests. Widening is unsafe, narrowing is safe."""

    def get(self, key: str = ""):
        """Describe a session; defaults to this one."""
        return self._ask(SESSION_GET, key=key)

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

    def read(self, key: str = ""):
        """Read a setting, or all of them."""
        return self._ask(CONFIG_READ, key=key or None)

    def write(self, key: str, value):
        """Change a setting."""
        return self._ask(CONFIG_WRITE, key=key, value=value)


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

    def list(self):
        """Everything registered, by family."""
        return self._ask(PLUGIN_LIST)

    def describe(self, name: str):
        """Metadata for one plugin."""
        return self._ask(PLUGIN_DESCRIBE, name=name)


class _Services(_Namespace):
    """Calling into loaded services."""

    def list(self):
        """Loaded services and whether each is ready."""
        return self._ask(SERVICE_LIST)

    def call(self, name: str, method: str, **kwargs):
        """Invoke an exported method. Simple data comes back, never objects."""
        return self._ask(SERVICE_CALL, name=name, method=method,
                         kwargs=kwargs)


class _Tools(_Namespace):
    """Calling other tools."""

    def list(self):
        """Tools the current scope exposes."""
        return self._ask(TOOL_LIST)

    def call(self, name: str, **kwargs):
        """Call another tool."""
        return self._ask(TOOL_CALL, name=name, kwargs=kwargs)


class _Commands(_Namespace):
    """Running slash commands."""

    def list(self):
        """Registered slash commands."""
        return self._ask(COMMAND_LIST)

    def run(self, name: str, **args):
        """Run a slash command in one shot."""
        return self._ask(COMMAND_CALL, name=name, args=args)


class _Agent(_Namespace):
    """The model, and other agents."""

    def complete(self, prompt: str = "", messages=None):
        """A model call. Keys and sockets stay kernel-side."""
        return self._ask(AGENT_COMPLETE, prompt=prompt,
                         messages=list(messages or []))

    def spawn(self, prompt: str, wait: bool = True):
        """Run a subagent now."""
        return self._ask(AGENT_SPAWN, prompt=prompt, wait=wait)

    def schedule(self, prompt: str, cron: str):
        """Run a subagent later. Unattended, so always checked."""
        return self._ask(AGENT_SCHEDULE, prompt=prompt, cron=cron)


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


class _Markdown:
    """Presentation helpers, mirroring the kernel's markdown-on-the-wire
    convention so sandboxed output renders identically."""

    @staticmethod
    def table(headers, rows) -> str:
        """Render a GitHub-flavored markdown table."""
        head = "| " + " | ".join(str(h) for h in headers) + " |"
        rule = "| " + " | ".join("---" for _ in headers) + " |"
        body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
        # Leading blank line: GFM folds a table into the preceding paragraph
        # without one.
        return "\n" + "\n".join([head, rule, *body])

    @staticmethod
    def card(title: str, pairs) -> str:
        """Render a detail card as a two-column table."""
        return f"**{title}**\n" + _Markdown.table(
            ["Field", "Value"], [[k, v] for k, v in pairs])


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
        self.users = _Users(self)
        self.plugins = _Plugins(self)
        self.services = _Services(self)
        self.tools = _Tools(self)
        self.commands = _Commands(self)
        self.agent = _Agent(self)
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

    # ── the channel ────────────────────────────────────────────────

    def _send(self, request: Request):
        """Send a Request and block until the kernel answers."""
        return self._channel.send(request)

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
