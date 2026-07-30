"""The Request vocabulary — what sandboxed code may ask the kernel to do.

A Request is an inert, serializable description of a desired effect. It is
never a callable and never holds a live object, so the same value crosses a
thread boundary (in-process runner) or a pipe (subprocess runner) unchanged.

Everything that touches disk, network, clock, or process is a Request.
Everything else belongs in the SDK and never reaches the kernel.

The catalogue below is the complete list of what any plugin can ever do, and
is documented family by family in ``docs/SECURITY_CONTRACT_APPENDIX.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── filesystem ────────────────────────────────────────────────────────
FS_READ = "fs.read"
FS_WRITE = "fs.write"
# Bytes cross as base64 — JSON has no bytes type, and the same value has to
# survive both a thread boundary and a pipe unchanged.
FS_READ_BYTES = "fs.read_bytes"
FS_WRITE_BYTES = "fs.write_bytes"
FS_LIST = "fs.list"
FS_SEARCH = "fs.search"
FS_DELETE = "fs.delete"
FS_MOVE = "fs.move"
FS_TEMP = "fs.temp"

# ── database ──────────────────────────────────────────────────────────
DB_QUERY = "db.query"
DB_WRITE = "db.write"
DB_DEFINE = "db.define"

# ── conversations ─────────────────────────────────────────────────────
CONV_CREATE = "conv.create"
CONV_READ = "conv.read"
CONV_LIST = "conv.list"
CONV_APPEND = "conv.append"
CONV_SET_TITLE = "conv.set_title"
CONV_SET_CATEGORY = "conv.set_category"
CONV_SET_NOTIFICATION_MODE = "conv.set_notification_mode"
CONV_LOAD = "conv.load"
CONV_CLEAR = "conv.clear"
CONV_DELETE = "conv.delete"

# ── sessions ──────────────────────────────────────────────────────────
SESSION_GET = "session.get"
SESSION_LIST = "session.list"
SESSION_PUSH = "session.push"
SESSION_STATE_GET = "session.state_get"
SESSION_STATE_SET = "session.state_set"
SESSION_CANCEL = "session.cancel"
SESSION_ADD_TOOL = "session.add_tool"
SESSION_REMOVE_TOOL = "session.remove_tool"
SESSION_ADD_PROMPT = "session.add_prompt_extra"
SESSION_REMOVE_PROMPT = "session.remove_prompt_extra"

# ── user interaction ──────────────────────────────────────────────────
UI_ASK = "ui.ask"
UI_APPROVE = "ui.approve"
UI_RENDER = "ui.render"

# ── configuration ─────────────────────────────────────────────────────
CONFIG_READ = "config.read"
CONFIG_WRITE = "config.write"
PATH_GET = "paths.get"

# ── users ─────────────────────────────────────────────────────────────
USER_READ = "user.read"
USER_LIST = "user.list"
USER_WRITE = "user.write"

# ── plugins ───────────────────────────────────────────────────────────
PLUGIN_LIST = "plugin.list"
PLUGIN_DESCRIBE = "plugin.describe"
# Lint a source file against the sandbox contract. Read-only in the strongest
# sense — the validator is a pure AST walk and never imports what it reads —
# which is what lets an authoring agent check its own work without a dialog.
PLUGIN_VALIDATE = "plugin.validate"
PLUGIN_REGISTER = "plugin.register"
PLUGIN_UNREGISTER = "plugin.unregister"
PLUGIN_RELOAD = "plugin.reload"
PLUGIN_INSTALL = "plugin.install"
PLUGIN_UNINSTALL = "plugin.uninstall"
PLUGIN_UPDATE = "plugin.update"

# ── services, tools, commands ─────────────────────────────────────────
SERVICE_LIST = "service.list"
SERVICE_CALL = "service.call"
SERVICE_LOAD = "service.load"
SERVICE_UNLOAD = "service.unload"
TOOL_LIST = "tool.list"
TOOL_CALL = "tool.call"
COMMAND_LIST = "command.list"
COMMAND_CALL = "command.call"

# ── agent ─────────────────────────────────────────────────────────────
AGENT_COMPLETE = "agent.complete"
AGENT_SPAWN = "agent.spawn"
AGENT_SCHEDULE = "agent.schedule"

# Joining and ending a subagent already started. The vocabulary grew here for
# the reason ``proc.start`` grew it: a background child outlives the Request
# that made it, so there is a handle to hand back, wait on, and eventually
# cancel — and no return value expresses that. ``spawn(wait=True)`` is still
# one Request answering with one result; these two exist so that
# ``wait=False`` is usable from code that has no agent turn to hold open, which
# is every script.
AGENT_COLLECT = "agent.collect"
AGENT_STOP = "agent.stop"

# Placing the model call an escort was handed. Only meaningful inside a
# ``llm_call`` hook: it is the escort dialing the phone it holds, and the
# kernel resolves it to the very call this hook was invoked for.
LLM_PROCEED = "llm.proceed"
LLM_DELTA = "llm.delta"

# ── scheduling ────────────────────────────────────────────────────────
CRON_LIST = "cron.list"
CRON_GET = "cron.get"
CRON_CREATE = "cron.create"
CRON_UPDATE = "cron.update"
CRON_REMOVE = "cron.remove"
CRON_ENABLE = "cron.enable"

# ── events ────────────────────────────────────────────────────────────
EVENT_EMIT = "event.emit"
EVENT_REQUEST = "event.request"

# ── frontends ─────────────────────────────────────────────────────────
# A frontend is the one family that acts *for a person*: it carries what
# someone typed into the state machine. These are the inbound half of that,
# and they are reachable only from inside a loaded frontend's box — the
# handler resolves that frontend's own adapter through a token, so nothing
# else can submit on a user's behalf or bind a session it does not own.
FRONTEND_SUBMIT = "frontend.submit"
FRONTEND_CANCEL = "frontend.cancel"
FRONTEND_BIND = "frontend.bind"
FRONTEND_ATTEND = "frontend.attend"
FRONTEND_RESOLVE = "frontend.resolve"
# Whether an approval is still waiting. A frontend could remember what it was
# asked to render instead, but that record goes stale the moment the approval
# is answered somewhere else or times out — and a frontend acting on a stale
# one would swallow the next thing a person typed as a yes/no.
FRONTEND_PENDING = "frontend.pending"

# The machine's console. Scoped like the rest of the family — the kernel reads
# stdin on its own thread and the guest drains what arrived, so nothing blocks
# a box and a child process never opens stdin at all.
CONSOLE_READ = "console.read"
CONSOLE_WRITE = "console.write"

# ── pipeline ──────────────────────────────────────────────────────────
TASK_ENQUEUE = "task.enqueue"
TASK_STATUS = "task.status"
TASK_OUTPUT = "task.output"
TASK_LIST = "task.list"
TASK_GRAPH = "task.graph"
TASK_PAUSE = "task.pause"
TASK_RESET = "task.reset"
TASK_TRIGGER = "task.trigger"
FILE_REGISTER = "file.register"
FILE_LIST = "file.list"

# ── parsing, ledger, network, process, self ───────────────────────────
PARSE_FILE = "parse.file"
PARSE_MODALITY = "parse.modality"
LEDGER_RECORD = "ledger.record"
LEDGER_READ = "ledger.read"
NET_HTTP = "net.http"
PROC_RUN = "proc.run"

# Running a command and *keeping* it is a different act from running one to
# completion, and it is the one case where growing the vocabulary is right
# rather than a last resort: a live process outlives the Request that made it,
# so there is a handle to hand back, poll, and eventually kill — none of which
# a return value can express. A dev server is the motivating case, and the
# agent has to be able to clean one up without a dialog or it will not start
# one at all.
PROC_START = "proc.start"
PROC_STATUS = "proc.status"
PROC_STOP = "proc.stop"
PROC_LIST = "proc.list"

# Running a file of SDK code that is not a plugin. The vocabulary grew here
# rather than an argument because there was nothing to grow: a script is not a
# tool (nothing registers it), not a command (nobody typed it) and not a
# process (``proc.run`` starts an OS process outside the boundary entirely).
#
# It is the counterpart to free authorship. The agent may already write
# anything it likes under ``sandbox_plugins`` because everything there is
# contained before it runs; this is how it *runs* one without the file having
# to become a registered capability first. Every effect the script performs is
# an ordinary Request classified with the caller still in the chain, so routing
# work through a script launders nothing — which is what lets this be safe
# while ``proc.run`` never can be.
SCRIPT_RUN = "script.run"

ENV_READ = "env.read"
SECRET_REVEAL = "secret.reveal"
SELF_RESPOND = "self.respond"

# Ending the process. One type with a ``restart`` argument rather than two,
# because stopping and stopping-then-starting are the same act with a different
# tail — and growing an argument is cheaper than growing the vocabulary.
APP_STOP = "app.stop"


ALL_TYPES = {
    FS_READ, FS_WRITE, FS_READ_BYTES, FS_WRITE_BYTES,
    FS_LIST, FS_SEARCH, FS_DELETE, FS_MOVE, FS_TEMP,
    DB_QUERY, DB_WRITE, DB_DEFINE,
    CONV_CREATE, CONV_READ, CONV_LIST, CONV_APPEND, CONV_SET_TITLE,
    CONV_SET_CATEGORY, CONV_SET_NOTIFICATION_MODE, CONV_LOAD,
    CONV_CLEAR, CONV_DELETE,
    SESSION_GET, SESSION_LIST, SESSION_PUSH, SESSION_STATE_GET,
    SESSION_STATE_SET, SESSION_CANCEL, SESSION_ADD_TOOL, SESSION_REMOVE_TOOL,
    SESSION_ADD_PROMPT, SESSION_REMOVE_PROMPT,
    UI_ASK, UI_APPROVE, UI_RENDER,
    CONFIG_READ, CONFIG_WRITE, PATH_GET,
    USER_READ, USER_LIST, USER_WRITE,
    PLUGIN_LIST, PLUGIN_DESCRIBE, PLUGIN_VALIDATE,
    PLUGIN_REGISTER, PLUGIN_UNREGISTER,
    PLUGIN_RELOAD, PLUGIN_INSTALL, PLUGIN_UNINSTALL, PLUGIN_UPDATE,
    SERVICE_LIST, SERVICE_CALL, SERVICE_LOAD, SERVICE_UNLOAD,
    TOOL_LIST, TOOL_CALL, COMMAND_LIST, COMMAND_CALL,
    AGENT_COMPLETE, AGENT_SPAWN, AGENT_SCHEDULE, AGENT_COLLECT, AGENT_STOP,
    LLM_PROCEED, LLM_DELTA,
    CRON_LIST, CRON_GET, CRON_CREATE, CRON_UPDATE, CRON_REMOVE, CRON_ENABLE,
    EVENT_EMIT, EVENT_REQUEST,
    FRONTEND_SUBMIT, FRONTEND_CANCEL, FRONTEND_BIND, FRONTEND_ATTEND,
    FRONTEND_RESOLVE, FRONTEND_PENDING, CONSOLE_READ, CONSOLE_WRITE,
    TASK_ENQUEUE, TASK_STATUS, TASK_OUTPUT, TASK_LIST, TASK_GRAPH,
    TASK_PAUSE, TASK_RESET, TASK_TRIGGER, FILE_REGISTER, FILE_LIST,
    PARSE_FILE, PARSE_MODALITY, LEDGER_RECORD, LEDGER_READ,
    NET_HTTP, PROC_RUN, PROC_START, PROC_STATUS, PROC_STOP, PROC_LIST,
    SCRIPT_RUN,
    ENV_READ, SECRET_REVEAL, SELF_RESPOND,
    APP_STOP,
}

# Requests that read rather than change. The policy function leans on this,
# and so does anything asking whether a chain has done anything yet.
READ_ONLY = {
    FS_READ, FS_READ_BYTES, FS_LIST, FS_SEARCH,
    DB_QUERY, CONV_READ, CONV_LIST, SESSION_GET,
    SESSION_LIST, SESSION_STATE_GET, CONFIG_READ, PATH_GET, USER_READ, USER_LIST,
    PLUGIN_LIST, PLUGIN_DESCRIBE, PLUGIN_VALIDATE,
    SERVICE_LIST, TOOL_LIST, COMMAND_LIST,
    CRON_LIST, CRON_GET, TASK_STATUS, TASK_OUTPUT, TASK_LIST, TASK_GRAPH,
    FILE_LIST, PARSE_FILE,
    PARSE_MODALITY, LEDGER_READ, ENV_READ, CONSOLE_READ, FRONTEND_PENDING,
    PROC_STATUS, PROC_LIST,
    # Taking a finished child's report changes nothing about the world; the
    # child already did whatever it was going to do. Listed here mainly so the
    # ledger's sandbox sink drops it: ``collect(timeout=0)`` is a poll, and a
    # fan-out loop would otherwise write a row per tick forever — the same
    # problem ``console.read`` is in this set for.
    AGENT_COLLECT,
}


@dataclass(frozen=True)
class Request:
    """One asked-for effect.

    type:
        A member of :data:`ALL_TYPES`.
    args:
        Plain data only — no callables, no handles, no live objects.
    """
    type: str
    args: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in ALL_TYPES:
            raise ValueError(f"unknown request type: {self.type}")

    @property
    def family(self) -> str:
        """The resource family this Request belongs to."""
        return self.type.split(".")[0]

    @property
    def read_only(self) -> bool:
        """Whether this Request only reads."""
        return self.type in READ_ONLY

    def to_dict(self) -> dict:
        """Serialize for the subprocess runner."""
        return {"type": self.type, "args": dict(self.args)}

    @staticmethod
    def from_dict(raw: dict) -> "Request":
        """Rebuild from the wire."""
        return Request(type=raw["type"], args=dict(raw.get("args") or {}))


# ──────────────────────────────────────────────────────────────────────
# The return contract.
# ──────────────────────────────────────────────────────────────────────

from .codes import DENIAL_CODES, ERROR_DENIED  # noqa: E402

#: The word a refusal's message opens with. Cosmetic now that :attr:`Result.
#: code` carries the signal, but kept so a person or the model reading the
#: sentence still sees it. One definition, shared with the code vocabulary.
DENIED = ERROR_DENIED


class RequestFailed(Exception):
    """A Request did not succeed.

    Python's answer to "an operation that can fail" is an exception, so that
    is what the SDK gives plugin authors. Handling a failure is a ``try``;
    ignoring one lets the runner turn it into a failed result with the reason
    intact. Neither costs a line of ceremony.

    The :class:`Result` is still there on ``.result`` for anything that wants
    the whole shape.
    """

    def __init__(self, result: "Result", request_type: str = ""):
        self.result = result
        self.request_type = request_type
        super().__init__(f"{request_type or 'request'}: {result.error}"
                         if request_type else result.error)

    @property
    def error(self) -> str:
        """Why it failed."""
        return self.result.error

    @property
    def retryable(self) -> bool:
        """Whether trying again could plausibly work."""
        return self.result.retryable


class Denied(RequestFailed):
    """The kernel refused a Request — policy, not breakage.

    Separate from :class:`RequestFailed` so a plugin can react to "the user
    said no" without also swallowing "the disk is full"::

        try:
            page = sdk.net.http(url)
        except sdk.Denied:
            return "I need permission to fetch that."
    """


@dataclass(frozen=True)
class Result:
    """What a Request returns.

    Three outcomes share one shape, so plugin code has exactly one error path
    to learn. A denial is an ordinary failure — never an exception, never a
    kill — because code that treats denial as fatal is the most likely thing a
    careless author writes.

    Truthy on success, so the common check reads naturally::

        r = sdk.fs.read(path)
        if not r:
            return sdk.fail(r.error)
        text = r.data
    """
    ok: bool = True
    data: Any = None
    error: str = ""
    retryable: bool = False
    #: Why it failed, for code rather than for a person — one of the
    #: ``ERROR_*`` names in :mod:`guest.codes`, or empty. Empty is the norm and
    #: is not a bug: most failures are only ever read. See that module.
    code: str = ""

    # ── how a result reaches the agent and the frontend ────────────
    # ``data`` is for the caller; these carry the rest of what a plugin
    # produces, and dropping them would quietly lose behaviour the kernel
    # already depends on.
    #
    # Tools:
    llm_summary: str = ""            # what the *model* is told, when the raw
                                     # data is the wrong thing to show it
    attachment_paths: list = field(default_factory=list)  # files for the user
    #
    # Tasks:
    also_contains: list = field(default_factory=list)     # nested content
                                                          # found while parsing
    discovered_paths: list = field(default_factory=list)  # new files to
                                                          # register

    def __bool__(self) -> bool:
        return self.ok

    @property
    def denied(self) -> bool:
        """Whether this failure was a refusal rather than a breakage.

        Reads :attr:`code`, with the old message-prefix check kept as a
        fallback for a Result built by a peer that predates the field — a
        subprocess runner from an older build, or one rebuilt from a stored
        payload. A coded Result never consults the prefix.
        """
        if not self.ok and self.code:
            return self.code in DENIAL_CODES
        return not self.ok and self.error.startswith(DENIED)

    @staticmethod
    def failure(error: str, retryable: bool = False,
                code: str = "") -> "Result":
        """Build a failure report.

        ``code`` is optional and usually stays empty — see ``guest/codes.py``
        for when one is worth adding.
        """
        return Result(ok=False, error=error, retryable=retryable, code=code)

    @staticmethod
    def refusal(reason: str = "", code: str = ERROR_DENIED) -> "Result":
        """Build a denial — a failure whose cause is policy, not breakage.

        ``code`` defaults to the generic refusal, so every existing caller
        classifies correctly without being touched. Pass a narrower one from
        ``DENIAL_CODES`` where the reason is known.
        """
        detail = f"{DENIED}: {reason}" if reason else DENIED
        return Result(ok=False, error=detail, retryable=False, code=code)

    def to_dict(self) -> dict:
        """Serialize for the subprocess runner."""
        return {"ok": self.ok, "data": self.data,
                "error": self.error, "retryable": self.retryable,
                "code": self.code,
                "llm_summary": self.llm_summary,
                "attachment_paths": list(self.attachment_paths),
                "also_contains": list(self.also_contains),
                "discovered_paths": list(self.discovered_paths)}

    @staticmethod
    def from_dict(raw: dict) -> "Result":
        """Rebuild from the wire."""
        return Result(ok=raw["ok"], data=raw.get("data"),
                      error=raw.get("error", ""),
                      retryable=raw.get("retryable", False),
                      code=raw.get("code", ""),
                      llm_summary=raw.get("llm_summary", ""),
                      attachment_paths=list(raw.get("attachment_paths") or []),
                      also_contains=list(raw.get("also_contains") or []),
                      discovered_paths=list(raw.get("discovered_paths") or []))
