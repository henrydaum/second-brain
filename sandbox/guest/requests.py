"""The Request vocabulary — what sandboxed code may ask the kernel to do.

A Request is an inert, serializable description of a desired effect. It is
never a callable and never holds a live object, so the same value crosses a
thread boundary (in-process runner) or a pipe (subprocess runner) unchanged.

Everything that touches disk, network, clock, or process is a Request.
Everything else belongs in the SDK and never reaches the kernel.

The catalogue below is the complete list of what any plugin can ever do, and
is documented family by family in ``SECURITY_CONTRACT_APPENDIX.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── filesystem ────────────────────────────────────────────────────────
FS_READ = "fs.read"
FS_WRITE = "fs.write"
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

# ── users ─────────────────────────────────────────────────────────────
USER_READ = "user.read"
USER_LIST = "user.list"
USER_WRITE = "user.write"

# ── plugins ───────────────────────────────────────────────────────────
PLUGIN_LIST = "plugin.list"
PLUGIN_DESCRIBE = "plugin.describe"
PLUGIN_REGISTER = "plugin.register"
PLUGIN_UNREGISTER = "plugin.unregister"
PLUGIN_RELOAD = "plugin.reload"
PLUGIN_INSTALL = "plugin.install"
PLUGIN_UNINSTALL = "plugin.uninstall"

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

# ── pipeline ──────────────────────────────────────────────────────────
TASK_ENQUEUE = "task.enqueue"
TASK_STATUS = "task.status"
TASK_OUTPUT = "task.output"
FILE_REGISTER = "file.register"
FILE_LIST = "file.list"

# ── parsing, ledger, network, process, self ───────────────────────────
PARSE_FILE = "parse.file"
PARSE_MODALITY = "parse.modality"
LEDGER_RECORD = "ledger.record"
LEDGER_READ = "ledger.read"
NET_HTTP = "net.http"
PROC_RUN = "proc.run"
ENV_READ = "env.read"
SELF_RESPOND = "self.respond"


ALL_TYPES = {
    FS_READ, FS_WRITE, FS_LIST, FS_SEARCH, FS_DELETE, FS_MOVE, FS_TEMP,
    DB_QUERY, DB_WRITE, DB_DEFINE,
    CONV_CREATE, CONV_READ, CONV_LIST, CONV_APPEND, CONV_SET_TITLE,
    CONV_SET_CATEGORY, CONV_DELETE,
    SESSION_GET, SESSION_LIST, SESSION_PUSH, SESSION_STATE_GET,
    SESSION_STATE_SET, SESSION_CANCEL, SESSION_ADD_TOOL, SESSION_REMOVE_TOOL,
    SESSION_ADD_PROMPT, SESSION_REMOVE_PROMPT,
    UI_ASK, UI_APPROVE, UI_RENDER,
    CONFIG_READ, CONFIG_WRITE,
    USER_READ, USER_LIST, USER_WRITE,
    PLUGIN_LIST, PLUGIN_DESCRIBE, PLUGIN_REGISTER, PLUGIN_UNREGISTER,
    PLUGIN_RELOAD, PLUGIN_INSTALL, PLUGIN_UNINSTALL,
    SERVICE_LIST, SERVICE_CALL, SERVICE_LOAD, SERVICE_UNLOAD,
    TOOL_LIST, TOOL_CALL, COMMAND_LIST, COMMAND_CALL,
    AGENT_COMPLETE, AGENT_SPAWN, AGENT_SCHEDULE,
    CRON_LIST, CRON_GET, CRON_CREATE, CRON_UPDATE, CRON_REMOVE, CRON_ENABLE,
    EVENT_EMIT, EVENT_REQUEST,
    TASK_ENQUEUE, TASK_STATUS, TASK_OUTPUT, FILE_REGISTER, FILE_LIST,
    PARSE_FILE, PARSE_MODALITY, LEDGER_RECORD, LEDGER_READ,
    NET_HTTP, PROC_RUN, ENV_READ, SELF_RESPOND,
}

# Requests that read rather than change. The policy function leans on this,
# and so does anything asking whether a chain has done anything yet.
READ_ONLY = {
    FS_READ, FS_LIST, FS_SEARCH, DB_QUERY, CONV_READ, CONV_LIST, SESSION_GET,
    SESSION_LIST, SESSION_STATE_GET, CONFIG_READ, USER_READ, USER_LIST,
    PLUGIN_LIST, PLUGIN_DESCRIBE, SERVICE_LIST, TOOL_LIST, COMMAND_LIST,
    CRON_LIST, CRON_GET, TASK_STATUS, TASK_OUTPUT, FILE_LIST, PARSE_FILE,
    PARSE_MODALITY, LEDGER_READ, ENV_READ,
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

DENIED = "denied"


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

    def __bool__(self) -> bool:
        return self.ok

    @property
    def denied(self) -> bool:
        """Whether this failure was a refusal rather than a breakage."""
        return not self.ok and self.error.startswith(DENIED)

    @staticmethod
    def failure(error: str, retryable: bool = False) -> "Result":
        """Build a failure report."""
        return Result(ok=False, error=error, retryable=retryable)

    @staticmethod
    def refusal(reason: str = "") -> "Result":
        """Build a denial — a failure whose cause is policy, not breakage."""
        detail = f"{DENIED}: {reason}" if reason else DENIED
        return Result(ok=False, error=detail, retryable=False)

    def to_dict(self) -> dict:
        """Serialize for the subprocess runner."""
        return {"ok": self.ok, "data": self.data,
                "error": self.error, "retryable": self.retryable}

    @staticmethod
    def from_dict(raw: dict) -> "Result":
        """Rebuild from the wire."""
        return Result(ok=raw["ok"], data=raw.get("data"),
                      error=raw.get("error", ""),
                      retryable=raw.get("retryable", False))
