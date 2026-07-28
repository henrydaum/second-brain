# The Second Brain SDK

*How to write code that runs inside the sandbox.*

---

## The model, in one paragraph

Your code cannot act. It can only **ask**. Anything that touches disk, network,
clock, or process is a *Request* you make through `sdk`; the kernel decides
whether to allow it, does the work, and hands back the answer. Everything else
— arithmetic, string handling, your own logic — runs normally and costs
nothing. You are not writing async code and there are no callbacks: a Request
looks and behaves like an ordinary blocking function call.

---

## Writing something that runs

### A script

No base class, no declarations. A file with functions that take `sdk`:

```python
"""Summarize a file."""


def summarize(sdk, path):
    """Count the lines and words in a file."""
    lines = sdk.fs.read(path).splitlines()
    return {"lines": len(lines), "words": sum(len(l.split()) for l in lines)}
```

That is a complete, runnable sandbox program. Use this for one-off computation
and scratch work.

### A plugin

Subclass a base when the *kernel* has to register and schedule the thing.
The filename must carry the family prefix — `tool_*.py`, `task_*.py`,
`service_*.py`, `command_*.py`, `frontend_*.py` — because discovery finds
plugins by filename.

```python
"""Count the words in a file."""

from guest.bases import BaseTool


class WordCount(BaseTool):
    """Count the words in a text file."""

    name = "word_count"
    description = "Count the words in a text file."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def run(self, sdk, path):
        """Read the file and count."""
        return len(sdk.fs.read(path).split())
```

Entry points by family — note the **argument order differs**:

| Family | Entry point |
|---|---|
| tool | `run(self, sdk, **kwargs)` |
| task | `run(self, sdk, paths)` |
| command | `run(self, sdk, args)` and optionally `form(self, sdk, args)` |
| service | `start(self, sdk)`, `stop(self, sdk)`, plus its exported methods |

---

## The idiom

**A Request returns its value and raises if it fails.** No result object, no
branch that exists only to forward an error:

```python
text = sdk.fs.read(path)          # a str
rows = sdk.db.query("SELECT 1")   # a list of dicts
sdk.fs.write(path, text)          # just do it
```

**Return whatever you like.** The runner wraps it:

```python
return {"words": 12}       # fine
return "some markdown"     # fine
return None                # fine
```

Reach for `sdk.ok(...)` only when you need to attach something extra:

```python
return sdk.ok(rows, llm_summary="12 matching rows")   # what the model is told
return sdk.ok(data, attachments=["/tmp/chart.png"])   # files for the user
return sdk.fail("no such document")                   # fail without raising
```

**Catch a failure only when you can do something about it.** An uncaught one
becomes your plugin's failure, carrying the original reason — usually exactly
what you wanted:

```python
try:
    page = sdk.net.http(url)
except sdk.Denied:
    return "I need permission to fetch that."
```

`sdk.Denied` (the user or policy said no) is a subclass of `sdk.Failed`
(anything went wrong), so catching `sdk.Failed` catches both.

**Log through the SDK**, never the `logging` module — a subprocessed plugin's
log lines have to reach the kernel to be seen at all:

```python
sdk.log("starting the sweep")
sdk.log("could not reach the index", level="warning")
```

---

## The Request reference

Each namespace is exactly one Request family, so `sdk.fs.read` *is* the
`fs.read` Request.

### Files and processes

```python
sdk.fs.read(path)                          # -> str
sdk.fs.write(path, data, mode="overwrite") # mode="append" to add
sdk.fs.read_bytes(path)                    # -> bytes; use for anything non-text
sdk.fs.write_bytes(path, data, mode="overwrite")
sdk.fs.list(path, pattern="*")             # -> [str]
sdk.fs.list(path, details=True)            # -> [{path, name, is_dir, size, mtime}]
                                           # point it at a *file* for just that
                                           # one entry — this is how you ask
                                           # "does it exist / has it changed?"
                                           # mtime is st_mtime_ns; compare with !=
sdk.fs.search(pattern, root=".", glob="**/*")   # -> [{path, line, text}]
sdk.fs.delete(path)
sdk.fs.move(src, dst, copy=False)
sdk.fs.temp(directory=False, suffix="")    # scratch space; always allowed

sdk.net.http(url, method="GET", headers=None, body=None)  # -> {status, body}
sdk.proc.run(argv, timeout=120.0, cwd=None)               # -> {code, stdout, stderr}
sdk.env.read(name)                         # credentials come back as handles
sdk.secrets.reveal(name)                   # plaintext; always asks the user
```

`read` decodes UTF-8 with replacement, which quietly mangles anything that is
not text. Reach for `read_bytes` whenever the file is an image, audio, a PDF,
or an archive. Base64 on the wire is the SDK's problem, not yours — you hand
over `bytes` and get `bytes` back.

### Data

```python
sdk.db.query(sql, params)      # -> [dict]
sdk.db.write(sql, params)
sdk.db.define(ddl)             # create a table your plugin owns

sdk.conv.create(title, category=None, activate=False)
sdk.conv.read(conversation_id, details=False)
sdk.conv.list(category=None, limit=50, details=False)
sdk.conv.append(conversation_id, role, content)
sdk.conv.set_title(conversation_id, title)
sdk.conv.set_category(conversation_id, category)
sdk.conv.set_notification_mode(conversation_id, mode)
sdk.conv.load(conversation_id)
sdk.conv.clear(conversation_id=None) # defaults to the active conversation
sdk.conv.delete(conversation_id)

sdk.config.read(key)           # omit key for everything
sdk.config.read(details=True)  # visible, redacted setting descriptors
sdk.config.write(key, value)
sdk.paths.get(name)            # project, data, installed_plugins, sandbox_plugins
sdk.users.read(user_id=None)   # defaults to the current user
sdk.users.list()
sdk.users.write(user_id=None, **fields)
```

Secret-prefixed fields are proxied recursively, including fields inside
structured settings such as profiles. A returned handle can be written back
through `sdk.config.write`; the kernel restores its original value without
revealing it to the plugin.

**Reading rows of user-owned tables uses the `my_` name**, which the kernel
expands to the current user. Reading the base table is refused:

```python
sdk.db.query("SELECT * FROM my_conversations WHERE title LIKE ?", ["%tax%"])
```

### People and sessions

```python
sdk.ui.ask(prompt, title="Question", type="text", choices=None)
sdk.ui.approve(action, justification)
sdk.ui.render(paths, caption="")     # show files in the chat

sdk.session.get(key="")              # defaults to this session
sdk.session.list()
sdk.session.push(message, key="")    # message the user out of band
sdk.session.state_get(namespace="sandbox")
sdk.session.state_set(value, namespace="sandbox")
sdk.session.cancel(key="")
sdk.session.add_tool(tool) / remove_tool(tool)
sdk.session.add_prompt(text) / remove_prompt(handle)
```

### Other code

```python
from guest.forms import FormStep

sdk.tools.list()
sdk.tools.call(name, **kwargs)
sdk.commands.list(details=False, visible=False)
sdk.commands.run(name, **args)
sdk.services.list(details=False)
sdk.services.call(name, method, **kwargs)   # only exported methods
sdk.services.load(name) / unload(name)
sdk.plugins.list(source="registered", category="")
sdk.plugins.describe(name)
sdk.plugins.register(path)
sdk.plugins.unregister(path=...)          # or name=..., family=...
sdk.plugins.reload(path=...)              # or name=..., family=...
sdk.plugins.install(package_id)
sdk.plugins.uninstall(package_id)
sdk.plugins.update()

sdk.agent.complete(prompt)           # a model call
sdk.agent.spawn(prompt, wait=True)   # a subagent now
sdk.agent.schedule(prompt, cron)     # a subagent later
```

Plugin lifecycle mutations are approval-gated. Paths must resolve to a
recognized built-in, sandbox, or installed plugin file. A name-only unload or
reload must identify exactly one registered plugin; supply `family` when the
same name exists in more than one registry.

### Standing at a doorway

A **hook** is the one inbound thing here: the kernel calls *you*, once per turn,
at a labeled moment. Declare it on a service — there is nothing to register and
therefore nothing to leak:

```python
class Doorman(BaseService):
    name = "doorman"
    hooks = {"end_turn": "check_done"}

    def check_done(self, sdk, ctx, ending):
        if ending.reason == "budget_exhausted":
            return SendBack("Summarize what you found.", ephemeral=True)
        return None                      # abstain
```

Six moments: `turn_start`, `shape_scope`, `vet_permission`, `llm_call`,
`end_turn`, `turn_finish`. Every one is `method(self, sdk, ctx, payload)`, and
returning `None` abstains. Payloads and verdicts live in `guest.hooks`.

The `llm_call` escort holds the phone as well as the request:

```python
def escort(self, sdk, ctx, request):
    response = sdk.llm.proceed(request)     # place the call
    if not response.content.strip():
        request.messages += [{"role": "user", "content": "Answer."}]
        response = sdk.llm.proceed(request)  # go around again
    return response
```

`request.llm` is the backend's **name** — assign another loaded one to swap
brains for that call. `sdk.llm.proceed` works only inside an `llm_call`
hook; anywhere else there is no call in flight and it is refused.

A scope shaper is handed tool **names** and returns the ones to keep: it can
hide and reorder but never synthesize, so adding a tool is
`sdk.session.add_tool`. And a hook that raises, or whose service is unloaded,
simply abstains — it can never break a turn.

Hooks run synchronously on the drive thread, so they are paid on every turn
they touch. Keep them fast.

### Listening to the bus

`sdk.events.emit` is the outbound half and needs no declaration. Hearing back
does — and like a hook, it is declared rather than registered, so there is no
subscription to forget to drop:

```python
class Watcher(BaseService):
    name = "watcher"
    subscribed_channels = ["task_completed", "session_turn_completed"]

    def on_event(self, sdk, channel, payload):
        if channel == "task_completed":
            sdk.log(f"{payload['task_name']} finished")
```

One `on_event` receives every declared channel; a channel you did not declare
is never delivered. **Only services and frontends can subscribe** — a tool is a
call that ends, so there would be nothing to deliver to, and the validator says
so rather than letting the declaration sit there doing nothing.

Channel names are not a closed vocabulary. The kernel's are in
`events/event_channels.py`, but a plugin owns its own channels and you may
listen to another plugin's, so nothing validates the string — a typo is
silence, not an error.

### Polling a resident plugin

Services and frontends may ask the kernel to call a short `poll(self, sdk)`
method repeatedly:

```python
class Clock(BaseService):
    name = "clock"
    poll_interval = 1.0
    max_poll_failures = 5

    def poll(self, sdk):
        did_work = self._fire_due_jobs(sdk)
        return did_work
```

The kernel owns the thread, cadence, shutdown, and failure limit. A truthy
return means work remains and requests another call immediately; a falsy
return waits `poll_interval`. Calls are serialized through the resident box,
so exported methods, events, renders, and polls never mutate guest state at
the same time. Polling is disabled for a service unless it declares a positive
interval; frontends default to 0.05 seconds. It is invalid on commands, tools,
and tasks.

### Being a frontend

A frontend is inbound-driven, and that inverts the usual shape twice.

**There is no main loop.** `start` opens the transport and *returns*; the
kernel then calls `poll` over and over on a thread it owns. Blocking in `start`
would hold the box — a box takes one call at a time — and no `render` would
ever get in, so the frontend would go deaf the moment it started listening.

```python
class Chat(BaseFrontend):
    name = "chat"
    poll_interval = 0.05          # paid only when a poll finds nothing

    def start(self, sdk):
        self._cursor = 0
        return True

    def poll(self, sdk):
        updates = sdk.net.http(f"https://api.example.com/updates?after={self._cursor}")
        for update in updates["body"]["items"]:
            self._cursor = update["id"]
            sdk.frontend.submit_text(f"chat:{update['room']}", update["text"])
        return bool(updates["body"]["items"])     # truthy = call me straight back

    def render(self, sdk, session_key, kind, payload):
        if kind == "messages":
            for text in payload:
                sdk.net.http("https://api.example.com/send", method="POST",
                             json={"room": session_key, "text": text})
```

`poll` must return promptly — between polls is the only moment the kernel can
call `render`, so a slow poll is a frozen display. A long-poll with a short
server-side timeout is the right shape; an unbounded wait is not.

If handing input to the runtime can synchronously produce output, declare
`background_submit = True`. The host then schedules `sdk.frontend.submit_*`
off the `poll` call so output can render into the serialized frontend box.
For a frontend that must reopen its last conversation before accepting input,
declare `restore_on_start = True`; the host restores after `start` returns so
restored forms and approvals cannot re-enter a busy box.

**Showing things is not a Request** — `render` is called *on you*, with a
`kind` saying what: `messages`, `attachments`, `form_field`, `approval`,
`buttons`, `error`, `typing`, `tool_status`, `stream_delta`. Handle what your
transport can show and ignore the rest; a frontend that only renders
`messages` is a working frontend.

Carrying what a person *does* back the other way is:

```python
sdk.frontend.submit_text(session_key, text)
sdk.frontend.submit_attachment(session_key, path, extension="")
sdk.frontend.submit_action(session_key, action_type, payload=None)
sdk.frontend.cancel(session_key)
sdk.frontend.bind(session_key, external_id=None, user_type="user", config=None)
sdk.frontend.attended(session_key, present=True)
sdk.frontend.pending_approval(session_key)      # an id, or None
sdk.frontend.resolve(session_key, value, request_id="")
```

These work **only inside a loaded frontend**. Each resolves to your own
frontend's adapter through a handle the kernel parks when your box opens, so
you cannot submit on another frontend's behalf and a tool that imported the
same namespace reaches nothing at all.

An `approval` render carries an `id`; answer it with `sdk.frontend.resolve`.
Holding the id is enough to answer and *only* enough to answer — the action
being authorized never crosses.

**Ask what is pending; do not remember it.** A transport where a person answers
by typing "yes" has to know whether a yes/no is what the next line means. You
are told an approval exists — you were handed one to render — but not when it
stops existing: another frontend can answer it, or it can time out. Call
`sdk.frontend.pending_approval(key)` at the moment you need to decide, and
check `sdk.session.get(key)["phase"]` too: when the state machine is already
collecting the answer itself, interpreting the line as well consumes one
keystroke twice.

### The console

A frontend whose transport is *this machine's terminal* declares
`uses_console = True` and reads through the kernel:

```python
class Terminal(BaseFrontend):
    name = "terminal"
    uses_console = True

    def start(self, sdk):
        return True

    def poll(self, sdk):
        line = sdk.console.read_line()     # a line, or None. Never blocks.
        if line is None:
            return False
        sdk.frontend.submit_text("default", line)
        return True

    def render(self, sdk, session_key, kind, payload):
        if kind == "messages":
            for text in payload:
                sdk.console.write(sdk.md.plain(text))
```

**`input()` is refused and always will be**, for three compounding reasons.
It blocks, and a box takes one call at a time — so a frontend blocked on input
holds its own box and cannot render, meaning agent output would appear only
*after* the next thing you typed. A subprocess box's stdin **is** the wire
protocol, so reading it would eat the frames the box talks over. And a rule
that worked in-process and corrupted the protocol under isolation is the worst
kind, because nothing fails until someone sets `isolation`.

Inverting it fixes all three: the kernel reads on its own thread, you drain
what arrived. A console frontend can therefore be subprocess-isolated, which
`input()` could never allow.

`read_line()` returns `None` when nothing has arrived — return falsy from
`poll` and renders land in the pause. It *raises* once the console is closed
and drained; letting that propagate out of `poll` is how a frontend stops
itself at end of input on a piped stdin.

**The console is exclusive.** Two frontends reading one stdin would split a
person's keystrokes between them, which reads as the machine dropping
characters — so the kernel lends it to one claimant and refuses the second.

`sdk.md.plain(text)` renders markdown for a monospace surface: tables become
padded columns and code fences drop away. Pure, no Request.

`bind` is the "whose data is this?" axis, not permissions. With no
`external_id` the session takes your declared `default_user_id`; with one it is
upgraded to that identity's own user, which is what a `per_user` frontend does
on login. Authenticating is your job — the kernel stores what you give it.

Two things are worth knowing about the payload. Handlers run **on the thread
that emitted**, so a slow `on_event` slows down whoever published; do the work
in a task if it is not quick. And a payload only carries what can cross the
boundary — `bus.request`'s synchronous round-trip machinery is stripped, so a
sandboxed subscriber sees it as an ordinary event and cannot answer it.

### Talking to a model

An LLM backend is the one thing here that is **not a plugin**: no family, no
entry point, nothing discovery registers. It is a class in
`helpers/llm_<provider>.py`, found by declaration, loaded into a box, and
called. Copy `templates/llm_backend_template.py` rather than starting blank.

```python
dependencies_pip = ["some-provider-sdk"]
lifetime = "persistent"
supports_streaming = True
supports_tool_choice = True
display_name = "Some Provider"

from guest.llm import BaseLLMBackend, LLMResponse


class SomeProvider(BaseLLMBackend):
    """Reach a model through some-provider-sdk."""

    def start(self, sdk):
        """Import the library once, for this box's whole life."""
        import some_provider_sdk

        self._client = some_provider_sdk
        return True

    def chat(self, sdk, request):
        """Answer one request with one response."""
        answer = self._client.chat(
            model=request.model_name, messages=request.messages,
            tools=request.tools or None, api_key=request.api_key or None,
            **request.params)
        return LLMResponse(content=answer.text, prompt_tokens=answer.tokens)
```

**Everything about the model arrives on the request, nothing lives on you.**
`model_name`, `api_key`, `base_url`, `messages`, `tools`, `params`,
`attachments`. That is what lets the kernel run a *pool* of these boxes for one
model and serve concurrent calls in parallel — two boxes are interchangeable
only if neither remembers who it was talking to. Keep in `start` what is truly
per-process: the imported library, a connection pool.

**Streaming pushes and returns.** When `request.stream` is set, call
`sdk.llm.delta(text)` as text arrives *and* return the accumulated response.
The deltas are for the user's eyes; the response is what gets recorded.

```python
pieces = []
for chunk in self._client.stream(...):
    pieces.append(chunk.text)
    sdk.llm.delta(chunk.text)
return LLMResponse(content="".join(pieces))
```

Notice there is no check for "did the user cancel?". There is nothing to check.
`delta` is one-way and answers nothing; if the user cancels, the kernel cancels
this execution and your next Request raises `Terminated`. Do not wrap a stream
loop in a bare `except Exception` — `Terminated` is a `BaseException` precisely
so that a careless catch cannot swallow it, but a careless `except BaseException`
still can.

**Raise on failure.** `chat` is wrapped: an exception is classified and turned
into an error response for you, and a context-overflow is recognised, which is
what makes the kernel compact the conversation and retry rather than fail the
turn.

**Attachments arrive pre-routed.** The kernel has already split the bundle
against this model's declared capabilities and appended a text fallback for
whatever it cannot read. Everything in `request.attachments` is meant to go on
the wire; read its bytes with `sdk.fs.read_bytes`.

`request.api_key` is plaintext, unlike every other credential in the SDK. A
provider library opens its own socket, so there is no `net.http` for the kernel
to substitute a handle into. If your provider speaks plain HTTP, prefer
`sdk.net.http` and keep the handle.

### Machinery

```python
sdk.cron.list() / get(name) / create(name, job) / update(name, patch)
sdk.cron.remove(name) / enable(name, enabled=True)

sdk.events.emit(channel, payload)
sdk.events.request(channel, payload, timeout=120.0)

sdk.llm.delta(text)      # LLM backends only, inside chat()
sdk.llm.proceed(request) # llm_call escorts only

sdk.tasks.enqueue(name, paths) / status(name, path) / output(name, path=None)
sdk.tasks.list(details=False) / graph()
sdk.tasks.pause(name, paused=True) / reset(name, failed_only=False)
sdk.tasks.trigger(name, payload=None)
sdk.files.register(path, **meta) / list(modality="")

sdk.parse.file(path, modality="text")
sdk.parse.modality(extension)

sdk.ledger.record(action, ok=True, data=None)
sdk.ledger.read(limit=50)
```

---

## Free helpers

These run inside the sandbox. No Request, no approval, no cost:

```python
sdk.text.truncate(text, limit)
sdk.text.cosine(vector_a, vector_b)
sdk.md.table(headers, rows)
sdk.md.card(title, pairs)

sdk.path.join(root, "helpers", "thing.py")
sdk.path.parent(p); sdk.path.name(p); sdk.path.stem(p); sdk.path.suffix(p)
sdk.path.absolute(p, base=sdk.paths.get("project"))
sdk.path.within(p, root)          # containment, separator-aware
sdk.path.normalize(p)             # canonical key for comparing two paths
```

`sdk.path` exists because you cannot import `pathlib` or `os.path` — both
reach the environment — while *manipulating* a path is only string
arithmetic. Two things it will not do, both deliberate:

- **It never consults the current directory.** Inside a box that is
  `sandbox/`, which means nothing to your plugin, so a relative path with no
  `base` stays relative rather than becoming confidently wrong. Pass the base
  you mean — usually `sdk.paths.get("project")`.
- **It never resolves symlinks**, because that is a disk read. Two names for
  one file therefore compare unequal.

Note `sdk.paths` (a Request — asks the kernel where things are) and
`sdk.path` (a helper — arithmetic on a string you already have) are different
namespaces. The plural one crosses the boundary; the singular one does not.

Plus the pure standard library — `json`, `re`, `math`, `datetime`, `time`,
`collections`, `itertools`, `hashlib`, `base64`, `csv`, `email`, `textwrap`,
`statistics`, `dataclasses`, `typing`, and friends. `croniter` and
`cron_descriptor` are available too.

**The test for what needs a Request:** does it touch disk, network, clock, or
process? If no, just write it.

---

## Declarations

Only declare what differs from the defaults. Saying nothing gets you an
ephemeral, in-process box of your own — right for most tools.

```python
class Indexer(BaseTool):
    name = "indexer"           # required; must be unique
    description = "..."        # shown to the agent
    parameters = {...}         # JSON schema for the arguments

    box = "search"             # share a process with helper files
    timeout = 120              # seconds; the kernel clamps it
    memory_mb = 512            # subprocess only, POSIX only
    requests = ["fs.read", "db.query"]   # what you may do; see below

    # Frontends only:
    background_submit = True   # submit off poll() when replies may render
    restore_on_start = True    # host restores after start() releases the box

    dependencies_pip = ["numpy"]
    dependencies_files = ["helpers/shared.py"]
```

Declarations are **intent**. The kernel reads them without importing your file,
resolves them, and clamps them. Asking for a longer timeout does not grant one.

`requests` is the exception that goes the other way: it does not grant, it
*limits*. When a command declares `require_approval = True` and the user says
yes, that single approval covers exactly the Request types listed here —
anything else still prompts on its own. So list what you actually use and
nothing more, and expect the validator to reject a name that is not a real
Request type. A misspelling grants nothing and shows up as a dialog the user
thought they had already answered.

Services declare what other code may reach:

```python
class Embedder(BaseService):
    name = "embedder"
    exports = ["embed", "similarity"]   # everything else stays internal
    hooks = {"end_turn": "check"}       # doorways to stand at
    subscribed_channels = ["task_completed"]   # bus channels to hear
```

Those last three are the same idea three times: **the kernel reads the
declaration and does the registering**, so a plugin holds no handle it could
leak and uninstalling the file takes the wiring with it.

**Declaring a file makes it importable.** `dependencies_files` names files
from other folders; they join your box's namespace, so you reach them as
siblings:

```python
class Caption(BaseTool):
    dependencies_files = ["helpers/parse_image.py"]

# then, in the same file
from .parse_image import parse_image
```

You still write the import. Declaring is what makes the name *available* —
exactly as `dependencies_pip = ["numpy"]` installs numpy and you still write
`import numpy`. Nothing appears in your namespace by magic.

This is how you reach a **parser**:

```python
from guest.parsing import ParseResult, clean_text, max_chars, register


def parse_thing(sdk, path, config=None):
    """One signature, whoever calls it."""
    return ParseResult(modality="text", output=clean_text(sdk.fs.read(path)))


register([".thing"], "text", parse_thing)
```

Import the contract from **`guest.parsing`**, never from the kernel's
`parsing` — a child process cannot see the kernel, so a kernel import loads
in-process and fails in a subprocess, which is exactly where a heavy parser
wants to run. Avoid `pathlib` too; match suffixes with `str.endswith`.

Modalities whose result is a live object — image, audio, video, tabular — can
only be used *inside* the box that imports the parser, because a PIL image or
an open container cannot cross a boundary. Text and extracted paths can, so
`sdk.parse.file` handles those and refuses the rest, pointing you here.

**Helper files** need no class. Give them the same `box` as the plugin that
imports them and use relative imports:

```python
# helper_words.py
box = "wordcount"

def count_words(text):
    return len(text.split())
```

```python
# tool_wordcount.py
from .helper_words import count_words
```

Files in the same box share a process and can import each other. Files in
different boxes cannot reach each other at all — the only way across is a
Request.

---

## What gets rejected, and what to write instead

The validator reads your file before it runs. It never imports it, so being
checked cannot execute anything.

| Instead of | Write |
|---|---|
| `open(p).read()` | `sdk.fs.read(p)` |
| `Path(p).write_text(s)` | `sdk.fs.write(p, s)` |
| `import os`, `import pathlib` | `sdk.fs`, `sdk.env` |
| `import subprocess` | `sdk.proc.run` |
| `import requests`, `urllib.request` | `sdk.net.http` |
| `import logging` | `sdk.log` |
| `context.db`, `context.services` | `sdk.db`, `sdk.services` |
| `db.conn.execute(...)` | `sdk.db.write(sql, params)` |
| `import paths`, `import runtime.*` | a Request for whatever you needed |
| `eval`, `exec`, `__import__` | build the value directly |

Importing a third-party library that isn't vouched for is **not** an error —
it loads with a disclaimer, and the kernel puts it in a subprocess, because
that library's actions cannot be mediated. You do not ask for this and cannot
decline it: **isolation is not something code declares.** Everything can be
written this way; nothing is off-limits for needing one.

A few stdlib modules get the same treatment for the same reason: `sqlite3`,
`zipfile` and `tarfile` open a file *you* name and do their own I/O. Reading a
user's `.db` read-only or extracting an archive is legitimate, so they are
disclaimed rather than refused — subprocess them. Reaching around the kernel's
own database is still an error, caught at `db.conn` rather than at the import.

---

## Things worth knowing

**Safe work is silent.** Reads, database queries, scratch writes, calling
tools and services — none of these interrupt anyone. The user is asked only
for things that reach outside or change what the system can do: network
requests, shell commands, writes outside scratch, config changes, installing
plugins, creating scheduled work.

**Widening capability is always checked; narrowing it never is.** Adding a
tool to a session asks; removing one does not.

**You never see the chain of provenance.** The kernel tracks who called whom
and shows it to the user when it asks. You cannot read it or affect it.

**Name a credential setting `secret_something`.** That prefix is the whole
declaration — the same way `tool_`, `command_` and `service_` prefixes tell
discovery what a file is:

```python
config_settings = [
    ("Brave key", "secret_brave_api_key", "API key for search.", "", {}),
]
```

`sdk.config.read("secret_brave_api_key")` then returns
`<secret:secret_brave_api_key>` rather than the value. Pass that straight into
`sdk.net.http` and the kernel substitutes the real thing on the way out, so
your code uses a credential it never held and cannot leak one by accident. A
setting *without* the prefix is not a secret and is handed over as-is — the
validator warns if one looks like it should have been marked.

Environment variables are the exception, judged by their names, because
nothing declares them: `OPENAI_API_KEY` was named by somebody else entirely.

That works because the *kernel* makes the call. If you are driving a library
that performs its own network I/O — an OAuth client, a provider SDK — there is
no Request to substitute into and you genuinely need the value:

```python
key = sdk.secrets.reveal("gmail_client_secret")   # always asks the user
```

**Asking for your own credential does not interrupt anyone.** A plugin that
declares a setting in its `config_settings` owns that key: configuring it was
the consent, and re-asking on every load would be pointless noise. A
*different* plugin reaching for the same key does get a dialog — that is the
question actually worth asking.

Use a handle wherever a handle can work, because once you hold plaintext you
are responsible for it. And be honest about the ceiling: a credential inside a
foreign library is beyond the kernel's reach.

**Nothing survives between ephemeral runs.** Module state is discarded after
each call. A service, or a persistent box, is how you keep something.

**Your code cannot end itself except by returning.** Returning is the normal
exit; `sdk.respond(value)` is an early one. Timeouts and shutdown are the
kernel's decision.
