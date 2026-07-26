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
sdk.fs.list(path, pattern="*")             # -> [str]
sdk.fs.search(pattern, root=".", glob="**/*")   # -> [{path, line, text}]
sdk.fs.delete(path)
sdk.fs.move(src, dst, copy=False)
sdk.fs.temp(directory=False, suffix="")    # scratch space; always allowed

sdk.net.http(url, method="GET", headers=None, body=None)  # -> {status, body}
sdk.proc.run(argv, timeout=120.0, cwd=None)               # -> {code, stdout, stderr}
sdk.env.read(name)                         # credentials come back as handles
```

### Data

```python
sdk.db.query(sql, params)      # -> [dict]
sdk.db.write(sql, params)
sdk.db.define(ddl)             # create a table your plugin owns

sdk.conv.create(title)         # -> conversation id
sdk.conv.read(conversation_id) # -> {conversation, messages}
sdk.conv.list()
sdk.conv.append(conversation_id, role, content)
sdk.conv.set_title(conversation_id, title)
sdk.conv.set_category(conversation_id, category)
sdk.conv.delete(conversation_id)

sdk.config.read(key)           # omit key for everything
sdk.config.write(key, value)
sdk.users.read(user_id=None)   # defaults to the current user
sdk.users.list()
sdk.users.write(user_id=None, **fields)
```

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
sdk.tools.list()
sdk.tools.call(name, **kwargs)
sdk.commands.list()
sdk.commands.run(name, **args)
sdk.services.list()
sdk.services.call(name, method, **kwargs)   # only exported methods
sdk.plugins.list() / describe(name)

sdk.agent.complete(prompt)           # a model call
sdk.agent.spawn(prompt, wait=True)   # a subagent now
sdk.agent.schedule(prompt, cron)     # a subagent later
```

### Machinery

```python
sdk.cron.list() / get(name) / create(name, job) / update(name, patch)
sdk.cron.remove(name) / enable(name, enabled=True)

sdk.events.emit(channel, payload)
sdk.events.request(channel, payload, timeout=120.0)

sdk.tasks.enqueue(name, paths) / status(name, path) / output(name, path=None)
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
```

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
    isolation = "subprocess"   # needed if you import a foreign library
    timeout = 120              # seconds; the kernel clamps it
    memory_mb = 512            # subprocess only, POSIX only
    requests = ["fs.read", "db.query"]   # advisory; shown at install time

    dependencies_pip = ["numpy"]
    dependencies_files = ["services/helpers/shared.py"]
```

Declarations are **intent**. The kernel reads them without importing your file,
resolves them, and clamps them. Asking for a longer timeout does not grant one.

Services declare what other code may reach:

```python
class Embedder(BaseService):
    name = "embedder"
    exports = ["embed", "similarity"]   # everything else stays internal
```

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
| `import sqlite3` | `sdk.db` |
| `import logging` | `sdk.log` |
| `context.db`, `context.services` | `sdk.db`, `sdk.services` |
| `db.conn.execute(...)` | `sdk.db.write(sql, params)` |
| `import paths`, `import runtime.*` | a Request for whatever you needed |
| `eval`, `exec`, `__import__` | build the value directly |

Importing a third-party library that isn't vouched for is **not** an error —
it loads with a disclaimer, and you should declare
`isolation = "subprocess"`, because that library's actions cannot be mediated.

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

**Secrets are usable but not readable.** `sdk.config.read("brave_api_key")`
returns `<secret:brave_api_key>`. Pass that straight into `sdk.net.http` and
the kernel substitutes the real value on the way out. You can use a credential
you were never given — which means you cannot leak one by accident.

**Nothing survives between ephemeral runs.** Module state is discarded after
each call. A service, or a persistent box, is how you keep something.

**Your code cannot end itself except by returning.** Returning is the normal
exit; `sdk.respond(value)` is an early one. Timeouts and shutdown are the
kernel's decision.
