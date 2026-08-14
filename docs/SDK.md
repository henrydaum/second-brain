# The Second Brain SDK

*How to write code that runs inside the sandbox.*

---

## How to use this guide

Read this document **before writing any new script or extension**. Then read the
matching file in `templates/` from top to bottom. The two documents have
different jobs:

- `docs/SDK.md` defines what all sandbox code can ask the kernel to do, how
  Requests return and fail, what validation rejects, and where the underlying
  contracts live.
- `templates/<type>_template.py` defines that type's filename, folder,
  declarations, lifecycle, entry-point signature, and common failure modes. It
  also contains executable examples.

Do not infer an API from a similar framework or an older Second Brain plugin.
If this guide and the template do not answer a detail, follow their code
pointer and inspect the implementation. The shortest safe authoring loop is:

1. Choose the code type with the table below.
2. Read this guide and its template.
3. Write the file under `<DATA_DIR>/workspace/<root>/`.
4. Validate it with `sdk.plugins.validate(path)` (or the installed validation
   tool that exposes the same operation).
5. Run a script or register/reload a plugin, then verify its smallest useful
   behavior. Writing a file and activating it are separate operations.

| Need | Write | Read next |
|---|---|---|
| One-off computation or a reusable private routine | script | `templates/script_template.py` |
| An action the model can call | tool | `templates/tool_template.py` |
| File- or event-driven pipeline work | task | `templates/task_template.py` |
| A persistent shared capability | service | `templates/service_template.py` |
| A slash workflow the user invokes | command | `templates/command_template.py` |
| A new user interaction surface | frontend | `templates/frontend_template.py` |
| A file-type reader | parser | `templates/parser_template.py` |
| A model-provider connection | LLM backend | `templates/llm_backend_template.py` |
| A service that influences agent turns | hook | `templates/hook_template.py` and `sandbox/guest/hooks.py` |

### Where to look when this guide stops

These are implementation references, not additional prerequisites:

| Question | Authoritative code |
|---|---|
| Which roots and filename prefixes exist? | `trees.py` |
| What may each plugin family declare? | `sandbox/guest/bases.py` |
| What methods and signatures does `sdk` expose? | `sandbox/guest/sdk.py` |
| What Request names exist and how failures are represented? | `sandbox/guest/requests.py` |
| Why was a Request safe, gated, or refused? | `sandbox/policy.py`, then `docs/PERMISSIONS_MAP.md` |
| What source patterns does validation allow? | `sandbox/validator.py` |
| Why does code run in- or out-of-process? | `sandbox/isolation.py` |
| How are plugins discovered and adapted? | `plugins/plugin_discovery.py`, `sandbox/bridge.py` |
| How are parsers or model backends discovered? | `parsing/registry.py`, `llm/registry.py` |

`CLAUDE.md` is the broad architecture map when a change crosses several of
these areas. `docs/MIGRATING_PLUGINS.md` is only for converting old native
plugins; do not use the old contract for new code.

---

## The model, in one paragraph

Your code cannot act. It can only **ask**. Anything that touches disk, network,
clock, or process is a *Request* you make through `sdk`; the kernel decides
whether to allow it, does the work, and hands back the answer. Everything else
— arithmetic, string handling, your own logic — runs normally and costs
nothing. You are not writing async code and there are no callbacks: a Request
looks and behaves like an ordinary blocking function call.

Validation, isolation, and permission are different layers. Validation limits
what code may do directly. Isolation decides which process holds it. Permission
decides whether the kernel will perform a particular Request. Passing validation
or running in a subprocess never grants a Request.

The agent owns `<DATA_DIR>/workspace/`: it may freely create, replace, move, or
delete files there. The user may separately configure `fs_writable_dirs`.
Those are **user-owned folders opened to the agent**, not extra agent workspace.
Writes there do not prompt, but code must touch them only when the user's task
calls for it and must preserve unrelated work. The source tree and installed
package tree remain protected from that standing grant. See
`docs/PERMISSIONS_MAP.md` and `sandbox/policy.py` for the exact decision path.

---

## Writing something that runs

### A script

No base class, no declarations. A file with functions that take `sdk`:

```python
"""Summarize a file."""


def main(sdk, path):
    """Count the lines and words in a file."""
    lines = sdk.fs.read(path).splitlines()
    return {"lines": len(lines), "words": sum(len(l.split()) for l in lines)}
```

That is a complete, runnable sandbox program.

**Put agent-authored scripts in `scripts/`** —
`<DATA_DIR>/workspace/scripts/<name>.py`,
which `sdk.paths.get("scripts")` will tell you. The directory is the whole
declaration: there is no prefix, base class, or keyword that could say what
this file is, so where it sits has to.

Run it with `sdk.scripts.run(path)`, which calls `main(sdk)` by default and
hands back what it returned. **This is what to reach for instead of
`sdk.proc.run`.** A shell command is an OS process outside the boundary, so it
is asked about every single time and no phrasing changes that. A script is
contained — every effect inside it comes back through the gate on its own and
is judged there — so running one costs no dialog at all. Anything expressible
in Python should be a script.

The exception is a script importing a library the validator cannot see inside.
That is asked about once per run, and the library is named, because a foreign
library's own actions are the one part of a script that does not come back as a
Request. Stdlib and SDK only means no interruption.

Scripts are always run in a subprocess, wherever they live. Nothing registers
one and nobody reviewed it, so containment is the whole of what makes running
it cheap.

A name is a filename, not a family: `sync_photos.py` is fine, but do not call a
script `tool_something.py` or `service_something.py` — a family prefix makes
the validator expect a plugin class and refuse the file.

### A plugin

Subclass a base when the *kernel* has to register and schedule the thing.
The filename must carry the family prefix — `tool_*.py`, `task_*.py`,
`service_*.py`, `command_*.py`, `frontend_*.py` — because discovery finds
plugins by filename.

Before writing it, read the template for its family. The template is more
specific than this overview and wins if an example here omits a declaration or
lifecycle detail.

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

**Let `sdk.retry` decide what is worth trying again.** Every failure carries
whether it was transient, set by the handler that failed — which is the only
place that knows. A locked file, an HTTP timeout and a box that died all say
yes; a malformed query says no:

```python
page = sdk.retry(lambda: sdk.net.http(url))
```

Three attempts by default, waiting `backoff` seconds and then double. A
**refusal is never retried** whatever else you ask for — policy is not a
transient condition, and asking again is a second dialog in front of somebody
who already answered. Pass `on=` a predicate over the exception to decide for
yourself, `attempts=1` to turn it off without unwrapping the call.

The backoff sleeps, and sleeping counts against your deadline — which is the
reason the next one exists.

**Ask `sdk.budget()` how long is left**, so long work can stop itself:

```python
done = []
for doc in documents:
    if sdk.budget()["running"] < 20:
        break
    done.append(analyse(sdk, doc))
return sdk.ok({"done": done, "resume_at": len(done)})
```

It answers `{running, wall, deadline, ceiling}` in seconds. `running` is what
your declared `timeout` measures — elapsed time *minus* whatever the kernel
spent answering your Requests, so four minutes inside `sdk.proc.run` costs you
almost nothing. `wall` is the ceiling that bounds the run however it spends the
time, and it is not declarable. Both are `None` when nothing is enforcing a
deadline.

Without this the only thing that ends an over-long run is the watchdog, and it
ends it by killing the box — so a loop three-quarters of the way through a
corpus returns *nothing at all*. It is read-only, so calling it every iteration
costs nothing.

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
sdk.fs.read_bytes(path, offset=0, length=0)  # -> bytes; anything non-text
sdk.fs.iter_bytes(path, chunk_size=4 * 1024 * 1024,
                  offset=0, limit=None)      # -> lazy byte chunks
sdk.fs.write_bytes(path, data, mode="overwrite")
sdk.fs.stat(path)                           # -> {path, name, is_file, is_dir,
                                            #     is_symlink, size, mtime}
sdk.fs.exists(path)                         # -> bool
sdk.fs.list(path, pattern="*")             # -> [str]
sdk.fs.list(path, details=True)            # -> [{path, name, is_dir, size, mtime}]
                                           # directory entry metadata
sdk.fs.list(path, recursive=True, files_only=True,
            sort="mtime", limit=100)       # -> {root, entries, truncated,
                                           #     scan_truncated}
sdk.fs.search(pattern, root=".", glob="**/*")   # -> [{path, line, text}]
sdk.fs.search(pattern, root=".", regex=True, mode="content",
              case_insensitive=False, multiline=False,
              context_lines=0, limit=100)  # -> {root, mode, results, ...}
sdk.fs.delete(path)
sdk.fs.move(src, dst, copy=False)
sdk.fs.temp(directory=False, suffix="")    # workspace/temp scratch; always allowed
```

`sdk.fs.stat` inspects exactly one file or directory and raises when it is
missing. `sdk.fs.exists` uses the same `fs.stat` Request but turns that expected
missing case into `False`; denials and real I/O failures still raise.

```python
sdk.net.http(url, method="GET", headers=None, body=None,
             params=None, json=None)       # -> {status, body, headers}
sdk.net.http_json(url, method="GET", headers=None, body=None,
                  params=None, json=None)  # same, with parsed JSON body

sdk.proc.run(argv, timeout=120.0, cwd=None, shell=None)   # -> {code, stdout,
                                           #     stderr, command}
sdk.proc.start(argv, cwd=None, shell=None, label="")      # -> {id, pid, log,
                                           #     command, running, ...}
sdk.proc.status(id, tail=4000)             # -> the same, plus `output`
sdk.proc.stop(id)                          # -> {id, code, log}
sdk.proc.list()                            # -> [ ... ]

sdk.scripts.run(path, entry="main", wait=True, **args)  # -> whatever it
                                           #     returned; no dialog
                                           #     wait=False -> {id, script,
                                           #     started}
sdk.scripts.collect(ids=None, timeout=None) # join detached scripts;
                                           #     ids=None = all mine
sdk.scripts.stop(id)                       # cancel one

sdk.env.read(name)                         # credentials come back as handles
sdk.secrets.reveal(name)                   # plaintext; always asks the user

sdk.app.stop(restart=False)                # end the process; always asks
```

`sdk.app.stop` returns the message to show the user, and the kernel defers the
actual stop for a moment so that message arrives first. It is what `/quit` and
`/restart` are; a plugin has no other reason to reach for it.

**Prefer a script to a command.** `sdk.scripts.run` runs a file of SDK code
from `scripts/` and does not ask, because everything inside it is classified
individually. `sdk.proc.run` starts a process the kernel cannot see into and
always asks. The two are not close in cost, so reach for the shell only when
the work genuinely is another program. Keyword arguments go to the entry
function — `sdk.scripts.run(p, total=3)` calls `main(sdk, total=3)` — and
`wait=False` answers as soon as it has started rather than when it finishes.

**Several at once.** `wait=False` hands back an `id`, and that is how work is
fanned out over ordinary code rather than over subagents. Each script is a box
of its own, so they genuinely run in parallel; none of it involves a model.

```python
ids = [sdk.scripts.run("analyse.py", wait=False, doc=d)["id"]
       for d in documents]
for report in sdk.scripts.collect(ids):
    sdk.log(f"{report['script']}: {report['state']} {report['data']}")
```

A report carries `id`, `script`, `state`, `ok`, `data` and `error`, where
`state` is `running`, `done`, `failed` or `cancelled` — the same four a
subagent reports. `timeout=0` polls without waiting, and anything still going
comes back `running` and stays collectable. Each finished report is
**delivered once**, so two collectors cannot both act on one answer. Reach for
`sdk.agent.spawn` instead when the work needs judgement rather than code.

**Reaching the network.** `sdk.net.http` answers with
`{status, body, headers}`, and an HTTP *error* status is an ordinary answer
rather than a failure — check `status` the way you would for a 200. That is
deliberate: a 429's body is where an API tells you which limit you hit and for
how long, and only a request that got no reply at all (DNS, refused, timed out)
raises. `body` is decoded text; there is no binary download.

`params` URL-encodes query values, including repeated list values. `json`
encodes a request body and supplies `Content-Type: application/json` unless
you supplied one; it cannot be combined with `body`. Use `http_json` when the
response should be decoded too. It keeps the same envelope and replaces its
text `body` with the parsed value; an empty body becomes `None`.

Redirects are also ordinary answers and are **not followed automatically**.
Read the 3xx response's `headers["location"]` and make another `sdk.net.http`
call if you want to follow it. That second call is intentional: a server may
choose a new host, but it cannot authorize that host on the plugin's behalf.

Whether it asks the user depends entirely on **where** it is going. The kernel
config setting `net_allowed_hosts` lists hosts a plugin may reach without a
dialog, and a bare domain covers its subdomains. Anything else prompts, naming
the host. You cannot declare your own hosts and should not try — the point is
that a person decides what this app talks to, so a service that needs an
endpoint says so in its install notes and the user adds it once.

**Running commands.** `run` blocks and hands back what the command printed;
`start` hands back a *handle* to something still running, which `status`,
`stop` and `list` speak about. Both ask the user first — every command does,
and no amount of phrasing changes that, so ask for what you actually need.
Stopping does not ask: a server the agent cannot kill without a dialog is a
server it will not start.

`shell` is the difference between an argv and a command *line*. Leave it
`None` and the argv is executed directly — no pipes, no globbing, no
metacharacters, which is what you want when you built the list yourself. Pass
`"default"` for the platform shell (`cmd.exe` on Windows, `/bin/sh`
elsewhere) when you need `|`, `>` or `&&`, or `"powershell"` for that one.
The kernel builds the invocation, because a guest that wrapped its own
command as `["cmd", "/c", line]` would have every embedded quote mangled —
`cmd.exe` does not understand the escaping `subprocess` produces.

A started process cannot stream across the boundary, so its output is teed to
a log file: `status` returns the tail, and `log` is a path `sdk.fs.read` will
open. The registry is in memory and does not survive a restart, so anything
still running when Second Brain goes down is orphaned rather than killed.

`stop` answers `{id, code, stopped, pid, log}`. On POSIX it sends `SIGTERM`
first — a dev server should get to close its socket — and escalates to
`SIGKILL` if that is ignored; on Windows `taskkill /T /F` is already a hard
kill of the whole tree. `stopped` is `False` in the rare case something
outlived even that: it is untracked and still running, which is worth saying
rather than reporting a clean stop.

Two `sdk.paths.get` names are not directories and belong to this section:
`"python"` is the interpreter hosting the app — invoke `pip` through it or
packages land in whatever environment is first on `PATH` — and `"platform"`
is `sys.platform`. They are here because the validator refuses `sys`, and
these are the only facts behind it a plugin has a real claim on.

`read` decodes UTF-8 with replacement, which quietly mangles anything that is
not text. Reach for `read_bytes` whenever the file is an image, audio, a PDF,
or an archive. Base64 on the wire is the SDK's problem, not yours — you hand
over `bytes` and get `bytes` back.

One *answer* has to fit in one wire message, so a whole-file `read_bytes` is
capped around 11 MB and says so. `iter_bytes` makes the successive windowed
Requests for you and stops at EOF. Join it when the whole file genuinely needs
to be in memory, or process each chunk as it arrives:

```python
data = b"".join(sdk.fs.iter_bytes(path))
```

**`list` and `search` each have two shapes**, and passing any of the extra
arguments switches to the second. Plain, they are a flat glob and a substring
scan returning bare lists. With extras, they walk the tree properly — pruning
`.git`, `node_modules`, `__pycache__` and friends, never following a symlink,
capping the enumeration — and answer with a dict carrying `truncated` and
`scan_truncated` alongside the results. Use the second shape for anything
tree-shaped; a plain `**/*` glob over a project descends into `.git` and hands
back tens of thousands of paths nobody wanted.

Search modes: `"content"` gives `rel:lineno: text` lines, `"files"` gives
matching paths, `"count"` gives `[path, n]` pairs. `regex=True` reads the
pattern as Python `re` (not PCRE — escape literal braces). `glob` filters which
files are searched, where `"*.py"` is top level only and `"**/*.py"` is any
depth. Binary and oversized files are skipped and counted, and ripgrep is used
when it happens to be installed — none of which changes the answer's shape.

### Data

```python
sdk.db.query(sql, params, max_rows=0)  # -> [dict]; reads only, capped at 500
                                       # (max_rows may only lower that cap;
                                       #  exactly the cap back means more)
sdk.db.write(sql, params)
sdk.db.define(ddl)             # create a table your plugin owns

sdk.conv.create(title, category=None, activate=False)
sdk.conv.read(conversation_id, details=False)
sdk.conv.list(category=None, limit=50, offset=0, details=False)
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
sdk.paths.get(name)            # project, data, bundled, installed, workspace,
                               # scripts, python, platform
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

**A `conv.read` message row carries `author`, and `role` alone will mislead
you.** The kernel writes `role='user'` rows the person never typed — a cancel
notice, a doorman's note, the summary bridge compaction leaves behind, a note
that a slash command ran — and each carries a non-empty `author`; a row
somebody actually typed has `author` of `None`. `sdk.conv.append` stamps the
calling plugin's name, read off the provenance chain rather than accepted as an
argument. When you want what the *user* said, filter for it:

```python
sdk.db.query("SELECT content FROM conversation_messages"
             " WHERE conversation_id = ? AND role = 'user'"
             "   AND COALESCE(author, '') = ''"
             " ORDER BY id DESC LIMIT 1", [conversation_id])
```

(`role = 'system'` is a separate exclusion again: those rows are state and
compaction markers, not messages.)

**Writing: rows yes, schemas no.** You may write rows in the kernel's own
tables — data cannot change how the kernel works, only structure can. Prefer
the named Request where one exists (`sdk.conv.set_title` over an `UPDATE`,
`sdk.conv.append` over an `INSERT`), because those carry the owner check and
emit the event frontends redraw from. Reach for `sdk.db.write` when there is no
verb for what you need, which is mostly bookkeeping columns:

```python
sdk.db.write("UPDATE conversations SET last_title_check_message_count = ?"
             " WHERE id = ?", [count, conversation_id])
```

Refused outright: `CREATE`/`DROP`/`ALTER` naming a kernel table, anything
touching `sqlite_master`, any `PRAGMA`, `ATTACH`/`DETACH`/`VACUUM`, and
`users.password_hash`. Asked about rather than refused: `DELETE` from a kernel
table — it is the one row write you cannot undo by writing again. Your own
tables, made with `sdk.db.define`, are yours to delete from freely.

**The 500-row cap means the answer crosses, not the data.** If your query
would return more than that, the reduction belongs in SQL — an aggregate, a
`GROUP BY`, an `ORDER BY … LIMIT`. Paging the whole table over and computing
in Python works and is nearly always the wrong shape.

Vectors are the case where that used to be impossible, so SQLite carries an
extra operator for them:

```python
sdk.db.query(
    "SELECT path, chunk_index, vec_cosine(embedding, ?) AS score "
    "FROM text_embeddings WHERE model_name = ? AND length(embedding) = ? "
    "ORDER BY score DESC LIMIT ?",
    [query_vector, model, len(query_vector), 5])
```

`vec_cosine(a, b)` takes two float32 BLOBs — bytes cross as bytes, so the
query vector binds as an ordinary parameter — and answers with cosine
similarity, or `NULL` for anything it cannot compare (different lengths, a
non-BLOB column, a zero vector). It never raises, because a scalar function
that raised would fail the whole statement over one stale row. Filtering on
`length(embedding)` first skips vectors left behind by a different model
before the arithmetic runs.

### People and sessions

```python
sdk.ui.ask(prompt, title="Question", type="string", choices=None,
           required=True, default=None, timeout=300.0)
                                     # type: string | integer | number |
                                     #       boolean | array | object
                                     # cancel raises sdk.Denied; no answer fails
sdk.ui.approve(action, justification)
                                     # the Request *is* the question: it is
                                     # always unsafe, so the kernel's approver
                                     # asks it. Returns True, or raises
                                     # sdk.Denied when the answer was no.
sdk.ui.render(paths, caption="")     # show files in the chat

sdk.session.get(key="")              # defaults to this session
sdk.session.list()
sdk.session.push(message, key="")    # message the user out of band
sdk.session.push(message, title="Indexed", notify=True, level="success")
                                     # ...or as a *notification*: see below
sdk.session.state_get(namespace="sandbox")
sdk.session.state_set(value, namespace="sandbox")
sdk.session.cancel(key="")
sdk.session.add_tool(tool) / remove_tool(tool)
sdk.session.add_prompt(text) / remove_prompt(handle)
sdk.session.add_attachment(path)     # show the *model* a file
sdk.session.set_mode(mode, scope="conversation")
                                     # "lockdown" | "ask" | "yolo".
                                     # scope="turn" expires at turn end.
```

`sdk.session.get()["mode"]` reads how this conversation answers approval
dialogs. Setting it is the one place the widening/narrowing rule below has
teeth for a *plugin*: `"lockdown"` narrows and goes through, anything else
raises a dialog — and one you cannot answer for yourself, since unattended
work is refused before a mode is ever consulted. Reach for it only when a
plugin genuinely owns the conversation's posture. The person's own route is
`/mode`.

Two different destinations, and the names are close enough to be worth stating
apart. `sdk.session.add_attachment(path)` puts a file in front of the **model**
on its next call — that is how a tool lets the agent actually look at an image.
`sdk.ok(..., attachments=[path])` shows a file to the **user**. A tool that
produced a file should return it; a tool that wants the agent to see one stages
it; `sdk.ui.render(paths)` is the same user-facing move for a task, command or
service, none of which returns anything with room for a file.

You never need to check whether the model can read the modality. If it cannot,
the kernel substitutes the file's parsed text, and failing that a line naming
where the file is — so staging is always the right call, and a capability test
in your plugin would only get the answer wrong.

### Notifications

`notify=True` turns a push into a **notification**: the system telling the user
something, rather than something said in the conversation. A frontend with
somewhere to put those — a panel, a badge, a toast — draws it there; one
without shows it in the chat exactly as a plain push would. Nothing is lost by
asking, so the question is only which it *is*.

```python
sdk.session.push("Indexed 12 files.", title="Nightly index",
                 notify=True, level="success")
```

Reach for it when the user did not just ask for this and is not watching: a
background write finishing, something needing their attention later. A plain
push is right when you are speaking *into* the conversation — a tool narrating
what it is doing mid-turn is not a notification.

`level` is `info` | `success` | `warning` | `error` and only styles the result.
`title` is what a collapsed panel shows, so make it say what happened.

**You cannot state who sent it.** The kernel stamps `source` from the
provenance chain, so a plugin cannot claim to be the plugin watcher and a
reader can trust the attribution. Same reason a box cannot state its own chain
root.

Reading them back is for a frontend drawing a panel, not for an ordinary tool:

```python
sdk.notifications.list(limit=50, since_id=None, unread_only=False)
sdk.notifications.mark_read(ids=None, before_id=None)
```

Both are scoped to the calling user in SQL — there is no `user_id` argument to
pass and none to get wrong. `mark_read` answers with how many rows actually
changed, so calling it twice is idempotent. Persisted notifications survive a
restart and are swept by `data_retention_days` like everything else; transient
ones (progress, e.g. "Compacting conversation…") are delivered and never
stored.

### Other code

```python
from guest.forms import FormStep

sdk.tools.list()
sdk.tools.call(name, **kwargs)
sdk.commands.list(details=False, visible=False)
sdk.commands.run(name, **args)
sdk.services.list(details=False)
sdk.services.call(service, method, *args, **kwargs)  # exported methods only
                                          # service and method are
                                          # positional-only, so an export may
                                          # take its own name= without colliding
sdk.services.load(name) / unload(name)
sdk.plugins.list(source="registered", category="")
sdk.plugins.describe(name)
sdk.plugins.validate(path)                # -> {ok, disclaimed, findings,
                                          #     unmediated, declarations, digest}
sdk.plugins.register(path)
sdk.plugins.unregister(path=...)          # or name=..., family=...
sdk.plugins.reload(path=...)              # or name=..., family=...
sdk.plugins.install(package_id)
sdk.plugins.uninstall(package_id)
sdk.plugins.update()

sdk.agent.complete(prompt, messages=None, session_key=None, profile="")
                                          # a model call. `profile` names an
                                          # LLM profile; "" follows the
                                          # session's, or the default.
sdk.agent.spawn(prompt, title="Subagent", attachments=None,
                wait=True, timeout_seconds=None)
sdk.agent.collect(ids=None, timeout=None) # join children; ids=None = all mine
sdk.agent.stop(id)                        # cancel one
sdk.agent.schedule(prompt, cron, title="Scheduled subagent",
                   attachments=None, one_time=False, name="")
```

Plugin lifecycle mutations are approval-gated. Paths must resolve to a
recognized built-in, sandbox, or installed plugin file. A name-only unload or
reload must identify exactly one registered plugin; supply `family` when the
same name exists in more than one registry.

### Orchestrating subagents

A **subagent** is a whole agent turn running in its own conversation, on its
own thread. `spawn` starts one, and everything else is about getting the answer
back.

Waiting for one is the easy case — `wait=True` blocks and hands you the report:

```python
report = sdk.agent.spawn("Summarise every TODO in docs/, grouped by file.")
sdk.log(report["text"])
```

The interesting case is several at once, and that is what a **script** is for.
`spawn(wait=False)` returns straight away with a handle, so N children start in
the time one would have taken; `collect` joins them. Nothing in between is in
your context until you ask for it — which is the real reason to write the
script instead of making N tool calls:

```python
def main(sdk, questions):
    """Research several questions at once, then write up what came back."""
    started = [
        sdk.agent.spawn(
            f"Research this and report your findings:\n\n{question}",
            title=question[:40],
            wait=False,
            timeout_seconds=600,
        )["id"]
        for question in questions
    ]

    # Nothing is waiting yet, so anything here happens while they run.
    sdk.log(f"{len(started)} agents working")

    findings, lost = [], []
    for report in sdk.agent.collect(started):
        if report["ok"]:
            findings.append(f"## {report['title']}\n\n{report['text']}")
        else:
            # A cancelled child hit its deadline and produced nothing. Say so;
            # never fill the gap with a guess.
            lost.append(f"{report['title']} ({report['state']})")

    if not findings:
        return sdk.fail("every agent failed: " + ", ".join(lost))
    return {"report": "\n\n".join(findings), "lost": lost}
```

Run it with `sdk.scripts.run("research.py", questions=[...])`. One Request, one
result. (Write the report out with `sdk.fs.write` if you want it on disk —
that is a separate act and gets its own approval, which is why it is not part
of the example.)

Four things worth knowing before you write one of these:

- **A report is delivered once.** `collect` takes it; a later `collect` will
  not hand it over again. Inside an agent turn, anything you did not collect is
  collected for you before the turn ends, so `wait=False` never loses a result
  even if you forget it.
- **`collect(timeout=0)` polls** — running children come back with
  `state == "running"` and stay uncollected, so a progress display is cheap.
  With no timeout it waits until each child's own deadline, which is usually
  what you want.
- **A deadline is a hard cutoff.** A child still going at `timeout_seconds` is
  cancelled and comes back as `state == "cancelled"` with no text. That is a
  real answer, not a missing one, and reporting anything on its behalf is the
  one mistake worth guarding against. Its partial transcript survives in
  `conversation_id`.
- **A subagent cannot spawn subagents,** and nobody can answer a dialog on its
  behalf — its Requests are unattended, so anything unsafe is refused outright
  rather than asked about. Give it work that needs no permission.

`sdk.agent.stop(id)` ends one early. `sdk.agent.schedule(prompt, cron)` is the
same thing on a timer; it is unsafe by policy, because unattended future work
is the one case where nobody will be there to notice it going wrong.

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

`end_turn` is the richest of them and the least used. Its verdicts are `Allow`,
`SendBack(note)` (push the agent back inside with a note), `RequireTool(name)`
(make it call something before it may leave), and `Redrive()`. A doorman can
never trap the agent: past `DOORMAN_FIRE_LIMIT` interventions in one turn the
kernel lets it out regardless.

`turn_start` is the mirror image — it runs before the agent begins, and it
adjusts rather than judges. Staging an attachment into the coming turn is
`sdk.session.add_attachment(path)`, which is reachable from anywhere in a box
rather than only from this hook: a tool mid-turn stages onto the *next* model
call, which is the same queue, drained at every agent action.

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

### Setting up and cleaning up

Two optional methods bracket a package's life on the machine. Any of the five
families may write either; both are found by AST, so a plugin that wants
neither costs nothing.

```python
class Notes(BaseService):
    name = "notes"
    requests = ["paths.get", "config.read", "config.write", "db.define"]

    def on_install(self, sdk):
        folder = sdk.path.join(sdk.paths.get("workspace"), "notes")
        watched = sdk.config.read("sync_directories") or []
        if folder not in watched:
            sdk.config.write("sync_directories", [*watched, folder])
        sdk.db.define("CREATE TABLE IF NOT EXISTS notes_seen ("
                      " id INTEGER PRIMARY KEY, path TEXT)")

    def on_uninstall(self, sdk):
        sdk.db.define("DROP TABLE IF EXISTS notes_seen")
```

`on_install` runs right after the package's files land and **before its own
services load**, so anything it arranges is in place by the time the plugin
starts. It runs again on any update that changes the file, so **write it
idempotent** — read, skip what is already done, then write. That is also what
keeps it from overwriting a value the user has since edited.

`on_uninstall` runs as the *first* step of an uninstall: the file is still on
disk, the plugin is still registered, and its pip dependencies are still
installed. A dependency somebody else still needs is not being uninstalled and
gets no call.

Both run in an ordinary ephemeral box under the chain of the `/packages`
command the user typed, and that is where the permissions come from. Reads and
SQL are free — dropping a table you created needs no dialog. A **config write
raises one approval dialog** naming the setting and the value, because the
chain is attended: somebody is right there, having asked for this package. It
is the only moment a plugin can write a kernel setting at all; from a
service's own chain the same write is refused outright, with nobody to ask.

Raising is reported and changes nothing else. A package whose setup was
declined is still installed; a package whose cleanup failed is still removed.
Be conservative about what counts as yours on the way out: a table you created
is unambiguous, a folder the user has been syncing for months is not.

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
        updates = sdk.net.http_json(
            "https://api.example.com/updates", params={"after": self._cursor})
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

**A client library that owns an event loop.** Most modern transport libraries
are asyncio-only and expect to run the process. You may not give one a thread —
`threading` is refused, because the kernel schedules — so instead create the
loop in `start` and lend it slices from `poll`:

```python
class Chat(BaseFrontend):
    poll_interval = 0.0           # poll already spends its time in the loop

    def start(self, sdk):
        self._queue = []
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())   # handlers append to _queue
        return True

    def poll(self, sdk):
        self._loop.run_until_complete(asyncio.sleep(0.08))   # the library's turn
        pending, self._queue = self._queue, []
        for item in pending:
            sdk.frontend.submit_text(item["key"], item["text"])
        return bool(pending)

    def render(self, sdk, session_key, kind, payload):
        # Between polls the loop is idle, so a send is just awaited.
        self._loop.run_until_complete(self._send(session_key, payload))
```

The library's handlers should *queue* rather than submit: `poll` is where
Requests belong, and everything is one thread, so a plain list is enough
synchronisation. Work that outlives a single call — a streaming pump, a
keepalive — goes on the loop with `create_task` and makes progress during
later slices.

This shape needs **subprocess isolation** and does not announce it. A
subprocess box serves every call on one thread, so a loop bound in `start` is
still that thread's loop when `poll` comes back; an in-process resident box
runs each call on a fresh worker, where `run_until_complete` would be talking
to a loop belonging to somebody else. Isolation is read off the file's tree and
imports, never declared, so make sure it comes out right: an installed package
importing a foreign library always gets a subprocess, and a plugin whose imports
are all pure will not.

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
sdk.frontend.submit_attachment(session_key, path, extension="", file_name="",
                               caption="", is_photo=False, ingest=False)
sdk.frontend.submit_attachments(session_key, files, caption="", ingest=False)
sdk.frontend.submit_action(session_key, action_type, payload=None)
sdk.frontend.cancel(session_key)
sdk.frontend.bind(session_key, external_id=None, user_type="user", config=None)
sdk.frontend.attended(session_key, present=True)
sdk.frontend.pending_input(session_key, details=False)      # an id, or None
                                                # details=True: the question,
                                                # {"kind": "approval"|"form_field",
                                                #  "payload": {...}}, or None
sdk.frontend.resolve(session_key, value, request_id="")
```

These work **only inside a loaded frontend**. Each resolves to your own
frontend's adapter through a handle the kernel parks when your box opens, so
you cannot submit on another frontend's behalf and a tool that imported the
same namespace reaches nothing at all.

An `approval` render carries an `id`; answer it with `sdk.frontend.resolve`.
Holding the id is enough to answer and *only* enough to answer — the action
being authorized never crosses.

`resolve`, `cancel` and the `submit_*` calls all drive the state machine, and
the turn they start renders back into *your* box. The kernel therefore runs
them off your thread when you declare `background_submit` — without that, the
render would block on the call lock you are still holding inside `poll`, and
your transport would freeze permanently. What comes back is whether there was
anything to do (`resolve` returns False for an approval that was already
answered or timed out), not whether the turn succeeded; the turn reaches you
as renders, like everything else.

**A file that arrived over your transport wants `ingest=True`.** Point it at
scratch from `sdk.fs.temp()`, let the client library download straight there,
and the kernel moves it into the attachment cache — a watched directory, so the
pipeline extracts and indexes it like any other incoming file. Leaving it in
temp would skip all of that. The bytes never cross the boundary, which is also
the only way a file bigger than one wire message gets in:

```python
temp = sdk.fs.temp(suffix=sdk.path.suffix(name))
await handle.download_to_drive(temp)
sdk.frontend.submit_attachment(key, temp, file_name=name,
                               caption=caption, ingest=True)
```

**Several files are one message, so they are one submit.** A person who picks
three files and types a line has not sent three messages, and sending them as
three does not work: a `send_attachment` hands the turn to the agent, so the
second one arrives at a session that is already busy and is told to wait.
`submit_attachments` carries the whole message, and the model sees every file
in the same call. `caption` and `ingest` are the message's; a file may still
say its own.

```python
sdk.frontend.submit_attachments(key, [
    {"path": first, "file_name": "chart.png"},
    {"path": second, "file_name": "notes.pdf"},
], caption="what do these have in common?", ingest=True)
```

### Acting as one of your sessions

A frontend often has to do something a person asked for that is not "carry this
line into the state machine" — load a conversation, read the command list,
write a setting. Calling the SDK directly gets you the read-only half and
silently nothing else, because your box's chain is rooted `frontend:<name>`,
which names no session: nobody is watching it, so anything unsafe is **refused
rather than asked**, and anything reading `ctx.session_key` acts on nothing.

`act` says whose request this actually is:

```python
handle = sdk.frontend.act(session_key, "conv.load", {"id": 7})
...
result = sdk.frontend.collect(handle)     # None until it finishes
```

The Request runs rooted at that session, with that session's context, and is
classified exactly as it would be anywhere else. What changes is that
attendance now decides — and attendance is *what you declared*:

```python
sdk.frontend.attended(key, True)      # a socket connected
sdk.frontend.attended(key, False)     # it went away
```

So an unsafe Request raises a real dialog, which arrives back at you as an
`approval` render for the same session, and you answer it with
`sdk.frontend.resolve`. Say nobody is watching and the authority goes with it.

**It does not wait, and must not.** Your box serves one call at a time, and the
dialog has to render *into that box* to be seen — so waiting inline would
deadlock until the dialog timed out. Start it, return from `poll`, and collect
on a later tick:

```python
def poll(self, sdk):
    for handle, waiting_on in list(self._waiting.items()):
        outcome = sdk.frontend.collect(handle)
        if outcome is not None:              # a dict: ok, data, error, code
            del self._waiting[handle]
            self._answer(sdk, waiting_on, outcome)
    ...
```

`collect` hands back the `Result` as a plain dict rather than raising, because
a refusal is an ordinary answer to pass on to whoever asked — not a failure of
yours. Delivery is one-shot, and an answer nobody collects is swept.

Two things `act` will not carry: itself (recursion) and the `http.*` family,
which belongs to your transport rather than to any session.

**Ask what is pending; do not remember it.** A transport where a person answers
by typing "yes" has to know whether a yes/no is what the next line means. You
are told an approval exists — you were handed one to render — but not when it
stops existing: another frontend can answer it, or it can time out. Call
`sdk.frontend.pending_input(key)` at the moment you need to decide, and
check `sdk.session.get(key)["phase"]` too: when the state machine is already
collecting the answer itself, interpreting the line as well consumes one
keystroke twice.

**And say what an answer did — the kernel no longer does.** An approval's
outcome crosses as the phase leaving `approving_request` and as
`ActionResult.data`, not as prose. It used to be a sentence on the `messages`
kind, which is what the agent's own words ride, so a frontend that draws its own
dialog could not tell them apart and printed "Approval required." into the chat
beside the dialog that already said so. Word it however your transport should.

**A render is an event, not state.** Nothing is re-sent because you asked. A
transport that can reconnect — a browser, a socket that dropped — needs
`pending_input(key, details=True)` to get back to a question it was never
handed, and it covers a suspended form as well as an approval, because both are
"this session is blocked until a person answers". It is answered from the
session's own persisted phase stack when you have no record of one, so a restart
does not report a blocked session as an idle one.

**And you are told when a question stops waiting.** `render_approval_settled`
is the counterpart to `render_approval_request`, and the only way a surface that
drew a dialog learns it may take it down: another frontend can answer the same
question, and the approver denies by name after 300 seconds. Neither is
something you did. It is defaulted to nothing rather than raising, so a frontend
written before it existed is correct as it stands — just chattier than it needs
to be, since it has to keep asking to find out.

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
padded columns and code fences drop away. `sdk.md.align_tables(text)` is the
first half alone, for a surface that renders fences itself and wants the tables
padded inside one. Both pure, no Request.

### A port

A frontend a client connects *to* — a web UI, anything speaking SSE — declares
`serves_http = <port>`. The kernel binds it on loopback, accepts and parses on
its own threads, and you drain what arrived:

```python
import json


class Web(BaseFrontend):
    name = "web"
    serves_http = 8787

    def start(self, sdk):
        self._streams = {}
        self._seq = 0
        return True

    def poll(self, sdk):
        requests = sdk.http.drain()        # a list, possibly empty. Never blocks.
        if not requests:
            return False
        for request in requests:
            if request["path"] == "/events":
                # Held open. Renders push into it later, from render().
                sdk.http.respond(request["id"], stream=True)
                self._streams[request["id"]] = True
            else:
                sdk.frontend.submit_text("web", request["body"])
                sdk.http.respond(request["id"], body='{"ok": true}')
        return True

    def render(self, sdk, session_key, kind, payload):
        self._seq += 1
        for stream_id in list(self._streams):
            try:
                sdk.http.push(stream_id, json.dumps(
                    {"kind": kind, "payload": payload}), ident=str(self._seq))
            except sdk.Failed:
                # The client went away. Ordinary, and you are told so you can
                # stop rendering a whole turn into a closed socket.
                self._streams.pop(stream_id, None)
```

**`socket` is refused and always will be**, and this is the mediated route
rather than an exception to it. The reasoning is the console's: a guest that
opened its own socket would block its box, and a rule that worked in-process
and broke under isolation would be the worst kind. Inverting it means the child
process never opens a socket at all, so an HTTP frontend can be
subprocess-isolated.

**A reply may outlive the call that opened it**, which is why there are four
Requests and not two. `respond(..., stream=True)` writes the SSE headers and
leaves the connection open; `push` adds one frame; `close` ends it. The
connection stays kernel-side and you hold only an id — enough to answer, and
only enough to answer.

`drain()` returns `{id, method, path, query, headers, body}` and never blocks;
return falsy from `poll` when it is empty and renders land in the pause.
`push` **fails** once the client has gone — the ordinary end of a stream rather
than a fault, but told to you, because a frontend that never hears it goes on
rendering a whole turn into a closed socket. It is learned on the write *after*
they left, which is how SSE works everywhere and is soon enough.

**Number your frames.** `push(..., ident=...)` becomes the frame's `id:`, which
a browser hands straight back as `Last-Event-ID` when `EventSource` reconnects
— and `EventSource` reconnects on its own, for free. Keep a short per-session
buffer, replay from that header, and a page refresh resumes instead of losing
whatever was said while it was reloading. Note the browser cannot send an
`Authorization` header on an `EventSource` at all, so a token-protected stream
needs it in the query string; that is the API, not an oversight.

`Content-Length` and `Connection` are computed for you on a non-streaming
response, and the SSE content type on a streaming one. Anything you put in
`headers` wins — the kernel fills gaps, it does not overrule you.

**Serving a browser means CORS, and it is yours to set.** The kernel adds no
`Access-Control-*` headers, because which origins may talk to your frontend is
a decision about *your* deployment, not something a sandbox can guess. A page
served from anywhere other than this exact port is a cross-origin request, so a
client on a different host — or the same host on a different port — needs at
minimum:

```python
CORS = {"Access-Control-Allow-Origin": "https://example.com",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"}


def poll(self, sdk):
    """Answer preflight before anything else."""
    for request in sdk.http.drain():
        if request["method"] == "OPTIONS":
            sdk.http.respond(request["id"], status=204, headers=CORS)
            continue
        sdk.http.respond(request["id"], headers=CORS, body="{}")
    return True
```

Forget the preflight and a browser reports a CORS failure with no detail and no
server-side trace, which is an hour lost to an opaque error. Prefer serving
your client's HTML from this same frontend when you can — same origin, no CORS,
nothing to get wrong.

**The port is exclusive**, for a blunter reason than the console's: two
frontends cannot bind one port. The kernel lends it to the first claimant and
refuses the second. Binding is the kernel's act, so nothing you declare can
reach a public interface — exposing the port is a tunnel's job.

**Your declaration is a default.** The config key `<name>_port` overrides it,
so a person can move the port without editing your plugin. Declare it in
`config_settings` and it shows up in `/config` like any other setting.

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
`llm/llm_<provider>.py`, found by the LLM registry, loaded into a box, and
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

sdk.parse.file(path, modality="text")   # local parser if declared, else kernel
sdk.parse.modality(extension)

sdk.ledger.record(action, ok=True, data=None)
sdk.ledger.read(limit=50)

sdk.notifications.list(limit=50, since_id=None, unread_only=False)
sdk.notifications.mark_read(ids=None, before_id=None)
```

---

## Free helpers

These run inside the sandbox. No Request, no approval, no cost:

```python
sdk.text.truncate(text, limit)
sdk.md.table(headers, rows)
sdk.md.card(title, pairs)
sdk.md.plain(text)                # monospace: padded tables, fences stripped
sdk.md.align_tables(text)         # padded tables only, fences left alone

sdk.path.join(root, "helpers", "thing.py")
sdk.path.parent(p); sdk.path.name(p); sdk.path.stem(p); sdk.path.suffix(p)
sdk.path.absolute(p, base=sdk.paths.get("project"))
sdk.path.within(p, root)          # containment, separator-aware
sdk.path.normalize(p)             # canonical key for comparing two paths
sdk.path.as_posix(p)              # forward slashes for display and APIs
sdk.path.relative(p, root)        # textual relative path; never reads disk
sdk.path.with_suffix(p, ".json")  # replace the final suffix
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
`statistics`, `dataclasses`, `typing`, `tomllib`, `ipaddress`, `zlib`, and
friends. `croniter` and `cron_descriptor` are available too.

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

### `narration`, a reserved parameter name

A tool call renders in the chat as `⋯ tool_name`. When the arguments do not say
*why* you were called, declare a `narration` property and the model's own words
render beside the name:

```python
class RunCommand(BaseTool):
    name = "run_command"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run."},
            "narration": {
                "type": "string",
                "description": "A few words on what you are doing and why, "
                               "shown to the user. E.g. 'checking what "
                               "changed since the last commit'.",
            },
        },
        "required": ["command"],
    }

    def run(self, sdk, command):        # note: no `narration` parameter
        """Run it."""
        return sdk.proc.run(command, shell=True)
```

Three things to know. It is **optional** — leave it out of `required` so a model
that skips it is not an error. The kernel **strips it** before calling you, so
`run` never receives it and must not accept it. And it is **not free**: the
narration stays in the conversation history and is re-sent on every later call of
the turn, so declare it where the intent is genuinely invisible from the
arguments (running a command, editing a file) and skip it where the arguments
already say everything (reading a named path).

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

### Teaching the agent about yourself

`agent_prompt` is your text in the agent's system prompt, added when you are in
scope and gone when you are uninstalled. This is where point-of-use guidance
belongs — not in the kernel's static prompt, which every model pays for on
every turn whether or not your plugin is installed.

One name, two shapes. A string when the text never changes:

```python
class Todo(BaseTool):
    name = "todo"
    agent_prompt = "## Todos\nKeep the list short. Close items as you finish."
```

A method when it depends on something live:

```python
class Scripts(BaseTool):
    name = "scripts"

    def agent_prompt(self, sdk):
        return f"## Scripts\nThey go in {sdk.paths.get('scripts')}."
```

Prefer the string. It is read from your file without importing it, costs
nothing, and lands in the cacheable block at the top of the prompt.

The method is a real call into your box, so it is cached — but only until
something changes. Any effect performed anywhere in the sandbox (a write, an
install, a config change; reads do not count) invalidates it, and the next
model call recomputes it. So a listing you build from `sdk.fs.list` is allowed
to be genuinely live: a file the agent writes this turn shows up on the next
call, which is the point of the shape. A live contribution rides in the dynamic
block at the tail of the prompt rather than in the cacheable prefix, so
refreshing it costs nothing beyond your own call.

Keep it cheap anyway — it runs on the turn thread, and an agent that reads and
thinks for ten iterations without changing anything should pay for it once.
Never perform an *effect* from `agent_prompt`: you would invalidate your own
cache and recompute on every call, forever.

Write markdown with an `##` heading, and write it for the weakest model you
expect to run — it is competing for attention with everything else in the
prompt.

> Older plugins spell the method `agent_prompt_for`. That name is gone — a
> plugin still using it contributes nothing to the prompt, silently. Rename it
> to `agent_prompt`; the signature is unchanged.

**Declaring a file makes it importable.** `dependencies_files` names files
from other folders; they join your box's namespace, so you reach them as
siblings:

```python
class Caption(BaseTool):
    dependencies_files = ["parsers/parse_image.py"]

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

### Parsing a heavy modality: declare it

Modalities whose result is a live object — image, audio, video, tabular — can
only be used *inside* the box that holds the parser, because a PIL image or an
open container cannot cross a boundary. Text and extracted paths can.

So you do not move the result, you move the parser. Declare what you need and
the kernel resolves it against whichever parser packages are installed, loads
those files into **your** box before anything runs, and `sdk.parse.file` calls
one directly:

```python
parse_modalities = ["image"]

class OCRImages(BaseTask):
    name = "ocr_images"
    modalities = ["image"]

    def run(self, sdk, paths):
        for path in paths:
            for image in sdk.parse.file(path, "image"):   # live, and local
                scratch = sdk.fs.temp(suffix=".png")
                image.save(scratch, format="PNG")
                sdk.services.call("ocr", "process_image", image_path=scratch)
                sdk.fs.delete(scratch)
        return sdk.ok()
```

Three things follow, and they are the whole contract:

- **Declare nothing and `sdk.parse.file` stays a Request.** The kernel routes
  to the parser and answers with what fits on a wire. This is the cheaper path
  and the right default for text — the parser's dependencies stay out of your
  box. Asking for a heavy modality you did not declare is refused, and the
  refusal names the declaration that fixes it.
- **The kernel resolves, you do not.** Which files provide `"image"` depends on
  what is installed, which your box has no way to know. Naming a modality
  nothing provides is not an error; you find out when you parse a file and are
  told there is no route for that extension.
- **Declaring puts you in a subprocess.** Provisioned parsers are foreign code
  by construction, so the declaration tightens your isolation. That is the
  point of declaring rather than importing the parser file yourself: a relative
  import is invisible to the isolation decision, and would run a C library
  inside the kernel's own process.

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

**Safe work is silent.** Reads, database queries, scratch and agent-workspace
writes, writes inside a user-configured `fs_writable_dirs` folder, and calls to
tools or services do not interrupt anyone. The two write grants are not the
same: workspace is agent-owned; configured writable folders are user-owned and
must be handled as user data. The user is asked for Requests that reach an
unapproved destination or change what the system can do, including other
network requests, shell commands, other filesystem writes, configuration
changes, package installation, and scheduled work.

**Widening capability is always checked; narrowing it never is.** Adding a
tool to a session asks; removing one does not.

**You never see the chain of provenance.** The kernel tracks who called whom
and shows it to the user when it asks. You cannot read it or affect it.

**Name a credential setting `secret_something`.** That prefix is the whole
declaration — the same way `tool_`, `command_` and `service_` prefixes tell
discovery what a file is:

```python
config_settings = [
    ("Provider key", "secret_example_api_key", "API key for this service.", "", {}),
]
```

`sdk.config.read("secret_example_api_key")` then returns
`<secret:secret_example_api_key>` rather than the value. Pass that straight into
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
key = sdk.secrets.reveal("provider_client_secret")   # always asks the user
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
