# The HTTP protocol

Everything a client can do with Second Brain. This is the whole surface — there
is no second API, no direct database, no kernel import. Whatever is not here,
a client cannot do.

Served by the `frontend_http` store package. Hand this document to whoever is
building the client; `docs/http_reference_client.html` is a working example to
check the bridge against when the client misbehaves.

There are three endpoints.

```
GET  /events?thread=<t>&token=<T>      every render, as it happens (SSE)
POST /sdk/<request.type>?thread=<t>    any of the 121 Requests
GET  /files?path=<absolute path>       one host file, as a real HTTP body
```

Plus static hosting on `GET /*` when `http_static_dir` is configured, and
`OPTIONS *` for CORS preflight.

---

## 1. Getting connected

**Auth.** Every request carries `Authorization: Bearer <secret_http_token>`,
including static assets. `/events` *also* accepts `?token=`, because a browser's
`EventSource` cannot send headers — that is the API, not an oversight. No other
route accepts a query token; a token in a URL reaches logs and history, so it
buys exactly the one thing that cannot be done without it.

**Threads.** `?thread=main` selects a session, keyed `http:main`. Two threads
are two independent conversations. Omitting it means `default`. **The client
never names a session any other way** — a `session_key`, `token`, or (for the
`session.*` family) `key` in a request body is stripped and replaced, because
identity is the server's to state.

**CORS.** Set `http_allowed_origins` unless the app is served from this same
port. Forget it and the browser refuses the preflight while telling you almost
nothing about why.

**Ports.** The kernel binds loopback only. Expose it with a tunnel.

---

## 2. `GET /events` — the render stream

An SSE stream. Every frame is one `render` call the kernel made, verbatim:

```
id: 41
data: {"kind":"stream_delta","session_key":"http:main","payload":{…}}
```

There is no translation layer. These are the same ten payloads a native
frontend receives, so a client that handles all ten can do what the REPL can.

**Use `EventSource`.** It reconnects on its own and sends back the last `id:`
it saw as `Last-Event-ID`; the server replays from there, so a page refresh
resumes instead of losing the turn it happened during. The buffer holds the
last **500** frames per session — a longer disconnect drops the middle, so a
client that has been away a while should re-read state rather than trust the
replay.

**The stream is the attendance signal.** Opening it declares that a person is
watching this session; a failed push means they left. This is not bookkeeping —
attendance decides whether an unsafe Request raises a dialog or is refused
outright. No stream, no dialogs.

**One stream per thread.** A second `GET /events` for the same thread replaces
the first.

### The twelve kinds

Handle what you can show and ignore the rest; a client that only renders
`messages` is a working client.

#### `messages` — `list[str]`

GitHub-flavored markdown, including tables and fenced code blocks. This is the
interchange format everywhere in Second Brain; it is also what the model emits,
so one rendering path covers both.

**This kind is the conversation and nothing else** — the agent's replies and
the person's own words. Everything that used to be smuggled through it now has
its own kind: a refusal is `error`, an announcement is `notification`, and what
a slash command answered with is `callable_output`. If you are drawing a chat
transcript, `messages` is the only kind that belongs in it.

#### `stream_delta` — `dict`

The reply arriving token by token. Only sent when `supports_streaming` is
declared, which `frontend_http` does.

| Field | Type | Notes |
|---|---|---|
| `stream_id` | `str` | Groups the fragments. Use it as the message key. |
| `seq` | `int` | 1-based, increments per fragment. |
| `delta` | `str` | The fragment. Empty on the final frame. |
| `done` | `bool` | `False` while running, `True` once. |
| `aborted` | `bool` | The stream was cut off. |
| `final_text` | `str` | **Only when `done` and not `aborted`.** |
| `kind` | `str` | Only alongside `final_text`. Usually `"final"`. |

Two rules that matter. On `done` **with** `final_text`, replace what you
accumulated — it is the cleaned text, and the deltas agree with it. On
`done` with `aborted: true` there is **no** `final_text`: discard the partial
render rather than leaving half a sentence on screen.

A `messages` frame may repeat text you already streamed; track `stream_id`s you
have shown and skip the duplicate.

#### `typing` — `bool`

`true` when the agent takes the turn, `false` when it hands back.

**`false` means the *logical* turn ended**, not each internal drive. A turn held
open by a doorman, or re-driven after an escalation, keeps `typing` on until the
whole thing is done. A crash also forces it back. So it is a reliable "the agent
is finished" signal, and the only one.

#### `tool_status` — `dict`

Fires for tool calls and slash commands alike.

| Field | Type | Notes |
|---|---|---|
| `call_id` | `str` | Stable across started/finished. Update in place. |
| `status` | `str` | `"started"`, `"progressed"`, `"finished"`. |
| `kind` | `str` | `"command"` for slash commands; absent for tools. |
| `tool_name` | `str` | Tools only. |
| `command_name` | `str` | Commands only. |
| `args` | `dict` | On `started` / `progressed`. |
| `narration` | `str` | Short human blurb. Repeated on `finished`, deliberately, so a client overwriting one line still has it. |
| `ok` | `bool` | On `finished`. |
| `error` | `str \| None` | On `finished`. |
| `summary` | `str` | On `finished`. What the call amounted to. `""` on failure — `error` is the outcome then — and `""` for a tool that reported nothing. |

`summary` is the tool's own account of its result, capped exactly as the
transcript row is, so a client that renders it live and re-reads it from
`conv.read` shows the same bytes both times. It is written for the model, so
expect a sentence of prose rather than raw output. **`narration` and `summary`
are not interchangeable**: the first is what the agent set out to do, sent on
`started` and repeated here; the second is what came back.

#### `approval` — `dict`

A question the kernel is blocking a turn on.

> **The invariant everything below follows from.** A session in the
> `approving_request` phase is stopped until somebody answers or cancels.
> Nothing times it out except the kernel's own 300-second deadline, and until
> then the agent is not slow — it is waiting on you. Making sure that answer
> happens is the client's job, and a client that ignores this kind is
> indistinguishable from one that has hung.

That single rule is why the dialog cannot be dismissed without settling
something, why the composer should be disabled while one is up (plain text in
that phase is coerced into the *answer*), and why the `approval_settled` frame
below exists.

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | Answer with this. |
| `title` | `str` | The question. |
| `body` | `str` | Arguments, who is asking, and any extra note. |
| `type` | `str` | Usually `"boolean"` or `"string"`. |
| `enum` | `list \| None` | Allowed answers. |
| `enum_labels` | `list \| None` | **Paired with `enum` by index.** May be `None` even when `enum` is not. |
| `default` | any | |

Answer the **value**, show the **label** — putting internal spellings on a
person's buttons is the failure this pairing exists to prevent.

**This frame is the whole notification.** The approval lifecycle sends no
`messages` frame — not when a question is raised, not when it is answered or
refused. It used to, and the prose rode the same kind the agent's own words ride,
so a client with a dialog could not tell them apart and drew "Approval required."
into the chat beside the dialog that already said so. Whatever a person should
read about an approval is the client's to write.

**This kind is not only about permission.** `runtime.request_input` backs all of
it — a sandbox permission gate, `ui.ask`, a tool asking the person a question —
so `type` may be any of `boolean`, `string`, `integer`, `number`, `array` or
`object`, with or without an `enum`. A client that renders this as a permission
prompt will mislabel an ordinary question as a permission grant.

```
POST /sdk/frontend.resolve?thread=main
{"value": true, "request_id": "<id>"}
```

Answers `false` if there was nothing left to answer (already resolved
elsewhere, or timed out — dialogs expire after 300s). Treat that as "the
dialog is stale", not as an error.

#### `approval_settled` — `dict`

A question stopped waiting. `{request_id, reason}`, where `reason` is
`"answered"` or `"cancelled"`.

**The counterpart to `approval`, and the only thing that says a dialog may come
down.** You will get one for a question you answered yourself, and — the reason
this kind exists — for one you did not: another client can answer the same
question, and the kernel denies by name after 300 seconds. Neither is something
you did, and before this frame existed neither was something you could learn
except by asking on a timer.

It says how the question ended, not what the answer was. The answer went to
whoever was blocked on it and is not repeated to a bystander.

##### Getting back to one after a reload

A render is an event, and events are not re-sent on demand. A client that was not
connected when the question was asked — a browser that reloaded, a transport that
dropped — has one route back to it:

```
POST /sdk/frontend.pending?thread=main
{"details": true}

→ {"kind": "approval",   "payload": {id, title, body, type, enum, …}}
→ {"kind": "form_field", "payload": {name, field, collected, display}}
→ null
```

The payloads are the same projections the two renders make, so what comes back
draws the real dialog rather than a reconstruction of one. Both kinds are here
because they are one thing — a session blocked until a person answers — and a
client that restores one but not the other still strands people. Approvals are
reported first, because they nest: a form step can raise one, and the inner
question is the one to answer.

**Ask on every reconnect, not once at boot, and act on `null` as well as on an
answer.** The 500-frame replay usually re-delivers a live `approval`, but it is
a race against your own startup reads, and a kernel restart empties the buffer
entirely. `null` while you are showing a dialog means it was settled while you
were not listening, and the dialog should close.

It is answered from the session's own persisted phase stack when the frontend
has no record of one, so a restart does not report a blocked session as an idle
one. That distinction is the whole reason to prefer this over remembering: a
frontend's memory of what it rendered dies with its process, and the question
does not.

**Once connected, stop asking.** `approval` and `approval_settled` cover the
whole life of a question between them; this call is for the gap a stream cannot
cover, which is what happened while nobody was listening. Polling it on a timer
is a client working around a protocol that could announce a question and not its
end, and that is no longer the protocol you have.

#### `form_field` — `dict`

A command is collecting arguments one step at a time. `{name, field, collected,
display}`.

- `name` — the command or tool being filled in.
- `collected` — arguments gathered so far.
- `field` — the raw step: `name`, `prompt`, `required`, `type`, `enum`,
  `enum_labels`, `default`, `prompt_when_missing`, `columns`.
- `display` — what to actually draw:

| Field | Type | Notes |
|---|---|---|
| `prompt` | `str` | Always present. |
| `assist` | `str` | Hint text. |
| `choices` | `list[{value, label}]` | Empty for free text. Booleans get True/False. |
| `input_mode` | `str` | A hint for the input widget. |
| `allow_skip` | `bool` | The step is optional. |
| `allow_cancel` | `bool` | Effectively always true. |
| `allow_back` | `bool` | Only true once a previous step exists. |

**Answer a form by submitting plain text**, not by a special Request:

```
POST /sdk/frontend.submit?thread=main
{"input_kind": "text", "text": "Main"}
```

The literal strings `/back`, `/skip` and `/cancel` drive the affordances above.
An empty string skips an optional step.

#### `buttons` — `list[dict]`

Quick replies, conventionally `{value, label}` like form `choices`. Nothing in
the kernel currently emits this; it exists for store plugins. Submit the
`value` as text, same as a form choice.

#### `error` — `dict`

`{code, message, details, retry_phase}`. `message` is the human-readable part.

#### `attachments` — `list[str]`

Filesystem paths on the host, not URLs and not bytes. A browser client cannot
open them directly; fetch the contents with `POST /sdk/fs.read_bytes`
(base64-encoded in the response) if you want to display them.

#### `notification` — `dict`

The system telling the user something, as distinct from the conversation
saying it: a plugin registering, a scheduled agent finishing, a setting
changing, a background write completing.

```json
{"title": "Nightly index finished", "body": "Indexed 12 files.",
 "source": "session", "source_id": "spawn_subagent:41", "level": "info",
 "conversation_id": 41, "notification_id": 108,
 "load_hint": "/conversations Main 41 'Load conversation'",
 "sent_at": 1765412880.4}
```

**You only get this kind if you ask for it.** Declare
`supports_notifications` in `capabilities`; without it the kernel flattens
each notification into markdown and sends it as a `messages` render, which is
what every frontend saw before this kind existed. That is the same bargain
`stream_delta` offers, and for the same reason — a client that quietly ignored
the kind would look merely quiet rather than broken.

`source` is stamped by the kernel, never by whoever raised it. For a plugin it
is read off the live provenance chain, so a plugin cannot claim to be the
plugin watcher. Treat it as trustworthy attribution and show it.

`level` is `info` | `success` | `warning` | `error` and only styles the result.

`load_hint` is a pre-rendered slash command, there for surfaces with no better
way to reach a conversation. **Ignore it** — you have `conversation_id` and can
open the conversation yourself; rendering a terminal command in a web UI is the
failure it exists to avoid.

`notification_id` is absent when the notification was not persisted (transient
progress, e.g. "Compacting conversation…"). Everything else is in the
`notifications` table and can be read back — see below.

#### `callable_output` — `list[str]`

What a slash command or a user-invoked tool **returned**: a `/config` listing,
a `/conversations` table, a `/debug` dump. GitHub-flavored markdown, same wire
convention as `messages`.

Separate because it is the answer to something the person typed rather than
something anybody said. It was much the largest population making `messages`
unreadable to a client — the agent's reply and a settings table arrived as the
same kind of thing, with no field to tell them apart.

**You only get this kind if you ask for it.** Declare
`supports_callable_output` in `capabilities`, which `frontend_http` does;
without it the kernel sends command output as a `messages` render, exactly as
every frontend saw it before this kind existed. Same bargain as
`notification` and `stream_delta` above.

A good client gives this its own treatment — a collapsible block, a monospace
panel beside the transcript — rather than a chat bubble. It is output, not
speech.

#### Filling the panel on a fresh load

The stream only ever answers "what happened since you connected". A panel that
draws notifications needs the rest:

```
POST /sdk/notification.list    {"limit": 50}
POST /sdk/notification.list    {"since_id": 108}      # incremental
POST /sdk/notification.list    {"unread_only": true}
POST /sdk/notification.mark_read {"ids": [108, 109]}
POST /sdk/notification.mark_read {"before_id": 109}   # mark all read
```

Both are scoped to the thread's own user in SQL — there is no `user_id`
argument to pass and none to get wrong. `mark_read` answers with how many rows
actually changed, so calling it twice is idempotent rather than double-counted.

---

## 3. `POST /sdk/<type>` — doing things

The body is the Request's arguments as JSON; the answer is its result.

```
POST /sdk/conv.list?thread=main
{"details": true}

200 {"data": [...]}
```

Failures carry the kernel's own error code:

```
403 {"error": "denied: config.write …", "code": "approval_declined"}
```

| Status | Meaning |
|---|---|
| 200 | `{"data": …}` |
| 400 | Bad arguments, or an unknown Request type |
| 403 | Refused — declined, or nobody was there to ask |
| 404 | The thing does not exist |
| 499 | Cancelled |
| 503 | That subsystem is not available in this kernel |
| 504 | Timed out |
| 500 | Anything else |

**Requests are not pre-filtered.** There is no allowlist; policy decides, the
same way it decides for a tool or a script. Two exceptions, refused by the
kernel: `frontend.act`/`frontend.collect` (recursion) and the whole `http.*`
family, which belongs to the transport rather than to any session.

### Unsafe Requests raise a dialog

Anything consequential — `config.write`, `conv.delete`, a gated
`command.call` — is not simply refused. It raises a real approval, which
arrives **on your event stream** as an `approval` frame while the POST is still
open. Answer it with `frontend.resolve` and the original POST completes with
the result.

So a client needs the `approval` handler wired before it tries anything
interesting, and the POST may legitimately take as long as a person takes.

This works only while the session is attended, i.e. while `/events` is open.
Without a stream, unsafe Requests come back `403 approval_declined` with nobody
having been asked.

### The Requests a client actually wants

The full vocabulary is 121 types; `docs/SECURITY_CONTRACT_APPENDIX.md`
catalogues all of them with their policy inputs. The useful subset:

**Conversation**

| Request | Arguments |
|---|---|
| `conv.list` | `category`, `limit`, `details` |
| `conv.read` | `id`, `details` |
| `conv.create` | `title`, `category`, `activate` |
| `conv.load` | `id` |
| `conv.set_title` | `id`, `title` |
| `conv.set_category` | `id`, `category` |
| `conv.clear` | `id` |
| `conv.delete` | `id` — **unsafe, raises a dialog** |

A `conv.read` message row is
`{id, conversation_id, role, content, tool_call_id, tool_name, timestamp,
attachments, author}`.

**`author` is who actually wrote the row**, and it is `null` for almost all of
them, which is the answer "the row is what its `role` says". A non-null value
means the kernel synthesized the row wearing somebody else's role — the values
it uses are `cancel_notice`, `doorman_note`, `command_note`, `compaction` and
`truncation`, and a plugin's `conv.append` is stamped with the plugin's own
name. `role` cannot carry this because `role: "system"` was already taken by
state and compaction markers, which are not messages at all and should be
skipped. Do not render an authored row as something the person said; hiding
them outright is a reasonable default, since each one is addressed to the
model rather than to a reader.

**`attachments` is the files that message carried** — a list of
`{path, file_name, modality, extension}`, empty for the overwhelming majority
of rows — and `content` is what the person actually typed. They used to be one
field: the pointer line `[Attached image file: chart.png (cached at …)]` was
welded onto the text, so the only way to know a message had a file was to parse
prose, and a person typing those characters looked exactly like an attachment.
Render them however your UI renders a file; `GET /files?path=` turns one into
something an `<img>` or a `<video>` can load.

Conversations written before the column keep the old welded line in `content`
and have no `attachments`. Nothing rewrote them — guessing where prose ends and
a file begins is not worth doing to somebody's own words — so a client that
wants to show those has to accept the sentence as-is.

**Talking**

| Request | Arguments |
|---|---|
| `frontend.submit` | `input_kind: "text"`, `text` — chat and slash commands alike |
| `frontend.submit` | `input_kind: "attachment"`, `path`, `file_name`, `caption`, `ingest` — one file |
| `frontend.submit` | `input_kind: "attachment"`, `files: [{path, file_name, extension, is_photo, caption}]`, `caption`, `ingest` — one message, several files |
| `frontend.resolve` | `value`, `request_id` |
| `frontend.cancel` | — stop the current turn |
| `frontend.pending` | — the id of the approval still waiting, or `null`. With `details: true`, the question itself as `{kind, payload}` — approval **or** form step — which is how a reconnecting client gets back to one |

**Introspection**

| Request | Arguments |
|---|---|
| `command.list` | `details`, `visible` — build a command palette from this |
| `command.call` | `name`, `args` — structured, no string parsing |
| `session.get` | `details` — includes `phase` and `attended` |
| `config.read` | `key` (a *setting* name), `keys`, `present`, `details` |
| `tool.list`, `llm.list`, `plugin.list`, `task.list`, `cron.list` | |
| `user.read` | |

Note `session.get`'s `key` argument is the *session*, and is overwritten with
your thread — as it is for the whole `session.*` family. `config.read`'s `key`
is a setting name and is left alone.

**Files**

| Request | Arguments |
|---|---|
| `fs.read` | `path` |
| `fs.read_bytes` | `path`, `offset`, `length` — answers **base64** |
| `fs.list` | `path`, `glob`, `recursive`, … |
| `fs.search` | `path`, `pattern`, `regex`, … |
| `fs.stat` | `path` |

One answer has to fit in one wire message and base64 costs a third on top, so a
whole-file `read_bytes` is capped well below the sizes a UI deals in. Ask for
successive `offset`/`length` windows and join them; a short read means you
reached the end, so the loop terminates without your having to learn the size
first.

### Sending files

There is no upload route, and there does not need to be one: a file becomes a
path first, and the path is what a submit carries.

```
POST /sdk/fs.temp        {"suffix": ".png"}          → a scratch path
POST /sdk/fs.write_bytes {"path": …, "data": "<base64>"}
POST /sdk/fs.write_bytes {"path": …, "data": "<base64>", "mode": "append"}
POST /sdk/frontend.submit {"input_kind": "attachment",
                           "files": [{"path": …, "file_name": "chart.png"},
                                     {"path": …, "file_name": "notes.pdf"}],
                           "caption": "what do these have in common?",
                           "ingest": true}
```

Scratch is a safe write, so none of this raises a dialog. One `write_bytes`
message is capped exactly as one `read_bytes` answer is, so a large file goes
up in `append` chunks. `ingest` then moves each file into the attachment cache
— a watched directory, so the pipeline indexes it like anything else — and
removes the scratch copy.

**Attach several files in one submit, not several submits.** A submitted
attachment hands the turn to the agent immediately, so a second submit arrives
at a session that is already busy and comes back `busy`. `files` is the whole
message: one action, one turn, and the model sees every file in the same call.
`caption` is the line the person typed and belongs to the message — it is
recorded once, on the first file, rather than repeated per file. A file may
still carry its own `caption` when the transport gave it one (Telegram does).

The one-file form (`path`, `file_name`, `caption`, `ingest` at the top level)
is unchanged and still works; `files` with one entry means exactly the same
thing.

## 4. `GET /files` — a host file as a URL

Everything else here answers JSON. This answers **bytes**, because some things
are not renderable any other way.

`fs.read_bytes` already reads any file a client is allowed to read, so this
grants nothing new — it is the same read, through the same policy, recorded in
the same ledger. What it adds is a *transport*. A Request answers base64 inside
JSON, and `<img>`, `<video>` and `<audio>` want a URL. Rebuilding a Blob works
for a picture and is hopeless for media: it buffers the whole file before the
first frame and cannot seek.

```
GET /files?path=%2Fsrv%2Fapp%2Fchart.png
Authorization: Bearer <secret_http_token>
```

Percent-encode the path (`encodeURIComponent`) — it is an absolute host path,
and on Windows it contains `\` and `:`. Same bearer token as every other route.

| Answer | When |
|---|---|
| `200` + body | The whole file, when it fits in one message |
| `206` + `Content-Range` | A `Range` was asked for, **or** the file is larger than one message |
| `400` | No `?path=`, or it names a directory |
| `401` | Missing or wrong token |
| `403` | Policy refused the read |
| `404` | No such file |
| `416` + `Content-Range: bytes */<size>` | The range starts past the end |

`Accept-Ranges: bytes` is always sent, and `Range` is honored in the single-range
forms (`bytes=0-1023`, `bytes=1024-`, `bytes=-4`). That is what lets a `<video>`
seek instead of downloading everything before the point you clicked.

**A large file always comes back `206`, even with no `Range` header.** One
response body crosses the box boundary in one wire message and that message is
capped, so the route serves the first window and advertises the real total in
`Content-Range`. Media elements follow up on their own; a plain `fetch` should
loop on `Content-Range` until it has the whole length. `HEAD` answers the full
`Content-Length` without a body, which is the cheap way to size a file first.

**Every extension works.** The bytes are served whatever the file is; the
extension only decides the `Content-Type` label. A small explicit table is
consulted first — so the answer is identical on every host — then Python's
`mimetypes`, then `application/octet-stream`, which the browser treats as a
download rather than a guess.

Every extension in the kernel's own modality map (`parsing._NATIVE_DEFAULTS`,
what `parse.modality` answers from) is guaranteed a label whose top-level type
matches its modality, so a client that categorises by modality and then picks
an element will never hand a `<video>` something it refuses to play. That
agreement is pinned by
`test_every_native_modality_gets_a_playable_type` — two tables answering one
question is how they drift.

Use it for anything the browser renders natively: images, video, audio, PDF,
SVG, plain text. For formats it cannot — `.docx`, `.xlsx`, `.pptx` — use
`parse.file` with `modality: "text"` instead; that is what the parser packages
are for, and text is one of the two things a parse result may carry.

---

**What the agent did** — the flight recorder, and the only place some of it is
kept.

| Request | Arguments |
|---|---|
| `ledger.read` | `conversation_id`, `action_types`, `since_id`, `origin`, `session_key`, `limit` |

Renders are events, not state, so anything a frontend only *saw* is gone on
reload — and two of those things are worth getting back. Files the agent
**edited** arrive as `attachments` render frames and are recorded as
`origin: "sandbox"` rows for `fs.write` / `fs.write_bytes` / `fs.delete` /
`fs.move`; files it **showed** you are on the `origin: "agent_enact"` row for the
tool call, under `data_json.attachments`. Neither is in `conv.read` — they are
things the agent *did*, not part of any message.

Files the **person** sent are the opposite case and are on the message itself:
`conv.read` gives every row an `attachments` list (see below). Don't go to the
ledger for those.

Shell commands count too. A successful `proc.run` / `proc.start` whose line is
a recognised file command — `rm`, `mv`, `cp`, `mkdir`, `touch`, `rmdir`, `ln`
and the Windows spellings — records the paths it touched, tagged
`data_json.via: "shell"` because a path read out of a command line is a weaker
claim than one the kernel serviced. `data_json.deleted` is the subset that no
longer exists. Anything it cannot read honestly records no paths at all: an
unlisted program, a glob (`rm *.log` names nothing until a shell expands it),
a redirect, `$(…)`, a subshell, or a command that exited non-zero.

Read the paths from `data_json.paths`, never by parsing `args_json` — that field
is capped, and past the cap the object is replaced by a `head`/`tail` wrapper.
The argument that blows the cap is the file's own contents, so parsing it would
lose exactly the largest edits. `data_json.bytes` carries the size of a
successful write.

```jsonc
// "which files has this conversation touched?"
POST /sdk/ledger.read
{"conversation_id": 7, "action_types": ["fs.write", "fs.delete", "fs.move"]}
```

Rows come back newest first. Keep the highest `id` you have seen and pass it as
`since_id` to ask only for what followed, rather than re-reading the
conversation every time a `tool_status` frame lands. A row with `ok: 0` and
`error_code: "approval_declined"` is a change you refused — worth showing, since
nothing else records that it was attempted.

Naming a conversation you do not own is refused, not asked about.

**Consequential** — all raise a dialog: `config.write`, `plugin.install`,
`plugin.uninstall`, `proc.run`, `session.set_mode`, `agent.spawn`.

---

## 5. Building a client: the short version

1. `GET /events?thread=main&token=…` with `EventSource`. Handle `messages`,
   `stream_delta` and `typing` first — that is a working chat.
2. Wire `approval` **and `approval_settled`** early. Without the first, the
   first consequential thing the agent does hangs with no explanation; without
   the second, a question somebody else answers leaves a dialog on your screen
   that can no longer be answered.
3. Send chat with `frontend.submit` / `input_kind: "text"`. Slash commands go
   through the same call; the state machine works out which it was.
4. Add `form_field` when you want `/config`, `/packages`, `/llm` and the rest
   to be usable — they are all one generic renderer.
5. Call `frontend.pending {details: true}` on every reconnect — not on a timer —
   and act on `null` as well as on an answer. Without this, an unanswered
   question does not survive a page reload.
6. Build the rest of the UI from `conv.list`, `command.list`, `session.get`.
7. Render `messages` as GitHub markdown.
8. Declare `supports_notifications` and give `notification` its own area —
   otherwise plugin registrations and scheduled-agent results land in the chat
   alongside the agent's replies, which is where they used to have to go.
   Backfill the area with `notification.list` on load; the stream only carries
   what happened while you were connected.
9. Declare `supports_callable_output` and give it somewhere that is not a chat
   bubble. Between this and step 8, `messages` is left meaning exactly one
   thing — the conversation — which is what makes a transcript view possible
   at all.
10. When you read history back with `conv.read`, honour `author` (below).
    A row with one is not something the person said.

### Things that will bite

- **No stream, no dialogs.** Attendance follows the event stream, so a client
  that POSTs before connecting gets silent refusals.
- **`typing: false` is the only end-of-turn signal.** There is no "done" event.
- **Aborted streams have no `final_text`.** Discard the partial.
- **`enum` and `enum_labels` pair by index**, and labels may be absent.
- **Paths are host paths.** `attachments` are not URLs.
- **The replay buffer is 500 frames.** Long disconnects lose the middle.
- **A POST can block on a human.** That is the design, not a hang.
- **Renders are events, not state.** Nothing is re-sent because you asked; a
  question you were not connected for is reachable only through
  `frontend.pending {details: true}`.
- **`approval` is not only permission.** The same kind carries any question a
  tool asks, so do not word the dialog as a permission grant.
- **`notification` needs `supports_notifications`, and `callable_output` needs
  `supports_callable_output`.** Without them you still get everything,
  flattened into `messages` renders — which looks like nothing is wrong and is
  the reason the opt-ins are worth checking.
- **`role` does not tell you who wrote a history row.** `conv.read` returns
  rows the kernel synthesized in the person's slot — a cancel notice, a
  doorman's note, the summary bridge compaction leaves behind, a note that a
  slash command ran. Every one of them has a non-empty `author`; a row the
  person actually typed has `author: null`. Drawing a transcript on `role`
  alone puts the kernel's own bookkeeping in the user's bubble. Note
  `role: "system"` is separate again — those are state markers, not messages,
  and should be skipped outright.
- **Not every notification is persisted.** `notification_id` is absent for
  transient progress, so keying a panel's list on it will silently drop those.
- **You are not the only one who can answer.** A question raised on your session
  can be settled from another client or by the 300s timeout, which is what
  `approval_settled` is for.
