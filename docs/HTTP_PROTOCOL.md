# The HTTP protocol

Everything a client can do with Second Brain. This is the whole surface — there
is no second API, no direct database, no kernel import. Whatever is not here,
a client cannot do.

Served by the `frontend_http` store package. Hand this document to whoever is
building the client; `docs/http_reference_client.html` is a working example to
check the bridge against when the client misbehaves.

There are two endpoints.

```
GET  /events?thread=<t>&token=<T>      every render, as it happens (SSE)
POST /sdk/<request.type>?thread=<t>    any of the 121 Requests
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

There is no translation layer. These are the same nine payloads a native
frontend receives, so a client that handles all nine can do what the REPL can.

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

### The nine kinds

Handle what you can show and ignore the rest; a client that only renders
`messages` is a working client.

#### `messages` — `list[str]`

GitHub-flavored markdown, including tables and fenced code blocks. This is the
interchange format everywhere in Second Brain; it is also what the model emits,
so one rendering path covers both.

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

#### `approval` — `dict`

The agent (or a plugin) wants permission. **A turn is blocked until this is
answered**, so a client that ignores this kind will appear to hang.

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

```
POST /sdk/frontend.resolve?thread=main
{"value": true, "request_id": "<id>"}
```

Answers `false` if there was nothing left to answer (already resolved
elsewhere, or timed out — dialogs expire after 300s). Treat that as "the
dialog is stale", not as an error.

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

**Talking**

| Request | Arguments |
|---|---|
| `frontend.submit` | `input_kind: "text"`, `text` — chat and slash commands alike |
| `frontend.submit` | `input_kind: "attachment"`, `path`, `file_name`, `caption`, `ingest` |
| `frontend.resolve` | `value`, `request_id` |
| `frontend.cancel` | — stop the current turn |
| `frontend.pending` | — the id of the approval still waiting, or `null` |

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

**Consequential** — all raise a dialog: `config.write`, `plugin.install`,
`plugin.uninstall`, `proc.run`, `session.set_mode`, `agent.spawn`.

---

## 4. Building a client: the short version

1. `GET /events?thread=main&token=…` with `EventSource`. Handle `messages`,
   `stream_delta` and `typing` first — that is a working chat.
2. Wire `approval` early. Without it, the first consequential thing the agent
   does hangs with no explanation.
3. Send chat with `frontend.submit` / `input_kind: "text"`. Slash commands go
   through the same call; the state machine works out which it was.
4. Add `form_field` when you want `/config`, `/packages`, `/llm` and the rest
   to be usable — they are all one generic renderer.
5. Build the rest of the UI from `conv.list`, `command.list`, `session.get`.
6. Render `messages` as GitHub markdown.

### Things that will bite

- **No stream, no dialogs.** Attendance follows the event stream, so a client
  that POSTs before connecting gets silent refusals.
- **`typing: false` is the only end-of-turn signal.** There is no "done" event.
- **Aborted streams have no `final_text`.** Discard the partial.
- **`enum` and `enum_labels` pair by index**, and labels may be absent.
- **Paths are host paths.** `attachments` are not URLs.
- **The replay buffer is 500 frames.** Long disconnects lose the middle.
- **A POST can block on a human.** That is the design, not a hang.
