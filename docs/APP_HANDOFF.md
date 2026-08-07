# Handoff — building the Second Brain web app

Copy everything below the line into a fresh agent session in the new, empty
repo. It assumes nothing about Second Brain being present on disk; the contract
it needs is stated here.

---

## Who you are working with

Henry is an experienced Python developer and the sole author of Second Brain, a
local-first AI agent kernel. **He does not know React.** Treat that as the
central fact of this project:

- Explain React concepts when they first matter, briefly, in the flow of the
  work — not as a tutorial up front.
- Prefer boring, legible React over clever React. No custom hooks where a
  function will do, no state libraries, no abstraction he would have to learn
  in order to change a colour.
- When you make a structural choice (where state lives, why a component
  re-renders), say why in a sentence. He will be maintaining this.
- He writes unusually thorough code comments explaining *why*, and expects the
  same. Match that.

He works on Windows; the server runs on a Mac Mini.

## What you are building

A dedicated desktop/web client for an AI agent, built on **assistant-ui** and
speaking the **AG-UI protocol**. This replaces Telegram, which was never
designed to host an agent — its commands are clunky and its buttons round-trip
through Telegram's servers before anything happens.

The goal in one line: **the chat is only ever between the person and the agent;
everything administrative gets its own surface.**

### Stack

- React + TypeScript + Vite
- `@assistant-ui/react` and `@assistant-ui/react-ag-ui`
- Tailwind is fine if assistant-ui's setup wants it; do not add a component
  library on top
- No backend of your own. The server already exists.

Pin `@assistant-ui/react-ag-ui` early and **read its source** (`src/runtime/`)
rather than trusting docs — the docs omit real API. That is how the contract
below was established.

## The server side, which already works

Second Brain runs on the Mac Mini and serves this on loopback, exposed through
a tunnel. **One endpoint plus a small read API is the entire connection.** There
is no second API, no database access, no SDK.

Every request carries `Authorization: Bearer <token>`, including static assets.
`OPTIONS` is answered without a token (a browser sends none on preflight).
CORS is off unless the server sets `agui_allowed_origins`, so during development
either set that to your Vite origin or serve the built app from the server's own
static directory.

### `POST /agui` — the conversation

Standard AG-UI. Body is `RunAgentInput`:

```jsonc
{
  "threadId": "default",     // selects the conversation; opaque to you
  "runId": "run-abc",
  "messages": [{ "id": "m1", "role": "user", "content": "hello" }],
  "resume": [],              // see interrupts below
  "state": {}, "tools": [], "context": [], "forwardedProps": {}
}
```

Response is `text/event-stream`, `data: {json}` per frame, one run per POST,
ending in `RUN_FINISHED`.

The server keeps its own history, so it reads **only the last user message**.
Resending the whole thread is harmless but pointless.

Events you will receive: `RUN_STARTED`, `TEXT_MESSAGE_START/CONTENT/END`,
`TOOL_CALL_START/ARGS/END/RESULT`, `RUN_ERROR`, `RUN_FINISHED`, and `CUSTOM`.
Text is **GitHub markdown, verbatim** — tables and fenced code included.

### The read API

AG-UI describes a conversation and nothing else, so everything a real client
needs beyond chat is plain REST beside it:

| | |
|---|---|
| `GET /conversations` | list — this is what populates a thread list |
| `GET /conversations/{id}` | messages and metadata |
| `POST /conversations` | `{title, activate}` |
| `POST /conversations/{id}/load` | point the session at a conversation |
| `GET /commands` | every slash command, with its form metadata |
| `GET /config` | settings; secrets come back as opaque handles |
| `GET /session?thread=X` | session and user state |

**There is deliberately no write API.** Deleting a conversation, changing a
setting, installing a package — all of it happens by sending the slash command
as an ordinary chat message (`/config`, `/packages install x`). That is not a
limitation to work around: it is the only path that earns the server's approval
gates, so a destructive action asks the person first. Anything you tried to do
directly would be silently refused.

## Interrupts — the part that matters most

The agent stops and asks the person things: permission to run a shell command,
a value for a form field, a choice among options. This is expressed as an
**AG-UI interrupt**, and handling it is not optional — a client that ignores
interrupts leaves the agent blocked on a question nobody can see, which looks
exactly like the app hanging.

A run that stops short ends like this:

```jsonc
{
  "type": "RUN_FINISHED",
  "threadId": "default", "runId": "run-abc",
  "outcome": {
    "type": "interrupt",
    "interrupts": [{
      "id": "a1",
      "reason": "confirmation",        // or "input_required"
      "message": "Run shell commands\n\ngit status",
      "responseSchema": {
        "type": "object",
        "properties": { "value": { "type": "string",
                                   "enum": ["allow", "deny"],
                                   "enumLabels": ["Allow", "Deny"] } },
        "required": ["value"]
      },
      "metadata": { "second_brain": { "kind": "approval", "request": { } } }
    }]
  }
}
```

You answer by POSTing the **next** run with a `resume` array:

```jsonc
{ "threadId": "default", "runId": "run-def", "messages": [],
  "resume": [{ "interruptId": "a1", "status": "resolved",
               "payload": { "value": "allow" } }] }
```

`status: "cancelled"` is a real answer (it denies, rather than leaving the agent
waiting) — wire it to closing the dialog.

That resumed run **stays open** and the unblocked turn renders into it. So a
resume is a continuation, not a separate conversation.

assistant-ui supports all of this natively: the AG-UI runtime parses interrupts
into `AgUiInterrupt`, and the runtime extras give you `interrupts` and
`submitInterruptResponses(entries)`. Use those rather than hand-rolling fetch
calls. `useAgUiSubmitInterruptResponses` is the hook to look for.

### Three reasons, three UIs

- `confirmation` — a permission dialog. **This is a safety surface.** Show the
  full `message`, make Allow and Deny equally easy to hit, never default or
  auto-dismiss. Some carry an `enum` with more than two options (things like
  "allow once" versus "always allow this host") — render every one.
- `input_required` with an `enum` — a choice. Render one button per entry.
- `input_required` without — a text/number input, typed from
  `responseSchema.properties.value.type`.

**`enum` and `enumLabels` pair by index. Answer with the value, display the
label.** Getting this backwards puts internal spellings like
`always:api.search.brave.com` on a person's buttons. This has bitten this
system before.

### The metadata escape hatch

Every interrupt carries `metadata.second_brain` with the server's own richer
payload. For `input_required` from a form it looks like:

```jsonc
{ "kind": "form_field",
  "form": { "field":   { "name": "port", "type": "integer", "enum": null,
                         "enum_labels": null, "default": 8787,
                         "required": true, "columns": 2 },
            "display": { "prompt": "Which port?", "assist": "…",
                         "choices": [{ "value": "x", "label": "X" }],
                         "allow_back": true, "allow_skip": false,
                         "allow_cancel": true } } }
```

Build the good version from this — labelled choices, a Back button, Skip when
`allow_skip`, Cancel. But **render correctly from the generic fields first**,
then enrich. That ordering keeps the app honest about the protocol and means a
missing metadata field degrades instead of crashing.

The payoff is large and worth understanding: **one form renderer covers every
admin screen.** `/config`, `/packages`, `/permissions`, `/mode`, `/schedule`,
`/llm`, `/conversations` all arrive as this same shape. You are not building
twenty screens.

### CUSTOM events

`{ "type": "CUSTOM", "name": "second_brain.attachments", "value": [...] }` —
file paths the agent produced, and `second_brain.tool_progress` for mid-tool
updates. Display-only; nothing answers them. Reachable via the runtime's
`onCustomEvent`. Safe to ignore in v1.

## What to build, in order

1. **Chat that works.** assistant-ui + the AG-UI runtime pointed at `/agui`,
   with the bearer token. Streaming text, tool call indicators, markdown. Stop
   here and confirm a real conversation works before anything else.
2. **Interrupts.** Approvals first — that is the safety surface and the thing
   Telegram did worst. Then the generic form renderer.
3. **Thread list** from `/conversations`, with create and switch.
4. **A command surface.** `/commands` gives you the catalogue; running one is
   just sending its text. This is the feature that motivated the whole project:
   real buttons instead of typing `/packages install foo`.
5. **Settings view** from `/config`, read-only at first — remember writes go
   through `/config` as a command, which will come back as a form interrupt.

## Getting it running

Ask Henry for the base URL and the bearer token. Do not commit the token; put
it in `.env.local` (gitignored) and read it via `import.meta.env`.

There is a **reference client** in the Second Brain repo at
`docs/agui_reference_client.html` — one file, no build step, renders only the
generic protocol fields. When something misbehaves, open that first: if it works
there, the server is fine and the bug is in this app. Ask Henry for it if you
want to read it; it is the shortest complete statement of the contract.

## Things that will waste your time if nobody says them

- **`RUN_FINISHED` does not mean the turn is over** when it carries an
  interrupt outcome. The agent is still mid-turn, blocked, waiting for you.
- **A stream that stays open and goes quiet** usually means an interrupt you
  did not handle.
- **The server is single-user by default.** Do not build login; the bearer
  token is the whole perimeter for now.
- **Do not add a state library.** assistant-ui owns conversation state; the
  read API is fetch-on-demand. Anything else is a second source of truth.
- **Read the react-ag-ui source when the docs are silent.** The documented
  runtime options omit `onCustomEvent`, which does exist. Assume the docs are
  incomplete rather than the feature missing.

## How to work with Henry

Show him running software early and often — he would rather see a working chat
box than a plan for one. When you hit a genuine fork (how the admin surface is
laid out, whether approvals are modal), ask; he has strong and well-reasoned
opinions about interaction design. When it is a React convention he has no stake
in, just pick one and say what you picked.
