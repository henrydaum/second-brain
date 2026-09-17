# Migration: the kernel now says who is speaking

**Status:** the kernel and store changes are done and on disk. This client is
the last piece. Nothing here is optional-but-nice — the client currently
*guesses* at three things the wire now states outright, and every guess has a
case it gets wrong.

**Before you start:** re-copy `docs/HTTP_PROTOCOL.md`, `docs/SDK.md`,
`docs/frontend_http.py` and `docs/frontend_template.py` from the kernel repo
(`Z:\My Code\Second Brain\docs\`, and `frontend_http.py` from
`Z:\My Code\SecondBrain-store\frontends\`). The copies in this repo predate the
change and describe eleven render kinds; there are twelve. Read the new
`HTTP_PROTOCOL.md` §"The twelve kinds" and the `conv.read` row shape before
touching code.

---

## What changed, in one sentence

`messages` used to carry five unrelated things. It now carries exactly one —
the conversation, meaning the agent's replies and the person's own words — and
the other four have their own channels.

| What | Was | Now |
|---|---|---|
| The agent's reply, the person's words | `messages` | `messages` (unchanged) |
| A command or tool's return value | `messages` | **`callable_output`** (new kind) |
| A refusal (bad args, unknown command, no access) | `messages`, sometimes ×2 | `error` |
| An announcement (`/new`, restore-on-start) | `messages` | `notification` |
| A kernel-synthesized history row | `role: "user"` | `role: "user"` + **`author`** |

---

## 1. `callable_output` — replaces the timing guess

**This is the big one.** [`src/runtime/store.ts:686`](../src/runtime/store.ts)
currently routes messages into the command panel like this:

```ts
// A running command's output belongs to the command, not the chat.
if (state.command) {
  return { ...state, command: { ...state.command,
    outcome: [...state.command.outcome, ...frame.payload] } };
}
```

That is a guess from *timing*: "a command is open, so this `messages` frame
must be its output." It is wrong whenever anything else renders while a command
is open — a notification landing mid-command, the agent finishing a turn that
overlaps one — and it is wrong in the invisible direction, since the misrouted
text simply appears in the wrong panel.

The kernel now says it. Declare the capability and the frames arrive already
separated:

- Add `supports_callable_output: true` to the capabilities this client declares.
  *(Note: `frontend_http` on the store branch already declares it — check
  whether this client's capabilities come from there or from its own handshake
  before adding anything.)*
- Add `| { kind: "callable_output"; payload: string[] }` to the `Frame` union in
  [`src/lib/events.ts:245`](../src/lib/events.ts).
- Add a `case "callable_output":` to the reducer in
  [`src/runtime/store.ts`](../src/runtime/store.ts) that appends to
  `state.command.outcome` — the same body the `if (state.command)` branch has
  now.
- **Delete the `if (state.command)` branch from `case "messages"`.** Leaving it
  in means command output routed twice once both paths are live.

Payload shape is `string[]` of GitHub-flavored markdown — identical to
`messages`, so your existing renderer works unchanged.

One thing to decide: `callable_output` arrives even when **no command panel is
open**, because a tool the user invoked directly also produces it. Today that
case falls through to the chat. Pick a home for it deliberately rather than
letting it vanish.

## 2. `error` — now carries things that used to be chat

The `error` kind already exists and [`store.ts:821`](../src/runtime/store.ts)
already handles it (`{ ...state, error: frame.payload }`). What changed is
*volume and content*: these now arrive as `error` instead of `messages`.

| `code` | When |
|---|---|
| `bad_command_args` | `/config` with a bad enum value — your original complaint |
| `unknown_command` | `/nosuchthing` |
| `command_not_allowed` | command blocked by this frontend's profile |
| `not_found` | naming a conversation you don't own |
| `no_conversation` | chat action with nothing loaded |
| `no_llm` | first run, before `/setup` |
| `busy` | mid-turn, or a cron handoff landing on an open form |

Two consequences:

- **Check that `state.error` is actually surfaced.** It is set by the reducer;
  make sure something renders it. These messages used to appear in the chat, so
  if the error slot is currently unrendered or transient, this change makes
  them *disappear* rather than move — a silent regression, and the one failure
  mode worth testing by hand.
- **An error now arrives once.** It used to arrive up to three times (the
  kernel set `message` and `error` on the same result, and both were appended
  to `messages`). If anything here dedupes error text against chat text, that
  workaround is now dead code.

## 3. `notification` — two more producers

No new work; the kind is already handled at
[`provider.tsx:561`](../src/runtime/provider.tsx). Just know that "New
conversation started" and "Loaded last conversation" now arrive here instead of
in the chat. Both are `persist: false`, so they banner and never enter the
panel — check the banner path handles them sensibly.

## 4. `author` — replaces reading English

`conv.read` message rows gained an `author` column.

```ts
/** One row of the `messages` table, as `conv.read` hands it over. */
export type StoredMessage = {
  id: number;
  role: string;
  content: string;
  /** Who actually wrote it. `null` for anything a person or the model said —
   *  which is almost every row. Non-null means the kernel synthesized this row
   *  wearing someone else's role. */
  author?: string | null;
  // …
};
```

Values: `command_note`, `doorman_note`, `cancel_notice`, `compaction`,
`truncation`, or a plugin's own name (stamped from the provenance chain, so it
cannot be forged).

**Why you need it.** [`src/lib/history.ts:218`](../src/lib/history.ts) branches
on `message.role === "user"` and renders the row as something the person said.
Six kernel mechanisms write `role: "user"` rows the person never typed, so
right now your scrollback shows these as **your own messages**:

- `[SYSTEM NOTE] The user ran the slash command /clear.` (with
  `reveal_user_commands` on)
- `[The user cancelled the previous turn. Everything it had started…]`
- `[Conversation summary from earlier]…` after any compaction
- `[Earlier conversation dropped to fit context…]` after an overflow
- a doorman hook's feedback note

`prose()` at [`history.ts:126`](../src/lib/history.ts) does not strip any of
them, and it should not start — regexing English is the trap this column exists
to remove.

**The change** is one condition in the `role === "user"` branch:

```ts
if (message.role === "user") {
  if (message.author) continue;   // kernel bookkeeping, not the person
  // … existing body unchanged
}
```

Hiding them outright is the right default: each is addressed *to the model*,
not to a reader. If you'd rather show them, render them as a system aside —
never in the user bubble.

Note `role: "system"` is a **separate** exclusion that still applies: those are
state and compaction markers (packed JSON), not messages. The existing
`if (message.role !== "assistant") continue;` at
[`history.ts:255`](../src/lib/history.ts) keeps handling them.

**Old rows read correctly.** `author` is `null` for every row written before
the migration — there was no backfill, deliberately — so the field being
optional in the type is honest, not defensive.

## 5. Dead workarounds to remove

Once the above is in, check whether these are still earning their place in
[`src/runtime/store.ts`](../src/runtime/store.ts):

- `suppressNextCancellationNotice` (`:170`, `:557`, `:669`)
- `suppressedCommand` (`:167`, `:448`, `:676`, `:723`)
- the regex `/^cancelled\.?$/i` at `:671` and `:678`

They exist to keep a command's `"Cancelled."` echo out of the chat by matching
**English text**. With command output on its own kind, most of what they were
guarding is structural now.

**Do not delete them blind.** One case genuinely still routes through
`messages`: `/cancel` answering "Nothing to cancel." is the return value of the
`session.cancel` Request, and the kernel deliberately kept it on `messages`
because `command_cancel` reads that field to decide what to say. So verify
against the real app rather than reasoning it out — cancel with a command open,
cancel with nothing running, cancel mid-turn.

---

## How to verify

The kernel side is covered by 2433 passing tests; what needs manual checking is
this client against a live app.

1. `/config` with a bad value → an error surface, **not** a chat bubble, and
   visible rather than swallowed.
2. `/config` with no args, completed through the form → output in the command
   panel, arriving via `callable_output`.
3. `/cancel` with nothing running → still says "Nothing to cancel."
4. `/cancel` with a command open → no stray "Cancelled." in the chat.
5. `/new` → a notification banner, no chat line.
6. Turn on `reveal_user_commands`, run `/clear`, reload the page → the
   `[SYSTEM NOTE]` row does **not** appear as your message.
7. Reload a conversation long enough to have compacted → no
   `[Conversation summary from earlier]` bubble.
8. A conversation created right after a `/cancel` → its **title** is your
   actual message, not the cancel notice. (This was broken kernel-side and is
   fixed; worth confirming end to end.)

## Order

1, 4 and 5 are independent — do them in any order. Do **2** before **5**, and
do the `case "callable_output"` and the deletion of the `if (state.command)`
branch in the same commit, or command output routes twice in between.
