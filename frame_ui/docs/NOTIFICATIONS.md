# Notifications

Things the system tells you, as opposed to things the agent says to you: a
plugin registering, a scheduled agent finishing, a setting changing, a
background memory write. They arrive on the render stream as their own kind,
and the persistent ones are also in a table you can read back.

The plan this document is written for: **an ephemeral banner for every
notification as it arrives, plus a panel listing the persisted ones.** Those two
sets are not the same set, which is the one thing worth reading carefully.

---

## Before any of this works

`frontend_http.py` must declare the capability. On the store branch, in the
`capabilities` dict:

```python
capabilities = {
    "supports_typing": True,
    "supports_streaming": True,
    ...
    "supports_notifications": True,      # ← without this, none of the below happens
}
```

**Without it the feature silently does not exist.** The kernel does not drop
notifications for a frontend that has not opted in — it flattens each one into
markdown and sends it as an ordinary `messages` frame, exactly as it did before
this kind was added. So the symptom is not an error: it is plugin
registrations appearing in the chat transcript as if a person had typed them,
and a notification panel that stays empty forever while everything is working
as designed.

The same bargain `supports_streaming` makes for `stream_delta`, and for the
same reason — a client quietly ignoring a kind looks merely quiet.

---

## What is and is not a notification

The kernel draws the line at **who was speaking**. Two things travel out of
band that are *not* notifications, and both are the agent's own turn:

- **Mid-turn narration** — the model's "let me check that file" alongside a
  tool call. Arrives as `messages` (deduped against `stream_delta`).
- **`sdk.ui.render`** — a tool showing you a file. Arrives as `attachments`.

Everything else that reaches you unprompted is a notification: the plugin
watcher, config changes, a scheduled agent's result or failure, compaction
notices.

You do not have to reproduce this rule. It is decided kernel-side at each emit
site, and by the time a frame reaches you the classification is already made.

---

## The frame

```ts
export type NotificationPayload = {
  /** Short header — what happened. This is the collapsed-state line. */
  title: string;
  /** The detail. Multi-line, and **sometimes markdown** — see below. */
  body: string;
  /** Who raised it. Stamped by the kernel — see below. */
  source: string;
  /** Styling only. */
  level: "info" | "success" | "warning" | "error";
  /** Epoch **seconds**, fractional — same units as `LedgerRow.ts`. */
  sent_at: number;

  /** Producer-specific id: a session key, a handle id, a config scope. */
  source_id?: string;
  /** The session it came *from*, which is usually not one you are looking at. */
  source_session_key?: string;
  /** The conversation it is about, when there is one. */
  conversation_id?: number;
  /** A pre-rendered slash command. **Ignore it** — see below. */
  load_hint?: string;
  /** The row id, **absent when this was not persisted**. Read this carefully. */
  notification_id?: number;
};
```

Add it to the `Frame` union in `src/lib/events.ts`:

```ts
export type Frame =
  | { kind: "messages"; payload: MessagesPayload }
  ...
  | { kind: "notification"; payload: NotificationPayload };
```

### `source` is worth showing

It is stamped by the kernel, never stated by whoever raised the notification.
For plugin code it is read off the live provenance chain, which is the part of
a chain a sandboxed plugin cannot state about itself — so a plugin cannot claim
to be `plugin_watcher`. Treat it as trustworthy attribution.

Values you will see in practice: `plugin_watcher`, `config`, `runtime`,
`subagents`, `session`, and the leaf name of whichever plugin called
`sdk.session.push(notify=True)` (e.g. `tool_memory`).

### `body` is inconsistently markdown, and you have to pick a side

Most producers send plain prose — the plugin watcher sends a file stem, the
config announcer sends a comma-separated list of setting names. But
`source: "session"` notifications carry a **background agent's final answer**,
which is the model's own output and therefore GitHub-flavoured markdown, tables
and fenced code included.

Nothing declares which you have. Render `body` through the markdown renderer
you already use for `messages`: plain prose survives it unchanged, whereas
plain-text rendering of an agent's answer shows raw `**bold**` and unformatted
tables. Getting it wrong in that direction is the visible one.

`title` is always plain text and is safe to render as such.

### `load_hint` is not for you

It is a literal `/conversations Main 7 'Load conversation'` string, there for
surfaces whose only affordance is text — the REPL and Telegram. You have
`conversation_id` and can open the conversation yourself. Rendering a terminal
command as the way to open a conversation in a web UI is the failure the field
exists to avoid.

---

## The banner set and the panel set are different

This is the part that will bite an implementation that assumes otherwise.

**Not every notification is persisted.** Transient progress — "Compacting
conversation…", overflow recovery — is delivered and deliberately never
stored, because a panel that fills up with progress lines is one nobody reads.
Those frames arrive with **no `notification_id`**.

So:

| | Source | Contains |
|---|---|---|
| **Banners** | the stream, live | everything, persisted or not |
| **Panel** | `notification.list` + streamed frames | only rows with an id |

Which maps cleanly onto your plan:

```ts
function onNotification(payload: NotificationPayload) {
  showBanner(payload);                       // always
  if (payload.notification_id !== undefined) {
    addToPanel(payload);                     // only if it has a row
  }
}
```

Keying a panel list on `notification_id` is right — but only *after* that
check. Using it as a React key without it means every transient notification
gets key `undefined`.

---

## Reading the panel's contents

Two Requests, both `ALWAYS_SAFE` — they never raise an approval dialog, and
polling them costs nothing worth thinking about.

```ts
import { sdk } from "@/lib/client";

/** One row, as `notification.list` hands it over. Newest first. */
export type Notification = {
  id: number;
  ts: number;                        // epoch seconds, fractional
  title: string;
  body: string;
  source: string;
  source_id: string | null;
  level: string;
  session_key: string | null;
  conversation_id: number | null;
  user_id: number | null;
  /** Epoch seconds when it was settled, or null. This is the unread flag. */
  read_at: number | null;
};

export async function listNotifications(opts: {
  limit?: number;
  since_id?: number;
  unread_only?: boolean;
} = {}) {
  return sdk<Notification[]>("notification.list", opts);
}

/** Returns how many rows actually changed. */
export async function markRead(opts: {
  ids?: number[];
  before_id?: number;
}) {
  return sdk<number>("notification.mark_read", opts);
}
```

Neither takes a `user_id`. Both scope to the calling thread's own user inside
the SQL, so there is no argument to pass and none to get wrong.

`mark_read` requires **either** `ids` or `before_id`. Calling it with neither
is a `400` (`RequestFailed`, code `invalid_argument`) rather than a silent
"settle everything" — worth knowing, because the obvious "dismiss all visible"
implementation passes an array that is empty when the panel is. Guard the call.

It counts only rows that actually changed, so calling it twice is idempotent
rather than double-counted.

```ts
await markRead({ ids: [108, 109] });   // dismiss what was clicked
await markRead({ before_id: 109 });    // "mark all read"
```

---

## Filling the panel

**The stream only ever answers "what happened since you connected."** Anything
from before that — which is most of it, for a client that was closed while a
scheduled agent ran — is only in the table. So the panel needs a backfill on
load, not just a stream subscription.

```ts
const rows = await listNotifications({ limit: 50 });
```

Then keep it current from the stream rather than refetching. If you do both,
**dedupe on `id` / `notification_id`** — a persisted notification that arrives
while you are connected is in both the frame and the table, and a naive
`[...rows, ...streamed]` shows it twice.

An unread badge is `unread_only: true`, or just counting `read_at === null` in
what you already hold.

### On reconnect

`EventSource` replays from `Last-Event-ID` and the buffer holds 500 frames per
session, so a short drop is covered for free. A longer one loses the middle —
the same caveat `events.ts` already documents for every other kind. After a
reconnect that was away a while, prefer `listNotifications({ since_id })` over
trusting the replay:

```ts
const fresh = await listNotifications({ since_id: highestIdWeHold });
```

That is the incremental read the `since_id` argument exists for, and it is an
index seek rather than a scan.

---

## Where to wire it

Notifications belong to the **session**, not to the conversation. A plugin
registering has nothing to do with the thread you are reading, and routing one
through the conversation store means a history read can discard it.

`provider.tsx` already fans exactly this case out at the doorway, for
approvals:

```ts
const close = connect((frame) => {
  if (frame.kind === "approval") { askDispatch(...); return; }
  if (frame.kind === "approval_settled") { askDispatch(...); return; }
  if (frame.kind === "notification") {           // ← same shape of decision
    notifyDispatch({ type: "raised", notification: frame.payload });
    return;
  }
  dispatch({ type: "frame", frame });
}, setStatus);
```

Keeping the split at the doorway is what stops anything downstream from having
to remember that notifications are not conversation.

---

## Levels

`info` · `success` · `warning` · `error`. Styling only — nothing branches on
them kernel-side, and an unrecognised value is normalised to `info` before it
reaches you, so the union is closed.

In practice: `success` for a plugin registering, `error` for a registration
failing or a scheduled agent dying, `info` for everything else. The old text
carried `✓` and `✕` glyphs inside the message; those are gone precisely so a
client can style rather than parse.

---

## Things that will bite

- **No `supports_notifications`, no `notification` frames.** They arrive as
  `messages` instead, which looks like nothing is wrong.
- **`notification_id` is optional.** Transient progress has none. Do not use it
  as a key without checking.
- **The banner set ⊋ the panel set.** They are not two views of one list.
- **`body` is markdown *sometimes*.** A background agent's answer is the model's
  own GFM; a watcher line is plain prose. Render both as markdown.
- **`mark_read` with neither `ids` nor `before_id` is a 400**, not a no-op.
- **`sent_at` and `ts` are seconds, not milliseconds.** Same trap
  `LedgerRow.ts` documents; `Turn.createdAt` is milliseconds.
- **A notification's `conversation_id` is usually not the open one.** That is
  the point — it came from a background session. Offer to switch to it; do not
  assume the user is already there.
- **`source_session_key` is not a delivery target.** It names where the work
  happened, which is a session with no frontend attached. Nothing to route on.
- **Don't render `load_hint`.**

---

## Related

- `docs/HTTP_PROTOCOL.md` — the `notification` kind alongside the other ten,
  and the transport rules that apply to all of them.
- `docs/SDK.md` — `sdk.session.push(notify=True)`, for when you want a plugin
  to raise one.
- `docs/AGENT_FILE_ACTIVITY.md` — the same shape of problem (events are not
  state, so a panel needs a table behind it) solved for files.
