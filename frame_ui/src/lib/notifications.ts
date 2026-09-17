/**
 * The notifications table, and how to read it back.
 *
 * **Frames are events; the table is state.** A `notification` frame only ever
 * answers "what happened since you connected", and for a client that was closed
 * while a scheduled agent ran that is none of it. So the panel is sourced from
 * here and merely *kept current* by the stream — the same division `lib/ledger.
 * ts` draws for files, and for the same reason: a panel sourced from frames
 * alone is a panel that empties when the page refreshes.
 *
 * Both Requests are `ALWAYS_SAFE`, so neither raises an approval dialog, and
 * `notification.list` is `READ_ONLY` besides — it writes no ledger row per call
 * and does not bump `sandbox.epoch`, which is what makes refilling on every
 * reconnect cost nothing worth thinking about.
 *
 * Neither takes a `user_id`. Both scope to the calling thread's own user inside
 * the SQL, so there is no argument to pass and none to get wrong.
 */

import { sdk } from "@/lib/client";
import type { NotificationPayload } from "@/lib/events";

/** One row, as `notification.list` hands it over. Newest first. */
export type Notification = {
  /** Monotonic, and also the `since_id` cursor. */
  id: number;
  /** Epoch **seconds**, fractional. Use `atOf`. */
  ts: number;
  title: string;
  body: string;
  source: string;
  source_id: string | null;
  /** Widened from the frame's closed union on purpose: this comes back out of
   *  SQLite as whatever was written, and `levelOf` is what narrows it. */
  level: string;
  session_key: string | null;
  conversation_id: number | null;
  user_id: number | null;
  /** Epoch seconds when it was settled, or null. **This is the unread flag.** */
  read_at: number | null;
};

/** Whether anything here still wants attention. */
export const isUnread = (row: Notification) => row.read_at === null;

/**
 * A wire timestamp as a `Date`.
 *
 * **The wire speaks seconds and `Turn.createdAt` speaks milliseconds**, and the
 * two meet wherever a notification is drawn next to anything else. One helper so
 * the multiplication has a single place to be got wrong rather than one per call
 * site — the same trap `lib/ledger.ts` documents on `LedgerRow.ts`.
 */
export const atOf = (seconds: number): Date => new Date(seconds * 1000);

/** The frame's four levels, narrowed from whatever the table returns. The
 *  kernel normalises unrecognised values to `info` before sending, so this is
 *  only really guarding the *stored* column. */
export type Level = NotificationPayload["level"];
const LEVELS: readonly string[] = ["info", "success", "warning", "error"];
export const levelOf = (level: string): Level =>
  (LEVELS.includes(level) ? level : "info") as Level;

/**
 * A persisted frame in row shape.
 *
 * The stream and the table describe the same thing with different field names
 * (`sent_at`/`ts`, `source_session_key`/`session_key`), and a panel that held
 * both shapes would need every consumer to know which one it had. Converting at
 * the edge means there is one type downstream.
 *
 * **Only call this when `notification_id` is set.** A transient notification has
 * no row and never belongs in the list; the caller does that check, because the
 * decision is the caller's — see `reduceNotifications`.
 */
export function rowFromFrame(
  payload: NotificationPayload,
  id: number,
): Notification {
  return {
    id,
    ts: payload.sent_at,
    title: payload.title,
    body: payload.body,
    source: payload.source,
    source_id: payload.source_id ?? null,
    level: payload.level,
    session_key: payload.source_session_key ?? null,
    conversation_id: payload.conversation_id ?? null,
    // Not on the frame, and not worth inventing: the row is this user's by
    // construction, since the kernel only streamed it to this user's frontend.
    user_id: null,
    // A frame is the notice that it *happened*, which is the definition of
    // unread. The backfill will carry the real value if it was settled
    // elsewhere in the meantime.
    read_at: null,
  };
}

/**
 * The settings a "Settings changed" notification is about.
 *
 * The config announcer puts a comma-separated list of setting keys in `body` —
 * `"scheduled_jobs"`, or `"http_port, http_token"`. That is a convention rather
 * than a field, so this reads it defensively: anything that is not shaped like a
 * setting key is dropped rather than pasted into a command line.
 *
 * **The filter is what makes the result safe to interpolate.** A key is an
 * identifier, so a name that survives this has no spaces and needs no quoting —
 * which is the property `/config all <name>` depends on. A body that turns out
 * to be prose yields nothing, and the caller falls back to the section.
 *
 * Only meaningful for `source: "config"`; every other producer uses `body` for
 * something else entirely.
 */
export function settingNamesOf(body: string): string[] {
  return body
    .split(",")
    .map((name) => name.trim())
    .filter((name) => /^[a-z][a-z0-9_.]*$/i.test(name));
}

/**
 * Whether `/config` can actually open this setting.
 *
 * **Being a real setting is not enough.** A declaration carrying `hidden` is
 * left out of `/config`'s catalogue on purpose — `llm_profiles` is managed
 * through `/llm`, `scheduled_jobs` through the timekeeper — and the command
 * validates `setting_name` against exactly that catalogue. So the name a
 * "Settings changed" notification carries is a name `/config` may well refuse:
 * `announce_config_change` reports whatever was written, hidden or not, which is
 * right for an announcement and wrong for a command line. Sending one anyway
 * printed the enum of every settable key into the chat as an error.
 *
 * `config.read` with `details` is the catalogue itself, filtered server-side by
 * `key` so the answer is one item or none rather than the whole list.
 *
 * **A failed read answers "no".** The fallback is the Configuration section,
 * which is merely less specific; the other way to be wrong is a rejected
 * command in the transcript.
 */
export async function settingIsBrowsable(name: string): Promise<boolean> {
  try {
    const found = await sdk<{ key?: string }[] | null>("config.read", {
      details: true,
      key: name,
    });
    return (found ?? []).some((item) => item.key === name);
  } catch {
    return false;
  }
}

/** Newest first, like the table. `since_id` is an index seek rather than a
 *  scan, which is what makes the reconnect top-up cheap. */
export function listNotifications(
  opts: { limit?: number; since_id?: number; unread_only?: boolean } = {},
): Promise<Notification[]> {
  return sdk<Notification[]>("notification.list", opts);
}

/**
 * Settle rows. Answers how many actually changed, so calling it twice is
 * idempotent rather than double-counted.
 *
 * **Neither argument is a `400`, not a no-op** — the kernel refuses to settle
 * the whole table by omission. That is easy to hit by accident: the obvious
 * "dismiss everything visible" passes an `ids` array that is empty exactly when
 * the panel is, so this refuses locally rather than letting a `RequestFailed`
 * out for a call that had nothing to do.
 *
 * `before_id` is **inclusive** — the handler filters `id <= ?` — so passing the
 * newest id held settles everything up to and including it.
 */
export async function markRead(opts: {
  ids?: number[];
  before_id?: number;
}): Promise<number> {
  const hasIds = Boolean(opts.ids?.length);
  if (!hasIds && opts.before_id === undefined) return 0;
  return sdk<number>("notification.mark_read", opts);
}
