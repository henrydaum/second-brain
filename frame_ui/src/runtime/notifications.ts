/**
 * Session-wide notifications have two independent collections:
 * - notificationQueue holds live arrivals until their status-line display ends.
 *   It includes transient updates that have no persisted notification ID.
 * - rows holds persisted notifications, including history fetched on connection.
 *
 * The queue is stored newest first and displayed oldest first, one at a time.
 * Every severity expires from the status line. Expiration does not remove a
 * persisted row or mark it read; the popup and unread dot own that behavior.
 * Neither collection is reset when the user switches conversations.
 */

import type { NotificationPayload } from "@/lib/events";
import { isUnread, rowFromFrame, type Notification } from "@/lib/notifications";

/** A queued arrival has its own key because transient updates have no row ID. */
export type QueuedNotification = {
  key: string;
  notification: NotificationPayload;
};

export type NotificationState = {
  /** Newest first in storage; the status line consumes from the end. */
  notificationQueue: QueuedNotification[];
  /** Newest first, matching the order `notification.list` returns. */
  rows: Notification[];
  /** Why the panel is empty, when the reason is not "nothing happened". Said in
   *  the panel rather than the error banner, for the reason
   *  `FileActivityProvider` gives about a missing `ledger.read`: a kernel
   *  without the Request would otherwise raise an error banner on every boot, about a
   *  surface you may never open. */
  failure: string | null;
};

export const initialNotifications: NotificationState = {
  notificationQueue: [],
  rows: [],
  failure: null,
};

export type NotificationAction =
  /** A `notification` frame off the event stream. */
  | { type: "raised"; notification: NotificationPayload; key: string }
  /** Remove a queued status message after its display and fade complete. */
  | { type: "dismissed"; key: string }
  /** Rows from `notification.list`: the opening read, or a reconnect top-up. */
  | { type: "backfilled"; rows: Notification[] }
  /** The opening read failed. Distinct from an empty backfill. */
  | { type: "failed"; message: string }
  /** Rows just settled, so the badge drops without waiting for a refetch.
   *  `before` settles everything at or below an id, matching the inclusive
   *  `id <= ?` the handler applies. */
  | { type: "read"; ids?: number[]; before?: number };

export function reduceNotifications(
  state: NotificationState,
  action: NotificationAction,
): NotificationState {
  switch (action.type) {
    case "raised": {
      const queuedNotification: QueuedNotification = { key: action.key, notification: action.notification };
      const notificationQueue = [queuedNotification, ...state.notificationQueue];

      // **Only the persisted ones reach the panel.** The check is here rather
      // than at the call site so there is one place that knows the two sets
      // differ, and it is the place that holds both.
      const id = action.notification.notification_id;
      if (id === undefined) return { ...state, notificationQueue };

      return {
        ...state,
        notificationQueue,
        rows: merge(state.rows, [rowFromFrame(action.notification, id)]),
      };
    }

    case "dismissed":
      return {
        ...state,
        notificationQueue: state.notificationQueue.filter((queuedNotification) => queuedNotification.key !== action.key),
      };

    case "backfilled":
      return {
        ...state,
        failure: null,
        rows: merge(state.rows, action.rows),
      };

    case "failed":
      return { ...state, failure: action.message };

    case "read": {
      const at = Date.now() / 1000;
      const ids = action.ids ? new Set(action.ids) : null;
      let changed = false;
      const rows = state.rows.map((row) => {
        if (!isUnread(row)) return row;
        const settled =
          (ids?.has(row.id) ?? false) ||
          (action.before !== undefined && row.id <= action.before);
        if (!settled) return row;
        changed = true;
        return { ...row, read_at: at };
      });
      return changed ? { ...state, rows } : state;
    }
  }
}

/**
 * Fold incoming rows into the ones already held, newest first.
 *
 * **By id, always.** The same notification reaches this state two ways — as a
 * frame while connected, and again in the backfill that runs on the next
 * reconnect — and a naive concatenation shows it twice. `EventSource` replays
 * from `Last-Event-ID` too, so even the frame alone can arrive more than once.
 *
 * Incoming wins on conflict, because it is the fresher account: a row settled
 * from another client comes back with a `read_at` this side has never seen, and
 * that is exactly the fact worth taking.
 */
function merge(held: Notification[], incoming: Notification[]): Notification[] {
  if (!incoming.length) return held;
  const byId = new Map(held.map((row) => [row.id, row]));
  for (const row of incoming) byId.set(row.id, row);
  return [...byId.values()].sort((a, b) => b.id - a.id);
}

/** How many still want attention. Counted off what is held rather than asked
 *  for with `unread_only`, so the badge answers instantly and the dot never
 *  lags a round trip behind the panel it labels. */
export const unreadCount = (state: NotificationState): number =>
  state.rows.reduce((count, row) => count + (isUnread(row) ? 1 : 0), 0);

/** The highest row id held, which is the `since_id` cursor for the reconnect
 *  top-up and the `before_id` for "mark everything read". Zero when empty —
 *  `notification.list` treats that as "from the beginning". */
export const highestId = (state: NotificationState): number =>
  state.rows.reduce((max, row) => Math.max(max, row.id), 0);
