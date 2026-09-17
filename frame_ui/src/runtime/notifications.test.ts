/**
 * The two sets, pinned.
 *
 * Every case here is something `docs/NOTIFICATIONS.md` warns will bite an
 * implementation that assumes the status queue and the panel list are two views
 * of one list. They are not, and the reducer is pure so that saying so costs a
 * test rather than a browser.
 */

import { describe, expect, it, vi } from "vitest";

// `client.ts` reads `window.location` as it loads, to work out which thread this
// browser is. Nothing under test here makes a Request — the reducer is pure —
// but `lib/notifications.ts` imports it for the two Requests it wraps, and that
// is enough to need a stub in a suite that runs without a DOM. Same reason
// `lib/ledger.test.ts` opens the same way.
vi.mock("@/lib/client", () => ({ sdk: vi.fn() }));

const { highestId, initialNotifications, reduceNotifications, unreadCount } =
  await import("@/runtime/notifications");

import type { NotificationPayload } from "@/lib/events";
import type { Notification } from "@/lib/notifications";
import type { NotificationAction } from "@/runtime/notifications";

/** A frame. `notification_id` absent means transient — delivered, never
 *  stored — which is the distinction the whole file is about. */
const frame = (
  over: Partial<NotificationPayload> = {},
): NotificationPayload => ({
  title: "Plugin registered",
  body: "weather.py",
  source: "plugin_watcher",
  level: "success",
  sent_at: 1_770_000_000,
  ...over,
});

/** A row, as `notification.list` hands it back. */
const row = (over: Partial<Notification> = {}): Notification => ({
  id: 1,
  ts: 1_770_000_000,
  title: "Scheduled agent finished",
  body: "**Done.**",
  source: "subagents",
  source_id: null,
  level: "info",
  session_key: null,
  conversation_id: null,
  user_id: null,
  read_at: null,
  ...over,
});

/** Fold a list of actions over the empty state. */
function run(...actions: NotificationAction[]) {
  return actions.reduce(reduceNotifications, initialNotifications);
}

describe("the status queue is larger than the panel set", () => {
  it("queues a transient notification and files nothing", () => {
    // "Compacting conversation…" and overflow recovery arrive with no
    // `notification_id` on purpose: a panel that fills with progress lines is
    // one nobody reads. The status line is the only surface they ever get, so
    // filtering the status queue to persisted rows would send them nowhere.
    const state = run({
      type: "raised",
      notification: frame({ title: "Compacting conversation…", level: "info" }),
      key: "k1",
    });

    expect(state.notificationQueue).toHaveLength(1);
    expect(state.rows).toHaveLength(0);
  });

  it("queues and files a persisted one", () => {
    const state = run({
      type: "raised",
      notification: frame({ notification_id: 7 }),
      key: "k1",
    });

    expect(state.notificationQueue).toHaveLength(1);
    expect(state.rows.map((one) => one.id)).toEqual([7]);
  });

  it("keys queued arrivals independently of persisted IDs", () => {
    // Keying a React list on `notification_id` gives every transient one
    // `key={undefined}`, which collapses them into one entry and then animates
    // the wrong one out.
    const state = run(
      { type: "raised", notification: frame(), key: "k1" },
      { type: "raised", notification: frame(), key: "k2" },
    );

    expect(state.notificationQueue.map((one) => one.key)).toEqual(["k2", "k1"]);
  });
});

describe("the same notification never lands twice", () => {
  it("survives the reconnect replay", () => {
    // `EventSource` replays from `Last-Event-ID`, so a reconnect re-delivers
    // frames this client already saw. Appending would put a second copy in the
    // panel that nothing ever removes.
    const state = run(
      { type: "raised", notification: frame({ notification_id: 7 }), key: "k1" },
      { type: "raised", notification: frame({ notification_id: 7 }), key: "k2" },
    );

    expect(state.rows).toHaveLength(1);
    // Both still queued: the replay is a duplicate row, but two arrivals are
    // two arrivals, and the status queue is keyed on its own ids anyway.
    expect(state.notificationQueue).toHaveLength(2);
  });

  it("merges a frame and the backfill that also carries it", () => {
    // A persisted notification that arrives while connected is in both the
    // frame and the table, and a naive `[...rows, ...streamed]` shows it twice.
    const state = run(
      { type: "raised", notification: frame({ notification_id: 7 }), key: "k1" },
      { type: "backfilled", rows: [row({ id: 7 }), row({ id: 6 })] },
    );

    expect(state.rows.map((one) => one.id)).toEqual([7, 6]);
  });

  it("takes the backfill's account of a row it already held", () => {
    // Settled from another client while this one was away. The frame said
    // unread — it was, when it was sent — and the table is the fresher answer.
    const state = run(
      { type: "raised", notification: frame({ notification_id: 7 }), key: "k1" },
      { type: "backfilled", rows: [row({ id: 7, read_at: 1_770_000_100 })] },
    );

    expect(unreadCount(state)).toBe(0);
  });

  it("keeps rows newest first however they arrived", () => {
    const state = run(
      { type: "backfilled", rows: [row({ id: 4 }), row({ id: 2 })] },
      { type: "raised", notification: frame({ notification_id: 9 }), key: "k1" },
      { type: "backfilled", rows: [row({ id: 6 })] },
    );

    expect(state.rows.map((one) => one.id)).toEqual([9, 6, 4, 2]);
  });
});

describe("what the bell is drawn from", () => {
  it("drops the count without waiting for a refetch", () => {
    const state = run(
      { type: "backfilled", rows: [row({ id: 3 }), row({ id: 2 })] },
      { type: "read", before: 3 },
    );

    expect(unreadCount(state)).toBe(0);
    expect(state.rows.every((one) => one.read_at !== null)).toBe(true);
  });

  it("settles at or below the id, matching the handler's `id <= ?`", () => {
    const state = run(
      { type: "backfilled", rows: [row({ id: 3 }), row({ id: 2 })] },
      { type: "read", before: 2 },
    );

    expect(unreadCount(state)).toBe(1);
  });

  it("leaves an already-settled row's timestamp alone", () => {
    // Not cosmetic: re-stamping would make "read at" drift forward every time
    // the panel opened, and the server counts only rows that actually changed.
    const state = run(
      { type: "backfilled", rows: [row({ id: 3, read_at: 1_770_000_050 })] },
      { type: "read", before: 3 },
    );

    expect(state.rows[0].read_at).toBe(1_770_000_050);
  });

  it("counts nothing for transient notifications", () => {
    // They have no row, so there is nothing to settle and nothing to badge —
    // the status message already was the whole of their delivery.
    const state = run({ type: "raised", notification: frame(), key: "k1" });

    expect(unreadCount(state)).toBe(0);
    expect(highestId(state)).toBe(0);
  });
});

describe("the cursor", () => {
  it("is the highest row held, whichever way it arrived", () => {
    const state = run(
      { type: "backfilled", rows: [row({ id: 4 })] },
      { type: "raised", notification: frame({ notification_id: 11 }), key: "k1" },
    );

    // `since_id` for the reconnect top-up, `before_id` for "mark all read".
    expect(highestId(state)).toBe(11);
  });

  it("is zero when empty, which `notification.list` reads as the beginning", () => {
    expect(highestId(initialNotifications)).toBe(0);
  });
});

describe("an empty panel says which kind of empty", () => {
  it("distinguishes a failed read from nothing having happened", () => {
    expect(initialNotifications.failure).toBeNull();
    expect(run({ type: "backfilled", rows: [] }).failure).toBeNull();
    expect(run({ type: "failed", message: "no" }).failure).toBe("no");
  });

  it("clears a past failure once a read succeeds", () => {
    const state = run(
      { type: "failed", message: "no" },
      { type: "backfilled", rows: [row()] },
    );

    expect(state.failure).toBeNull();
  });
});

describe("status messages expire", () => {
  it("removes one by key", () => {
    const state = run(
      { type: "raised", notification: frame(), key: "k1" },
      { type: "raised", notification: frame(), key: "k2" },
      { type: "dismissed", key: "k1" },
    );

    expect(state.notificationQueue.map((one) => one.key)).toEqual(["k2"]);
  });

  it("leaves the row behind when a persisted status message expires", () => {
    // Expiring the status message is not settling the notification. The panel is
    // where it lives afterwards, and the unread dot should still be up.
    const state = run(
      { type: "raised", notification: frame({ notification_id: 7 }), key: "k1" },
      { type: "dismissed", key: "k1" },
    );

    expect(state.notificationQueue).toHaveLength(0);
    expect(unreadCount(state)).toBe(1);
  });
});
