/** @vitest-environment jsdom */

/**
 * Coming back to a page a phone suspended while it was in a pocket.
 *
 * The failure this guards against is silent by construction: the tab resumes
 * holding a stream that says `OPEN` and carries nothing, the status line stays
 * green, and the only symptom is a conversation that never moves again. There
 * is no error to assert on, so what gets pinned instead is the decision — when
 * a stream is thrown away, when it is left alone, and that the replacement asks
 * for what it missed.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/client", () => ({
  serverUrl: (path: string) => new URL(path, "http://localhost:4173"),
  authHeaders: () => ({}),
}));

const { connect } = await import("@/lib/events");

/** Enough of an `EventSource` to be opened, driven and closed. jsdom has none
 *  of its own, and the real one would want a server. */
class FakeEventSource {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;

  static opened: FakeEventSource[] = [];

  readonly url: string;
  readyState = FakeEventSource.CONNECTING;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;

  constructor(url: URL | string) {
    this.url = String(url);
    FakeEventSource.opened.push(this);
  }

  /** The server accepted it. */
  accept() {
    this.readyState = FakeEventSource.OPEN;
    this.onopen?.();
  }

  /** One frame, with the sequence number the server sends as `id:`. */
  deliver(id: string, frame: unknown) {
    this.onmessage?.({
      data: JSON.stringify(frame),
      lastEventId: id,
    } as MessageEvent);
  }

  /**
   * The answer was not an event stream, so the browser gives up for good.
   *
   * What a restarting backend looks like from the page: Caddy stays up and
   * answers `502`, and the spec's rule for a response that is not
   * `200 text/event-stream` is to fail the connection permanently — one
   * `error`, `readyState` `CLOSED`, and no retry of its own ever again.
   */
  fail() {
    this.readyState = FakeEventSource.CLOSED;
    this.onerror?.();
  }

  /** The connection dropped mid-stream, which the browser retries itself. */
  drop() {
    this.readyState = FakeEventSource.CONNECTING;
    this.onerror?.();
  }

  close() {
    this.readyState = FakeEventSource.CLOSED;
  }
}

/** The most recent stream handed out, which is the live one. */
const latest = () => FakeEventSource.opened.at(-1)!;
const openedCount = () => FakeEventSource.opened.length;

function hide() {
  vi.spyOn(document, "hidden", "get").mockReturnValue(true);
  document.dispatchEvent(new Event("visibilitychange"));
}

function show() {
  vi.spyOn(document, "hidden", "get").mockReturnValue(false);
  document.dispatchEvent(new Event("visibilitychange"));
}

/** Long enough that an `OPEN` stream stops being believed. */
const A_WHILE = 61_000;

beforeEach(() => {
  FakeEventSource.opened = [];
  vi.stubGlobal("EventSource", FakeEventSource);
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("returning to a foregrounded page", () => {
  it("deduplicates replayed delivery IDs but permits a new sharing event", () => {
    const receive = vi.fn();
    const close = connect(receive, vi.fn());
    latest().accept();
    const frame = { kind: "attachments", payload: ["/image.png"] };
    latest().deliver("1", frame);
    latest().deliver("1", frame);
    latest().deliver("2", frame);
    expect(receive).toHaveBeenCalledTimes(2);
    close();
  });

  it("reopens a stream the browser has given up on, however brief the absence", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();

    // The connection died while the page was suspended — which is the state a
    // suspended page cannot have noticed, since none of its code ran.
    latest().close();
    hide();
    show();

    expect(openedCount()).toBe(2);
    close();
  });

  it("reopens a stream that still claims to be open, after a long absence", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();

    hide();
    vi.advanceTimersByTime(A_WHILE);
    show();

    // The zombie: `OPEN`, carrying nothing, and no event will ever say so.
    expect(openedCount()).toBe(2);
    close();
  });

  it("leaves a healthy stream alone when the absence was brief", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();

    hide();
    vi.advanceTimersByTime(2_000);
    show();

    // Reopening hands the session back and forth on the server. A glance at
    // another window is not evidence of anything.
    expect(openedCount()).toBe(1);
    close();
  });

  it("asks for the frames it missed", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();
    latest().deliver("41", { kind: "typing", payload: true });

    hide();
    vi.advanceTimersByTime(A_WHILE);
    show();

    // **The point of the whole exercise.** A `new EventSource` sends no
    // `Last-Event-ID`, so without this the reconnect would drop the turn that
    // ran while the phone was asleep — the one thing somebody came back for.
    expect(latest().url).toContain("since=41");
    close();
  });

  it("does not interrupt a first connection that is still being made", () => {
    // Opening the app in a background tab and switching to it a moment later.
    // Nothing has gone wrong yet; replacing the in-flight request with an
    // identical one would only make the first paint later.
    const close = connect(vi.fn(), vi.fn());
    hide();
    vi.advanceTimersByTime(2_000);
    show();

    expect(openedCount()).toBe(1);
    close();
  });

  it("does not ask for a replay on the first connection", () => {
    const close = connect(vi.fn(), vi.fn());
    expect(latest().url).not.toContain("since=");
    close();
  });

  it("keeps handing frames to the caller across a reopen", () => {
    const onFrame = vi.fn();
    const close = connect(onFrame, vi.fn());
    latest().accept();

    hide();
    vi.advanceTimersByTime(A_WHILE);
    show();
    latest().accept();
    latest().deliver("42", { kind: "typing", payload: false });

    expect(onFrame).toHaveBeenCalledWith({ kind: "typing", payload: false });
    close();
  });

  it("says it is reconnecting, so the resync effects run again", () => {
    const onStatus = vi.fn();
    const close = connect(vi.fn(), onStatus);
    latest().accept();
    onStatus.mockClear();

    hide();
    vi.advanceTimersByTime(A_WHILE);
    show();
    latest().accept();

    // The provider keys `reconcile`, the notification backfill and
    // `syncSession` on the transition back to "open" — which is exactly the
    // work a page returning from a suspension needs done.
    expect(onStatus.mock.calls.map(([status]) => status)).toEqual([
      "reconnecting",
      "open",
    ]);
    close();
  });

  it("opens nothing once the caller has closed it", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();
    close();

    hide();
    vi.advanceTimersByTime(A_WHILE);
    show();

    // StrictMode unmounts and remounts this in development. A listener that
    // outlived its stream would open a second one against the same thread and
    // quietly evict the live mount's.
    expect(openedCount()).toBe(1);
  });
});

/**
 * Restarting Second Brain while the page is open and being looked at.
 *
 * The old failure needed no suspension and no phone: Caddy answers `502` for
 * the seconds its backend is missing, the browser reads that as "not an event
 * stream" and stops for good, and the status line goes on saying
 * "Reconnecting…" about a client that has stopped reconnecting. Nothing but a
 * manual refresh brought it back.
 */
describe("a backend that went away", () => {
  it("tries again once the browser has given up", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();
    latest().fail();

    expect(openedCount()).toBe(1);
    vi.advanceTimersByTime(1_000);
    expect(openedCount()).toBe(2);
    close();
  });

  it("takes over an attempt that reports nothing at all", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();
    latest().drop();

    // **The failure that stranded this client.** `drop` is `CONNECTING` with no
    // further event — the browser waiting on a request Caddy is holding while
    // its backend restarts. There is no `CLOSED` coming, so a retry that waits
    // for one waits forever, which is what the manual refresh was working
    // around.
    expect(openedCount()).toBe(1);
    vi.advanceTimersByTime(3_000);
    expect(openedCount()).toBe(2);
    close();
  });

  it("keeps taking over, for as long as nothing answers", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();
    latest().drop();

    // Each replacement is itself on a deadline, so silence never becomes rest.
    // A backend down for half a minute is found within three seconds of
    // returning, with nobody touching the page.
    vi.advanceTimersByTime(30_000);
    expect(openedCount()).toBe(11);

    latest().accept();
    vi.advanceTimersByTime(A_WHILE);
    expect(openedCount()).toBe(11);
    close();
  });

  it("stops replacing attempts the moment one is accepted", () => {
    const onStatus = vi.fn();
    const close = connect(vi.fn(), onStatus);
    latest().accept();
    latest().drop();

    vi.advanceTimersByTime(3_000);
    latest().accept();
    onStatus.mockClear();

    // An accepted stream is the one state worth trusting, so the deadline is
    // cancelled rather than left to churn a working connection.
    vi.advanceTimersByTime(A_WHILE);
    expect(openedCount()).toBe(2);
    expect(onStatus).not.toHaveBeenCalled();
    close();
  });

  it("backs off, so an outage measured in hours is not a request a second", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();

    const gaps = [1_000, 2_000, 4_000, 5_000, 5_000];
    let expected = 1;
    for (const gap of gaps) {
      latest().fail();
      vi.advanceTimersByTime(gap - 1);
      expect(openedCount()).toBe(expected);
      vi.advanceTimersByTime(1);
      expect(openedCount()).toBe(++expected);
    }
    close();
  });

  it("comes back on its own, and says so", () => {
    const onStatus = vi.fn();
    const close = connect(vi.fn(), onStatus);
    latest().accept();
    onStatus.mockClear();

    latest().fail();
    vi.advanceTimersByTime(1_000);
    latest().accept();

    // "open" is what the provider keys the catalogue re-read, the pending-input
    // reconcile and the conversation resync on. Without it the page is
    // connected to a server it has not spoken to since before the restart.
    expect(onStatus.mock.calls.map(([status]) => status)).toEqual([
      "reconnecting",
      "reconnecting",
      "open",
    ]);
    close();
  });

  it("asks the restarted server for the frames it missed", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();
    latest().deliver("41", { kind: "typing", payload: true });

    latest().fail();
    vi.advanceTimersByTime(1_000);

    expect(latest().url).toContain("since=41");
    close();
  });

  it("starts quick again after a recovery, rather than inheriting the ceiling", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();

    // A long first outage, which walks the delay up to its cap.
    for (const gap of [1_000, 2_000, 4_000, 5_000]) {
      latest().fail();
      vi.advanceTimersByTime(gap);
    }
    latest().accept();

    // A second, unrelated restart must not be five seconds slower to notice
    // just because an earlier one went on for a while.
    const recovered = openedCount();
    latest().fail();
    vi.advanceTimersByTime(1_000);
    expect(openedCount()).toBe(recovered + 1);
    close();
  });

  it("recovers a page that was opened before the server was up", () => {
    // The LaunchAgent order on a cold boot: Caddy serves the build and answers
    // 502 for Second Brain until it is listening. Nothing here has ever been
    // connected, so there is no "reconnection" — but it is the same wait.
    const close = connect(vi.fn(), vi.fn());
    latest().fail();
    vi.advanceTimersByTime(1_000);

    expect(openedCount()).toBe(2);
    latest().accept();
    close();
  });

  it("ignores a failure from a stream it has already replaced", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();

    const superseded = latest();
    hide();
    vi.advanceTimersByTime(A_WHILE);
    show();
    latest().accept();

    superseded.fail();
    vi.advanceTimersByTime(A_WHILE);

    // Booking a retry against the live stream would throw away a working
    // connection because a dead one finally noticed it was dead.
    expect(openedCount()).toBe(2);
    close();
  });

  it("books nothing once the caller has closed it", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();
    latest().fail();
    close();

    vi.advanceTimersByTime(A_WHILE);
    expect(openedCount()).toBe(1);
  });
});

/**
 * Two tabs of the same app, which is one session split in two.
 *
 * The server keeps one stream per thread and a second `GET /events` replaces
 * the first, so both tabs reconnecting on sight is not two clients recovering —
 * it is one stream being taken back and forth every three seconds, forever.
 * What that costs is invisible from the status line: four re-reads a turn, and
 * a blink of nobody attending the session, in which an unsafe Request is
 * refused rather than asked about.
 */
describe("another tab taking the stream", () => {
  it("leaves it to the other tab while nobody is looking at this one", () => {
    const onStatus = vi.fn();
    const close = connect(vi.fn(), onStatus);
    latest().accept();
    const taken = latest();

    hide();
    // The eviction, as this tab sees it: the socket ends mid-stream, which is
    // indistinguishable from any other drop and is why the rule is about who is
    // looking rather than about what happened.
    taken.drop();
    vi.advanceTimersByTime(30_000);

    expect(openedCount()).toBe(1);
    // Closed, not merely left alone: an `EventSource` retries on its own, and a
    // page that only declined to *book* a retry would still be back in the
    // fight three seconds later without asking anybody.
    expect(taken.readyState).toBe(FakeEventSource.CLOSED);
    expect(onStatus).toHaveBeenLastCalledWith("reconnecting");
    close();
  });

  it("takes it back when somebody looks at this one again", () => {
    const close = connect(vi.fn(), vi.fn());
    latest().accept();
    latest().deliver("7", { kind: "typing", payload: true });

    hide();
    latest().drop();
    show();

    expect(openedCount()).toBe(2);
    // Standing down is not forgetting: the turn that ran in the other tab is
    // what this one comes back needing.
    expect(latest().url).toContain("since=7");
    close();
  });

  it("stands down on a deadline as well as on a failure", () => {
    const close = connect(vi.fn(), vi.fn());

    // Nothing has been accepted here, so this is the attempt's own deadline
    // coming due rather than anything the server said. It must not strand the
    // tab: `everOpened` is false and a brief absence proves nothing, so only
    // remembering the stream is owed brings it back.
    hide();
    vi.advanceTimersByTime(3_000);
    expect(openedCount()).toBe(1);

    show();
    expect(openedCount()).toBe(2);
    close();
  });
});
