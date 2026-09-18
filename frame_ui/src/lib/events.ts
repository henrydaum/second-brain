/**
 * The inbound half of the bridge: `GET /events`.
 *
 * Every frame is one `render` call the kernel made, verbatim. There is no
 * translation layer — these are the same twelve payloads a native Python
 * frontend receives, which is why a client that handles all twelve can do what
 * the REPL can.
 *
 * **Opening this stream is the attendance signal.** It declares that a person is
 * watching this session, and attendance is what decides whether an unsafe
 * Request raises a dialog or is refused outright. No stream, no dialogs. That is
 * why boot connects here before it POSTs anything.
 */

import { authHeaders, serverUrl } from "@/lib/client";

/* ── The eleven kinds ────────────────────────────────────────────────────
 *
 * Handle what you can show and ignore the rest; a client that only renders
 * `messages` is a working client. These types are transcriptions of the
 * protocol document, so every optional marker below is a real "may be absent",
 * not defensive typing.
 *
 * `notification` is the one exception to "ignore the rest": ignoring it is not
 * quiet, because the kernel only sends it to a frontend that declared it can
 * show it. See `NotificationPayload`.
 */

/** GitHub-flavoured markdown, including tables and fenced code blocks. This is
 *  the interchange format everywhere in Second Brain and also what the model
 *  emits, so one rendering path covers both. */
type MessagesPayload = string[];

/** Markdown returned by a slash command or a tool the person invoked
 *  directly. This is output, not part of the conversation. */
type CallableOutputPayload = string[];

/** The reply arriving token by token. */
type StreamDeltaPayload = {
  turn_id?: string | null;
  /** Groups the fragments. This is the message key. */
  stream_id: string;
  /** 1-based, increments per fragment. */
  seq: number;
  /** The fragment. Empty on the final frame. */
  delta: string;
  /** False while running, true exactly once. */
  done: boolean;
  /** The stream was cut off. */
  aborted?: boolean;
  /** **Only when `done` and not `aborted`.** The cleaned text; the deltas agree
   *  with it, so it replaces rather than appends. */
  final_text?: string;
  /** Only alongside `final_text`. Usually `"final"`. */
  kind?: string;
};

type TurnActivityPayload = {
  turn_id?: string | null;
  phase: "waiting" | "thinking";
};

/** Fires for tool calls and slash commands alike. */
export type ToolStatusPayload = {
  turn_id?: string | null;
  /** Stable across started/finished — update in place, do not append. */
  call_id: string;
  status: "started" | "progressed" | "finished";
  /** `"command"` for slash commands; absent for tools. */
  kind?: string;
  tool_name?: string;
  command_name?: string;
  args?: Record<string, unknown>;
  /** Short human blurb. Repeated on `finished`, deliberately, so a client
   *  overwriting one line still has it. */
  narration?: string;
  ok?: boolean;
  error?: string | null;
  /**
   * **On `finished` only: what the call amounted to.**
   *
   * The tool's own account of its result — prose, since it is written for the
   * model — capped exactly as the copy the kernel stores, so this and the row
   * `conv.read` hands back are the same bytes.
   *
   * Empty on failure, where `error` is the outcome, and empty for a tool that
   * reported nothing. **Not interchangeable with `narration`**: that is what
   * the agent set out to do, this is what came back.
   */
  summary?: string;
};

/**
 * A question the kernel is blocking a turn on.
 *
 * **A turn is blocked until this is answered**, so a client that ignores this
 * kind will appear to hang.
 *
 * **Not only permission.** One kernel primitive — `runtime.request_input` —
 * raises a sandbox permission gate, a `ui.ask`, and a tool asking the person
 * something, and all three arrive here. `type` may be any of `boolean`,
 * `string`, `integer`, `number`, `array` or `object`. A client that words this
 * as a permission grant will mislabel an ordinary question as one; everything
 * above the wire in this app therefore calls it an *input request*, and only
 * the wire spells it `approval`.
 */
export type ApprovalPayload = {
  /** Answer with this, via `frontend.resolve`. */
  id: string;
  title: string;
  /** Arguments, who is asking, and any extra note. Shown in full. */
  body?: string;
  /** Usually `"boolean"` or `"string"`. */
  type?: string;
  /** Allowed answers. */
  enum?: unknown[] | null;
  /** **Paired with `enum` by index.** May be null even when `enum` is not. */
  enum_labels?: string[] | null;
  default?: unknown;
};

/**
 * A question stopped waiting: answered, cancelled, or denied by timeout.
 *
 * **The counterpart to `approval`, and the only thing that says a dialog may
 * come down.** Another client can answer the same question, and the kernel
 * denies by name after 300s — neither is something this client did, and
 * without this frame neither is something it could learn except by polling.
 *
 * `reason` says how the question ended, not what the answer was: the answer
 * went to whoever was blocked on it and is deliberately not repeated here.
 */
type ApprovalSettledPayload = {
  /** The `id` of the `approval` this settles. */
  request_id: string;
  reason?: "answered" | "cancelled";
};

/** One step of a command collecting its arguments. */
export type FormFieldPayload = {
  /** The command or tool being filled in. */
  name?: string;
  /** The raw step. `display` is what to actually draw; this is here for the
   *  cases where the raw type matters (numeric inputs, say). */
  field?: {
    name?: string;
    prompt?: string;
    required?: boolean;
    type?: string;
    enum?: unknown[] | null;
    enum_labels?: string[] | null;
    default?: unknown;
    prompt_when_missing?: boolean;
    columns?: number;
  };
  /** Arguments gathered so far. */
  collected?: Record<string, unknown>;
  display?: {
    prompt: string;
    /** Hint text. */
    assist?: string;
    /** Empty for free text. Booleans get True/False. */
    choices?: { value: unknown; label?: string }[];
    /** A hint for the input widget. */
    input_mode?: string;
    allow_skip?: boolean;
    /** Effectively always true. */
    allow_cancel?: boolean;
    /** Only true once a previous step exists. */
    allow_back?: boolean;
  };
};

/** Quick replies. Nothing in the kernel currently emits this; it exists for
 *  store plugins. Submit the `value` as text, same as a form choice. */
export type ButtonsPayload = { value: unknown; label?: string }[];

export type ErrorPayload = {
  code?: string;
  message?: string;
  details?: unknown;
  retry_phase?: unknown;
};

/** Filesystem paths on the host — not URLs, and not bytes. A browser cannot
 *  open these directly; the contents come back through `fs.read_bytes`. */
type AttachmentsPayload = string[];

/** The conversation currently bound to this browser session. A null id means
 * the session was reset remotely and the client must show New Conversation. */
type ConversationPayload = {
  conversation_id: number | null;
  title: string;
};

/**
 * Something the *system* is telling you, as opposed to something the agent said.
 *
 * **This kind only arrives because the frontend asked for it.** The store's
 * `frontend_http.py` declares `supports_notifications`, and without that
 * declaration the kernel does not drop these — it flattens each one into
 * markdown and sends it as an ordinary `messages` frame. So the failure mode is
 * not an error: it is plugin registrations appearing in the transcript as if a
 * person had typed them. The same bargain `supports_streaming` makes for
 * `stream_delta`.
 *
 * The line the kernel draws is **who was speaking**. Mid-turn narration ("let me
 * check that file") and `sdk.ui.render` are the agent's own turn and stay on
 * `messages` and `attachments` respectively. Everything else that arrives
 * unprompted is one of these. The classification is made at each emit site, so
 * there is nothing to re-derive on this side.
 */
export type NotificationPayload = {
  /** Short header — what happened. Always plain text. */
  title: string;
  /** The detail. Multi-line, and **sometimes markdown** — see `body` in
   *  `lib/notifications.ts` for why it is rendered as markdown regardless. */
  body: string;
  /**
   * Who raised it.
   *
   * Stamped by the kernel off the live provenance chain, never stated by
   * whoever raised it — which is the part a sandboxed plugin cannot state about
   * itself, so a plugin cannot claim to be `plugin_watcher`. Trustworthy
   * attribution, and worth showing.
   *
   * In practice: `plugin_watcher`, `config`, `runtime`, `subagents`, `session`,
   * or the leaf name of a plugin that called `sdk.session.push(notify=True)`.
   */
  source: string;
  /** Styling only. Nothing branches on it kernel-side, and an unrecognised
   *  value is normalised to `info` before it arrives — so the union is closed. */
  level: "info" | "success" | "warning" | "error";
  /** Epoch **seconds**, fractional. Same units as `LedgerRow.ts`, and not the
   *  milliseconds `Turn.createdAt` uses. */
  sent_at: number;

  /** Producer-specific: a session key, a handle id, a config scope. */
  source_id?: string;
  /** The session it came *from*, which is usually not one being looked at.
   *  **Not a delivery target** — it names where the work happened, and that is
   *  a session with no frontend attached. */
  source_session_key?: string;
  /** The conversation it is about, when there is one. Usually **not** the open
   *  one; that is the point, since it came from a background session. */
  conversation_id?: number;
  /** A pre-rendered slash command, for surfaces whose only affordance is text.
   *  **Never rendered here** — this client has `conversation_id` and can open
   *  the conversation itself. */
  load_hint?: string;
  /**
   * The row id — **absent when this was not persisted.**
   *
   * Transient progress ("Compacting conversation…", overflow recovery) is
   * delivered and deliberately never stored, because a panel that fills with
   * progress lines is one nobody reads. Live status messages can include
   * transient updates absent from the panel; this field tells them apart.
   */
  notification_id?: number;
};

/** One frame off the stream, discriminated by `kind`. */
export type Frame =
  | { kind: "messages"; payload: MessagesPayload }
  | { kind: "callable_output"; payload: CallableOutputPayload }
  | { kind: "stream_delta"; payload: StreamDeltaPayload }
  | { kind: "typing"; payload: boolean }
  | { kind: "turn_activity"; payload: TurnActivityPayload }
  | { kind: "tool_status"; payload: ToolStatusPayload }
  | { kind: "approval"; payload: ApprovalPayload }
  | { kind: "approval_settled"; payload: ApprovalSettledPayload }
  | { kind: "form_field"; payload: FormFieldPayload }
  | { kind: "buttons"; payload: ButtonsPayload }
  | { kind: "error"; payload: ErrorPayload }
  | { kind: "attachments"; payload: AttachmentsPayload }
  | { kind: "conversation"; payload: ConversationPayload }
  | { kind: "notification"; payload: NotificationPayload };

/** What the connection itself is doing, for the status line. This is not part
 *  of the protocol — it is the transport's own state, and worth showing because
 *  a silently dropped stream otherwise looks exactly like a hung agent. */
export type StreamStatus = "connecting" | "open" | "reconnecting";

/**
 * How long away makes an *apparently healthy* stream worth doubting.
 *
 * Only the ambiguous case needs a number. A stream the browser has given up on
 * announces itself in `readyState` and is reopened on sight, whatever the clock
 * says. The one this threshold is about is the stream that claims to be `OPEN`
 * and is not — and without a heartbeat on the wire there is nothing to ask, so
 * how long the page was away is the only evidence there is.
 *
 * A minute, because being wrong costs something in both directions. Reopening
 * hands the session briefly back and forth on the server, so churning it on
 * every glance at another window is not free; waiting too long leaves somebody
 * looking at a green dot and an empty conversation. A pocketed phone is away
 * for minutes and a glance at another window is away for seconds, so a minute
 * separates them without being close to either.
 */
const LONG_ENOUGH_AWAY_MS = 60_000;

/**
 * How soon to try again after a connection *failed*, and the ceiling.
 *
 * The thing being waited for is a backend coming back, which takes seconds —
 * so the first retry is quick enough to feel like nothing happened, and the
 * backoff exists only so an outage measured in hours is not also a request per
 * second for hours.
 */
const FIRST_RETRY_MS = 1_000;
const LONGEST_RETRY_MS = 5_000;

/**
 * How long an attempt may go without being accepted before it is replaced.
 *
 * **The one that actually recovers a restart.** A failure that announces itself
 * is the easy case; the case that stranded this client is an attempt that
 * announces nothing — `CONNECTING` forever, no `error`, no `open`. Caddy's
 * `reverse_proxy` does not have to answer `502` the instant its upstream dies:
 * it can dial, wait and retry within one request, so the browser holds a
 * request that will never become a stream and fires no event saying so.
 *
 * Three seconds because both directions cost something. This runs against a
 * loopback gateway, where a live backend answers in milliseconds and anything
 * still unanswered after three seconds is waiting on something that is not
 * there. Longer leaves somebody watching a stale page; much shorter would start
 * replacing attempts that were about to succeed on a bad day.
 */
const STALLED_ATTEMPT_MS = 3_000;

/**
 * Open the render stream. Returns a function that closes it.
 *
 * **`EventSource`, deliberately.** It reconnects on its own and sends back the
 * last `id:` it saw as `Last-Event-ID`, and the server replays from there — so a
 * page refresh resumes instead of losing the turn it happened during. A
 * fetch-based reader would have to reimplement both halves of that.
 *
 * ## Why restarting the server needed a manual refresh, and what fixed it
 *
 * **`EventSource`'s own reconnection cannot be relied on to end.** The spec
 * gives it two behaviours: a stream that dies mid-flight is reopened on a
 * timer, and a *response* that is not `200 text/event-stream` fails the
 * connection permanently — `error` once, `readyState` `CLOSED`, nothing further
 * attempted. Neither guarantees the page ever hears about a backend coming
 * back, and a restart can land in a third state that is worse than both:
 * `CONNECTING`, indefinitely, with no event of any kind.
 *
 * That third state is what this deployment produces. Caddy stays up while its
 * backend is missing (see `docs/MACOS_DEPLOYMENT.md`) and its `reverse_proxy`
 * may hold a request rather than answer `502` — dialling, waiting, retrying
 * inside the one request the browser is waiting on. So the page saw the stream
 * drop, said "Reconnecting…", and then nothing happened ever again. A version
 * of this that retried only from `CLOSED` did not help, because `CLOSED` was
 * never reached.
 *
 * **So the rule below does not read `readyState` to decide whether to act.**
 * There is one state worth trusting, `OPEN`, and every attempt that has not
 * reached it is on a deadline from the moment it is made. A definite failure is
 * still worth noticing — it means the next attempt need not wait out the
 * deadline — but nothing depends on one arriving.
 *
 * The replay buffer holds the last 500 frames per session, so a long disconnect
 * drops the middle. A client that has been away a while should re-read state
 * rather than trust the replay; that is what `onRefetchThread` is for.
 *
 * One stream per thread: a second `GET /events` for the same thread replaces the
 * first, so mounting this twice silently steals the connection from the first
 * mount. React 19's development StrictMode double-invokes effects, which is
 * exactly that situation — hence the cleanup function, which must actually run.
 *
 * ## Coming back to a suspended page
 *
 * **`EventSource`'s own recovery only covers failures it noticed.** A phone that
 * suspends a backgrounded web app tears the connection down underneath it
 * without the page ever running the code that would see it, so the tab resumes
 * holding a socket that is `OPEN` and dead: no `error` event, the status line
 * still green, and nothing arriving until somebody reloads. On a tunnelled
 * connection to a machine that may itself have been asleep, that is the ordinary
 * way this app goes quiet rather than an exotic one.
 *
 * So returning to the foreground is treated as a reason to check rather than as
 * nothing, and `reopen` below is what a check that fails does about it.
 */
export function connect(
  onFrame: (frame: Frame) => void,
  onStatus: (status: StreamStatus) => void,
): () => void {
  /** The last `id:` seen, so a reopen can ask for what it missed. */
  let lastEventId = "";
  const delivered = new Set<string>();
  /** The caller has torn this down; nothing may open another stream. */
  let abandoned = false;
  /** This stream has been accepted at least once. A connection still being
   *  made for the first time is not evidence of anything having gone wrong. */
  let everOpened = false;
  let stream: EventSource | null = null;
  /**
   * When to give up on the attempt in flight and make another.
   *
   * **Always running except while a stream is accepted.** That is the whole
   * correction: there is exactly one state worth trusting, and it is `OPEN`.
   * Everything else — `CONNECTING` on the browser's own retry, `CONNECTING` on
   * a request Caddy is holding, `CLOSED` after it gave up — is the same fact
   * from here, which is that nothing is arriving.
   */
  let attempt: number | null = null;
  /** Grows per *definite* failure and resets the moment a stream is accepted,
   *  so the delay describes the current outage rather than the page's history. */
  let retryDelay = FIRST_RETRY_MS;
  /** A stream this page gave up rather than fought for, and owes itself the
   *  moment somebody looks at it again. See `standDown`. */
  let deferred = false;

  const cancelAttempt = () => {
    if (attempt === null) return;
    window.clearTimeout(attempt);
    attempt = null;
  };

  /** Book the next attempt, replacing whatever was booked before. */
  const armAttempt = (ms: number) => {
    cancelAttempt();
    if (abandoned) return;
    attempt = window.setTimeout(() => {
      attempt = null;
      reopen();
    }, ms);
  };

  /**
   * Give the stream up rather than take it back, until somebody looks.
   *
   * **Two tabs on one thread cannot both have it.** The server keeps one
   * stream per thread and a second `GET /events` replaces the first, so a page
   * that reconnects the instant it is displaced recovers nothing — it displaces
   * the other one, is displaced again three seconds later, and the two hand the
   * session back and forth for as long as both are open. That is not free: each
   * turn of it re-reads the catalogue, the conversation, the pending questions
   * and the notifications, and blinks the session *unattended* in between —
   * and an unsafe Request landing in one of those gaps is refused outright
   * rather than raising a dialog.
   *
   * A hidden tab has no claim worth any of that. So it stops competing: the
   * `EventSource` is closed, which is also what stops the browser's own retry
   * from quietly re-entering the fight, and `deferred` remembers that this page
   * owes itself a stream. Foregrounding pays it — see `onVisibilityChange`.
   * The tab somebody is reading wins, which is the right one to win.
   */
  const standDown = () => {
    cancelAttempt();
    stream?.close();
    stream = null;
    deferred = true;
  };

  const open = (status: StreamStatus) => {
    const url = serverUrl("/events");
    // EventSource cannot send headers, so development puts its token in the URL.
    // Production has no browser token: Caddy authenticates the loopback hop.
    const authorization = authHeaders().Authorization;
    if (authorization) url.searchParams.set("token", authorization.slice(7));

    // **A reopen has to ask for the replay by hand.** The browser sends
    // `Last-Event-ID` only on retries it started itself; a `new EventSource` is
    // a fresh request that carries nothing, so without this the frames that
    // arrived while the page was suspended — which is the entire turn somebody
    // missed — would be dropped by the very thing meant to recover them. The
    // server reads the header first and falls back to this, so the two cannot
    // both apply and replay twice.
    if (lastEventId) url.searchParams.set("since", lastEventId);

    const source = new EventSource(url);
    stream = source;
    deferred = false;
    onStatus(status);

    // **Every attempt is on the clock from the moment it is made.** Nothing
    // else here is guaranteed to fire: a request the gateway is holding open
    // produces no event at all, and this is the timer that turns that silence
    // into another attempt.
    armAttempt(STALLED_ATTEMPT_MS);

    source.onopen = () => {
      // Guarded like `onerror` below, and for the mirror of its reason: a
      // stream already replaced can still deliver its own `open`, and acting on
      // it would cancel the live attempt's deadline and call a connection this
      // page no longer holds good.
      if (source !== stream) return;
      everOpened = true;
      // The outage is over, so the delay it accumulated describes nothing. A
      // later one starts quick again rather than inheriting this one's ceiling.
      retryDelay = FIRST_RETRY_MS;
      // IDs restart when the backend restarts; deduplicate within a connection.
      delivered.clear();
      cancelAttempt();
      onStatus("open");
    };

    // A definite failure, which is worth acting on sooner than the deadline
    // above — we are not waiting to find out any more. The ladder is here
    // rather than on the stall path because this is the branch that can repeat
    // as fast as the network answers.
    source.onerror = () => {
      // A stream we have already replaced can still deliver its own failure.
      // Acting on it would tear down a connection that is fine.
      if (source !== stream) return;
      onStatus("reconnecting");
      // Not while nobody is looking: the commonest reason a healthy stream ends
      // is another tab taking it, and this one is in no position to want it
      // back. See `standDown`.
      if (document.hidden) {
        standDown();
        return;
      }
      if (source.readyState === EventSource.CLOSED) {
        const delay = retryDelay;
        retryDelay = Math.min(retryDelay * 2, LONGEST_RETRY_MS);
        armAttempt(delay);
        return;
      }
      // `CONNECTING`: the browser says it is retrying, which is a claim about
      // intent and not about progress. It gets the same deadline as any other
      // attempt — this is the branch a dropped stream lands in, and leaving it
      // to the browser is exactly what left the page stranded.
      armAttempt(STALLED_ATTEMPT_MS);
    };

    source.onmessage = (event) => {
      if (source !== stream || abandoned) return;
      if (event.lastEventId) {
        if (delivered.has(event.lastEventId)) return;
        delivered.add(event.lastEventId);
        if (delivered.size > 500) delivered.delete(delivered.values().next().value!);
      }
      // Kept even for a frame that will not parse: the id is the server's
      // sequence number and is what a later reopen resumes from, so skipping it
      // would ask for one frame that was already handled.
      if (event.lastEventId) lastEventId = event.lastEventId;

      let frame: Frame;
      try {
        frame = JSON.parse(event.data) as Frame;
      } catch {
        // A frame we cannot parse is a bug on the wire, not a reason to tear
        // down a working stream — the next frame is probably fine.
        console.error("second brain: unparseable frame", event.data);
        return;
      }
      onFrame(frame);
    };
  };

  /** Throw this stream away and take another. Closing first matters twice
   *  over: the server keeps one stream per thread and leaving the old one for
   *  it to evict makes the two racy, and an attempt the gateway is still
   *  holding open would otherwise stay held. */
  function reopen() {
    if (abandoned) return;
    // The deadline on an attempt can come due on a page nobody is looking at —
    // it was made just before the tab went away. Same answer as a failure while
    // hidden, and settled the same way on the way back.
    if (document.hidden) {
      standDown();
      return;
    }
    cancelAttempt();
    stream?.close();
    open("reconnecting");
  }

  /**
   * Foregrounding the page, and what it is worth doing about it.
   *
   * **The first rule is the one this page wrote itself.** A tab that stood down
   * while hidden holds no stream on purpose — see `standDown` — and being
   * looked at is the whole of what it was waiting for, so it takes one back
   * without consulting anything else.
   *
   * The rest is about a stream that was *lost* rather than given up, and it
   * takes two rules because the two cases are not equally certain. A stream
   * that had been working and is no longer `OPEN` — `CLOSED`, or stuck `CONNECTING` on a
   * backoff a suspended page never got to run down — is unambiguously carrying
   * nothing, and is reopened however brief the absence was. A stream that says
   * `OPEN` may be perfectly healthy, so only a long absence is taken as
   * evidence against it. See `LONG_ENOUGH_AWAY_MS`.
   *
   * A connection that has never been accepted is neither: it is a first attempt
   * still in flight, and interrupting it to start an identical one would make
   * opening the app in a background tab slower rather than more reliable.
   */
  let hiddenAt = 0;
  const onVisibilityChange = () => {
    if (document.hidden) {
      hiddenAt = Date.now();
      return;
    }

    const away = hiddenAt === 0 ? 0 : Date.now() - hiddenAt;
    hiddenAt = 0;

    // A stream given up rather than lost, so neither rule applies: there is no
    // stream to read a `readyState` off and no absence worth measuring. This
    // page stood down because nobody was reading it, and somebody is now.
    if (deferred) {
      reopen();
      return;
    }

    const dropped = everOpened && stream?.readyState !== EventSource.OPEN;
    if (!dropped && away < LONG_ENOUGH_AWAY_MS) return;
    reopen();
  };

  document.addEventListener("visibilitychange", onVisibilityChange);
  open("connecting");

  return () => {
    abandoned = true;
    cancelAttempt();
    document.removeEventListener("visibilitychange", onVisibilityChange);
    stream?.close();
  };
}
