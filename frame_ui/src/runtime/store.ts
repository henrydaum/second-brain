/**
 * Frames in, conversation out.
 *
 * This is the only place that knows what the render kinds mean. Everything
 * above it works on `Turn[]`, and everything below it works on wire frames.
 *
 * ## Why turns and not messages
 *
 * `stream_delta` and `tool_status` frames arrive **interleaved** during one
 * agent turn, and nothing in a `tool_status` frame says which message it belongs
 * to — there is no message id on the wire at all. So the unit here is a *turn*,
 * opened by `typing: true` and closed by `typing: false`, holding an ordered
 * list of parts.
 *
 * Frames land by identity rather than by appending: a `stream_delta` finds its
 * part by `stream_id`, a `tool_status` finds its by `call_id` and updates in
 * place. The resulting order is arrival order, which is the order it actually
 * happened in, and it converts straight into the interleaved text/tool-group
 * rendering assistant-ui already draws.
 *
 * ## Why a plain reducer
 *
 * The server is the single source of truth and this is a projection of it. A
 * state library would be a second source that could disagree. `useReducer` is
 * React's built-in for "state that changes in a fixed set of ways", and keeping
 * the transition function pure means the awkward protocol rules below are
 * testable without a browser.
 */

import type {
  ButtonsPayload,
  ErrorPayload,
  Frame,
  FormFieldPayload,
} from "@/lib/events";

/* ── Shape ──────────────────────────────────────────────────────────── */

export type TextPart = {
  kind: "text";
  id?: string;
  /** The `stream_id` that produced it, or a synthetic id for `messages` text
   *  that never streamed. */
  streamId: string;
  text: string;
  done: boolean;
};

export type ToolPart = {
  kind: "tool";
  callId: string;
  /** `tool_name` or `command_name` — the wire uses different fields for tools
   *  and slash commands, but they render identically. */
  name: string;
  /** True when this was a slash command rather than a tool. */
  isCommand: boolean;
  /** What came back, from `finished`. A different fact from the narration in
   *  `args` — see `ToolStatusPayload`. Empty until the call finishes, and empty
   *  after it if it failed, where `error` is the outcome. */
  summary: string;
  status: "started" | "progressed" | "finished";
  /**
   * What the call was made with — **including `narration`**, which the wire
   * also sends as a field of its own. See `toolArgs`: this is the one home for
   * it, because it is the only home a conversation read back from the database
   * can offer.
   */
  args?: Record<string, unknown>;
  ok?: boolean;
  error?: string | null;
};

/**
 * Files in a turn — which are two different things wearing one shape.
 *
 * From the agent they are **host paths** reported by a render frame. From the
 * person they are durable message-attachment records read from `conv.read`.
 * `sent` distinguishes the two; `attachments` carries the richer user shape.
 */
export type MessageAttachment = {
  /** The durable cached path. Absent only for the brief optimistic interval
   * between submitting a browser File and reading its stored message row. */
  path?: string;
  fileName: string;
  modality: string;
  extension: string;
};

export type FilesPart = {
  kind: "files";
  id?: string;
  callId?: string;
  receivedAt?: number;
  paths: string[];
  sent?: boolean;
  /** Structured only for files a user message carried. Agent files continue
   * to be plain paths supplied by `attachments` frames. */
  attachments?: MessageAttachment[];
};

/**
 * A slash command being run.
 *
 * **Commands are administration, not conversation**, so none of this reaches
 * the transcript. The chat is between the person and the agent; choosing a tool
 * from a list is a different activity that happens to travel over the same wire,
 * and letting it interleave turns the conversation into a log of button
 * presses.
 *
 * The wire makes this easy to separate: a `tool_status` frame for a command
 * carries `kind: "command"`, and its `args` accumulate the answers as they are
 * given — so this one object is both "what is running" and "what has been
 * collected so far", with no bookkeeping of our own.
 */
export type CommandRun = {
  /** `cmd:<name>:<hash>`, stable for the whole server run. A local
   *  `pending:<name>` identity exists only until its started frame arrives. */
  callId: string;
  name: string;
  /** Arguments collected so far. Cumulative, straight off the wire. */
  args: Record<string, unknown>;
  status: "started" | "progressed" | "finished";
  narration?: string;
  ok?: boolean;
  error?: string | null;
  /** Whatever it printed — captured here rather than in the chat. */
  outcome: string[];
};

export type Part = TextPart | ToolPart | FilesPart;

export type Turn = {
  id: string;
  /** Several visual messages may belong to one logical agent turn. */
  turnId?: string;
  continues?: boolean;
  source?: "live" | "history";
  activity?: { phase: "waiting" | "thinking" | "working" | "writing"; since: number; streamId?: string };
  /**
   * Who this turn is from — and `system`, which is nobody.
   *
   * A system turn is a **compaction marker**: a stored row saying the agent's
   * view of everything above it was replaced by a summary. It is not a message,
   * nothing here ever opens one, and no frame produces one; it arrives from
   * `lib/history.ts` with the scrollback, or from `compacted` below. See
   * `components/compaction-marker.tsx`.
   */
  role: "user" | "assistant" | "system";
  parts: Part[];
  /**
   * When this turn began, as epoch milliseconds.
   *
   * **Optional, and the absence is meaningful.** A turn that happens while the
   * page is open is stamped as it opens, which is accurate to the second. A
   * turn read back from `conv.read` is only stamped if the stored row carried a
   * time — see `history.ts`. Defaulting a historical message to "now" would put
   * a confident, wrong time under every message in the scrollback, so nothing
   * is shown instead.
   */
  createdAt?: number;
  /** Still being written. Drives the message's `running` status, and with it
   *  the working indicator. */
  running: boolean;
  /** The turn was cut off — cancelled, or the stream aborted. */
  aborted: boolean;
};

export type State = {
  turns: Turn[];
  /** The agent has the turn. **This is the only end-of-turn signal there is**:
   *  `false` means the *logical* turn ended, not each internal drive, and a
   *  crash forces it back too. */
  typing: boolean;
  /**
   * A command collecting its arguments, one step at a time.
   *
   * Note what is *not* beside it: a pending approval. That is session state,
   * not conversation state — the kernel holds it on the session's phase stack
   * and persists it there — and keeping it here meant `history` below threw it
   * away. See `runtime/input-requests.ts`.
   */
  form: FormFieldPayload | null;
  /** The command that form belongs to, and everything it has produced. Lives
   *  beside `form` rather than inside it because a command outlives its steps:
   *  it still has an outcome to show once the last question is answered. */
  command: CommandRun | null;
  /** A command the person cancelled. Frames and the HTTP response travel on
   *  independent paths, so its final status can arrive after the panel was
   *  dismissed. Keep its identity long enough to ignore that late tail. */
  suppressedCommand: { callId: string; name: string } | null;
  /** The approval dialog was explicitly closed, so the kernel's matching
   *  "Cancelled." acknowledgement is redundant UI chrome, not conversation. */
  suppressNextCancellationNotice: boolean;
  /** Quick replies offered by a store plugin. */
  buttons: ButtonsPayload;
  error: ErrorPayload | null;
  /**
   * Text already shown from a completed stream.
   *
   * A `messages` frame may repeat text that already arrived as deltas, and
   * rendering both puts the reply on screen twice. The protocol's advice is to
   * track what you have shown and skip the duplicate; since a `messages` frame
   * carries text and no stream id, the comparison has to be on the text itself.
   * Bounded, because this only ever needs to catch a repeat of something recent.
   */
  /**
   * Where the scrollback on screen starts, and whether more precedes it.
   *
   * Beside the turns rather than in the provider, because it is a fact *about*
   * them: `conv.read` answers with a page, and which page is not something a
   * separate piece of state can be trusted to stay in step with.
   */
  scrollback: { hasMore: boolean; oldestId: number | null };
  shownText: string[];
  /**
   * Per live stream, the text an earlier half of a split reply already shows.
   *
   * Empty for every conversation nobody interrupts. It fills only when a
   * message is sent while the agent is mid-sentence, which closes the turn the
   * stream was writing into and continues it in the next one — see
   * `splitOpenTurn`, and `tailOf` for what reads it.
   */
  carried: Record<string, string>;
  streams: Record<string, { seq: number; done: boolean }>;
};

export const initialState: State = {
  turns: [],
  typing: false,
  form: null,
  command: null,
  suppressedCommand: null,
  suppressNextCancellationNotice: false,
  buttons: [],
  error: null,
  scrollback: { hasMore: false, oldestId: null },
  shownText: [],
  carried: {},
  streams: {},
};

export type Action =
  /** One frame off the event stream. */
  | { type: "frame"; frame: Frame }
  | { type: "resumeTurn"; turnId: string }
  /** The person sent something. Echoed locally because `frontend.submit` does
   *  not send the user's own line back down the stream.
   *
   *  `isCommand` is decided by the caller against the server's own catalogue —
   *  see `looksLikeCommand` in `lib/commands.ts`. The reducer cannot work it
   *  out alone, and the guess it used to make was wrong in a way that lost
   *  people's messages. */
  | {
      type: "said";
      text: string;
      attachments?: MessageAttachment[];
      isCommand?: boolean;
    }
  /** Replace the latest optimistic user-file records with the canonical
   * cached paths read from its newly stored conversation row. */
  | { type: "hydrateSentAttachments"; attachments: MessageAttachment[] }
  /**
   * Scrollback, read from `conv.read` at boot. Replaces everything.
   *
   * **The paging cursor rides along, and that is the point.** It used to be
   * set separately by the provider after each of these, which meant six call
   * sites had to remember — and one of them (deleting the conversation you are
   * reading) did not, leaving a "load earlier" affordance on screen for a
   * conversation that no longer existed. Carrying it here makes forgetting a
   * type error instead of a silent one.
   */
  | { type: "history"; turns: Turn[]; hasMore: boolean; oldestId: number | null }
  /**
   * An older page, read as somebody scrolls up. Goes on the *front*.
   *
   * Deliberately not `history`, for the reason `compacted` is not either: that
   * clause resets every transient thing the reducer holds — a half-answered
   * form, a command panel still being read, a streaming reply. None of those
   * have anything to do with rows arriving above them, and a person who
   * scrolled up to read something would watch the turn they were waiting on
   * disappear.
   */
  | {
      type: "olderTurns";
      turns: Turn[];
      hasMore: boolean;
      oldestId: number | null;
    }
  /**
   * A compaction marker read back after one happened, appended to what is on
   * screen.
   *
   * **Not `history`, deliberately.** Compaction is asked for with `/compact`,
   * and that command's own result card is on screen when this lands — a full
   * re-read would reset every transient thing the reducer holds, including the
   * panel the person is reading. The marker is the only new row, so it is the
   * only thing brought over.
   */
  | { type: "compacted"; turn: Turn }
  /** Put the finished command away. Its own affordance, because a command that
   *  has printed something is not done being read just because it is done
   *  running. */
  | { type: "clearCommand" }
  /** Closing an approval dialog is already visible. Suppress only the kernel's
   *  next exact cancellation acknowledgement, not arbitrary errors or text. */
  | { type: "suppressNextCancellationNotice" }
  | { type: "clearError" };

/* ── Helpers ────────────────────────────────────────────────────────── */

let counter = 0;
/** Turn ids only have to be unique and stable within one page life —
 *  assistant-ui keys messages by them and the server never sees them. */
const nextId = () => `turn-${++counter}`;

const HOW_MUCH_TEXT_TO_REMEMBER = 50;

/** The turn frames should land in: the open assistant turn, or a new one.
 *
 *  A new one is minted rather than assumed because frames do not strictly
 *  require a preceding `typing: true` — a `messages` frame can arrive on its
 *  own, and dropping it because no turn was open would lose real output.
 *
 *  Reading the *last* turn stays correct even though a message can now be sent
 *  mid-reply, because `splitOpenTurn` closes the reply as it appends the
 *  person's line. So the open turn is always last, or there is none. */
function openTurn(turns: Turn[]): { turns: Turn[]; turn: Turn } {
  const last = turns.at(-1);
  if (last && last.role === "assistant" && last.running) {
    return { turns, turn: last };
  }
  const turn: Turn = {
    id: nextId(),
    role: "assistant",
    source: "live",
    parts: [],
    running: true,
    aborted: false,
    // Stamped when the turn opens rather than when it finishes: this is when
    // the agent started answering, which is what a reader means by "when".
    createdAt: Date.now(),
  };
  return { turns: [...turns, turn], turn };
}

/**
 * Close the reply being written, so a message sent mid-turn lands after it.
 *
 * **A queued message belongs where it was typed.** The composer takes one while
 * the agent still has the turn — the kernel queues it for the next loop
 * boundary — and the only place it can honestly go is between what the agent
 * had already said and everything that comes after. Appending it and leaving
 * the reply open put the rest of the reply *above* it, so the transcript
 * answered a question it had not yet been asked.
 *
 * So the open turn is closed here and the next frame opens a fresh one below.
 * Closing rather than leaving it running also decides where the working
 * indicator draws: `convert.ts` turns `running` into the message status, and
 * two running assistant messages would show two of them.
 *
 * **What it costs is a stream told in two halves**, which is why `carried`
 * exists. A `stream_delta` marked `done` carries `final_text` for the *whole*
 * stream and replaces whatever accumulated — landing that in the second half
 * would put the first half on screen twice. So the text already shown is
 * recorded per stream id on the way past, and `tailOf` takes it back off.
 *
 * **Text stays above the line; work still in flight moves below it.** Text
 * already on screen has been read, so splitting it is honest. A tool call has
 * not finished, and a tool-call part with no `result` inherits its *message's*
 * status (see `convert.ts`) — so leaving one in the message we are closing
 * would draw it as finished, and its real `finished` frame would then arrive
 * with nothing to update and appear again below. Moving it is what keeps one
 * call one block. That is why this returns a turn to place *after* the
 * person's: the caller owns the ordering, and the continuation has to come
 * second.
 *
 * **The continuation is opened here rather than left to the next frame**, even
 * when it starts empty. The agent still has the turn, and an empty running
 * message is what draws the working indicator — without one the transcript
 * goes silent between the moment you press send and whatever the agent says
 * next, which for a long tool call is a long time to look stopped. It also
 * makes the invariant exact: after this, exactly one assistant message is
 * running, so exactly one indicator can draw.
 */
function splitOpenTurn(
  turns: Turn[],
  carried: Record<string, string>,
): {
  turns: Turn[];
  carried: Record<string, string>;
  /** The reply's second half, to be appended after the person's message. Null
   *  only when no reply was open — an ordinary message, and much the commoner
   *  case. */
  continued: Turn | null;
} {
  const open = turns.at(-1);
  if (!open || open.role !== "assistant" || !open.running) {
    return { turns, carried, continued: null };
  }

  const kept: Part[] = [];
  const moved: Part[] = [];
  for (const part of open.parts) {
    if (part.kind === "tool" && part.status !== "finished") moved.push(part);
    else kept.push(part);
  }

  const next = { ...carried };
  for (const part of kept) {
    // Only a stream still being written can be continued below. A part already
    // `done` had its `final_text` applied in the half it belongs to.
    if (part.kind === "text" && !part.done) {
      next[part.streamId] = (next[part.streamId] ?? "") + part.text;
    }
  }

  const continued: Turn = {
    id: nextId(),
    turnId: open.turnId ?? open.id,
    role: "assistant",
    source: "live",
    parts: moved,
    running: true,
    aborted: false,
    createdAt: Date.now(),
  };

  // Nothing left to keep — either the agent had taken the turn without saying
  // anything yet, or everything it had was still in flight and has just moved.
  // An empty closed message is what `typing: false` filters away anyway, and
  // keeping one would draw a blank row above the person's line.
  if (kept.length === 0) {
    return { turns: turns.slice(0, -1), carried: next, continued };
  }

  return {
    turns: replace(turns, open.id, {
      ...open,
      turnId: open.turnId ?? open.id,
      continues: true,
      running: false,
      parts: kept.map((part) =>
        part.kind === "text" ? { ...part, done: true } : part,
      ),
    }),
    carried: next,
    continued,
  };
}

/**
 * The part of a finished stream's `final_text` that is not already on screen.
 *
 * `carried` is what an earlier half of a split reply is showing. The comparison
 * is on the text rather than on a character count because `final_text` is the
 * *cleaned* stream, and cleaning can move an offset while leaving the prose
 * recognisable — hence the second attempt against the untrimmed start.
 *
 * When neither lines up, the deltas that arrived in this half are kept. They
 * are what actually came over the wire, and showing the first half twice is a
 * worse failure than missing whatever the cleanup tidied.
 */
function tailOf(final: string, carried: string, accumulated: string): string {
  if (!carried) return final;
  if (final.startsWith(carried)) return final.slice(carried.length);
  const trimmed = carried.trimStart();
  if (trimmed && final.startsWith(trimmed)) return final.slice(trimmed.length);
  return accumulated;
}

/** Drop one stream from the carry map, which only ever holds live streams. */
function stopCarrying(
  carried: Record<string, string>,
  streamId: string,
): Record<string, string> {
  if (!(streamId in carried)) return carried;
  const next = { ...carried };
  delete next[streamId];
  return next;
}

/** End visible text at a presentation boundary while retaining final-text carry. */
function sealText(turn: Turn, carried: Record<string, string>) {
  const next = { ...carried };
  const parts = turn.parts.map((part) => {
    if (part.kind !== "text" || part.done) return part;
    next[part.streamId] = (next[part.streamId] ?? "") + part.text;
    return { ...part, done: true };
  });
  return { turn: { ...turn, parts }, carried: next };
}

/** Replace one turn in the list, leaving a new array behind.
 *
 *  Every update goes through here so that React sees a changed array identity;
 *  mutating a turn in place would leave assistant-ui rendering stale content. */
function replace(turns: Turn[], id: string, next: Turn): Turn[] {
  return turns.map((turn) => (turn.id === id ? next : turn));
}

/**
 * A tool call's arguments, with its narration among them.
 *
 * **`narration` arrives two ways and must end up in one.** The model writes it
 * as an argument — it is a reserved parameter name, and the kernel strips it
 * before the tool runs — and the wire *also* lifts it out to a field of its own
 * on `tool_status`. A conversation replayed from the database has only the
 * first, because only the arguments are stored. Keeping both would mean a field
 * that is reliably filled while you watch and reliably empty afterwards, which
 * is a worse trap than not having it: the transcript would quietly say
 * different things about the same call depending on when you looked.
 *
 * So the field folds back into the arguments it was taken from, here, once per
 * frame — rather than in `convert.ts`, which runs per message on every render
 * and is meant to make no decisions this reducer has not already made.
 *
 * The written argument wins where both exist; it is what the model actually
 * said. And the narration is *carried*, not just copied: it is repeated on
 * `finished` deliberately, but a frame that omits it while bringing new
 * arguments must not blank out what an earlier one established.
 */
function toolArgs(
  payload: { args?: Record<string, unknown>; narration?: string },
  existing: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
  const args = payload.args ?? existing;
  if (args && "narration" in args) return args;
  const carried = existing?.narration;
  const narration =
    payload.narration || (typeof carried === "string" ? carried : "");
  if (!narration) return args;
  return { ...args, narration };
}

/**
 * The kernel's bare acknowledgement that something was cancelled.
 *
 * **Recognised by its prose, which is nobody's idea of a good time.** Both
 * kinds it can arrive on — `messages` and `callable_output` — are a plain
 * `list[str]`, the one shape on this wire with nowhere to hang a flag, so the
 * alternative is a dict payload behind a capability flag: a protocol change
 * rather than a fix. What the kernel *does* say structurally is that a
 * cancellation happened at all (`approval_settled.reason`), and that is used as
 * the trigger below. This only has to pick the echo out of the frames that
 * follow it.
 *
 * Deliberately narrow: the *whole* frame has to be this one word, so a command
 * whose output merely mentions cancelling still prints.
 */
function isCancellationEcho(payload: string[]): boolean {
  return (
    payload.length > 0 &&
    payload.every((text) => /^cancelled\.?$/i.test(text.trim()))
  );
}

/* ── The reducer ────────────────────────────────────────────────────── */

export function reduce(state: State, action: Action): State {
  switch (action.type) {
    case "history":
      // A conversation switch or a cold boot. Everything transient goes with
      // it — a form belonging to the previous conversation is not answerable
      // any more.
      //
      // **A pending question does not live here, and that is load-bearing.**
      // This clause used to take one with it, which meant boot raced itself:
      // the stream replays the real question within a round trip, this read
      // takes two or three, and the later dispatch threw away what had just
      // arrived. See `runtime/input-requests.ts`.
      return {
        ...initialState,
        turns: action.turns,
        scrollback: { hasMore: action.hasMore, oldestId: action.oldestId },
      };

    case "olderTurns": {
      // Ids are the guard rather than a nicety. A page boundary can be re-read
      // — a retried request, a cursor asked for twice — and a duplicated turn
      // is not merely untidy: `toTurns` keys stored turns by row id, so React
      // would see two children with one key and drop one of them, silently.
      const known = new Set(state.turns.map((turn) => turn.id));
      const fresh = action.turns.filter((turn) => !known.has(turn.id));
      // The cursor moves even when the page carried nothing this can render —
      // a page of rows that are all bookkeeping still advances where the next
      // one starts, and `hasMore` going false is exactly how the affordance
      // knows to disappear.
      const scrollback = { hasMore: action.hasMore, oldestId: action.oldestId };
      const settled =
        scrollback.hasMore === state.scrollback.hasMore &&
        scrollback.oldestId === state.scrollback.oldestId;
      if (fresh.length === 0 && settled) return state;
      const joined = [...fresh, ...state.turns];
      const lastByTurn = new Map(joined.filter((turn) => turn.role === "assistant" && turn.turnId)
        .map((turn) => [turn.turnId, turn.id]));
      return { ...state, turns: joined.map((turn) =>
        turn.role === "assistant" && turn.turnId && lastByTurn.get(turn.turnId) !== turn.id
          ? { ...turn, continues: true } : turn), scrollback };
    }

    case "said": {
      // **Command interaction never enters the transcript.** Two shapes of it:
      // invoking one (`/tools`), and answering a step it asked — either by
      // typing or by pressing one of its buttons. Both are administration, and
      // echoing them is what turned the chat into a list of button presses.
      //
      // Note what is *not* here: a finished command still on screen does not
      // suppress anything. Once its questions are answered, the next line the
      // person types is an ordinary message again.
      //
      // **What counts as a command is the server's list, not the leading
      // slash.** Suppressing anything starting with "/" swallowed ordinary
      // messages: "/Users/henry/notes is where I keep this" went to the agent,
      // got a reply, and never appeared in the transcript — a conversation
      // answering a question nobody could see it asked. `isCommand` is checked
      // against `command.list` before the dispatch.
      const answering = state.form !== null;
      if (action.text.trim().toLowerCase() === "/cancel" && state.command) {
        return {
          ...state,
          form: null,
          buttons: [],
          suppressedCommand: {
            callId: state.command.callId,
            name: state.command.name,
          },
        };
      }
      if (answering || action.isCommand) {
        // The step has been sent, so the form goes; the command itself stays,
        // because it is about to say what it did.
        const commandName = action.isCommand
          ? action.text.trim().split(/\s/, 1)[0]?.slice(1).toLowerCase()
          : null;
        return {
          ...state,
          form: null,
          buttons: [],
          // Establish ownership before the server's status frame arrives.
          // `callable_output` and command status travel independently; without
          // this placeholder, output that wins the race is mistaken for a
          // directly invoked tool and sent to the chat-area fallback.
          command:
            commandName
              ? {
                  callId: `pending:${commandName}`,
                  name: commandName,
                  args: {},
                  status: "started",
                  outcome: [],
                }
              : state.command,
          // Starting another slash command retires the previous tombstone.
          suppressedCommand: answering ? state.suppressedCommand : null,
        };
      }

      const parts: Part[] = [];
      if (action.attachments?.length) {
        parts.push({
          kind: "files",
          paths: action.attachments.map(
            (attachment) => attachment.path ?? attachment.fileName,
          ),
          sent: true,
          attachments: action.attachments,
        });
      }
      if (action.text) {
        parts.push({
          kind: "text",
          streamId: nextId(),
          text: action.text,
          done: true,
        });
      }
      const turn: Turn = {
        id: nextId(),
        role: "user",
        parts,
        running: false,
        aborted: false,
        createdAt: Date.now(),
      };
      // Sent mid-reply, this closes the reply first so the line lands after
      // what the agent had already said and before whatever it says next. A
      // no-op for the ordinary case, where no turn is open.
      const split = splitOpenTurn(state.turns, state.carried);
      // An ordinary message also puts any finished command away — the person
      // has moved on, and its panel would otherwise sit there catching output
      // meant for the conversation.
      return {
        ...state,
        turns: split.continued
          ? [...split.turns, turn, split.continued]
          : [...split.turns, turn],
        carried: split.carried,
        form: null,
        command: null,
        suppressedCommand: null,
        buttons: [],
      };
    }

    case "hydrateSentAttachments": {
      const expected = action.attachments.map((file) => file.fileName);
      for (let index = state.turns.length - 1; index >= 0; index--) {
        const turn = state.turns[index];
        if (!turn || turn.role !== "user") continue;
        const files = turn.parts.find(
          (part): part is FilesPart => part.kind === "files" && part.sent === true,
        );
        if (!files) continue;
        const held = (files.attachments ?? []).map((file) => file.fileName);
        if (
          held.length !== expected.length ||
          held.some((name, at) => name !== expected[at])
        ) {
          continue;
        }
        const parts = turn.parts.map((part) =>
          part === files
            ? {
                ...files,
                paths: action.attachments.map((file) => file.path ?? file.fileName),
                attachments: action.attachments,
              }
            : part,
        );
        return {
          ...state,
          turns: state.turns.map((item, at) =>
            at === index ? { ...turn, parts } : item,
          ),
        };
      }
      return state;
    }

    case "compacted":
      // Keyed on the stored row's id, so asking twice — or a `/compact` that
      // found nothing to compact and left the previous marker as the newest —
      // cannot draw a second line for one compaction.
      if (state.turns.some((turn) => turn.id === action.turn.id)) return state;
      return { ...state, turns: [...state.turns, action.turn] };

    case "clearCommand":
      return { ...state, command: null, form: null };
    case "suppressNextCancellationNotice":
      return { ...state, suppressNextCancellationNotice: true };
    case "clearError":
      return { ...state, error: null };

    case "resumeTurn":
      return identifyTurn(state, action.turnId, true);

    case "frame": {
      const identity = frameTurnId(action.frame);
      if (identity) {
        const held = state.turns.filter((turn) => turn.role === "assistant" && turn.turnId === identity);
        // Identified completed history is authoritative over replayed frames.
        if (held.length && !held.some((turn) => turn.running)) return state;
        const active = state.turns.findLast((turn) => turn.running && turn.role === "assistant");
        if (active?.turnId && active.turnId !== identity && !active.turnId.startsWith("turn-")) return state;
        state = identifyTurn(state, identity);
      }
      const next = applyFrame(state, action.frame);
      if (next === state) return state;
      return {
        ...next,
        turns: next.turns.map((turn) => {
          if (turn.role !== "assistant" || !turn.running) return turn;
          const stream = turn.parts.findLast(
            (part): part is TextPart => part.kind === "text" && !part.done,
          );
          const phase = turn.activity?.phase === "waiting" ? "waiting" : stream ? "writing" : turn.parts.some((part) =>
            part.kind === "tool" && part.status !== "finished") ? "working" : "thinking";
          if (turn.activity?.phase === phase &&
              turn.activity.streamId === stream?.streamId) return turn;
          return {
            ...turn,
            activity: { phase, since: Date.now(), streamId: stream?.streamId },
          };
        }),
      };
    }
  }
}

function frameTurnId(frame: Frame): string | undefined {
  if (frame.kind === "turn_activity" || frame.kind === "stream_delta" ||
      (frame.kind === "tool_status" && frame.payload.kind !== "command")) {
    return frame.payload.turn_id || undefined;
  }
}

/** Bind a provisional live group, or resume its durable final segment on reload. */
function identifyTurn(state: State, turnId: string, resume = false): State {
  const last = state.turns.at(-1);
  const existing = state.turns.findLast((turn) => turn.role === "assistant" && turn.turnId === turnId);
  // A delayed frame must never rename a different, already identified turn.
  if (existing && existing !== last && !last?.running) {
    if (!resume) return state;
    const opened = openTurn(state.turns);
    return { ...state, typing: true, turns: opened.turns.map((turn) =>
      turn === opened.turn ? { ...turn, turnId } :
      turn.turnId === turnId ? { ...turn, continues: true } : turn) };
  }
  if (last?.role === "assistant" && last.turnId === turnId) {
    if (last.running) return state;
    return { ...state, typing: true, turns: state.turns.map((turn) =>
      turn === last ? { ...turn, running: true, continues: false } : turn) };
  }
  const opened = openTurn(state.turns);
  const provisional = opened.turn.turnId ?? opened.turn.id;
  // Only local provisional identities can be adopted by a new server ID.
  if (opened.turn.turnId && !opened.turn.turnId.startsWith("turn-")) return state;
  return { ...state, typing: true, turns: opened.turns.map((turn) =>
    (turn.turnId ?? turn.id) === provisional ? { ...turn, turnId } : turn) };
}

function applyFrame(state: State, frame: Frame): State {
  switch (frame.kind) {
    case "turn_activity": {
      const turn = state.turns.findLast((item) => item.role === "assistant" &&
        item.running && (!frame.payload.turn_id || item.turnId === frame.payload.turn_id));
      if (!turn || turn.activity?.phase === frame.payload.phase) return state;
      return { ...state, turns: replace(state.turns, turn.id, {
        ...turn, activity: { phase: frame.payload.phase, since: Date.now() },
      }) };
    }

    /* The agent takes or hands back the turn. */
    case "typing": {
      if (frame.payload) {
        const { turns } = openTurn(state.turns);
        return { ...state, typing: true, turns };
      }
      // Closing: everything still open is finished, including any text part
      // whose `done` frame never arrived (a crash forces `typing` back, and a
      // message stuck mid-write would otherwise pulse forever).
      const turns = state.turns
        .map((turn) =>
          turn.running
            ? {
                ...turn,
                running: false,
                parts: turn.parts.map((part) =>
                  part.kind === "text" ? { ...part, done: true } : part,
                ),
              }
            : turn,
        )
        // A turn that produced nothing at all leaves no row. `typing: true`
        // opens one before there is anything to put in it, so a turn the agent
        // ends without speaking — or one whose only output was a command's,
        // which belongs to the panel — would otherwise sit in the transcript as
        // a blank message.
        .filter((turn) => turn.parts.length > 0 || turn.running ||
          (turn.turnId && !turn.continues && state.turns.some((other) =>
            (other.turnId ?? other.id) === turn.turnId && other.parts.length > 0)));
      // Nothing is being written any more, so nothing can be continued into a
      // later turn. A backstop rather than the main path — a stream normally
      // retires its own entry on `done`.
      return { ...state, typing: false, turns, carried: {} };
    }

    /* The reply, token by token. */
    case "stream_delta": {
      const { stream_id, seq, delta, done, aborted, final_text } = frame.payload;
      const previous = state.streams[stream_id];
      if (previous?.done || (seq !== undefined && previous && seq <= previous.seq)) {
        return state;
      }
      state = {
        ...state,
        streams: {
          ...state.streams,
          [stream_id]: { seq: seq ?? (previous?.seq ?? 0) + 1, done },
        },
      };
      const opened = openTurn(state.turns);
      // A newly opened stream completes the preceding visible segment even if
      // its final frame was lost. A delayed final frame must not stop the new one.
      const boundary = !done && opened.turn.parts.some((part) =>
        part.kind === "text" && !part.done && part.streamId !== stream_id)
        ? sealText(opened.turn, state.carried)
        : { turn: opened.turn, carried: state.carried };
      const turns = opened.turns;
      const turn = boundary.turn;
      state = { ...state, carried: boundary.carried };
      const existing = turn.parts.find(
        (part): part is TextPart =>
          part.kind === "text" &&
          part.streamId === stream_id &&
          !part.done,
      );

      // **An aborted stream has no `final_text`.** Discard the partial rather
      // than leaving half a sentence on screen.
      if (done && aborted) {
        const parts = turn.parts.filter(
          (part) =>
            !(
              part.kind === "text" &&
              part.streamId === stream_id &&
              !part.done
            ),
        );
        return {
          ...state,
          carried: stopCarrying(state.carried, stream_id),
          turns: replace(turns, turn.id, { ...turn, parts, aborted: true }),
        };
      }

      // What an earlier half of this same stream is already showing, when a
      // message sent mid-sentence split the reply. Empty otherwise, which is
      // every conversation nobody interrupts.
      const shown = state.carried[stream_id] ?? "";
      const accumulated = (existing?.text ?? "") + (delta ?? "");

      // **On `done` with `final_text`, replace what accumulated.** It is the
      // cleaned text and the deltas agree with it, so appending would double
      // the reply and trusting the deltas would keep whatever it cleaned up.
      // It covers the *whole* stream, so a split one takes only the tail.
      const text =
        done && final_text !== undefined
          ? tailOf(final_text, shown, accumulated)
          : accumulated;

      const part: TextPart = { kind: "text", id: existing?.id ?? nextId(), streamId: stream_id, text, done };
      const parts = existing
        ? turn.parts.map((candidate) =>
            candidate === existing ? part : candidate,
          )
        : [...turn.parts, part];

      // The *whole* stream is what a later `messages` frame would repeat, so
      // that is what has to be remembered — plus the tail on its own, since
      // either could come back. Identical strings when nothing was split.
      const shownText = done
        ? [
            (shown + text).trim(),
            text.trim(),
            ...state.shownText,
          ].slice(0, HOW_MUCH_TEXT_TO_REMEMBER)
        : state.shownText;

      return {
        ...state,
        shownText,
        // A finished stream can no longer be continued in a later turn.
        carried: done ? stopCarrying(state.carried, stream_id) : state.carried,
        turns: replace(turns, turn.id, { ...turn, parts }),
      };
    }

    /* Whole messages, already complete. */
    case "messages": {
      if (
        state.suppressNextCancellationNotice &&
        isCancellationEcho(frame.payload)
      ) {
        return { ...state, suppressNextCancellationNotice: false };
      }

      // `session.cancel` deliberately keeps its acknowledgement on
      // `messages`: `/cancel` reads that Request result to distinguish a real
      // cancellation from "Nothing to cancel." A command cancelled from
      // Settings already disappeared visibly, so suppress only its exact
      // acknowledgement. This is not command-output routing; every other
      // message still belongs to the conversation even while a command exists.
      if (state.suppressedCommand && isCancellationEcho(frame.payload)) {
        return state;
      }

      let turns = state.turns;
      let turn: Turn | null = null;
      for (const text of frame.payload) {
        // Skip anything already streamed. See `shownText`.
        if (state.shownText.includes(text.trim())) continue;
        const opened = openTurn(turns);
        turns = opened.turns;
        turn = opened.turn;
        const parts: Part[] = [
          ...turn.parts,
          { kind: "text", id: nextId(), streamId: nextId(), text, done: true },
        ];
        turn = { ...turn, parts };
        turns = replace(turns, turn.id, turn);
      }
      return { ...state, turns };
    }

    /* A callable's return value is operational output, never conversation.
       Every such result belongs in Settings. When status has not established
       a named command (or the person invoked a tool directly), synthesize a
       completed run there rather than giving output a fallback in the thread. */
    case "callable_output": {
      /**
       * The cancellation acknowledgement, which is chrome rather than output.
       *
       * **Both of these used to live only in `messages`, and the frame moved
       * out from under them.** "Cancelled." is the result of a Request, so a
       * client declaring `supports_callable_output` — this one — receives it
       * here instead, where nothing was watching for it. What followed was the
       * fallback below inventing a command named `output` to hold it, and the
       * sidebar dutifully raising Settings to display a word.
       *
       * Two triggers because a cancellation has two origins. The flag is set
       * both optimistically when this client closes an approval and from the
       * kernel's own `approval_settled` — so a question cancelled by a peer or
       * timed out after 300s is covered too, and neither depends on which of
       * the two frames the kernel emits first. `suppressedCommand` is the other
       * origin: a command cancelled from Settings, which is already visibly
       * gone. That one deliberately does not test `state.command`, because
       * dismissing the panel nulls it before this frame arrives — which is
       * exactly how the Settings dialog came to reopen announcing its own
       * cancellation.
       */
      if (
        state.suppressNextCancellationNotice &&
        isCancellationEcho(frame.payload)
      ) {
        return { ...state, suppressNextCancellationNotice: false };
      }
      if (state.suppressedCommand && isCancellationEcho(frame.payload)) {
        return state;
      }

      if (
        state.command &&
        state.suppressedCommand?.callId === state.command.callId
      ) {
        return state;
      }
      if (!state.command) {
        return {
          ...state,
          command: {
            callId: `callable:${nextId()}`,
            name: "output",
            args: {},
            status: "finished",
            ok: true,
            outcome: [...frame.payload],
          },
        };
      }
      return {
        ...state,
        command: {
          ...state.command,
          outcome: [...state.command.outcome, ...frame.payload],
        },
      };
    }

    /* Tools and slash commands, which the wire reports the same way but which
       belong in different places. */
    case "tool_status": {
      const p = frame.payload;

      // A command runs the admin panel, never the transcript. A *tool* is the
      // agent working during a reply and stays in the message, because that is
      // genuinely part of what it said.
      if (p.kind === "command") {
        if (state.suppressedCommand?.callId === p.call_id) return state;
        const commandName = p.command_name ?? state.command?.name ?? "command";
        const same =
          state.command?.callId === p.call_id ||
          (state.command?.callId === `pending:${commandName}` &&
            state.command.name === commandName) ||
          // Output can cross before both the local placeholder and the status
          // frame (for example after a transient history reconciliation).
          // Its synthetic Settings run is still the only callable awaiting a
          // server identity, so adopt it without dropping what it printed.
          state.command?.callId.startsWith("callable:") === true;
        return {
          ...state,
          suppressedCommand: null,
          command: {
            callId: p.call_id,
            name: commandName,
            // Cumulative on the wire, so the latest frame is the whole answer
            // set — this is what makes the panel update as each one is given.
            args: p.args ?? (same ? state.command!.args : {}),
            status: p.status,
            narration: p.narration ?? (same ? state.command?.narration : undefined),
            ok: p.ok,
            error: p.error,
            // A new command replaces the last one's output; the same command
            // keeps accumulating it.
            outcome: same ? state.command!.outcome : [],
          },
        };
      }

      const opened = openTurn(state.turns);
      const knownCall = opened.turn.parts.some((part) => part.kind === "tool" && part.callId === p.call_id);
      const sealed = knownCall
        ? { turn: opened.turn, carried: state.carried }
        : sealText(opened.turn, state.carried);
      const { turn } = sealed;
      const turns = opened.turns;
      state = { ...state, carried: sealed.carried };
      const existing = turn.parts.find(
        (part): part is ToolPart =>
          part.kind === "tool" && part.callId === p.call_id,
      );
      const part: ToolPart = {
        kind: "tool",
        callId: p.call_id,
        name: p.tool_name ?? p.command_name ?? existing?.name ?? "tool",
        isCommand: p.kind === "command" || (existing?.isCommand ?? false),
        // Only `finished` carries this, so the earlier frames must not blank
        // out what a later one brought — and a kernel older than the field
        // simply leaves it empty forever, which renders as it did before.
        summary: p.summary ?? existing?.summary ?? "",
        status: p.status,
        // Which is also where the narration lands — see `toolArgs`.
        args: toolArgs(p, existing?.args),
        ok: p.ok ?? existing?.ok,
        error: p.error ?? existing?.error,
      };
      // `call_id` is stable across started/finished — update in place.
      const parts = existing
        ? turn.parts.map((x) =>
            x.kind === "tool" && x.callId === p.call_id ? part : x,
          )
        : [...turn.parts, part];
      return { ...state, turns: replace(turns, turn.id, { ...turn, parts }) };
    }

    /* Files the agent produced. Host paths, not URLs. */
    case "attachments": {
      if (!frame.payload.length) return state;
      // Attachment delivery may follow typing:false. It still belongs to the
      // last reply, not a new running message with a second footer.
      const last = state.turns.at(-1);
      const { turns, turn } = last?.role === "assistant"
        ? { turns: state.turns, turn: last }
        : openTurn(state.turns);
      const paths = [...new Set(frame.payload)];
      const callId = turn.parts.findLast((part) => part.kind === "tool")?.callId;
      const duplicate = turn.parts.some((part, index) =>
        part.kind === "files" && !part.sent &&
        (callId ? part.callId === callId : index === turn.parts.length - 1) &&
        part.paths.length === paths.length &&
        part.paths.every((path, at) => path === paths[at]),
      );
      if (duplicate) return state;
      const carried = { ...state.carried };
      const closed = turn.parts.map((part) => {
        if (part.kind !== "text" || part.done) return part;
        carried[part.streamId] =
          (carried[part.streamId] ?? "") + part.text;
        return { ...part, done: true };
      });
      const parts: Part[] = [
        ...closed,
        { kind: "files", id: nextId(), paths, callId, receivedAt: Date.now() },
      ];
      return {
        ...state,
        carried,
        turns: replace(turns, turn.id, { ...turn, parts }),
      };
    }

    /* ── Handled elsewhere, on purpose ──────────────────────────────────
     *
     * A question the kernel is blocking a turn on, the frame saying one stopped
     * waiting, and something the system is telling you. **Session state, not
     * conversation state**, all three: the kernel holds a pending question on
     * the session's phase stack and persists it there, and a notification is
     * usually not about the open conversation at all — a plugin registering is
     * about the install, a scheduled agent's report about a background session.
     * So both must survive everything `history` above resets, and the provider
     * fans them into `runtime/input-requests.ts` and `runtime/notifications.ts`
     * before this reducer ever sees them.
     *
     * They are listed anyway rather than left to a `default`, because an
     * unhandled kind here is a *compile* error — which is how the eleventh kind
     * got noticed instead of being silently dropped, exactly as this comment
     * said it would when there were ten. Any of these firing would mean the
     * fan-out in `provider.tsx` had been removed.
     */
    case "approval":
    case "notification":
    case "conversation":
      return state;

    /**
     * A question stopped being open — and, when it was cancelled, the one
     * structured warning that its acknowledgement is on its way.
     *
     * The dialog itself is reconciled in `runtime/input-requests.ts` like the
     * two above, which is why this was discarded outright until now. What it
     * adds here is a cancellation *this client may not have caused*: the frame
     * exists for a peer answering and for the kernel's own 300s timeout, and
     * both produce the same "Cancelled." echo as pressing the X does.
     */
    case "approval_settled":
      if (frame.payload.reason !== "cancelled") return state;
      return { ...state, suppressNextCancellationNotice: true };

    case "form_field":
      if (
        state.suppressedCommand?.name &&
        state.suppressedCommand.name === frame.payload.name
      ) {
        return state;
      }
      return { ...state, form: frame.payload };
    case "buttons":
      return { ...state, buttons: frame.payload };

    case "error":
      return { ...state, error: frame.payload };
  }
}
