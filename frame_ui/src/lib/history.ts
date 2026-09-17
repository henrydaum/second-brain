/**
 * Second Brain's stored messages, as turns the thread can be seeded with.
 *
 * **The filtering is the important part.** A conversation's rows include
 * `role: "system"` entries carrying the state machine's serialised state —
 * `{"__second_brain_state_machine__": true, ...}` — which is kernel bookkeeping
 * stored in the same table, not anything a person said or was told. Rendering
 * those would put the kernel's internals in the chat window.
 *
 * **One system row is the exception**, and it is the one that is not
 * bookkeeping: a compaction marker. It says the agent's view of this
 * conversation was replaced by a summary at that point, and the rows above it
 * are still shown because they are still what happened — see `compaction`.
 *
 * **Tool calls are rebuilt, not dropped.** A `tool` row is only meaningful
 * beside the assistant row whose `tool_calls` it answers, so the pairing has to
 * be reconstructed from flat rows — done below, because the alternative is a
 * scrollback that quietly disagrees with what you watched happen: tools appear
 * while a turn runs and vanish when the page reloads.
 *
 * That pairing is also why consecutive assistant and tool rows are merged into
 * one turn. The kernel writes an agent turn as several rows — the call, its
 * result, the reply that follows — and the live stream renders exactly that as a
 * single message. One turn per row would split it into two or three.
 *
 * Ported from the AG-UI draft, where this logic was arrived at the hard way. The
 * only thing that changed is where the rows come from.
 */

import { sdk } from "@/lib/client";
import type { Conversation } from "@/lib/conversations";
import type {
  MessageAttachment,
  ToolPart,
  Turn,
} from "@/runtime/store";

type StoredAttachment = {
  path?: unknown;
  file_name?: unknown;
  modality?: unknown;
  extension?: unknown;
};

/** One row of the `messages` table, as `conv.read` hands it over. */
export type StoredMessage = {
  id: number;
  turn_id?: string | null;
  role: string;
  content: string;
  tool_call_id: string | null;
  tool_name: string | null;
  /** Who actually wrote this row. Non-null means the kernel synthesized it
   *  for the model, even when its role is `user`. */
  author?: string | null;
  /** Files this message carried. New kernels always return a list; optional so
   * conversations from the older row shape remain readable. */
  attachments?: StoredAttachment[] | null;
  /**
   * When the row was stored, as **fractional epoch seconds** — e.g.
   * `1786158850.6803284`.
   *
   * Confirmed against a live `conv.read`, which is worth saying because the
   * protocol document specifies that call's envelope but not its columns, and
   * because the units are the trap here: the conversation rows `conv.list`
   * returns spell their times `created_at`/`updated_at` while message rows
   * spell theirs `timestamp`. Reading seconds as milliseconds dates every
   * message to January 1970.
   */
  timestamp?: number | null;
};

function messageAttachments(
  stored: StoredAttachment[] | null | undefined,
): MessageAttachment[] {
  if (!Array.isArray(stored)) return [];
  const attachments: MessageAttachment[] = [];
  for (const item of stored) {
    if (!item || typeof item !== "object") continue;
    if (typeof item.path !== "string" || !item.path) continue;
    const fileName =
      typeof item.file_name === "string" && item.file_name
        ? item.file_name
        : item.path.split(/[\\/]/).pop() || item.path;
    attachments.push({
      path: item.path,
      fileName,
      modality:
        typeof item.modality === "string" ? item.modality : "unknown",
      extension:
        typeof item.extension === "string" ? item.extension : "",
    });
  }
  return attachments;
}

/** A stored row's time as epoch milliseconds, which is what `Date` wants. */
function momentOf(message: StoredMessage): number | undefined {
  const seconds = message.timestamp;
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds <= 0) {
    return undefined;
  }
  return seconds * 1000;
}

/** One entry of an assistant row's `tool_calls`, in the provider's shape. */
type StoredToolCall = {
  id?: string;
  function?: { name?: string; arguments?: string };
};

/** A JSON object a row's `content` might be, or null when it is not one. */
function asObject(raw: string): Record<string, unknown> | null {
  if (!raw.trimStart().startsWith("{")) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    return parsed !== null && typeof parsed === "object"
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    // Not JSON after all — a message that merely opens with a brace.
    return null;
  }
}

/**
 * The text of one stored row, or null if it holds no prose.
 *
 * **Assistant rows are not plain text.** They hold the provider payload
 * serialised whole — `{"content": "…", "tool_calls": [...]}` — so rendering
 * `content` directly puts raw JSON in the chat window. A row whose `content` is
 * null carries only tool calls and has nothing to show.
 *
 * User rows are plain strings, so the parse is attempted and simply declined.
 */
function prose(raw: string): string | null {
  let text = raw;
  const parsed = asObject(raw);
  if (parsed) {
    if (typeof parsed.content !== "string") return null;
    text = parsed.content;
  }
  // Reasoning is stripped rather than shown. The live stream does not render it
  // inline either, so leaving it in would make scrollback disagree with what you
  // watched arrive.
  text = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  return text === "" ? null : text;
}

/**
 * The marker a compaction leaves behind, or null for any other system row.
 *
 * The kernel packs one of these when it folds the history into a summary —
 * `pack_compaction` in `state_machine/serialization.py` — and from then on
 * `messages_to_history` builds the agent's context from the summary plus
 * whatever was stored *after* the marker. Nothing ever removes one, so this is
 * a permanent fact about the conversation rather than a passing state.
 *
 * `tail_count` is deliberately not read. It records how many trailing rows the
 * live session kept beside the summary, which is true of the session that did
 * the compacting and not of the conversation once it is reloaded — a number
 * that stops being true is worse than no number.
 */
function compaction(
  raw: string,
): { summary: string; createdAt?: number } | null {
  const parsed = asObject(raw);
  if (!parsed || parsed.__second_brain_compaction__ !== true) return null;
  const created = parsed.created_at;
  return {
    summary: typeof parsed.summary === "string" ? parsed.summary : "",
    // Fractional epoch seconds, like every other time on this wire.
    createdAt:
      typeof created === "number" && Number.isFinite(created) && created > 0
        ? created * 1000
        : undefined,
  };
}

/** The calls an assistant row announces, as parts waiting for their answers. */
function toolParts(raw: string): ToolPart[] {
  const calls = asObject(raw)?.tool_calls;
  if (!Array.isArray(calls)) return [];

  const parts: ToolPart[] = [];
  for (const call of calls as StoredToolCall[]) {
    const callId = call?.id;
    if (typeof callId !== "string") continue;
    // Arguments are a *string* of JSON inside the row's JSON — the provider's
    // own double encoding, not a quirk of this table.
    let args: Record<string, unknown> | undefined;
    const raw_args = call.function?.arguments;
    if (typeof raw_args === "string") args = asObject(raw_args) ?? undefined;
    parts.push({
      kind: "tool",
      callId,
      name: call.function?.name ?? "tool",
      isCommand: false,
      summary: "",
      // A stored call is over by definition; whether it worked is decided by
      // the answering row below.
      status: "finished",
      args,
      ok: true,
    });
  }
  return parts;
}

/**
 * Fold one `tool` row into the call it answers.
 *
 * The inverse of the kernel's `_format_tool_result`: a failure is stored as
 * `{"error": "…"}` and anything else is the summary verbatim. Reading a failure
 * as prose would print a JSON object at people, which is the one shape that
 * must not survive the trip.
 */
function answer(part: ToolPart, content: string): void {
  const parsed = asObject(content);
  const error = parsed?.error;
  if (typeof error === "string") {
    part.ok = false;
    part.error = error;
    return;
  }
  part.summary = content;
}

/**
 * Stored rows → turns.
 *
 * A user row always starts a fresh turn. Assistant and tool rows accumulate
 * into the one that is open, because the kernel spreads a single agent turn
 * across several of them — see the note at the top of this file.
 */
export function toTurns(stored: StoredMessage[]): Turn[] {
  const turns: Turn[] = [];
  // The assistant turn being assembled, and the calls inside it still waiting
  // for their result rows. Both are cleared together: a call whose answer never
  // arrives is not going to be answered by the next turn's rows.
  let open: Turn | null = null;
  let pending = new Map<string, ToolPart>();

  for (const message of stored) {
    if (typeof message.content !== "string") continue;

    if (message.role === "tool") {
      const part = message.tool_call_id
        ? pending.get(message.tool_call_id)
        : undefined;
      if (part) answer(part, message.content);
      // Newer kernels may persist returned attachments on the tool row itself.
      // These are output records, not guesses from tool arguments or prose.
      const attachments = messageAttachments(message.attachments);
      const owner = open ?? (message.turn_id
        ? turns.findLast((turn) => turn.role === "assistant" && turn.turnId === message.turn_id)
        : undefined);
      if (owner && attachments.length && !part?.error) owner.parts.push({
        kind: "files", id: `stored-files-${message.id}`,
        paths: attachments.map((file) => file.path!),
        receivedAt: momentOf(message),
      });
      continue;
    }

    if (message.role === "user") {
      // Kernel notes deliberately wear the user's role because they are
      // addressed to the model. `author` is the wire's attribution; rendering
      // these as something the person said would misrepresent the transcript.
      if (message.author) continue;
      open = null;
      if (!message.turn_id) pending = new Map();
      const text = prose(message.content);
      const attachments = messageAttachments(message.attachments);
      if (text === null && attachments.length === 0) continue;
      const parts: Turn["parts"] = [];
      if (attachments.length) {
        parts.push({
          kind: "files",
          paths: attachments.map((attachment) => attachment.path!),
          sent: true,
          attachments,
        });
      }
      if (text !== null) {
        parts.push({
          kind: "text",
          streamId: `stored-${message.id}`,
          text,
          done: true,
        });
      }
      turns.push({
        // Row ids are the database's own and unique within a conversation,
        // which is exactly the stability assistant-ui wants from a message key.
        id: `stored-${message.id}`,
        role: "user",
        parts,
        running: false,
        aborted: false,
        createdAt: momentOf(message),
      });
      continue;
    }

    if (message.role === "system") {
      // The one system row a person is meant to see. Everything else wearing
      // this role is the state machine's serialised state and is skipped below
      // with the rest of the bookkeeping.
      const marker = compaction(message.content);
      if (!marker) continue;
      // A marker closes whatever assistant turn was being assembled. The rows
      // after it are a fresh context, and merging across the line would draw
      // one turn straddling it — with a tool call above the line answered by a
      // result below it.
      open = null;
      pending = new Map();
      turns.push({
        id: `stored-${message.id}`,
        role: "system",
        // Exactly one text part, because that is what assistant-ui accepts for
        // this role — and the summary is the row's own text, so nothing has to
        // be invented to fill it. Nothing renders it today; see
        // `components/compaction-marker.tsx` for what is drawn instead.
        parts: [
          {
            kind: "text",
            streamId: `stored-${message.id}`,
            text: marker.summary,
            done: true,
          },
        ],
        running: false,
        aborted: false,
        // The row's own column first. `created_at` inside the payload is the
        // same instant recorded by the packer, and is the fallback for a
        // kernel whose `conv.read` does not hand back `timestamp`.
        createdAt: momentOf(message) ?? marker.createdAt,
      });
      continue;
    }

    // Everything else is kernel bookkeeping and belongs nowhere near the chat
    // window.
    if (message.role !== "assistant") continue;

    const calls = toolParts(message.content);
    const text = prose(message.content);
    const attachments = messageAttachments(message.attachments);
    if (text === null && calls.length === 0 && !attachments.length) continue;

    if (open && message.turn_id && open.turnId !== message.turn_id) open = null;
    if (open === null) {
      open = {
        turnId: message.turn_id ?? undefined,
        id: `stored-${message.id}`,
        role: "assistant",
        parts: [],
        running: false,
        aborted: false,
        // The moment the turn *started*, which is this first row — matching the
        // live path, where a turn is stamped as it opens.
        createdAt: momentOf(message),
      };
      turns.push(open);
    }

    // Text first: a row carrying both is the model saying what it is about to
    // do and then doing it, which is the order the live stream shows too.
    if (text !== null) {
      open.parts.push({
        kind: "text",
        streamId: `stored-${message.id}`,
        text,
        done: true,
      });
    }
    for (const part of calls) {
      open.parts.push(part);
      pending.set(part.callId, part);
    }
    if (attachments.length) open.parts.push({
      kind: "files", id: `stored-files-${message.id}`,
      paths: attachments.map((file) => file.path!),
      receivedAt: momentOf(message),
    });
  }
  // User interruptions split visual segments, never the durable logical turn.
  const last = new Map<string, Turn>();
  for (const turn of turns) {
    if (turn.role !== "assistant" || !turn.turnId) continue;
    const previous = last.get(turn.turnId);
    if (previous) previous.continues = true;
    last.set(turn.turnId, turn);
  }
  return turns;
}

export type ConversationRead = {
  turns: Turn[];
  /**
   * Whether the conversation continues above what came back.
   *
   * `conv.read` answers with a *page* — the newest one unless asked otherwise
   * — because a transcript grows without limit and a whole one eventually
   * exceeds what the kernel can put on one wire message. It used to answer
   * with everything, which is how an 18 MB conversation stopped the frontend
   * from serving anything at all.
   */
  hasMore: boolean;
  /** The id of the oldest row on this page — the cursor for the next one up.
   *  `null` only when the page is empty. */
  oldestId: number | null;
  /**
   * The conversation's own row.
   *
   * **The open conversation is session state, not a row in the sidebar's
   * list.** The list holds one page of one category now, so looking the open
   * conversation up in it fails the moment it is filed somewhere the filter
   * excludes — and the header, which is the thing that files it, would vanish
   * the instant you used it. `conv.read` has always answered this alongside
   * the messages.
   */
  conversation: Conversation | null;
};

/**
 * Read a conversation and turn it into scrollback.
 *
 * The response is unwrapped defensively because `conv.read` is the one Request
 * here whose envelope is not stated in the protocol document — it may hand back
 * the rows directly or nest them under a conversation record. Both are accepted
 * rather than guessed at, since guessing wrong shows an empty history and says
 * nothing about why.
 *
 * `details: true` was already being asked for, and the answer already carried
 * the conversation's own row; this only stopped throwing it away. The header
 * therefore costs no Request of its own — it reads what opening the
 * conversation had already fetched.
 */
/** How to page. `before` walks upwards from a row already on screen. */
export type ReadOptions = {
  /** Ask for the newest rows *older* than this id, rather than the newest. */
  before?: number | null;
  /** Rows to ask for. The server bounds the answer by bytes regardless, so
   *  this is a preference and not a guarantee — see `hasMore`. */
  limit?: number | null;
};

export async function readConversation(
  id: number,
  options: ReadOptions = {},
): Promise<ConversationRead> {
  const request: Record<string, unknown> = { id, details: true };
  if (typeof options.before === "number") request.before_id = options.before;
  if (typeof options.limit === "number") request.limit = options.limit;

  const data = await sdk<
    | StoredMessage[]
    | {
        messages?: StoredMessage[];
        conversation?: Conversation | null;
        has_more?: boolean;
        oldest_id?: number | null;
      }
    | null
  >("conv.read", request);

  // The bare-array shape is an older kernel, which had no paging and answered
  // with the whole conversation. Saying `hasMore: false` is right for it:
  // there was never another page to ask for.
  if (Array.isArray(data)) {
    return {
      turns: toTurns(data),
      conversation: null,
      hasMore: false,
      oldestId: data.length ? (data[0]?.id ?? null) : null,
    };
  }

  const rows = data?.messages ?? [];
  const row = data?.conversation;
  return {
    turns: toTurns(rows),
    // An older kernel answers `{}` rather than omitting it; a row with no id is
    // not a conversation, and treating it as one would put an untitled ghost in
    // the header.
    conversation: row && typeof row.id === "number" ? row : null,
    hasMore: data?.has_more === true,
    // Preferring the server's own cursor over `rows[0].id` is not pedantry:
    // the two agree today, and the server is the one that knows what it
    // skipped on the way — a page made entirely of rows a client does not
    // render would otherwise report a cursor it never saw.
    oldestId:
      typeof data?.oldest_id === "number"
        ? data.oldest_id
        : (rows[0]?.id ?? null),
  };
}

/** Scan persisted outputs independently of the transcript's loaded window.
 * Parse after joining pages so tool calls and results can straddle a boundary.
 */
export async function readConversationFileTurns(
  id: number, cancelled: () => boolean = () => false,
): Promise<Turn[]> {
  const pages: StoredMessage[][] = [];
  let before: number | undefined;
  while (!cancelled()) {
    const data = await sdk<StoredMessage[] | {
      messages?: StoredMessage[]; has_more?: boolean; oldest_id?: number | null;
    }>("conv.read", { id, details: true, ...(before === undefined ? {} : { before_id: before }) });
    if (cancelled()) return [];
    const rows = Array.isArray(data) ? data : data?.messages ?? [];
    pages.push(rows);
    if (Array.isArray(data) || !data?.has_more) break;
    const oldest = data.oldest_id ?? rows[0]?.id;
    if (typeof oldest !== "number" || (before !== undefined && oldest >= before)) {
      throw new Error("Conversation file history pagination did not advance.");
    }
    before = oldest;
  }
  return toTurns(pages.reverse().flat());
}
