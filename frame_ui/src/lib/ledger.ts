/**
 * Which files the agent touched, out of the kernel's ledger.
 *
 * The ledger durably records file effects and tool-returned attachments.
 * New records carry turn_id in their metadata so live and reloaded recaps can
 * join them to conversation messages without relying on browser receipt times.
 * A files drawer sourced from anything else is a files drawer that empties when
 * the page refreshes.
 *
 * One Request supplies all of it, and it is read-only — so it never raises an
 * approval dialog, and polling it costs nothing worth thinking about.
 *
 * What this deliberately does **not** claim to cover, because nothing records
 * it: files pushed by a task, service or slash command through
 * `runtime.push_message(attachments=…)`, which has no `ToolResult` to hang them
 * on and goes straight to the bus; `plugin.install`/`plugin.uninstall`, which
 * write through the package manager with no `fs.*` Request; and `config.write`.
 * The drawer shows what the agent did through the kernel, not every change on
 * disk, and it should not pretend otherwise.
 */

import { sdk } from "@/lib/client";

/** One row, as `ledger.read` hands it over. */
export type LedgerRow = {
  /** Monotonic, and also the `since_id` cursor. */
  id: number;
  /** Epoch **seconds**, fractional. Note the units — `Turn.createdAt` is
   *  milliseconds, and the two meet in `runtime/file-activity.ts`. */
  ts: number;
  origin: string;
  action_type: string;
  conversation_id: number | null;
  ok: 0 | 1;
  error_code: string | null;
  /** **Not to be parsed for paths.** See `toFileEvents`. */
  args_json: string;
  data_json: string;
};

/** The actions worth asking for. Everything else in the ledger — a Request
 *  answered, a policy consulted — records no paths and would just be rows to
 *  filter out on this side. */
export const FILE_ACTIONS = [
  "fs.write",
  "fs.write_bytes",
  "fs.delete",
  "fs.move",
  "proc.run",
  "proc.start",
  "call_tool",
] as const;

/**
 * One thing that happened to one file.
 *
 * A row can name several paths — `fs.move` names two, a shell command any
 * number — so this is deliberately per-path rather than per-row. The row id
 * rides along so several events from one row can still be told apart from
 * several rows about one file.
 */
export type FileEvent = {
  rowId: number;
  turnId?: string;
  /** Epoch **milliseconds**, converted here so nothing downstream has to
   *  remember that the ledger speaks seconds. */
  ts: number;
  path: string;
  effect: FileEffect;
  /**
   * The paths were read out of a command line rather than serviced by the
   * kernel.
   *
   * A weaker claim than the rest, and flagged rather than hidden: `rm -rf
   * build` is parsed for its arguments, so the kernel is reporting what the
   * command *said* it would do to a path that it then confirmed. Worth a badge;
   * not worth discarding.
   */
  viaShell: boolean;
  bytes?: number;
  command?: string;
};

export type FileEffect =
  | "shown"
  | "wrote"
  | "deleted"
  | "moved-from"
  | "moved-to";

/** What `data_json` might hold. Everything is optional because the shape is the
 *  discriminator: an `attachments` row is a file the agent showed you, a
 *  `paths` row is one it changed. */
type LedgerData = {
  turn_id?: unknown;
  /** Always present. `"http:web -> edit_file"` — where the call came from, and
   *  at the far end whatever actually made it. */
  chain?: unknown;
  paths?: unknown;
  attachments?: unknown;
  deleted?: unknown;
  via?: unknown;
  command?: unknown;
  bytes?: unknown;
};

/**
 * Whatever actually made the call: the last hop of the chain.
 *
 * **This, and not `origin`, is what tells the agent's writes from ours.** A
 * file this browser uploads goes through the same `fs.write_bytes` and is
 * recorded with `origin: "sandbox"` exactly like a file the agent wrote — the
 * two are indistinguishable there. The chain is not ambiguous:
 * `"http:web -> frontend:http"` for our own upload against
 * `"http:web -> edit_file"` for the agent's, because the last hop names the
 * caller rather than the transport.
 */
function actorOf(data: LedgerData): string {
  const chain = typeof data.chain === "string" ? data.chain : "";
  return chain.split("->").pop()?.trim() ?? "";
}

function parse(raw: string): LedgerData | null {
  try {
    const value: unknown = JSON.parse(raw);
    return value !== null && typeof value === "object"
      ? (value as LedgerData)
      : null;
  } catch {
    return null;
  }
}

/** The string members of a value that should have been an array of paths. */
function paths(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

/**
 * Rows → events, newest first as they arrived.
 *
 * Three rules worth stating, because each one was a decision:
 *
 * **Only `data_json` is parsed.** `args_json` is capped at 4000 characters and
 * past the cap the whole object is replaced by a `{_truncated_chars, head,
 * tail}` wrapper — and the argument that blows the cap is the file's own
 * contents. Reading paths out of it would work on every small edit and silently
 * lose exactly the large ones.
 *
 * **A failed row records nothing.** The kernel already declines to name paths
 * for a command that exited non-zero; this drops the rest. Under-reporting is
 * the deliberate direction: a file shown as deleted that is still there is
 * worse than one missing from the list.
 *
 * **The chain separates the agent's work from ours.** This browser writes to
 * the host too — `uploadToHost` puts an attachment in scratch through the same
 * `fs.write_bytes` — and listing those would put the person's own attachments
 * in a panel labelled as what the agent did. `origin` cannot tell them apart:
 * both are `"sandbox"`, verified against a live ledger. The chain's last hop
 * can, and does. See `actorOf`.
 */
export function toFileEvents(rows: LedgerRow[]): FileEvent[] {
  const events: FileEvent[] = [];

  for (const row of rows) {
    if (row.ok !== 1) continue;
    const data = parse(row.data_json);
    if (!data) continue;

    const ts = row.ts * 1000;
    const identity = typeof data.turn_id === "string" && data.turn_id ? { turnId: data.turn_id } : {};
    const shown = paths(data.attachments);

    // A file the agent chose to show you. `call_tool` from `agent_enact`, which
    // is `ToolResult.attachment_paths` — whatever a tool returned from
    // `sdk.ok(attachments=[...])`.
    for (const path of shown) {
      events.push({ ...identity, rowId: row.id, ts, path, effect: "shown", viaShell: false });
    }
    if (shown.length) continue;

    const changed = paths(data.paths);
    if (!changed.length) continue;
    if (actorOf(data).startsWith("frontend:")) continue;

    const viaShell = data.via === "shell";
    const command = typeof data.command === "string" ? data.command : undefined;
    const bytes = typeof data.bytes === "number" ? data.bytes : undefined;
    const gone = new Set(paths(data.deleted));

    for (const [index, path] of changed.entries()) {
      events.push({
        ...identity,
        rowId: row.id,
        ts,
        path,
        // `fs.move` carries both ends, source first, and the two ends are
        // different facts: one file stopped existing and another started.
        effect:
          row.action_type === "fs.move"
            ? index === 0
              ? "moved-from"
              : "moved-to"
            : row.action_type === "fs.delete" || gone.has(path)
              ? "deleted"
              : "wrote",
        viaShell,
        ...(bytes === undefined ? {} : { bytes }),
        ...(command === undefined ? {} : { command }),
      });
    }
  }

  return events;
}

/**
 * Read the conversation's file activity.
 *
 * `sinceId` filters in SQL on the `(conversation_id, id)` index, so an idle
 * poll is cheap — which is what makes keeping this current a matter of asking
 * again rather than of subscribing to something. There is no push notification
 * for ledger rows; the existing frames are the trigger.
 *
 * Rows come back newest first.
 */
export async function readLedger(
  conversationId: number,
  sinceId?: number,
): Promise<LedgerRow[]> {
  const all: LedgerRow[] = [];
  let before: number | undefined;
  const limit = 50;
  for (;;) {
    const rows = await sdk<LedgerRow[]>("ledger.read", {
      conversation_id: conversationId,
      action_types: [...FILE_ACTIONS],
      limit,
      ...(sinceId === undefined ? {} : { since_id: sinceId }),
      ...(before === undefined ? {} : { before_id: before }),
    });
    if (!Array.isArray(rows) || !rows.length) return all;
    const oldest = Math.min(...rows.map((row) => row.id));
    if (before !== undefined && oldest >= before) {
      throw new Error("File history pagination did not advance.");
    }
    all.push(...rows);
    if (rows.length < limit) return all;
    before = oldest;
  }
}
