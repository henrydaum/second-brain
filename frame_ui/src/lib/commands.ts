/**
 * The command catalogue, and what running one means.
 *
 * The server hands over its entire vocabulary — name, description, category —
 * so Settings can organize what actually exists without maintaining a second
 * command list.
 *
 * `command.list` also describes the form each command collects, and that is
 * deliberately not modelled here. Collecting arguments is the server's job,
 * arriving as `form_field` frames that `command-panel.tsx` draws; holding a
 * copy of the shape client-side is how the old draft ended up parsing slash
 * commands by hand.
 */

import { sdk } from "@/lib/client";

export type Command = {
  name: string;
  description?: string;
  /** "System", "Conversation", "Capabilities", "Automation". */
  category?: string;
};

/**
 * Every command this session can run.
 *
 * `visible: true` asks the server to filter by what makes sense on this
 * frontend. It currently returns all 22 either way, which is fine — the point
 * is that the filtering is the server's call to make, not ours to guess.
 */
export async function listCommands(): Promise<Command[]> {
  const data = await sdk<Command[] | { items?: Command[] } | null>(
    "command.list",
    { details: true, visible: true },
  );
  // `conv.list` answers `{items}` with `details` and a bare array without, so
  // the shape is worth being tolerant about rather than assuming.
  const commands = Array.isArray(data) ? data : (data?.items ?? []);
  return [...commands].sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Is this line an invocation of a command the server actually has?
 *
 * **A leading slash is not the test.** The store used to suppress every message
 * beginning with "/" — because command interaction belongs in the Settings
 * panel rather than the transcript — and that quietly ate real messages: a path
 * (`/Users/henry/notes`), a fraction, a closing tag. They still reached the
 * agent and the agent still answered; only the person's own line was missing,
 * which reads as the chat replying to nothing.
 *
 * Checking the first word against `command.list` is the same judgement the
 * server's own state machine makes, so the two cannot disagree about what a
 * command is. An empty catalogue answers `false` for everything, which is the
 * right way to be wrong: a command shown in the chat is untidy, a message
 * deleted from it is a bug.
 */
export function looksLikeCommand(text: string, commands: Command[]): boolean {
  const first = text.trim().split(/\s/, 1)[0];
  if (!first?.startsWith("/")) return false;
  const name = first.slice(1).toLowerCase();
  return commands.some((command) => command.name.toLowerCase() === name);
}
