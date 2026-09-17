/**
 * `Turn` → `ThreadMessageLike`, the shape assistant-ui renders.
 *
 * This is the whole of the translation layer, and it is deliberately dumb: no
 * fetching, no state, no decisions the store did not already make. Anything
 * clever here would be a second interpretation of the protocol competing with
 * `store.ts`.
 *
 * `ExternalStoreRuntime` calls this per message on every render, so it must stay
 * cheap and it must be pure.
 */

import type { ThreadMessageLike } from "@assistant-ui/react";
import type { ReadonlyJSONObject } from "assistant-stream/utils";
import type { MessageAttachment, Turn } from "@/runtime/store";

/** The `name` on the data part carrying the *person's* attachments. `thread.tsx`
 *  maps this name to the component that names them. */
export const SENT_ATTACHMENTS = "sentAttachments";
export const AGENT_FILES = "agentFiles";
export const PRESENTATION = "presentation";

/** Key under `metadata.custom` holding the turn's time in epoch milliseconds.
 *  Only present when the time is actually known — see `timing` below. */
export const SENT_AT = "sentAt";

/** Key under `metadata.custom` marking a message as a compaction marker rather
 *  than anything anybody said. `components/compaction-marker.tsx` draws it, and
 *  keys on this rather than on the role: a system message assistant-ui cannot
 *  explain is one it should keep drawing nothing for, which is its own
 *  default. */
export const COMPACTED = "compacted";

/** The element type of a message's content, named so the map below can be
 *  annotated — without it TypeScript widens the three branches into a union
 *  full of `undefined` members that no longer matches. */
type Part = Exclude<ThreadMessageLike["content"], string>[number];

export function convertMessage(turn: Turn): ThreadMessageLike {
  const sentAttachments = turn.parts.flatMap((part): MessageAttachment[] =>
    part.kind === "files" && part.sent ? (part.attachments ?? []) : [],
  );
  // `flatMap` rather than `map`, because one kind of part deliberately produces
  // nothing to render — see the `files` case.
  const content = turn.parts.flatMap((part, index): Part[] => {
    switch (part.kind) {
      case "text":
        return [{
          type: "text" as const,
          text: part.text,
          ...(turn.role === "assistant" ? {
            status: { type: part.done || !turn.running ? "complete" as const : "running" as const },
            providerMetadata: { secondBrain: { segmentId: part.id ?? `${turn.id}:text:${index}` } },
          } : {}),
        }];

      case "tool":
        return [{
          type: "tool-call" as const,
          toolCallId: part.callId,
          toolName: part.name,
          // The wire types `args` as an open record because that is what JSON
          // is; assistant-ui wants the narrower "this is really JSON" type. The
          // cast is the seam between the two vocabularies, and it is safe
          // because the value came off `JSON.parse`.
          //
          // `narration` is among these, put there by the store rather than
          // found here — see `toolArgs` in `store.ts`.
          args: (part.args ?? {}) as ReadonlyJSONObject,
          // A tool-call part with no `result` inherits the *message's* status,
          // so a finished tool inside a still-running turn would spin forever.
          // Supplying one on `finished` is what flips it to done — and the
          // empty string counts, because the test is `=== undefined`.
          //
          // The failure reason or the outcome, never both: the kernel leaves
          // `summary` empty when a call fails, precisely so that one of these
          // is the answer. `narration` is deliberately not in this chain — it
          // is what the agent set out to do, which is not a result.
          ...(part.status === "finished"
            ? {
                result: part.error || part.summary || "",
                isError: part.ok === false,
              }
            : {}),
        }];

      case "files":
        /**
         * The person's attachments live in metadata and render above their
         * prose. Agent attachments become named data parts so their exact
         * position among live text and tool parts survives conversion.
         *
         * The ledger remains the reload fallback because conversation history
         * does not persist attachment frames or their intra-message offsets.
         */
        return part.sent
          ? []
          : [{
              type: "data" as const,
              name: AGENT_FILES,
              data: { paths: part.paths, id: part.id ?? `${turn.id}:files:${index}`, receivedAt: part.receivedAt },
            }];
    }
  });

  // **`status` is only legal on an assistant message**, and assistant-ui throws
  // rather than ignoring it on a user one. That is reasonable — a message the
  // person already sent has no run to be in — but it means the field cannot be
  // set unconditionally, which is easy to miss because a user turn is always
  // `running: false` and looks harmless.
  /**
   * The turn's time, said twice, because assistant-ui cannot say "unknown".
   *
   * `fromThreadMessageLike` fills a missing `createdAt` with `new Date()`, so
   * reading `message.createdAt` back can never distinguish a message that
   * genuinely arrived now from one that was read out of the database with no
   * stored time — and the second would be dated to the page load. The real
   * value therefore also travels in `metadata.custom`, where nothing defaults
   * it, and that is the one the UI renders. `createdAt` is still set when known
   * so the library's own view of the message is correct.
   */
  const custom: Record<string, unknown> = {};
  if (turn.role === "assistant") {
    const activeStream = turn.parts.findLast(
      (part) => part.kind === "text" && !part.done,
    );
    custom[PRESENTATION] = {
      source: turn.source ?? "history",
      turnId: turn.turnId ?? turn.id,
      continues: turn.continues === true,
      phase: turn.activity?.phase === "waiting" ? "waiting" : activeStream ? "writing" :
        turn.parts.some((part) => part.kind === "tool" && part.status !== "finished")
          ? "working" : "thinking",
      since: turn.activity?.since ?? turn.createdAt,
    };
  }
  if (turn.createdAt !== undefined) custom[SENT_AT] = turn.createdAt;
  if (sentAttachments.length) custom[SENT_ATTACHMENTS] = sentAttachments;
  if (turn.role === "system") custom[COMPACTED] = true;
  const timing = {
    ...(turn.createdAt === undefined
      ? {}
      : { createdAt: new Date(turn.createdAt) }),
    metadata: { custom },
  };

  if (turn.role === "system") {
    return {
      id: turn.id,
      role: "system",
      // **Exactly one text part, and assistant-ui throws rather than trimming**
      // — so this is assembled here instead of being handed `content`, which
      // holds however many parts the turn happens to have. The text is the
      // summary the agent was left with; nothing renders it, but it is what the
      // row actually says, and a placeholder would be words in the transcript
      // that nothing wrote.
      content: [
        {
          type: "text" as const,
          text: turn.parts
            .map((part) => (part.kind === "text" ? part.text : ""))
            .join(""),
        },
      ],
      ...timing,
    };
  }

  if (turn.role !== "assistant") {
    return { id: turn.id, role: turn.role, content, ...timing };
  }

  return {
    id: turn.id,
    role: turn.role,
    content,
    ...timing,
    status: turn.running
      ? { type: "running" }
      : turn.aborted
        ? { type: "incomplete", reason: "cancelled" }
        : { type: "complete", reason: "stop" },
  };
}
