/**
 * Which activity indicator a running assistant turn is allowed to draw.
 *
 * There are two of them in this app and they say different things. The dot the
 * markdown package pulses at the end of `.aui-md[data-status="running"]` means
 * *this sentence is still arriving*; the "Working" line means *the turn is
 * running and there is nothing on screen to hang a cursor off*.
 *
 * Nothing coordinated the two, because each was keyed on a condition the other
 * could not see: the dot on one part's own status, the "Working" line on
 * assistant-ui's `no-text` indicator mode, which asks only whether the last
 * part is *typed* `text` and never whether that text says anything. A turn
 * whose last text part held nothing but whitespace — a model that printed a
 * blank line before calling another tool — therefore satisfied both at once,
 * and the transcript pulsed twice about the same wait.
 *
 * So the rule lives here, is applied once per message, and both indicators are
 * keyed on its answer.
 */
export type Activity = "none" | "streaming" | "working";

export type ActivityInput = {
  /** The server's own statement that it still has the turn (`typing`). */
  threadRunning: boolean;
  /** Whether this is the last message in the transcript. */
  isLast: boolean;
  /** `message.status.type`, absent on a message that has no run. */
  messageStatus: string | undefined;
  /** The message's final part, or `undefined` when it has none yet. */
  lastPart: { readonly type: string; readonly text?: string } | undefined;
};

export function activityFor({
  threadRunning,
  isLast,
  messageStatus,
  lastPart,
}: ActivityInput): Activity {
  // Every one of these is a way of not being the turn in progress. `isLast`
  // earns its place among them: a transcript holding two turns that both
  // believe they are running is in an odd state, and the answer to that is one
  // indicator at the bottom, not two stacked up the page.
  if (!threadRunning || !isLast || messageStatus !== "running") return "none";

  // `trim`, not emptiness. A part holding only whitespace is a model that said
  // nothing, and a cursor trailing nothing is not a cursor — it is a second
  // "Working" that happens to be round.
  if (lastPart?.type === "text" && (lastPart.text ?? "").trim() !== "") {
    return "streaming";
  }

  // No parts yet, a tool call, or text with nothing visible in it: there is
  // nothing for a cursor to sit after, so the turn says so in words.
  return "working";
}
