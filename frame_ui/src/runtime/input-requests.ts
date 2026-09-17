/**
 * The questions a person still owes an answer to.
 *
 * **Deliberately not part of the conversation store.** A pending question is
 * session state, not conversation state: the kernel holds it on the session's
 * phase stack and persists it, and it outlives anything the transcript does.
 * Keeping it here used to mean `{type: "history"}` — a cold boot, a
 * conversation switch, a refetch — returned `{...initialState, turns}` and
 * silently discarded a live question along with the scrollback. On boot that is
 * a race the client runs against itself: the event stream replays the real
 * frame within a round trip, while the history read takes two or three, so a
 * reload usually wiped the very question it had just been handed and the agent
 * sat blocked until its 300s timeout. Separating the two is what makes that
 * unrepresentable rather than merely fixed.
 *
 * ## Why a queue
 *
 * The kernel approves on a thread pool, so two gated calls in one turn each
 * raise their own question, and it already models this as an ordered list
 * (`_pending_approval_order`). A single slot meant the second overwrote the
 * first, and the first was then unanswerable from the UI while the turn stayed
 * blocked on it.
 *
 * ## Where the truth is
 *
 * Two frames, and between them the whole life of a question: `approval` says
 * one appeared, `approval_settled` says one stopped waiting — because it was
 * answered somewhere else, or because the kernel denied it by name after 300
 * seconds. Neither of those is something this client did, and before the second
 * frame existed neither was something it could learn except by asking on a
 * timer.
 *
 * `reconciled` — the answer to `frontend.pending {details: true}` — covers the
 * gap a stream cannot: what happened while nobody was connected. It runs when
 * the stream opens and not otherwise, because from then on the frames say it.
 */

import type { FormFieldPayload } from "@/lib/events";
import type { InputRequest } from "@/lib/input-requests";

/**
 * What a reconciliation concluded.
 *
 * Shaped like `PendingInput` — the wire answer is assignable to it — but with
 * the *domain* request, whose id may be null. The server never sends a question
 * it cannot name; the client can construct one, for a kernel too old to
 * describe what it is holding.
 */
type ReconciledInput =
  | { kind: "approval"; payload: InputRequest }
  | { kind: "form_field"; payload: FormFieldPayload }
  | null;

export type InputRequestState = {
  /** Head first, which is the order `resolve_next_approval` works down. */
  queue: InputRequest[];
};

export const initialInputRequests: InputRequestState = { queue: [] };

export type InputRequestAction =
  /** An `approval` frame off the event stream. */
  | { type: "raised"; request: InputRequest }
  /** An `approval_settled` frame: this question stopped waiting, whoever
   *  ended it. Also how an answer from here comes back, so there is one way a
   *  question leaves the queue rather than two that can disagree. */
  | { type: "settled"; id: string }
  /** Answered from here, optimistically — the dialog closes before the POST
   *  lands, because that POST only completes once the *original* blocked
   *  Request finishes and a dialog left up meanwhile invites a second click.
   *  `null` answers the head, which is what a question with no id means. */
  | { type: "answered"; id: string | null }
  /** What `frontend.pending {details: true}` said.
   *
   *  There is deliberately no "clear everything" action beside it. The one
   *  thing entitled to empty this queue is the server saying it is empty —
   *  anything else is a client guessing, which is how a live question came to
   *  be thrown away by a conversation switch in the first place. */
  | { type: "reconciled"; pending: ReconciledInput };

/** The copy for a question we know exists but were never handed. */
const UNSEEN: Omit<InputRequest, "id"> = {
  title: "The agent is waiting on an answer",
  body:
    "This was asked before the page was open, and its details could not be " +
    "read back. Denying is the safe answer.",
};

export function reduceInputRequests(
  state: InputRequestState,
  action: InputRequestAction,
): InputRequestState {
  switch (action.type) {
    case "raised": {
      // **By identity, not by appending.** `EventSource` replays from
      // `Last-Event-ID` after a reconnect, so the same frame arrives twice; a
      // second copy in the queue would be a dialog that outlives its answer.
      const at = indexOf(state.queue, action.request.id);
      if (at === -1) return { queue: [...state.queue, action.request] };
      const queue = [...state.queue];
      queue[at] = action.request;
      return { queue };
    }

    case "settled":
    case "answered": {
      // A settled frame for a question we never held is ordinary: another
      // client answered it, or it expired while this page was elsewhere.
      if (action.id === null) return { queue: state.queue.slice(1) };
      const at = indexOf(state.queue, action.id);
      if (at === -1) return state;
      return { queue: state.queue.filter((_, index) => index !== at) };
    }

    case "reconciled": {
      const pending = action.pending;

      // Nothing is waiting. Anything still on screen was settled while this
      // page was not listening, and must come down: a question the kernel has
      // forgotten cannot be answered and says nothing about that when you try.
      if (pending === null) {
        return state.queue.length ? initialInputRequests : state;
      }

      // A form is a different surface with its own panel. It is reported here
      // only so one call covers both; it says nothing about the queue, and in
      // particular does not mean the queue is empty.
      if (pending.kind === "form_field") return state;

      const request = pending.payload;

      // Already held: the stream got there first, which is the common case and
      // the reason this is an identity check rather than a boot-scoped "did a
      // frame arrive" flag. That flag could not tell "the frame arrived" from
      // "the frame arrived and was then thrown away", and suppressed the
      // recovery in exactly the case that needed it.
      if (indexOf(state.queue, request.id) !== -1) return state;

      // **Only the head is authoritative.** `frontend.pending` reports the
      // front of the kernel's queue, so a non-null answer is no evidence about
      // anything behind it — appending rather than replacing is what keeps a
      // second question from being dropped.
      return { queue: [...state.queue, request] };
    }
  }
}

/**
 * Where a question with this id sits, or -1.
 *
 * `null` matches nothing rather than matching the other id-less entry: two
 * questions we cannot name are still two questions, and collapsing them would
 * lose one.
 */
function indexOf(queue: InputRequest[], id: string | null): number {
  if (id === null) return -1;
  return queue.findIndex((request) => request.id === id);
}

/** Build the stand-in for a question the kernel named but could not describe.
 *
 *  Only reachable against a kernel too old to answer `details`, which hands
 *  back a bare id (or `true`, meaning "one exists, unordered"). Worse than the
 *  real dialog, and far better than a turn blocked on a question with nothing
 *  on screen. */
export function unseenRequest(id: string | null): InputRequest {
  return { ...UNSEEN, id };
}
