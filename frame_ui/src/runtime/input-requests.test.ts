/**
 * The rules that were silently wrong, pinned.
 *
 * Every case here is a bug that shipped, not a hypothetical. The reducer is
 * pure precisely so these can be checked without a browser — `store.ts` has
 * claimed that as a design justification since it was written, and this is the
 * first file to collect on it.
 */

import { describe, expect, it } from "vitest";

import type { InputRequest } from "@/lib/input-requests";
import {
  initialInputRequests,
  reduceInputRequests,
  unseenRequest,
  type InputRequestAction,
  type InputRequestState,
} from "@/runtime/input-requests";
import { initialState, reduce } from "@/runtime/store";

const ask = (id: string | null, title = "Run a shell command"): InputRequest => ({
  id,
  title,
  body: "rm -rf /tmp/x",
  type: "string",
  enum: ["allow", "deny"],
  enum_labels: ["Allow", "Deny"],
});

const held = (state: InputRequestState) => state.queue.map((one) => one.id);

/** Fold a list of actions over the empty queue. */
function run(...actions: InputRequestAction[]) {
  return actions.reduce(reduceInputRequests, initialInputRequests);
}

describe("a question survives what the conversation does", () => {
  it("is not reachable from the conversation reducer at all", () => {
    // The regression test for the whole bug. A question used to live on the
    // conversation store, where `history` — a cold boot, a conversation switch,
    // a refetch — returned `{...initialState, turns}` and took it with it. On
    // boot that is a race the client runs against itself: the stream replays
    // the real question within one round trip, the history read takes two or
    // three, so a reload usually threw away what it had just been handed and
    // the agent sat blocked until its 300s timeout.
    const questions = run({ type: "raised", request: ask("approve_1") });

    reduce(initialState, { type: "history", turns: [], hasMore: false, oldestId: null });

    expect(held(questions)).toEqual(["approve_1"]);
  });

  it("keeps the answer's own id rather than reading the current one", () => {
    // Answering used to send whichever question was current at the moment of
    // the click. A second one arriving between the draw and the press made
    // that a different question than the one on screen.
    const state = run(
      { type: "raised", request: ask("approve_1") },
      { type: "raised", request: ask("approve_2") },
      { type: "answered", id: "approve_2" },
    );

    expect(held(state)).toEqual(["approve_1"]);
  });
});

describe("the queue", () => {
  it("holds a second question instead of overwriting the first", () => {
    // The kernel approves on a thread pool, so two gated calls in one turn
    // each raise their own. A single slot dropped the first, and the turn
    // stayed blocked on a question nothing could answer.
    const state = run(
      { type: "raised", request: ask("approve_1") },
      { type: "raised", request: ask("approve_2") },
    );

    expect(held(state)).toEqual(["approve_1", "approve_2"]);
  });

  it("promotes the next one when the head is answered", () => {
    const state = run(
      { type: "raised", request: ask("approve_1") },
      { type: "raised", request: ask("approve_2") },
      { type: "answered", id: "approve_1" },
    );

    expect(held(state)).toEqual(["approve_2"]);
  });

  it("does not duplicate a frame the stream replayed", () => {
    // `EventSource` resumes from `Last-Event-ID` after a reconnect, so the
    // same frame arrives twice. A second copy would be a dialog outliving its
    // own answer.
    const state = run(
      { type: "raised", request: ask("approve_1") },
      { type: "raised", request: ask("approve_1", "Run a shell command (again)") },
    );

    expect(held(state)).toEqual(["approve_1"]);
    expect(state.queue[0].title).toBe("Run a shell command (again)");
  });

  it("answers the head when the question could not be named", () => {
    const state = run(
      { type: "raised", request: ask(null) },
      { type: "answered", id: null },
    );

    expect(held(state)).toEqual([]);
  });

  it("keeps two unnamed questions apart", () => {
    // `null` must not match `null`: two questions we cannot name are still two
    // questions, and collapsing them loses one.
    const state = run(
      { type: "raised", request: ask(null) },
      { type: "raised", request: ask(null) },
    );

    expect(state.queue).toHaveLength(2);
  });
});

describe("a question that stops waiting", () => {
  it("comes down when the server says it settled", () => {
    // The half `approval` was missing. Another client can answer the same
    // question and the kernel denies by name after 300s; neither is something
    // this client did, and without this frame the dialog would sit there
    // unanswerable, saying nothing about it when tried.
    const state = run(
      { type: "raised", request: ask("approve_1") },
      { type: "settled", id: "approve_1" },
    );

    expect(held(state)).toEqual([]);
  });

  it("takes only the one that settled", () => {
    const state = run(
      { type: "raised", request: ask("approve_1") },
      { type: "raised", request: ask("approve_2") },
      { type: "settled", id: "approve_2" },
    );

    expect(held(state)).toEqual(["approve_1"]);
  });

  it("shrugs at one it never held", () => {
    // Ordinary: another client answered it, or it expired while this page was
    // somewhere else entirely.
    const before = run({ type: "raised", request: ask("approve_1") });
    const after = reduceInputRequests(before, {
      type: "settled",
      id: "approve_other",
    });

    expect(after).toBe(before);
  });

  it("is not put back by a reconcile that raced it", () => {
    // The settled frame and a reconcile can cross. Whichever lands second must
    // not resurrect the dialog, which is what the old suppress-once heuristic
    // existed to arrange and what a real settlement now does outright.
    const state = run(
      { type: "raised", request: ask("approve_1") },
      { type: "settled", id: "approve_1" },
      { type: "reconciled", pending: null },
    );

    expect(state).toEqual(initialInputRequests);
  });
});

describe("reconciling against the server", () => {
  it("closes a dialog settled while the page was away", () => {
    // Frames cover the connected case; this covers the gap a stream cannot —
    // what happened while nobody was listening.
    const state = run(
      { type: "raised", request: ask("approve_1") },
      { type: "reconciled", pending: null },
    );

    expect(held(state)).toEqual([]);
  });

  it("restores a question the page was never handed", () => {
    const request = ask("approve_1");
    const state = run({
      type: "reconciled",
      pending: { kind: "approval", payload: request },
    });

    // The real question, not a stand-in: `details` hands back the same
    // projection the render made.
    expect(state.queue).toEqual([request]);
  });

  it("leaves a question it already holds alone", () => {
    // This identity check replaced a boot-scoped "did a frame arrive" flag,
    // which could not tell "the frame arrived" from "the frame arrived and was
    // then thrown away" — and so suppressed the recovery in exactly the case
    // that needed it.
    const before = run({ type: "raised", request: ask("approve_1") });
    const after = reduceInputRequests(before, {
      type: "reconciled",
      pending: { kind: "approval", payload: ask("approve_1") },
    });

    expect(after).toBe(before);
  });

  it("does not evict what sits behind the head", () => {
    // `frontend.pending` reports the front of the kernel's queue, so a
    // non-null answer is no evidence about anything behind it.
    const state = run(
      { type: "raised", request: ask("approve_1") },
      { type: "raised", request: ask("approve_2") },
      {
        type: "reconciled",
        pending: { kind: "approval", payload: ask("approve_1") },
      },
    );

    expect(held(state)).toEqual(["approve_1", "approve_2"]);
  });

  it("says nothing about the queue when a form is what is pending", () => {
    // A form is a different surface with its own panel. It is reported through
    // the same call only so one round trip covers both, and in particular does
    // not mean the queue is empty.
    const state = run(
      { type: "raised", request: ask("approve_1") },
      {
        type: "reconciled",
        pending: { kind: "form_field", payload: { name: "packages" } },
      },
    );

    expect(held(state)).toEqual(["approve_1"]);
  });

  it("draws a stand-in for a question an old kernel cannot describe", () => {
    const state = run({
      type: "reconciled",
      pending: { kind: "approval", payload: unseenRequest(null) },
    });

    // `id: null` is the load-bearing part. The kernel answers `true` here for
    // "one exists, unordered", and using that as an id gets it stringified
    // into one that matches nothing — the dialog closes and reports success
    // while the turn stays blocked.
    expect(state.queue[0].id).toBeNull();
    expect(state.queue[0].title).toMatch(/waiting/i);
  });
});
