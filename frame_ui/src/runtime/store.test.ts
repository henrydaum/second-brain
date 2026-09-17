/**
 * What a message sent mid-turn does to the transcript.
 *
 * The composer can now send while the agent still has the turn — the kernel
 * queues it and drains it at the next loop boundary. The line has to land
 * *between* what the agent had already said and whatever it says next, which
 * means closing the open reply and continuing it below.
 *
 * That is a stream told in two halves, and the `done` frame carries
 * `final_text` for the whole of it. Most of what follows is about that: the
 * failure it causes is not a missing message but a doubled one. See
 * `splitOpenTurn` and `tailOf`.
 */

import { describe, expect, it } from "vitest";

import type { Frame } from "@/lib/events";
import {
  initialState,
  reduce,
  type State,
  type TextPart,
  type ToolPart,
} from "@/runtime/store";

/** Drive the reducer over a script, starting from nothing. */
const run = (...frames: Frame[]): State =>
  frames.reduce(
    (state, frame) => reduce(state, { type: "frame", frame }),
    initialState,
  );

const typing = (on: boolean) => ({ kind: "typing", payload: on }) as Frame;

describe("reply presentation ownership", () => {
  it("distinguishes a subagent barrier wait from ordinary thinking", () => {
    const state = run(typing(true),
      { kind: "turn_activity", payload: { phase: "waiting" } });
    expect(state.turns[0].activity?.phase).toBe("waiting");
    const resumed = reduce(state, { type: "frame", frame: {
      kind: "turn_activity", payload: { phase: "thinking" },
    } });
    expect(resumed.turns[0].activity?.phase).toBe("thinking");
  });

  it("does not leave an earlier segment writing when another stream completes", () => {
    const state = run(typing(true),
      { kind: "stream_delta", payload: { stream_id: "first", seq: 1, delta: "Earlier text", done: false } },
      { kind: "stream_delta", payload: { stream_id: "second", seq: 1, delta: "Later text", done: false } },
      { kind: "stream_delta", payload: { stream_id: "second", seq: 2, delta: "", done: true } });
    expect(state.turns[0].activity?.phase).toBe("thinking");
    expect(state.turns[0].parts.every((part) => part.kind !== "text" || part.done)).toBe(true);
  });

  it("keeps late attachments on the completed reply, without reopening it", () => {
    const state = run(typing(true),
      { kind: "messages", payload: ["Here are the results."] }, typing(false),
      { kind: "attachments", payload: ["/one.png", "/note.md"] });
    expect(state.turns).toHaveLength(1);
    expect(state.turns[0].running).toBe(false);
    expect(state.turns[0].parts.at(-1)).toMatchObject({ kind: "files", paths: ["/one.png", "/note.md"] });
  });

  it("reconciles a final stream across several file boundaries without duplicate text", () => {
    const chunk = (seq: number, text: string, done = false): Frame => ({ kind: "stream_delta", payload: {
      stream_id: "ordered", seq, delta: text, done, ...(done ? { final_text: "ABC" } : {}),
    } });
    const state = run(typing(true), chunk(1, "A"), chunk(1, "A"),
      { kind: "attachments", payload: ["/a.png"] }, chunk(2, "B"),
      { kind: "attachments", payload: ["/b.png"] }, chunk(3, "C", true), chunk(3, "C", true), typing(false));
    expect(state.turns).toHaveLength(1);
    expect(state.turns[0].parts.filter((part) => part.kind === "text").map((part) => part.text)).toEqual(["A", "B", "C"]);
  });

  it("does not end writing when an earlier tool completes", () => {
    const state = run(typing(true),
      { kind: "tool_status", payload: { call_id: "t1", tool_name: "search", status: "started" } },
      { kind: "stream_delta", payload: { stream_id: "s", seq: 1, delta: "While that runs", done: false } },
      { kind: "tool_status", payload: { call_id: "t1", status: "finished" } });
    expect(state.turns[0].activity?.phase).toBe("writing");
  });
});

describe("durable logical turn identity", () => {
  const delta = (id: string, stream: string, done = false): Frame => ({
    kind: "stream_delta", payload: { turn_id: id, stream_id: stream, seq: 1, delta: "text", done },
  });
  it("adopts the server identity across provisional interrupted segments", () => {
    let state = run(typing(true), { kind: "messages", payload: ["Before"] });
    state = reduce(state, { type: "said", text: "Interrupt" });
    state = reduce(state, { type: "frame", frame: delta("kernel-a", "a", true) });
    const replies = state.turns.filter((turn) => turn.role === "assistant");
    expect(replies.map((turn) => turn.turnId)).toEqual(["kernel-a", "kernel-a"]);
    expect(replies[0].continues).toBe(true);
    expect(replies[1].running).toBe(true);
    state = reduce(state, { type: "frame", frame: typing(false) });
    const settled = state;
    state = reduce(state, { type: "frame", frame: delta("kernel-a", "a", true) });
    expect(state).toBe(settled);
  });
  it("resumes the last persisted segment without appending a second reply", () => {
    let state = run(delta("kernel-a", "a", true), typing(false));
    state = reduce(state, { type: "resumeTurn", turnId: "kernel-a" });
    expect(state.turns).toHaveLength(1);
    expect(state.turns[0].running).toBe(true);
    expect(state.typing).toBe(true);
  });
  it("resumes after a persisted mid-turn user row with one final owner", () => {
    const state = reduce({ ...initialState, turns: [
      { id: "before", role: "assistant", turnId: "a", parts: [{ kind: "text", streamId: "s", text: "Before", done: true }], running: false, aborted: false },
      { id: "user", role: "user", parts: [], running: false, aborted: false },
    ] }, { type: "resumeTurn", turnId: "a" });
    expect(state.turns).toHaveLength(3);
    expect(state.turns[0].continues).toBe(true);
    expect(state.turns[2]).toMatchObject({ turnId: "a", running: true });
  });
  it("keeps a recap owner even when interrupted without further prose", () => {
    let state = run(typing(true), { kind: "messages", payload: ["Before"] },
      { kind: "attachments", payload: ["/a.png"] });
    state = reduce(state, { type: "said", text: "Interrupt" });
    state = reduce(state, { type: "frame", frame: typing(false) });
    expect(state.turns).toHaveLength(3);
    expect(state.turns[0].continues).toBe(true);
    expect(state.turns[2].turnId).toBe(state.turns[0].turnId);
    expect(state.turns[2].running).toBe(false);
  });
});

const delta = (text: string, over: Record<string, unknown> = {}): Frame =>
  ({
    kind: "stream_delta",
    payload: { stream_id: "s1", delta: text, done: false, ...over },
  }) as Frame;

const toolStatus = (status: "started" | "progressed" | "finished"): Frame =>
  ({
    kind: "tool_status",
    payload: {
      kind: "tool",
      call_id: "c1",
      tool_name: "read_file",
      status,
      narration: "Reading",
      ...(status === "finished" ? { ok: true, summary: "Read it." } : {}),
    },
  }) as Frame;

/** Every text part of every turn, in order, as plain strings. */
const said = (state: State) =>
  state.turns.map((turn) =>
    turn.parts
      .filter((part): part is TextPart => part.kind === "text")
      .map((part) => part.text)
      .join(""),
  );

describe("agent attachment placement", () => {
  const files = (...paths: string[]): Frame =>
    ({ kind: "attachments", payload: paths }) as Frame;

  it("splits a stream so later text stays below the shown file", () => {
    let state = run(typing(true), delta("Before"), files("/chart.png"));
    state = reduce(state, { type: "frame", frame: delta("After") });

    expect(state.turns[0]?.parts).toEqual([
      expect.objectContaining({ kind: "text", text: "Before", done: true }),
      expect.objectContaining({ kind: "files", paths: ["/chart.png"] }),
      expect.objectContaining({ kind: "text", text: "After", done: false }),
    ]);
  });

  it("does not append the same shown path twice in one turn", () => {
    const state = run(
      typing(true),
      files("/chart.png"),
      files("/chart.png"),
    );

    expect(state.turns[0]?.parts).toEqual([
      expect.objectContaining({ kind: "files", paths: ["/chart.png"] }),
    ]);
  });
});

describe("a message sent while the agent is still talking", () => {
  it("lands under what was already said, not under the whole turn", () => {
    let state = run(typing(true), delta("Half a "));
    state = reduce(state, { type: "said", text: "one more thing" });
    state = reduce(state, { type: "frame", frame: delta("sentence.") });

    // The tail is its own message *below* the queued line — the thing the
    // single-turn reading got backwards, by writing it into the message above.
    expect(state.turns.map((turn) => turn.role)).toEqual([
      "assistant",
      "user",
      "assistant",
    ]);
    expect(said(state)).toEqual(["Half a ", "one more thing", "sentence."]);
  });

  it("leaves only one message running, so only one indicator draws", () => {
    let state = run(typing(true), delta("Half a "));
    state = reduce(state, { type: "said", text: "one more thing" });
    state = reduce(state, { type: "frame", frame: delta("sentence.") });

    expect(state.turns.filter((turn) => turn.running)).toHaveLength(1);
    expect(state.turns.at(-1)!.running).toBe(true);
  });

  it("does not repeat the first half when the stream finishes", () => {
    // `final_text` is the *whole* stream and replaces what accumulated. Landing
    // it whole in the second half is how the first half appeared twice.
    let state = run(typing(true), delta("Half a "));
    state = reduce(state, { type: "said", text: "one more thing" });
    state = reduce(state, {
      type: "frame",
      frame: delta("sentence.", { done: true, final_text: "Half a sentence." }),
    });

    expect(said(state)).toEqual(["Half a ", "one more thing", "sentence."]);
  });

  it("keeps the deltas when the cleaned text cannot be lined up", () => {
    // `final_text` is cleaned, so it need not start with what is on screen.
    // Falling back to the deltas shows the tail once; trusting `final_text`
    // would show the first half twice.
    let state = run(typing(true), delta("Half a "));
    state = reduce(state, { type: "said", text: "one more thing" });
    state = reduce(state, {
      type: "frame",
      frame: delta("sentence.", {
        done: true,
        final_text: "Something else entirely.",
      }),
    });

    expect(said(state)).toEqual(["Half a ", "one more thing", "sentence."]);
  });

  it("still recognises the whole reply if a messages frame repeats it", () => {
    let state = run(typing(true), delta("Half a "));
    state = reduce(state, { type: "said", text: "one more thing" });
    state = reduce(state, {
      type: "frame",
      frame: delta("sentence.", { done: true, final_text: "Half a sentence." }),
    });
    state = reduce(state, {
      type: "frame",
      frame: { kind: "messages", payload: ["Half a sentence."] } as Frame,
    });

    expect(said(state)).toEqual(["Half a ", "one more thing", "sentence."]);
  });

  it("moves the indicator below the line rather than leaving one above", () => {
    // `typing: true` opens a turn before there is anything to put in it. Closing
    // it in place would leave a blank running message above the person's line —
    // and dropping it without replacement would leave the agent looking stopped
    // while it works. It moves.
    let state = run(typing(true));
    state = reduce(state, { type: "said", text: "one more thing" });

    expect(state.turns.map((turn) => turn.role)).toEqual(["user", "assistant"]);
    expect(state.turns[1].parts).toEqual([]);
    expect(state.turns[1].running).toBe(true);
  });

  it("carries a call still in flight down with the reply", () => {
    // A tool-call part with no result inherits its message's status, so a
    // running call left in the closed half would draw as finished — and its
    // `finished` frame would then appear again below as a second block.
    let state = run(typing(true), delta("Looking. "), toolStatus("started"));
    state = reduce(state, { type: "said", text: "and check the logs" });

    expect(state.turns.map((turn) => turn.role)).toEqual([
      "assistant",
      "user",
      "assistant",
    ]);
    expect(state.turns[0].parts.map((part) => part.kind)).toEqual(["text"]);
    expect(state.turns[2].parts.map((part) => part.kind)).toEqual(["tool"]);

    // And the result updates that one block rather than making another.
    state = reduce(state, { type: "frame", frame: toolStatus("finished") });
    expect(state.turns).toHaveLength(3);
    expect(state.turns[2].parts).toHaveLength(1);
    expect((state.turns[2].parts[0] as ToolPart).status).toBe("finished");
  });

  it("leaves a call that already finished where it was made", () => {
    let state = run(
      typing(true),
      delta("Looked. "),
      toolStatus("started"),
      toolStatus("finished"),
    );
    state = reduce(state, { type: "said", text: "and now the logs" });

    expect(state.turns.map((turn) => turn.role)).toEqual([
      "assistant",
      "user",
      "assistant",
    ]);
    expect(state.turns[0].parts.map((part) => part.kind)).toEqual([
      "text",
      "tool",
    ]);
    expect(state.turns[2].parts).toEqual([]);
  });

  it("survives being sent twice in one reply", () => {
    let state = run(typing(true), delta("One "));
    state = reduce(state, { type: "said", text: "first" });
    state = reduce(state, { type: "frame", frame: delta("two ") });
    state = reduce(state, { type: "said", text: "second" });
    state = reduce(state, {
      type: "frame",
      frame: delta("three.", { done: true, final_text: "One two three." }),
    });

    expect(said(state)).toEqual([
      "One ",
      "first",
      "two ",
      "second",
      "three.",
    ]);
  });

  it("forgets the carried text once the turn ends", () => {
    let state = run(typing(true), delta("Half a "));
    state = reduce(state, { type: "said", text: "one more thing" });
    state = reduce(state, {
      type: "frame",
      frame: delta("sentence.", { done: true, final_text: "Half a sentence." }),
    });
    state = reduce(state, { type: "frame", frame: typing(false) });

    expect(state.carried).toEqual({});
  });
});

describe("an uninterrupted reply", () => {
  it("is still one message, with final_text replacing the deltas", () => {
    const state = run(
      typing(true),
      delta("Half a "),
      delta("sentance.", { done: true, final_text: "Half a sentence." }),
      typing(false),
    );

    expect(said(state)).toEqual(["Half a sentence."]);
    expect(state.carried).toEqual({});
  });
});

describe("approval cancellation acknowledgements", () => {
  it("keeps the exact Cancelled acknowledgement out of the conversation", () => {
    let state = reduce(initialState, {
      type: "suppressNextCancellationNotice",
    });
    state = reduce(state, {
      type: "frame",
      frame: { kind: "messages", payload: ["Cancelled."] } as Frame,
    });

    expect(state.turns).toEqual([]);
    expect(state.suppressNextCancellationNotice).toBe(false);
  });

  it("does not hide other text after an approval dialog closes", () => {
    let state = reduce(initialState, {
      type: "suppressNextCancellationNotice",
    });
    state = reduce(state, {
      type: "frame",
      frame: {
        kind: "messages",
        payload: ["The operation was cancelled after a timeout."],
      } as Frame,
    });

    expect(said(state)).toEqual([
      "The operation was cancelled after a timeout.",
    ]);
  });
});

describe("callable output", () => {
  it("routes command output to its panel while messages stay in chat", () => {
    const state = run(
      {
        kind: "tool_status",
        payload: {
          kind: "command",
          call_id: "cmd-1",
          command_name: "config",
          status: "started",
        },
      },
      { kind: "messages", payload: ["An agent reply."] },
      { kind: "callable_output", payload: ["| setting | value |"] },
    );

    expect(said(state)).toEqual(["An agent reply."]);
    expect(state.command?.outcome).toEqual(["| setting | value |"]);
  });

  it("keeps directly invoked tool output in a Settings output run", () => {
    const state = run({
      kind: "callable_output",
      payload: ["Direct result"],
    });

    expect(state.turns).toEqual([]);
    expect(state.command).toMatchObject({
      name: "output",
      status: "finished",
      outcome: ["Direct result"],
    });
  });

  it("adopts an early synthetic output run when command status follows", () => {
    let state = run({
      kind: "callable_output",
      payload: ["Project root /project"],
    });
    state = reduce(state, {
      type: "frame",
      frame: {
        kind: "tool_status",
        payload: {
          kind: "command",
          call_id: "cmd:locations:later",
          command_name: "locations",
          status: "finished",
          ok: true,
        },
      },
    });

    expect(state.command).toMatchObject({
      callId: "cmd:locations:later",
      name: "locations",
      outcome: ["Project root /project"],
    });
  });

  it("keeps command output in Settings when it beats the status frame", () => {
    let state = reduce(initialState, {
      type: "said",
      text: "/locations",
      isCommand: true,
    });
    state = reduce(state, {
      type: "frame",
      frame: { kind: "callable_output", payload: ["Project root /project"] },
    });

    expect(state.turns).toEqual([]);
    expect(state.command).toMatchObject({
      callId: "pending:locations",
      name: "locations",
      outcome: ["Project root /project"],
    });

    state = reduce(state, {
      type: "frame",
      frame: {
        kind: "tool_status",
        payload: {
          kind: "command",
          call_id: "cmd:locations:1234",
          command_name: "locations",
          status: "finished",
          ok: true,
        },
      },
    });

    expect(state.command).toMatchObject({
      callId: "cmd:locations:1234",
      status: "finished",
      outcome: ["Project root /project"],
    });
  });

  it("narrates a long command's progress on the command, not in chat", () => {
    // A package install reports what it is doing from deep inside the kernel.
    // It used to reach the person on `messages`, which put the progress of a
    // command run from Settings into the transcript — and, since a push writes
    // no history row, took it away again on the next reload.
    let state = reduce(initialState, {
      type: "said",
      text: "/packages",
      isCommand: true,
    });
    const frames: Frame[] = [
      {
        kind: "tool_status",
        payload: {
          kind: "command",
          call_id: "cmd:packages:1",
          command_name: "packages",
          status: "started",
          args: { action: "install", package_id: "memory" },
        },
      },
      {
        kind: "tool_status",
        payload: {
          kind: "command",
          call_id: "cmd:packages:1",
          command_name: "packages",
          status: "progressed",
          narration: "Copying package files",
        },
      },
    ];
    state = frames.reduce(
      (current, frame) => reduce(current, { type: "frame", frame }),
      state,
    );

    expect(said(state)).toEqual([]);
    expect(state.command).toMatchObject({
      callId: "cmd:packages:1",
      status: "progressed",
      narration: "Copying package files",
      // A progress frame says nothing about the answers already given, so it
      // must not blank them out — the panel is still showing them.
      args: { action: "install", package_id: "memory" },
    });
  });

  it("structurally ignores output from a cancelled command", () => {
    let state = run({
      kind: "tool_status",
      payload: {
        kind: "command",
        call_id: "cmd-1",
        command_name: "config",
        status: "started",
      },
    });
    state = reduce(state, { type: "said", text: "/cancel", isCommand: true });
    state = reduce(state, {
      type: "frame",
      frame: { kind: "callable_output", payload: ["Cancelled."] },
    });

    expect(state.command?.outcome).toEqual([]);
  });

  it("keeps the documented cancel Request acknowledgement out of chat", () => {
    let state = run({
      kind: "tool_status",
      payload: {
        kind: "command",
        call_id: "cmd-1",
        command_name: "locations",
        status: "started",
      },
    });
    state = reduce(state, { type: "said", text: "/cancel", isCommand: true });
    state = reduce(state, {
      type: "frame",
      frame: { kind: "messages", payload: ["Cancelled."] },
    });

    expect(state.turns).toEqual([]);
  });

  it("does not swallow a non-cancellation message after command cancellation", () => {
    let state = run({
      kind: "tool_status",
      payload: {
        kind: "command",
        call_id: "cmd-1",
        command_name: "locations",
        status: "started",
      },
    });
    state = reduce(state, { type: "said", text: "/cancel", isCommand: true });
    state = reduce(state, {
      type: "frame",
      frame: { kind: "messages", payload: ["An overlapping agent reply."] },
    });

    expect(said(state)).toEqual(["An overlapping agent reply."]);
  });
});

describe("sent attachment hydration", () => {
  it("replaces optimistic names with the cached paths from conv.read", () => {
    let state = reduce(initialState, {
      type: "said",
      text: "Look at this",
      attachments: [
        {
          fileName: "chart.png",
          modality: "image",
          extension: "png",
        },
      ],
    });
    state = reduce(state, {
      type: "hydrateSentAttachments",
      attachments: [
        {
          path: "/workspace/attachments/1_chart.png",
          fileName: "chart.png",
          modality: "image",
          extension: "png",
        },
      ],
    });

    expect(state.turns[0]?.parts[0]).toMatchObject({
      kind: "files",
      paths: ["/workspace/attachments/1_chart.png"],
      attachments: [
        { path: "/workspace/attachments/1_chart.png", fileName: "chart.png" },
      ],
    });
  });
});

/**
 * A compaction marker arriving after the fact.
 *
 * `/compact` writes a stored row and no frame announces it, so the provider
 * reads it back and hands it over — without disturbing the command panel that
 * is reporting the compaction at that moment.
 */
describe("compaction markers", () => {
  const marker = {
    id: "stored-7",
    role: "system" as const,
    parts: [
      { kind: "text" as const, streamId: "stored-7", text: "A summary", done: true },
    ],
    running: false,
    aborted: false,
    createdAt: 1786732595340,
  };

  it("lands at the end of the transcript without touching what is on screen", () => {
    const said = reduce(initialState, { type: "said", text: "Hello" });
    const state = reduce(said, { type: "compacted", turn: marker });

    expect(state.turns.map((turn) => turn.role)).toEqual(["user", "system"]);
    expect(state.turns[0]).toBe(said.turns[0]);
  });

  it("draws one line however many times the same marker is read", () => {
    const once = reduce(initialState, { type: "compacted", turn: marker });
    const twice = reduce(once, { type: "compacted", turn: marker });

    expect(twice).toBe(once);
    expect(twice.turns).toHaveLength(1);
  });
});

/**
 * The narration ends up in one place, whichever way the wire sent it.
 *
 * The model writes it as a reserved argument; the wire also lifts it out to a
 * field of its own. Only the argument is stored, so a conversation read back
 * has only that — and a client keeping both would show the blurb while you
 * watched and lose it on reload. See `toolArgs`.
 */
describe("tool narration", () => {
  const tool = (over: Record<string, unknown>): Frame =>
    ({
      kind: "tool_status",
      payload: { kind: "tool", call_id: "c1", tool_name: "read_file", ...over },
    }) as Frame;

  const argsOf = (state: State) => {
    const part = state.turns
      .flatMap((turn) => turn.parts)
      .find((p): p is ToolPart => p.kind === "tool");
    return part?.args;
  };

  it("folds the lifted field in beside the arguments", () => {
    const state = run(
      tool({ status: "started", args: { path: "/etc/x" }, narration: "Checking the config" }),
    );
    expect(argsOf(state)).toEqual({
      path: "/etc/x",
      narration: "Checking the config",
    });
  });

  it("leaves a narration the model wrote as an argument alone", () => {
    const state = run(
      tool({ status: "started", args: { narration: "as written" }, narration: "lifted" }),
    );
    expect(argsOf(state)).toEqual({ narration: "as written" });
  });

  it("adds nothing to a call that was never narrated", () => {
    const state = run(tool({ status: "started", args: { path: "/etc/x" } }));
    expect(argsOf(state)).toEqual({ path: "/etc/x" });
  });

  it("carries it across a later frame that brings arguments without it", () => {
    // `narration` is repeated on `finished` deliberately, but a kernel that
    // omitted it must not take back what `started` established.
    const state = run(
      tool({ status: "started", narration: "Checking the config" }),
      tool({ status: "finished", args: { path: "/etc/x" }, ok: true, summary: "Read it." }),
    );
    expect(argsOf(state)).toEqual({
      path: "/etc/x",
      narration: "Checking the config",
    });
  });
});

/**
 * Cancelling something must not put a modal on screen to say so.
 *
 * "Cancelled." is the kernel's acknowledgement of a Request, and it arrives on
 * `callable_output` for any client declaring `supports_callable_output`. The
 * guards for it were written when it arrived on `messages`, so it sailed past
 * them into the fallback that invents a command to hold stray output — and the
 * sidebar raises Settings for any named command. See `isCancellationEcho`.
 */
describe("the cancellation acknowledgement", () => {
  const cancelled = (): Frame =>
    ({ kind: "callable_output", payload: ["Cancelled."] }) as Frame;

  const settled = (reason: "answered" | "cancelled"): Frame =>
    ({
      kind: "approval_settled",
      payload: { request_id: "r1", reason },
    }) as Frame;

  it("invents no command when an approval was closed by this client", () => {
    // What `cancelInputRequest` dispatches the moment the X is pressed.
    const state = reduce(
      reduce(initialState, { type: "suppressNextCancellationNotice" }),
      { type: "frame", frame: cancelled() },
    );
    expect(state.command).toBeNull();
    expect(state.suppressNextCancellationNotice).toBe(false);
  });

  it("invents no command when the kernel says a question was cancelled", () => {
    // A peer answering, or the 300s timeout — no local dispatch happened.
    const state = run(settled("cancelled"), cancelled());
    expect(state.command).toBeNull();
  });

  it("leaves an answered question's output alone", () => {
    expect(run(settled("answered")).suppressNextCancellationNotice).toBe(false);
  });

  it("does not reopen Settings for a command cancelled from Settings", () => {
    // `say("/cancel")` records the tombstone, then the panel is dismissed —
    // which nulls `command` before the acknowledgement lands.
    let state = run(
      {
        kind: "tool_status",
        payload: {
          kind: "command",
          call_id: "cmd:llm:1",
          command_name: "llm",
          status: "started",
        },
      } as Frame,
    );
    state = reduce(state, { type: "said", text: "/cancel", isCommand: true });
    state = reduce(state, { type: "clearCommand" });
    state = reduce(state, { type: "frame", frame: cancelled() });
    expect(state.command).toBeNull();
  });

  it("still shows output that merely mentions cancelling", () => {
    const state = reduce(
      reduce(initialState, { type: "suppressNextCancellationNotice" }),
      {
        type: "frame",
        frame: {
          kind: "callable_output",
          payload: ["Cancelled. 3 jobs remain scheduled."],
        } as Frame,
      },
    );
    expect(state.command?.outcome).toEqual([
      "Cancelled. 3 jobs remain scheduled.",
    ]);
  });
});
