/** @vitest-environment jsdom */
import { afterEach, expect, it, vi } from "vitest";
import { appDocument } from "./html-app";

// Execute the actual injected script, with a parent that acknowledges writes
// on demand. This tests the debounce and message protocol without a live kernel.
function bridge() {
  vi.useFakeTimers();
  const parent = { postMessage: vi.fn() };
  const guest = new EventTarget() as EventTarget & {
    brain: { state: { set: (value: unknown) => void } };
  };
  const html = new DOMParser().parseFromString(appDocument("", "token"), "text/html");
  new Function("window", "parent", html.querySelector("script")!.textContent!)(guest, parent);
  const send = (data: object) => {
    const event = new MessageEvent("message", { data: {
      channel: "second-brain-html-v1", token: "token", ...data,
    } });
    Object.defineProperty(event, "source", { value: parent });
    guest.dispatchEvent(event);
  };
  const messages = (kind: string) => parent.postMessage.mock.calls
    .map(([message]) => message).filter(message => message.kind === kind);
  return { state: guest.brain.state, send, messages };
}

afterEach(() => { vi.useRealTimers(); });

it("flushes a debounced change immediately and waits for its acknowledgement", async () => {
  const { state, send, messages } = bridge();
  state.set({ text: "first" });
  state.set({ text: "latest" });
  expect(messages("call")).toHaveLength(0);
  send({ kind: "flush", id: "refresh" });
  expect(messages("call")).toEqual([expect.objectContaining({
    type: "widget.state_set", args: { value: { text: "latest" } },
  })]);
  expect(messages("flushed")).toHaveLength(0);
  send({ kind: "result", id: messages("call")[0].id, data: null });
  await vi.advanceTimersByTimeAsync(500);
  expect(messages("flushed")).toEqual([expect.objectContaining({ id: "refresh" })]);
  expect(messages("call")).toHaveLength(1);
});

it("waits for an in-flight save, then saves newer edits in order", async () => {
  const { state, send, messages } = bridge();
  state.set(1);
  await vi.advanceTimersByTimeAsync(500);
  const first = messages("call")[0];
  state.set(2);
  send({ kind: "flush", id: "switch" });
  expect(messages("call")).toHaveLength(1);
  send({ kind: "result", id: first.id, data: null });
  await vi.advanceTimersByTimeAsync(0);
  expect(messages("call")).toHaveLength(2);
  expect(messages("call")[1].args.value).toBe(2);
  expect(messages("flushed")).toHaveLength(0);
  send({ kind: "result", id: messages("call")[1].id, data: null });
  await vi.advanceTimersByTimeAsync(500);
  expect(messages("flushed")).toHaveLength(1);
  expect(messages("call")).toHaveLength(2);
});

it("reports failed writes and retains the dirty state for a retry", async () => {
  const { state, send, messages } = bridge();
  state.set({ draft: "keep me" });
  send({ kind: "flush", id: "first" });
  send({ kind: "result", id: messages("call")[0].id, error: { message: "Disk full" } });
  await vi.advanceTimersByTimeAsync(0);
  expect(messages("flushed")[0].error.message).toBe("Disk full");
  send({ kind: "flush", id: "retry" });
  expect(messages("call")[1].args.value).toEqual({ draft: "keep me" });
  send({ kind: "result", id: messages("call")[1].id, data: null });
  await vi.advanceTimersByTimeAsync(0);
  expect(messages("flushed")[1]).toEqual(expect.objectContaining({ id: "retry" }));
  expect(messages("flushed")[1].error).toBeUndefined();
});

it("acknowledges a widget with no changes without writing empty state", async () => {
  const { send, messages } = bridge();
  send({ kind: "flush", id: "clean" });
  await vi.advanceTimersByTimeAsync(0);
  expect(messages("call")).toHaveLength(0);
  expect(messages("flushed")).toHaveLength(1);
});
