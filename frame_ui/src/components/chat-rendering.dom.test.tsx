// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AssistantRuntimeProvider, ThreadPrimitive, useExternalStoreRuntime } from "@assistant-ui/react";
import { useReducer, type Dispatch } from "react";
import { initialState, reduce, type Action } from "@/runtime/store";
import { convertMessage } from "@/runtime/convert";
import { type Frame } from "@/lib/events";
import type { FileEvent } from "@/lib/ledger";

const controls = vi.hoisted(() => ({ waiting: false, rows: [] as unknown[], listeners: new Set<() => void>() }));
vi.mock("@/runtime/provider", async () => {
  const { useSyncExternalStore } = await import("react");
  return { useApprovals: () => {
    const waiting = useSyncExternalStore((listener) => {
      controls.listeners.add(listener);
      return () => { controls.listeners.delete(listener); };
    }, () => controls.waiting);
    return { inputRequests: waiting ? [{}] : [] };
  } };
});
vi.mock("@/lib/client", () => ({
  fileUrl: (path: string) => "http://localhost/files?path=" + encodeURIComponent(path),
  sdk: async () => null,
}));
vi.mock("@/lib/ledger", async (original) => ({
  ...await original<typeof import("@/lib/ledger")>(),
  readLedger: async () => controls.rows,
}));

const { AssistantMessage } = await import("@/components/thread");
const { FileActivityContext, currentFiles } = await import("@/runtime/file-activity-provider");
const { toSections, withStoreAttachments, fileTurns } = await import("@/runtime/file-activity");
const { ActivityLine } = await import("@/components/reply-activity");
const { ToolInput } = await import("@/components/tool-input");
const { toTurns } = await import("@/lib/history");
let dispatch: Dispatch<Action>;
const view = vi.fn();

function Harness({ events = [] }: { events?: FileEvent[] }) {
  const [state, send] = useReducer(reduce, initialState);
  dispatch = send;
  const runtime = useExternalStoreRuntime({
    messages: state.turns, convertMessage, isRunning: state.typing, onNew: async () => {},
  });
  const sections = toSections(withStoreAttachments(new Map([[state.turns[0]?.id ?? "unattributed", events]]), fileTurns(state.turns)), state.turns);
  const files = currentFiles(events, fileTurns(state.turns));
  return <AssistantRuntimeProvider runtime={runtime}>
      <FileActivityContext value={{
        sections, sectionFor: (id) => sections.find((section) => section.turnId === id) ?? null,
        fileFor: (path) => files.get(path), recoveredFor: () => [],
        entries: [...files.values()], total: files.size, failure: null,
        filesOpen: false, setFilesOpen: () => {}, openFilesAt: () => {},
        focusTurn: null, focusRequest: 0, clearFocus: () => {},
        viewing: null, view, stepView: () => {}, closeView: () => {},
      }}>
        <ThreadPrimitive.Root><ThreadPrimitive.Messages components={{ AssistantMessage, UserMessage: () => null }} /></ThreadPrimitive.Root>
      </FileActivityContext>
  </AssistantRuntimeProvider>;
}
const frame = (value: Frame) => act(async () => { dispatch({ type: "frame", frame: value }); });
const text = (stream_id: string, delta: string, seq = 1, done = false) =>
  frame({ kind: "stream_delta", payload: { stream_id, delta, seq, done } });

beforeEach(() => {
  controls.waiting = false;
  view.mockClear();
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
  vi.stubGlobal("matchMedia", () => ({
    matches: false, addEventListener() {}, removeEventListener() {},
  }));
});
afterEach(() => { cleanup(); vi.useRealTimers(); });

describe("assembled assistant-ui reply", () => {
  it("labels a child-agent barrier as Waiting without ending the reply", async () => {
    render(<Harness />);
    await frame({ kind: "typing", payload: true });
    await frame({ kind: "turn_activity", payload: { phase: "waiting" } });
    expect(screen.getByRole("status", { name: "Waiting" })).toHaveTextContent("Waiting");
    expect(screen.queryByText("Waiting for your response")).toBeNull();
    await frame({ kind: "turn_activity", payload: { phase: "thinking" } });
    expect(screen.getByRole("status", { name: "Thinking" })).toHaveTextContent("Thinking");
  });

  it("waits through user interruptions and model end tokens, then renders one recap and footer", async () => {
    const { container } = render(<Harness />);
    await frame({ kind: "typing", payload: true });
    await text("parent", "Before interruption", 1, true);
    await frame({ kind: "attachments", payload: ["/before.png"] });
    await act(async () => dispatch({ type: "said", text: "Also do this" }));
    expect(container.querySelector('[data-slot="assistant-message-footer"]')).toBeNull();
    expect(container.querySelector('[data-slot="attachment-group"]')).toBeNull();
    await text("after", "After interruption", 1, true);
    await frame({ kind: "attachments", payload: ["/after.md"] });
    // A completed model stream is not the end of the logical turn.
    expect(container.querySelector('[data-slot="attachment-group"]')).toBeNull();
    await frame({ kind: "typing", payload: false });
    expect(container.querySelectorAll('[data-slot="assistant-message-footer"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-slot="attachment-group"]')).toHaveLength(1);
    expect(screen.getByRole("button", { name: "2 files" })).toBeInTheDocument();
    const last = container.querySelectorAll('[data-role="assistant"]');
    const final = last[last.length - 1];
    expect(final.querySelectorAll('[data-slot="attachment-tile"]')).toHaveLength(2);
    expect(final.querySelector('[data-slot="assistant-message-footer"]')?.parentElement?.lastElementChild)
      .toBe(final.querySelector('[data-slot="assistant-message-footer"]'));
    expect(screen.queryByText("download")).toBeNull();
    expect(screen.queryByText("embed")).toBeNull();
    const copy = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: copy } });
    fireEvent.click(screen.getByRole("button", { name: "Copy" }));
    await waitFor(() => expect(copy).toHaveBeenCalledWith("Before interruption\n\nAfter interruption"));
  });

  it("rebuilds all four recap files from database rows without live attachment frames", async () => {
    const events: FileEvent[] = [{ rowId: 1, ts: 3000, path: "/test.txt", effect: "wrote", viaShell: false }];
    const { container } = render(<Harness events={events} />);
    const paths = ["/a.png", "/b.png", "/c.png"];
    await frame({ kind: "typing", payload: true });
    await text("live", "Images first, then edit.", 1, true);
    await frame({ kind: "tool_status", payload: { call_id: "show", tool_name: "show_files", args: { paths }, status: "finished", ok: true, summary: "Showed 3 files." } });
    await frame({ kind: "attachments", payload: paths });
    await frame({ kind: "typing", payload: false });
    const snapshot = () => [...container.querySelectorAll('[data-slot="attachment-tile"]')].map((node) => node.getAttribute("data-path"));
    expect(snapshot()).toEqual([...paths, "/test.txt"]);
    for (let reload = 0; reload < 2; reload++) {
      // Only persisted tool-call input/result and the edit ledger row remain.
      const turns = toTurns([
        { id: 1, role: "assistant", content: JSON.stringify({ content: "Images first, then edit.", tool_calls: [{ id: "show", function: { name: "show_files", arguments: JSON.stringify({ paths }) } }] }), tool_call_id: null, tool_name: null, timestamp: 1 },
        { id: 2, role: "tool", content: "Showed 3 files.", tool_call_id: "show", tool_name: "show_files", timestamp: 2 },
      ]);
      await act(async () => dispatch({ type: "history", turns, hasMore: false, oldestId: 1 }));
      expect(snapshot()).toEqual([...paths, "/test.txt"]);
      expect(screen.getAllByRole("img")).toHaveLength(3);
      expect(screen.getByRole("button", { name: "4 files" })).toBeInTheDocument();
      expect(screen.getAllByText("4 files")).toHaveLength(1);
      expect(container.querySelectorAll('[data-slot="attachment-group"]')).toHaveLength(1);
    }
  });

  it.each(["share-first", "edit-first"])("defers both file sources until completion (%s)", async (order) => {
    const { container, rerender } = render(<Harness />);
    await frame({ kind: "typing", payload: true });
    const events: FileEvent[] = [{ rowId: 1, ts: 1, path: "/test.txt", effect: "wrote", viaShell: false }];
    if (order === "edit-first") rerender(<Harness events={events} />);
    await frame({ kind: "attachments", payload: ["/image.png"] });
    expect(container.querySelector('[data-slot="attachment-group"]')).toBeNull();
    if (order === "share-first") rerender(<Harness events={events} />);
    await text("recap", "The text stream is finished, but the turn is still working.", 1, true);
    expect(container.querySelector('[data-slot="attachment-group"]')).toBeNull();
    await frame({ kind: "typing", payload: false });
    expect(container.querySelectorAll('[data-slot="attachment-group"]')).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Open test.txt" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open image.png" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "2 files" })).toBeInTheDocument();
  });

  it("combines late shares and edits under one footer, and replaces removed files in place", async () => {
    const { container, rerender } = render(<Harness />);
    await frame({ kind: "typing", payload: true });
    await text("late", "The requested outcomes", 1, true);
    await frame({ kind: "typing", payload: false });
    await frame({ kind: "attachments", payload: ["/image.png"] });
    const edited: FileEvent = { rowId: 1, ts: 1, path: "/note.md", effect: "wrote", viaShell: false };
    rerender(<Harness events={[edited]} />);
    expect(screen.getByRole("button", { name: "2 files" })).toBeInTheDocument();
    expect(container.querySelectorAll('[data-role="assistant"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-slot="attachment-group"]')).toHaveLength(1);
    rerender(<Harness events={[edited, { ...edited, rowId: 2, path: "/image.png", effect: "deleted" }]} />);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getAllByText("File removed")).toHaveLength(1);
    const footer = container.querySelector('[data-slot="assistant-message-footer"]')!;
    expect(footer.parentElement?.lastElementChild).toBe(footer);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("combines file events inside the reply, after text and above the footer", async () => {
    const { container } = render(<Harness />);
    await frame({ kind: "typing", payload: true });
    await text("s1", "Before the first images");
    await frame({ kind: "attachments", payload: ["/one.png", "/two.png"] });
    await text("s1", "Between the images", 2);
    await frame({ kind: "tool_status", payload: {
      call_id: "c2", tool_name: "show_files", status: "started", args: { paths: ["/three.png"] },
    } });
    await frame({ kind: "attachments", payload: ["/three.png"] });
    await frame({ kind: "tool_status", payload: { call_id: "c2", tool_name: "show_files", status: "finished", ok: true } });
    await text("s2", "After the images", 1, true);
    await frame({ kind: "typing", payload: false });
    await waitFor(() => expect(screen.getAllByRole("img")).toHaveLength(3));
    const replies = container.querySelectorAll('[data-role="assistant"]');
    expect(replies).toHaveLength(1);
    const reply = replies[0] as HTMLElement;
    const groups = reply.querySelectorAll('[data-slot="attachment-group"]');
    expect(groups).toHaveLength(1);
    const footer = reply.querySelector('[data-slot="assistant-message-footer"]')!;
    expect(footer.parentElement?.lastElementChild).toBe(footer);
    expect(screen.getByText("After the images").compareDocumentPosition(groups[0]) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(reply).getByRole("button", { name: "3 files" })).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open two.png" }));
    expect(view).toHaveBeenCalledWith(["/one.png", "/three.png", "/two.png"], 2);
  });

  it("switches one status line using stream completion, including waiting", async () => {
    render(<Harness />);
    await frame({ kind: "typing", payload: true });
    expect(screen.getByRole("status")).toHaveTextContent("Thinking");
    await text("s1", "A finished paragraph");
    expect(screen.getAllByRole("status")).toHaveLength(1);
    expect(screen.getByRole("status")).toHaveTextContent("Writing");
    await text("s1", "", 2, true);
    expect(screen.getByRole("status")).toHaveTextContent("Thinking");
    act(() => { controls.waiting = true; controls.listeners.forEach((listener) => listener()); });
    expect(screen.getByRole("status")).toHaveTextContent("Waiting for your response");
    await frame({ kind: "typing", payload: false });
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("expands large galleries and deduplicates paths within the reply outcome", async () => {
    render(<Harness />);
    await frame({ kind: "typing", payload: true });
    await frame({ kind: "attachments", payload: ["/1.png", "/2.png", "/3.png", "/4.png", "/5.png"] });
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    await frame({ kind: "typing", payload: false });
    expect(await screen.findAllByRole("img")).toHaveLength(4);
    fireEvent.click(screen.getByRole("button", { name: "Show 1 more" }));
    expect(screen.getAllByRole("img")).toHaveLength(5);
    await frame({ kind: "attachments", payload: ["/1.png"] });
    expect(screen.getAllByRole("img")).toHaveLength(5);
    fireEvent.error(screen.getAllByRole("img")[0]);
    expect(screen.getByText("Preview unavailable · Open file")).toBeInTheDocument();
  });
});

it("restarts Working elapsed time after Writing and waiting", () => {
  vi.useFakeTimers();
  const { rerender } = render(<ActivityLine phase="working" />);
  act(() => vi.advanceTimersByTime(4000));
  expect(screen.getByText("4s")).toBeInTheDocument();
  rerender(<ActivityLine phase="writing" />);
  rerender(<ActivityLine phase="working" />);
  expect(screen.queryByText("4s")).not.toBeInTheDocument();
  act(() => vi.advanceTimersByTime(3000));
  expect(screen.getByText("3s")).toBeInTheDocument();
  rerender(<ActivityLine phase="waiting" />);
  expect(screen.queryByText("3s")).not.toBeInTheDocument();
});

it("shows literal structured inputs, with a raw toggle and no copy control", () => {
  const args = { paths: ["/a folder/image.png"], caption: "*Literal* caption", options: { count: 2 } };
  render(<ToolInput args={args} argsText={JSON.stringify(args)} />);
  expect(screen.getByText("/a folder/image.png")).toBeInTheDocument();
  expect(screen.getByText("*Literal* caption")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Raw JSON" }));
  expect(screen.queryByRole("button", { name: "Copy input" })).not.toBeInTheDocument();
  expect(document.querySelector("pre")?.textContent).toBe(JSON.stringify(args, null, 2));
});

it("coalesces tools across hidden file and empty text parts", async () => {
  render(<Harness />);
  await frame({ kind: "typing", payload: true });
  for (let index = 0; index < 4; index++) {
    if (index === 1) {
      await frame({ kind: "attachments", payload: ["/one.png"] });
      await text("empty", " ", 1, true);
    }
    await frame({ kind: "tool_status", payload: { call_id: `group-${index}`, tool_name: "test", status: "started" } });
    await frame({ kind: "tool_status", payload: { call_id: `group-${index}`, tool_name: "test", status: "finished", ok: true } });
  }
  await frame({ kind: "typing", payload: false });
  expect(screen.getByText("4 tool calls")).toBeInTheDocument();
  expect(screen.queryByText("1 tool call")).toBeNull();
});
