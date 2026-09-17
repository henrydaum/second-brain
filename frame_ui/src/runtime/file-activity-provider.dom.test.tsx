// @vitest-environment jsdom
import { act, cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { Turn } from "@/runtime/store";
import type { LedgerRow } from "@/lib/ledger";

const controls = vi.hoisted(() => ({ conversationId: 1, state: { turns: [] as Turn[], typing: false } }));
vi.mock("@/runtime/provider", () => ({
  useConversations: () => ({ conversationId: controls.conversationId }),
  useSession: () => ({ state: controls.state }),
}));
vi.mock("@/lib/ledger", async (original) => ({
  ...await original<typeof import("@/lib/ledger")>(), readLedger: vi.fn(),
}));
vi.mock("@/lib/history", async (original) => ({
  ...await original<typeof import("@/lib/history")>(), readConversationFileTurns: vi.fn(async () => []),
}));
vi.mock("@/lib/files", () => ({ forgetFile: vi.fn() }));
vi.mock("@/lib/thumbnails", () => ({ forgetThumbnail: vi.fn() }));
const { readLedger } = await import("@/lib/ledger");
const { toTurns } = await import("@/lib/history");
const { FileActivityProvider, useFileActivity, currentFiles } = await import("./file-activity-provider");
let activity: ReturnType<typeof useFileActivity>;
function Probe() { activity = useFileActivity(); return null; }
const app = () => <FileActivityProvider><Probe /></FileActivityProvider>;
const turn = (id: string, live = true): Turn => ({
  id, role: "assistant", parts: [], running: false, aborted: false,
  source: live ? "live" : "history", createdAt: 1000,
});
const row = (id: number, path: string, action = "fs.write"): LedgerRow => ({
  id, ts: id + 1, origin: "agent", action_type: action, conversation_id: 1,
  ok: 1, error_code: null, args_json: "{}", data_json: JSON.stringify({ paths: [path] }),
});
function deferred() {
  let resolve!: (rows: LedgerRow[]) => void;
  const promise = new Promise<LedgerRow[]>((done) => { resolve = done; });
  return { promise, resolve };
}
beforeEach(() => { controls.conversationId = 1; controls.state = { turns: [turn("first")], typing: false }; vi.mocked(readLedger).mockReset(); });
afterEach(cleanup);

it("retains explorer origin while paging and resets it for other preview sources", async () => {
  vi.mocked(readLedger).mockResolvedValue([]);
  render(app());
  await act(async () => activity.view(["/one.txt", "/two.txt"], 0, "explorer"));
  act(() => activity.stepView(1));
  expect(activity.viewing).toMatchObject({ source: "explorer", index: 1 });
  act(() => activity.view(["/one.txt"], 0));
  expect(activity.viewing?.source).toBeUndefined();
});

it("rejects an opening ledger response after switching conversations", async () => {
  const stale = deferred();
  vi.mocked(readLedger).mockImplementation(async (id) => id === 1 ? stale.promise : []);
  const { rerender } = render(app());
  controls.conversationId = 2;
  controls.state = { turns: [turn("second")], typing: false };
  rerender(app());
  await act(async () => stale.resolve([row(1, "/wrong.md")]));
  expect(activity.entries).toEqual([]);
  expect(activity.sectionFor("second")).toBeNull();
});

it("captures poll ownership and queues overlapping requests without duplicate rows", async () => {
  const pending = deferred();
  vi.mocked(readLedger).mockResolvedValueOnce([]).mockReturnValueOnce(pending.promise).mockResolvedValue([]);
  const { rerender } = render(app());
  await waitFor(() => expect(readLedger).toHaveBeenCalledTimes(2));
  controls.state = { turns: [turn("first"), turn("second")], typing: true };
  rerender(app());
  expect(readLedger).toHaveBeenCalledTimes(2);
  await act(async () => pending.resolve([row(1, "/note.md"), row(1, "/note.md")]));
  await waitFor(() => expect(readLedger).toHaveBeenCalledTimes(3));
  expect(activity.entries).toHaveLength(1);
  expect(activity.sectionFor("first")?.touched[0].path).toBe("/note.md");
  expect(activity.sectionFor("second")).toBeNull();
});

it("combines explicit shares and ledger edits in one reply section", async () => {
  const reply = turn("first");
  reply.parts = [{ kind: "files", id: "share", paths: ["/photo.png"], receivedAt: 4000 }];
  controls.state.turns = [reply];
  vi.mocked(readLedger).mockResolvedValueOnce([]).mockResolvedValueOnce([row(1, "/note.md")]).mockResolvedValue([]);
  render(app());
  await waitFor(() => expect(activity.entries).toHaveLength(2));
  const section = activity.sectionFor("first")!;
  expect([...section.shown, ...section.touched].map((file) => file.path).sort()).toEqual(["/note.md", "/photo.png"]);
});

it("uses ledger order for deletion and recreation even with skewed timestamps", () => {
  const events = [
    { rowId: 2, path: "/a.png", ts: 1, effect: "deleted" as const, viaShell: false },
    { rowId: 1, path: "/a.png", ts: 99999, effect: "shown" as const, viaShell: false },
  ];
  expect(currentFiles(events, []).get("/a.png")?.gone).toBe(true);
  expect(currentFiles([{ ...events[0], rowId: 3, effect: "wrote" }, ...events], []).get("/a.png")?.gone).toBe(false);
});

it("cold-loads successful show_files outputs alongside ledger edits", async () => {
  controls.state.turns = toTurns([
    { id: 1, role: "assistant", content: JSON.stringify({ content: "Four outcomes", tool_calls: [{ id: "show", function: { name: "show_files", arguments: JSON.stringify({ paths: ["/a.png", "/b.png", "/c.png"] }) } }] }), tool_call_id: null, tool_name: null, timestamp: 1 },
    { id: 2, role: "tool", content: "Showed 3 files.", tool_call_id: "show", tool_name: "show_files", timestamp: 2 },
  ]);
  vi.mocked(readLedger).mockResolvedValueOnce([row(1, "/test.txt")]).mockResolvedValue([]);
  render(app());
  await waitFor(() => expect(activity.entries).toHaveLength(4));
  expect(activity.sections).toHaveLength(1);
  const section = activity.sectionFor("stored-1")!;
  expect([...section.shown, ...section.touched].map((file) => file.path).sort()).toEqual(["/a.png", "/b.png", "/c.png", "/test.txt"]);
});

it("uses durable turn IDs over timestamps and captured poll ownership", async () => {
  const pending = deferred();
  vi.mocked(readLedger).mockResolvedValueOnce([]).mockReturnValueOnce(pending.promise).mockResolvedValue([]);
  controls.state.turns = [{ ...turn("before"), turnId: "a", continues: true },
    { ...turn("after"), turnId: "a" }, { ...turn("newer"), turnId: "b" }];
  render(app());
  await waitFor(() => expect(readLedger).toHaveBeenCalledTimes(2));
  await act(async () => pending.resolve([{ ...row(1, "/one.md"), ts: 999999,
    data_json: JSON.stringify({ paths: ["/one.md"], turn_id: "a" }) }]));
  expect(activity.sectionFor("after")?.touched.map((entry) => entry.path)).toEqual(["/one.md"]);
  expect(activity.sectionFor("before")).toBeNull();
  expect(activity.sectionFor("newer")).toBeNull();
  // The same durable records must give the same assignment at reload.
  cleanup();
  vi.mocked(readLedger).mockResolvedValueOnce([{ ...row(1, "/one.md"), ts: 999999,
    data_json: JSON.stringify({ paths: ["/one.md"], turn_id: "a" }) }]).mockResolvedValue([]);
  controls.state.turns = controls.state.turns.map((turn) => ({ ...turn, source: "history" }));
  render(app());
  await waitFor(() => expect(activity.sectionFor("after")?.touched).toHaveLength(1));
  expect(activity.sectionFor("newer")).toBeNull();
});

it("keeps unknown durable ownership drawer-only until its message arrives", async () => {
  vi.mocked(readLedger).mockResolvedValueOnce([{ ...row(1, "/one.md"),
    data_json: JSON.stringify({ paths: ["/one.md"], turn_id: "later" }) }]).mockResolvedValue([]);
  const { rerender } = render(app());
  await waitFor(() => expect(activity.entries).toHaveLength(1));
  expect(activity.sectionFor("first")).toBeNull();
  controls.state.turns = [...controls.state.turns, { ...turn("later-segment"), turnId: "later" }];
  rerender(app());
  expect(activity.sectionFor("later-segment")?.touched).toHaveLength(1);
});

it("includes persisted files outside the loaded transcript", async () => {
  const { readConversationFileTurns } = await import("@/lib/history");
  vi.mocked(readConversationFileTurns).mockResolvedValueOnce([{
    ...turn("old", false), createdAt: 1,
    parts: [{ kind: "files", paths: ["/old.png"] }],
  }]);
  vi.mocked(readLedger).mockResolvedValue([]);
  const { rerender } = render(app());
  await waitFor(() => expect(activity.entries.map((entry) => entry.path)).toContain("/old.png"));
  expect(controls.state.turns.map((turn) => turn.id)).toEqual(["first"]);
  controls.state.turns = [{ ...turn("old", false), createdAt: 1 }, ...controls.state.turns];
  rerender(app());
  expect(activity.entries.map((entry) => entry.path)).toContain("/old.png");
});
