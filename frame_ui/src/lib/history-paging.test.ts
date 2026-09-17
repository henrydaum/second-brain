// @vitest-environment jsdom

/**
 * Reading a conversation a page at a time.
 *
 * `conv.read` used to answer with every row a conversation held. On a long one
 * that was 20 MB — 95% of it the state machine's own bookkeeping — which is
 * more than the kernel can put on a single wire message, so the answer became
 * undeliverable and the HTTP frontend's poll failed on every tick. The client
 * saw a request that never came back.
 *
 * Dropping the bookkeeping alone would only have moved the wall: a transcript
 * grows without limit whatever the model's context window is, because
 * compaction shrinks what the model sees and deletes nothing. So the read is
 * paged, and this is the client half of that.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

const sdk = vi.hoisted(() => vi.fn());
vi.mock("@/lib/client", () => ({ sdk }));

const { readConversation } = await import("@/lib/history");

const row = (id: number, content = `m${id}`) => ({
  id,
  role: "user",
  content,
  tool_call_id: null,
  tool_name: null,
  attachments: [],
});

beforeEach(() => sdk.mockReset());

describe("readConversation", () => {
  it("asks for the newest page when given no cursor", async () => {
    sdk.mockResolvedValue({ messages: [row(9)], has_more: false, oldest_id: 9 });

    await readConversation(7);

    expect(sdk).toHaveBeenCalledWith("conv.read", { id: 7, details: true });
  });

  it("passes a cursor and a limit through when asked to page", async () => {
    sdk.mockResolvedValue({ messages: [], has_more: false, oldest_id: null });

    await readConversation(7, { before: 42, limit: 20 });

    expect(sdk).toHaveBeenCalledWith("conv.read", {
      id: 7,
      details: true,
      before_id: 42,
      limit: 20,
    });
  });

  it("reports whether the conversation continues above the page", async () => {
    sdk.mockResolvedValue({
      messages: [row(4), row(5)],
      has_more: true,
      oldest_id: 4,
    });

    const read = await readConversation(7);

    expect(read.hasMore).toBe(true);
    expect(read.oldestId).toBe(4);
    expect(read.turns).toHaveLength(2);
  });

  it("prefers the server's cursor over the first row it rendered", async () => {
    // They agree today. The server is the one that knows what it *skipped* on
    // the way — a page whose leading rows the client renders none of would
    // otherwise report a cursor the client never saw, and page the same rows
    // forever.
    sdk.mockResolvedValue({
      messages: [row(80), row(81)],
      has_more: true,
      oldest_id: 61,
    });

    expect((await readConversation(7)).oldestId).toBe(61);
  });

  it("treats an older kernel's bare array as a conversation with no more", async () => {
    // That shape had no paging and answered with everything, so there was
    // never another page to ask for. Saying `hasMore: true` would offer a
    // button that fetches the same rows again forever.
    sdk.mockResolvedValue([row(1), row(2)]);

    const read = await readConversation(7);

    expect(read.hasMore).toBe(false);
    expect(read.turns).toHaveLength(2);
  });

  it("survives a kernel that omits the paging keys entirely", async () => {
    sdk.mockResolvedValue({ messages: [row(3)] });

    const read = await readConversation(7);

    expect(read.hasMore).toBe(false);
    expect(read.oldestId).toBe(3);
  });
});

it("scans file history across a tool call/result page boundary", async () => {
  const { readConversationFileTurns } = await import("./history");
  sdk.mockResolvedValueOnce({ messages: [{ ...row(2), role: "tool", tool_call_id: "show",
    content: "Shown", attachments: [{ path: "/old.png" }] }], has_more: true, oldest_id: 2 });
  sdk.mockResolvedValueOnce({ messages: [{ ...row(1), role: "assistant",
    content: JSON.stringify({ tool_calls: [{ id: "show", function: { name: "show_files", arguments: "{}" } }] }) }], has_more: false });
  const turns = await readConversationFileTurns(7);
  expect(sdk).toHaveBeenLastCalledWith("conv.read", { id: 7, details: true, before_id: 2 });
  expect(turns[0].parts).toContainEqual(expect.objectContaining({ kind: "files", paths: ["/old.png"] }));
  expect(turns[0].parts).toContainEqual(expect.objectContaining({ kind: "tool", status: "finished" }));
});
