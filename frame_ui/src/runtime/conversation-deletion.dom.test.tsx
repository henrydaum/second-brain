// @vitest-environment jsdom

/**
 * A deleted conversation leaves the sidebar, whichever row it was.
 *
 * The list is refreshed by merging a fresh page over the rows already held,
 * which is what lets a person keep pages they had asked for while a one-minute
 * poll re-reads the front. The merge has to know when it is *not* looking at a
 * truncated read, or the one row it preserves is the one row that was just
 * deleted — and it puts it back on every poll, so the row outlives its own
 * deletion until the page is reloaded.
 */

import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const sdk = vi.fn();
const connect = vi.fn();
const readConversation = vi.fn();

/** The rows the server has, newest first. Mutated by `conv.delete`. */
let server: { id: number; title: string; updated_at: number }[] = [];

vi.mock("@/lib/client", () => ({
  sdk: (...args: unknown[]) => sdk(...args),
  RequestFailed: class RequestFailed extends Error {},
}));
vi.mock("@/lib/events", () => ({
  connect: (...args: unknown[]) => connect(...args),
}));
vi.mock("@/lib/history", () => ({
  readConversation: (...args: unknown[]) => readConversation(...args),
}));
vi.mock("@/lib/commands", () => ({
  listCommands: async () => [],
  looksLikeCommand: () => false,
}));
vi.mock("@/lib/conversations", () => ({
  listConversations: async ({ limit = 50, offset = 0 } = {}) => ({
    items: server.slice(offset, offset + limit),
    hasMore: offset + limit < server.length,
    categories: [],
  }),
  CONVERSATION_PAGE: 50,
  setConversationTitle: async () => true,
  setConversationCategory: async () => true,
  conversationTitle: (row: { title: string }) => row.title,
}));
vi.mock("@/lib/notifications", () => ({
  listNotifications: async () => [],
  markRead: async () => undefined,
}));

const { SecondBrainProvider, useConversations } = await import(
  "@/runtime/provider"
);

const Probe = () => {
  const { conversations, deleteConversation } = useConversations();
  return (
    <>
      <span data-testid="list">
        {conversations.map((row) => row.id).join(",")}
      </span>
      {[3, 2, 1].map((id) => (
        <button
          key={id}
          type="button"
          onClick={() => void deleteConversation(id)}
        >
          {`Delete ${id}`}
        </button>
      ))}
    </>
  );
};

beforeEach(() => {
  vi.clearAllMocks();
  server = [3, 2, 1].map((id) => ({ id, title: `c${id}`, updated_at: id }));
  readConversation.mockResolvedValue({
    turns: [],
    conversation: null,
    hasMore: false,
    oldestId: null,
  });
  connect.mockImplementation(
    (_frame: unknown, setStatus: (status: string) => void) => {
      setStatus("open");
      return () => {};
    },
  );
  sdk.mockImplementation(
    async (type: string, args: Record<string, unknown>) => {
      if (type === "session.get")
        return { conversation_id: 9, mode: "ask", busy: false };
      if (type === "conv.delete") {
        server = server.filter((row) => row.id !== args.id);
        return true;
      }
      return null;
    },
  );
});

afterEach(cleanup);

describe("deleting a conversation", () => {
  // The oldest row is the one the merge reaches for, so it is the case that
  // regressed. Every other position happened to be covered by the fresh page.
  it.each([
    ["the oldest row", 1, "3,2"],
    ["a middle row", 2, "3,1"],
    ["the newest row", 3, "2,1"],
  ])("takes %s out of the list", async (_name, id, expected) => {
    render(
      <SecondBrainProvider>
        <Probe />
      </SecondBrainProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("list").textContent).toBe("3,2,1"),
    );

    await userEvent.click(screen.getByText(`Delete ${id}`));

    await waitFor(() =>
      expect(screen.getByTestId("list").textContent).toBe(expected),
    );
  });
});
