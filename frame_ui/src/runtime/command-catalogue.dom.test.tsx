// @vitest-environment jsdom

/**
 * A command installed while the page is open becomes usable while the page is
 * open.
 *
 * `command.list` was read once, by `loadCatalogue`, and never again — so
 * `/plugin install` reported success and the command it had just added was
 * absent from Settings and unrecognised by the composer until a reload. The
 * catalogue is now re-read at the end of every turn, and again on coming back
 * to the tab, which is the case where something else did the installing.
 */

import "@testing-library/jest-dom/vitest";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const sdk = vi.fn();
const connect = vi.fn();
const readConversation = vi.fn();
let receiveFrame: ((frame: unknown) => void) | null = null;

/** What the server would answer `command.list` with, right now. */
let installed: { name: string }[] = [];

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
  listCommands: async () => installed,
  looksLikeCommand: () => false,
}));
vi.mock("@/lib/conversations", () => ({
  listConversations: async () => ({ items: [], hasMore: false, categories: [] }),
  CONVERSATION_PAGE: 50,
  setConversationTitle: async () => true,
  setConversationCategory: async () => true,
}));
vi.mock("@/lib/notifications", () => ({
  listNotifications: async () => [],
  markRead: async () => undefined,
}));

const { SecondBrainProvider, useSettings } = await import("@/runtime/provider");

const Probe = () => {
  const { commands } = useSettings();
  return (
    <span data-testid="commands">
      {commands.map((command) => command.name).join(",")}
    </span>
  );
};

/** Drive `state.typing` the way the stream does, so the turn-end effect runs. */
const turn = async () => {
  await act(async () => {
    receiveFrame?.({ kind: "typing", payload: true });
  });
  await act(async () => {
    receiveFrame?.({ kind: "typing", payload: false });
  });
};

beforeEach(() => {
  vi.clearAllMocks();
  receiveFrame = null;
  installed = [{ name: "compact" }];
  readConversation.mockResolvedValue({
    turns: [],
    conversation: null,
    hasMore: false,
    oldestId: null,
  });
  connect.mockImplementation(
    (onFrame: (frame: unknown) => void, setStatus: (status: string) => void) => {
      receiveFrame = onFrame;
      setStatus("open");
      return () => {};
    },
  );
  sdk.mockImplementation(async (type: string) =>
    type === "session.get"
      ? { conversation_id: 9, mode: "ask", busy: false }
      : null,
  );
});

afterEach(cleanup);

describe("the command catalogue", () => {
  it("picks up a command installed during a turn", async () => {
    render(
      <SecondBrainProvider>
        <Probe />
      </SecondBrainProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("commands").textContent).toBe("compact"),
    );

    // What `/plugin install` does on the other side of the wire.
    installed = [{ name: "compact" }, { name: "weather" }];
    await turn();

    await waitFor(() =>
      expect(screen.getByTestId("commands").textContent).toBe(
        "compact,weather",
      ),
    );
  });

  /**
   * The same read-once mistake, one level up: the session was re-read only
   * after a command named `mode`, `llm` or `agent`. Any other route to the
   * model, the agent profile or the security mode — `/config`, or a command
   * installed from the store, which no such list can name in advance — left the
   * chrome showing what those settings used to be.
   */
  it("re-reads the session after any command, not three named ones", async () => {
    render(
      <SecondBrainProvider>
        <Probe />
      </SecondBrainProvider>,
    );
    await waitFor(() => expect(sdk).toHaveBeenCalled());
    sdk.mockClear();

    await act(async () => {
      receiveFrame?.({
        kind: "tool_status",
        payload: {
          kind: "command",
          call_id: "c1",
          command_name: "config",
          status: "running",
        },
      });
    });
    await act(async () => {
      receiveFrame?.({
        kind: "tool_status",
        payload: {
          kind: "command",
          call_id: "c1",
          command_name: "config",
          status: "finished",
          ok: true,
        },
      });
    });

    await waitFor(() =>
      expect(sdk).toHaveBeenCalledWith("session.get", { details: true }),
    );
  });

  it("picks up one installed elsewhere, on coming back to the tab", async () => {
    render(
      <SecondBrainProvider>
        <Probe />
      </SecondBrainProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("commands").textContent).toBe("compact"),
    );

    installed = [{ name: "compact" }, { name: "weather" }];
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
    });

    await waitFor(() =>
      expect(screen.getByTestId("commands").textContent).toBe(
        "compact,weather",
      ),
    );
  });
});
