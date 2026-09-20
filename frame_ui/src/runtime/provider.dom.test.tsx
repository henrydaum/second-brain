// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";

/**
 * What the provider believes about the turn when the page first loads.
 *
 * `typing` only ever moves on a `typing` frame, and a reload cannot receive the
 * one that opened a turn already in progress. Everything downstream of that is
 * cosmetic except one thing: `isRunning` is what puts Stop in the composer, so
 * getting it wrong takes away the only control that ends a turn — the case that
 * is worth a real render.
 *
 * The provider is *rendered* here, unlike the component tests next door, which
 * stub it deliberately. What is under test is its boot sequence, and there is no
 * way to stub that and still be testing it. The six modules it reaches the
 * server through are mocked instead, so nothing opens a connection.
 */

import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ComposerPrimitive,
  useAuiState,
  type PendingAttachment,
} from "@assistant-ui/react";

const sdk = vi.fn();
const connect = vi.fn();
const readConversation = vi.fn();
let receiveFrame: ((frame: unknown) => void) | null = null;

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
  listConversations: async () => ({ items: [], hasMore: false, categories: [] }),
  CONVERSATION_PAGE: 50,
  PLACEHOLDER_TITLE: "",
  setConversationTitle: async () => true,
  setConversationCategory: async () => true,
}));
vi.mock("@/lib/notifications", () => ({
  listNotifications: async () => [],
  markRead: async () => undefined,
}));

const { SecondBrainProvider } = await import("@/runtime/provider");
const { attachmentAdapter } = await import("@/runtime/attachment-adapter");
const { useConversations, useModels, useSession, useWidget } = await import(
  "@/runtime/domains"
);

/** Reads the one field under test out of the context. */
const Probe = () => {
  const { state } = useSession();
  const { conversationId } = useConversations();
  return <><span data-testid="typing">{String(state.typing)}</span>
    <span data-testid="conversation">{String(conversationId)}</span>
    <span data-testid="turns">{state.turns.length}</span></>;
};

const ModelProbe = () => {
  const { modelName, agentProfile, models, setModel } = useModels();
  return (
    <>
      <span data-testid="model">{modelName}</span>
      <span data-testid="agent">{agentProfile}</span>
      <span data-testid="models">{models.map((model) => model.model_name).join(",")}</span>
      <button type="button" onClick={() => void setModel("openrouter/gpt-5.4")}>
        Switch
      </button>
    </>
  );
};

const SubmitProbe = () => {
  const { submitting } = useSession();
  const running = useAuiState((s) => s.thread.isRunning);
  return (
    <>
      <span data-testid="submitting">{String(submitting)}</span>
      <span data-testid="runtime-running">{String(running)}</span>
      <ComposerPrimitive.Root>
        <ComposerPrimitive.Input aria-label="Message" />
        <ComposerPrimitive.Send>Send</ComposerPrimitive.Send>
      </ComposerPrimitive.Root>
    </>
  );
};

/** Answer `session.get` with `busy`, and everything else with nothing. */
const bootWith = (busy: boolean) => {
  sdk.mockImplementation(async (type: string) => {
    if (type === "session.get") {
      return { conversation_id: 7, mode: "ask", busy };
    }
    return null;
  });
};

beforeEach(() => {
  vi.clearAllMocks();
  receiveFrame = null;
  readConversation.mockResolvedValue({ turns: [], conversation: null });
  // The stream reports itself open, which is what the reconnect sync keys on.
  connect.mockImplementation(
    (onFrame: (frame: unknown) => void, setStatus: (s: string) => void) => {
      receiveFrame = onFrame;
      setStatus("open");
      return () => {};
    },
  );
});

describe("remote conversation handoff", () => {
  it("clears the transcript when the session is reset elsewhere", async () => {
    bootWith(false);
    readConversation.mockResolvedValue({
      turns: [{ id: "old", role: "user", parts: [] }],
      conversation: { id: 7, title: "Old" },
      hasMore: false,
      oldestId: null,
    });
    render(<SecondBrainProvider><Probe /></SecondBrainProvider>);
    await waitFor(() => expect(screen.getByTestId("conversation")).toHaveTextContent("7"));
    expect(screen.getByTestId("turns")).toHaveTextContent("1");

    receiveFrame?.({
      kind: "conversation",
      payload: { conversation_id: null, title: "New Conversation" },
    });

    await waitFor(() => expect(screen.getByTestId("conversation")).toHaveTextContent("null"));
    expect(screen.getByTestId("turns")).toHaveTextContent("0");
  });
});

afterEach(cleanup);

it("keeps the old widget binding until the new name and state arrive together", async () => {
  bootWith(false);
  const WidgetProbe = () => {
    const { widgetBinding, chooseWidget } = useWidget();
    return <>
      <span data-testid="binding">{JSON.stringify(widgetBinding)}</span>
      <button onClick={() => chooseWidget("b")}>Choose B</button>
    </>;
  };
  render(<SecondBrainProvider><WidgetProbe /></SecondBrainProvider>);
  await waitFor(() => expect(sdk).toHaveBeenCalledWith("widget.get", {}));
  const binding = { conversation_id: 7, name: "a", path: "/a.html", state: "1", installed: true };
  await act(async () => receiveFrame?.({ kind: "widget", payload: binding }));
  await userEvent.click(screen.getByText("Choose B"));
  expect(sdk).toHaveBeenCalledWith("widget.set", { name: "b" });
  expect(screen.getByTestId("binding")).toHaveTextContent('"name":"a"');
  expect(screen.getByTestId("binding")).toHaveTextContent('"state":"1"');
  await act(async () => receiveFrame?.({ kind: "widget", payload: {
    ...binding, name: "b", path: "/b.html", state: "2",
  } }));
  expect(screen.getByTestId("binding")).toHaveTextContent('"name":"b"');
  expect(screen.getByTestId("binding")).toHaveTextContent('"state":"2"');
});

describe("a page that loads in the middle of a turn", () => {
  it("comes up knowing the agent has the turn", async () => {
    bootWith(true);
    render(
      <SecondBrainProvider>
        <Probe />
      </SecondBrainProvider>,
    );

    // Without this the composer offers Send where Stop belongs, and reloading
    // becomes a way to lose the ability to interrupt.
    await waitFor(() =>
      expect(screen.getByTestId("typing").textContent).toBe("true"),
    );
  });

  it("survives the history read, which resets everything transient", async () => {
    // `history` is dispatched after the session is read and returns the store to
    // its initial state — so a seed placed before it would be thrown away, and
    // the assertion above would pass for one render and then stop being true.
    bootWith(true);
    render(
      <SecondBrainProvider>
        <Probe />
      </SecondBrainProvider>,
    );

    await waitFor(() => expect(readConversation).toHaveBeenCalledWith(7));
    await waitFor(() =>
      expect(screen.getByTestId("typing").textContent).toBe("true"),
    );
  });

  it("leaves an idle session idle", async () => {
    bootWith(false);
    render(
      <SecondBrainProvider>
        <Probe />
      </SecondBrainProvider>,
    );

    await waitFor(() => expect(readConversation).toHaveBeenCalledWith(7));
    expect(screen.getByTestId("typing").textContent).toBe("false");
  });
});

/**
 * The first message in a session, which is also the one that makes the
 * conversation.
 *
 * The server creates it on the way in and announces it, so a `conversation`
 * frame naming an id this client has never seen arrives *while the turn it
 * started is opening*. Read back as a switch, the scrollback replaces
 * everything transient — which is the running turn, the typing flag, and with
 * them the Thinking line and the Stop button, for the whole of the model's
 * first think.
 */
describe("the message that creates the conversation", () => {
  it("keeps the turn it started", async () => {
    sdk.mockImplementation(async (type: string) => {
      if (type === "session.get") return { mode: "ask", busy: false };
      return null;
    });
    readConversation.mockResolvedValue({
      turns: [{ id: "stored", role: "user", parts: [], running: false, aborted: false }],
      conversation: { id: 12, title: "New" },
      hasMore: false,
      oldestId: null,
    });
    const user = userEvent.setup();
    render(<SecondBrainProvider><Probe /><SubmitProbe /></SecondBrainProvider>);

    await user.type(screen.getByRole("textbox", { name: "Message" }), "Hello");
    await user.click(screen.getByRole("button", { name: "Send" }));

    // The server names the conversation it just made, then takes the turn.
    receiveFrame?.({ kind: "conversation", payload: { conversation_id: 12, title: "New" } });
    receiveFrame?.({ kind: "typing", payload: true });

    await waitFor(() => expect(screen.getByTestId("conversation")).toHaveTextContent("12"));
    // Long enough for a scrollback read to have landed, had one been made.
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.getByTestId("typing")).toHaveTextContent("true");
    expect(screen.getByTestId("runtime-running")).toHaveTextContent("true");
    // The row is still read, for the title — the transcript is what is left
    // alone, and `limit: 0` is how that is said.
    expect(readConversation).toHaveBeenCalledWith(12, { limit: 0 });
  });

  /**
   * The claim is consumed by the announcement, and the announcement can lose
   * the race it usually wins — `adoptConversation` asks for the same id and
   * sometimes answers first. A claim left standing afterwards would be spent
   * on whatever switched us next, which is the one case that genuinely needs
   * the scrollback.
   */
  it("does not let the claim outlive the message that made it", async () => {
    let bound: number | null = null;
    sdk.mockImplementation(async (type: string) => {
      if (type === "session.get") return { conversation_id: bound, mode: "ask", busy: false };
      if (type === "frontend.submit") { bound = 12; return true; }
      return null;
    });
    const user = userEvent.setup();
    render(<SecondBrainProvider><Probe /><SubmitProbe /></SecondBrainProvider>);
    await waitFor(() => expect(screen.getByTestId("conversation")).toHaveTextContent("null"));

    await user.type(screen.getByRole("textbox", { name: "Message" }), "Hello");
    await user.click(screen.getByRole("button", { name: "Send" }));
    // No `conversation` frame: the ask won, and the claim was never spent.
    await waitFor(() => expect(screen.getByTestId("conversation")).toHaveTextContent("12"));

    receiveFrame?.({ kind: "conversation", payload: { conversation_id: 20, title: "Other" } });
    await waitFor(() => expect(readConversation).toHaveBeenCalledWith(20, {}));
  });

  it("still reads a conversation it was merely pointed at", async () => {
    sdk.mockImplementation(async (type: string) => {
      if (type === "session.get") return { mode: "ask", busy: false };
      return null;
    });
    render(<SecondBrainProvider><Probe /></SecondBrainProvider>);
    await waitFor(() => expect(screen.getByTestId("conversation")).toHaveTextContent("null"));

    receiveFrame?.({ kind: "conversation", payload: { conversation_id: 9, title: "Theirs" } });

    await waitFor(() => expect(readConversation).toHaveBeenCalledWith(9, {}));
  });
});

describe("submission acknowledgement", () => {
  it("shows a run before the submit request or first server frame settles", async () => {
    let finishSubmit!: () => void;
    sdk.mockImplementation(async (type: string) => {
      if (type === "session.get") {
        return { conversation_id: 7, mode: "ask", busy: false };
      }
      if (type === "frontend.submit") {
        await new Promise<void>((resolve) => { finishSubmit = resolve; });
        return true;
      }
      return null;
    });
    const user = userEvent.setup();
    render(<SecondBrainProvider><SubmitProbe /></SecondBrainProvider>);

    await user.type(screen.getByRole("textbox", { name: "Message" }), "Hello");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(screen.getByTestId("submitting")).toHaveTextContent("true");
    await waitFor(() =>
      expect(screen.getByTestId("runtime-running")).toHaveTextContent("true"),
    );

    receiveFrame?.({ kind: "typing", payload: true });
    await waitFor(() =>
      expect(screen.getByTestId("submitting")).toHaveTextContent("false"),
    );
    expect(screen.getByTestId("runtime-running")).toHaveTextContent("true");
    finishSubmit();
  });
});

describe("global model synchronization", () => {
  it("reads the default model, agent profile, and configured models", async () => {
    sdk.mockImplementation(async (type: string, args?: { key?: string }) => {
      if (type === "session.get") {
        return {
          conversation_id: 7,
          mode: "ask",
          busy: false,
          agent_profile: "researcher",
        };
      }
      if (type === "llm.list") {
        return { profiles: [{ model_name: "anthropic/sonnet-4.6" }] };
      }
      if (type === "config.read") {
        return args?.key === "llm_profiles"
          ? { "anthropic/sonnet-4.6": { llm_extra_params: { reasoning_effort: "low" } } }
          : "anthropic/sonnet-4.6";
      }
      return null;
    });

    render(
      <SecondBrainProvider>
        <ModelProbe />
      </SecondBrainProvider>,
    );

    await waitFor(() =>
      expect(screen.getByTestId("model")).toHaveTextContent("anthropic/sonnet-4.6"),
    );
    expect(screen.getByTestId("agent")).toHaveTextContent("researcher");
    expect(screen.getByTestId("models")).toHaveTextContent("anthropic/sonnet-4.6");
  });

  it("switches the global default through config.write", async () => {
    bootWith(false);
    sdk.mockImplementation(async (type: string) => {
      if (type === "session.get") {
        return { mode: "ask", busy: false };
      }
      if (type === "llm.list") return { profiles: [] };
      if (type === "config.read") return "anthropic/sonnet-4.6";
      if (type === "config.write") return true;
      return null;
    });
    const user = userEvent.setup();
    render(
      <SecondBrainProvider>
        <ModelProbe />
      </SecondBrainProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("model")).toHaveTextContent("anthropic/sonnet-4.6"),
    );
    await user.click(screen.getByRole("button", { name: "Switch" }));
    await waitFor(() =>
      expect(screen.getByTestId("model")).toHaveTextContent("openrouter/gpt-5.4"),
    );
    expect(sdk).toHaveBeenCalledWith("config.write", {
      key: "default_llm_profile",
      value: "openrouter/gpt-5.4",
      scope: "plugin",
    });
  });

  /**
   * `config.write` takes one whole top-level setting, so a nested field means
   * writing `llm_profiles` back entire. Everything this does not mean to touch
   * has to survive that — the other profiles, the sibling fields of the one it
   * does touch, and above all the API key, which comes back as a handle and is
   * restored by the kernel only if the handle is written back unchanged.
   */


});

/**
 * What the composer is told when an attachment cannot be uploaded.
 *
 * **Yielded, never thrown**, and that is a fact about assistant-ui rather than
 * a preference: `ComposerPrimitive.AddAttachment` and the dropzone both await
 * `add` inside a `try {} catch {}` whose body is empty, so an exception is
 * discarded and the chip stays frozen on whatever it last showed. A file
 * refused for its size would then look exactly like one still uploading, for
 * ever. The red tile and its tooltip in `attachment.tsx` are drawn from the
 * status below and from nothing else.
 */
describe("an attachment that cannot be uploaded", () => {
  // The adapter's `add` is typed as "generator or promise", which is how
  // assistant-ui drives it too. Ours is always the generator.
  const drain = async (file: File) => {
    const states: PendingAttachment[] = [];
    const added = attachmentAdapter.add({ file });
    if (Symbol.asyncIterator in added) {
      for await (const state of added) states.push(state);
    } else {
      states.push(await added);
    }
    return states;
  };

  const fileOfSize = (bytes: number, name = "film.mov"): File =>
    ({
      name,
      type: "video/quicktime",
      size: bytes,
      slice: () => ({ arrayBuffer: async () => new ArrayBuffer(0) }),
    }) as unknown as File;

  it("reports the reason on the chip rather than throwing it away", async () => {
    // Over the cap in `lib/upload.ts`, which refuses before reading anything.
    const states = await drain(fileOfSize(200 * 1024 * 1024));

    expect(states.at(-1)?.status).toMatchObject({
      type: "incomplete",
      reason: "error",
      message: expect.stringContaining("100 MB"),
    });
  });

  it("claims the chip before it can fail, so there is somewhere to say so", async () => {
    const states = await drain(fileOfSize(200 * 1024 * 1024));
    expect(states[0]?.status).toMatchObject({ type: "running" });
  });

  it("finishes ready to send when the upload works", async () => {
    sdk.mockImplementation(async (type: string) =>
      type === "fs.temp" ? "/tmp/scratch.mov" : true,
    );

    const states = await drain(fileOfSize(1024));

    expect(states.at(-1)?.status).toEqual({
      type: "requires-action",
      reason: "composer-send",
    });
  });
});
