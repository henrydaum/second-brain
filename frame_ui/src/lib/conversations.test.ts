import { describe, expect, it, vi } from "vitest";

// `client.ts` reads `window.location` as it loads, to work out which thread
// this browser is. Nothing under test here makes a Request, but importing the
// module is enough to need a stub in a suite that runs without a DOM.
vi.mock("@/lib/client", () => ({ sdk: vi.fn() }));

const { conversationLoadError, conversationTitle } = await import("@/lib/conversations");
type Conversation = import("@/lib/conversations").Conversation;

const conversation = (fields: Partial<Conversation> = {}): Conversation => ({
  id: 1,
  title: "",
  ...fields,
});

describe("conversationTitle", () => {
  it("keeps a title the conversation earned", () => {
    expect(conversationTitle(conversation({ title: "SQLite Migration Bug" })))
      .toBe("SQLite Migration Bug");
  });

  it("says New chat for the kernel's own wording", () => {
    expect(conversationTitle(conversation({ title: "New conversation" }))).toBe(
      "New chat",
    );
    // What the kernel writes when a category is involved.
    expect(
      conversationTitle(conversation({ title: "New conversation (Main)" })),
    ).toBe("New chat");
  });

  it("says New chat for no title at all", () => {
    expect(conversationTitle(conversation({ title: "   " }))).toBe("New chat");
  });

  it("leaves a cleared conversation saying so", () => {
    // The sweep may replace this one, but until it does the suffix is telling
    // you something "New chat" would not.
    expect(
      conversationTitle(conversation({ title: "Virginia Holiday (cleared)" })),
    ).toBe("Virginia Holiday (cleared)");
  });
});

describe("conversationLoadError", () => {
  it("includes a structured kernel reason", () => {
    expect(conversationLoadError({
      ok: false,
      error: {
        code: "conversation_in_use",
        message: "This conversation is bound to another session.",
      },
    })).toBe(
      "Could not open it. This conversation is bound to another session.",
    );
  });

  it("retains the fallback when the server supplies no reason", () => {
    expect(conversationLoadError({ ok: false })).toBe("Could not open it.");
  });
});
