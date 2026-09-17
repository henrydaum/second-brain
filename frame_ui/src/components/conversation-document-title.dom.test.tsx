// @vitest-environment jsdom

import { cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ConversationDocumentTitle } from "@/components/conversation-document-title";
import type { Conversation } from "@/lib/conversations";
import * as provider from "@/runtime/provider";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  document.title = "";
});

const conversation = (id: number, title: string): Conversation => ({
  id,
  title,
});

describe("the browser tab title", () => {
  it("follows conversation switches and automatic retitles", async () => {
    let current = {
      conversations: [
        conversation(1, "New conversation"),
        conversation(2, "Remember Value 1272"),
      ],
      conversationId: 1,
      conversationsLoaded: true,
      openConversation: vi.fn(),
      newConversation: vi.fn(),
      deleteConversation: vi.fn(),
      openConversationRow: null,
      renameConversation: vi.fn(),
      categoriseConversation: vi.fn(),
      conversationsHasMore: false,
      scrollbackHasMore: false,
      loadingOlderMessages: false,
      loadOlderMessages: vi.fn(),
      loadMoreConversations: vi.fn(),
      conversationCategories: [],
      conversationFilter: { type: "category", category: null } as const,
      setConversationFilter: vi.fn(),
    };
    vi.spyOn(provider, "useConversations").mockImplementation(
      () => current as ReturnType<typeof provider.useConversations>,
    );

    const view = render(<ConversationDocumentTitle />);
    await waitFor(() =>
      expect(document.title).toBe("New chat - Second Brain"),
    );

    current = { ...current, conversationId: 2 };
    view.rerender(<ConversationDocumentTitle />);
    await waitFor(() =>
      expect(document.title).toBe("Remember Value 1272 - Second Brain"),
    );

    current = {
      ...current,
      conversations: [
        conversation(1, "New conversation"),
        conversation(2, "Remember Value 1272 Automatically"),
      ],
    };
    view.rerender(<ConversationDocumentTitle />);
    await waitFor(() =>
      expect(document.title).toBe(
        "Remember Value 1272 Automatically - Second Brain",
      ),
    );
  });

  it("uses the app name while no conversation is bound", async () => {
    vi.spyOn(provider, "useConversations").mockReturnValue({
      conversations: [],
      conversationsLoaded: true,
      conversationId: null,
      openConversation: vi.fn(),
      newConversation: vi.fn(),
      deleteConversation: vi.fn(),
      openConversationRow: null,
      renameConversation: vi.fn(),
      categoriseConversation: vi.fn(),
      conversationsHasMore: false,
      scrollbackHasMore: false,
      loadingOlderMessages: false,
      loadOlderMessages: vi.fn(),
      loadMoreConversations: vi.fn(),
      conversationCategories: [],
      conversationFilter: { type: "category", category: null } as const,
      setConversationFilter: vi.fn(),
    });

    render(<ConversationDocumentTitle />);
    await waitFor(() => expect(document.title).toBe("Second Brain"));
  });
});
