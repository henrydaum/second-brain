/** @vitest-environment jsdom */

import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const say = vi.fn();
const deleteConversation = vi.fn();

// Mutable so a test can put the composer mid-turn without a second mock.
let typing = false;
let status = "open";

vi.mock("@/runtime/provider", () => ({
  useSession: () => ({ say, state: { typing }, status }),
  useConversations: () => ({
    conversationId: 7,
    openConversationRow: { id: 7, title: "Kitchen rewire", category: null },
    conversationCategories: [],
    renameConversation: vi.fn(),
    categoriseConversation: vi.fn(),
    deleteConversation,
  }),
}));

const { ConversationMenu } = await import("@/components/conversation-menu");

beforeEach(() => {
  vi.clearAllMocks();
  typing = false;
  status = "open";
});
afterEach(cleanup);

const openMenu = async () => {
  const user = userEvent.setup();
  render(<ConversationMenu />);
  await user.click(
    screen.getByRole("button", { name: "Conversation: Kitchen rewire" }),
  );
  return user;
};

describe("ConversationMenu", () => {
  it("fills the available mobile header space but stays capped on desktop", () => {
    render(<ConversationMenu />);

    const trigger = screen.getByRole("button", {
      name: "Conversation: Kitchen rewire",
    });
    expect(trigger).toHaveClass("flex-1", "md:flex-none");
    expect(trigger).toHaveClass("md:max-w-[min(28rem,45vw)]");
  });

  it("runs the command behind each conversation action", async () => {
    const user = await openMenu();

    await user.click(screen.getByRole("menuitem", { name: "Clear" }));
    expect(say).toHaveBeenCalledWith("/clear");
  });

  it("compacts through the same path", async () => {
    const user = await openMenu();

    await user.click(screen.getByRole("menuitem", { name: "Compact" }));
    expect(say).toHaveBeenCalledWith("/compact");
  });

  it("does not submit a command onto a running turn", async () => {
    // Both are slash commands rather than direct API calls, and the state
    // machine is already occupied. Renaming and deleting are unaffected —
    // those are Requests, not commands.
    typing = true;
    await openMenu();

    // Asserted on the attribute as well as on the effect: an item that quietly
    // stopped rendering would also call nothing, and would pass a test that
    // only checked `say`.
    for (const name of ["Clear", "Compact"]) {
      expect(screen.getByRole("menuitem", { name })).toHaveAttribute(
        "aria-disabled",
        "true",
      );
    }
    expect(say).not.toHaveBeenCalled();
  });

  it("asks nobody before deleting, because the kernel will", async () => {
    // `conv.delete` raises the kernel's own approval dialog and this waits on
    // it, so a confirmation here would be a second one. Same for clear and
    // compact above, which is why neither has a dialog either.
    const user = await openMenu();

    await user.click(screen.getByRole("menuitem", { name: "Delete" }));
    expect(deleteConversation).toHaveBeenCalledWith(7);
  });

  it("does not start an approval-backed delete while disconnected", async () => {
    status = "reconnecting";
    await openMenu();

    expect(screen.getByRole("menuitem", { name: "Delete" })).toHaveAttribute(
      "aria-disabled",
      "true",
    );
    expect(deleteConversation).not.toHaveBeenCalled();
  });
});
