// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
const controls = vi.hoisted(() => ({ count: 2, more: true, loading: false, load: vi.fn(async () => {}) }));
vi.mock("@/runtime/provider", () => ({ useConversations: () => ({
  scrollbackHasMore: controls.more, loadingOlderMessages: controls.loading,
  loadOlderMessages: controls.load,
}) }));
vi.mock("@assistant-ui/react", async (original) => ({
  ...await original<typeof import("@assistant-ui/react")>(), useAuiState: () => controls.count,
}));
const { LoadOlder } = await import("./thread");
afterEach(cleanup);
it("preserves the viewport position when the last page removes the load button", () => {
  const { container, rerender } = render(<div data-slot="chat-viewport"><LoadOlder /></div>);
  const viewport = container.firstElementChild as HTMLElement;
  let height = 1000;
  Object.defineProperty(viewport, "scrollHeight", { get: () => height });
  viewport.scrollTop = 20;
  viewport.scrollTo = vi.fn();
  fireEvent.click(screen.getByRole("button", { name: "Load earlier messages" }));
  controls.loading = true;
  rerender(<div data-slot="chat-viewport"><LoadOlder /></div>);
  expect(viewport.scrollTo).not.toHaveBeenCalled();
  height = 1600;
  controls.count = 5;
  controls.more = false;
  controls.loading = false;
  rerender(<div data-slot="chat-viewport"><LoadOlder /></div>);
  expect(screen.queryByRole("button")).toBeNull();
  expect(viewport.scrollTo).toHaveBeenCalledWith({ top: 620, behavior: "instant" });
});
