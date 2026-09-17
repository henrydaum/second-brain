// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { QueuedNotification } from "@/runtime/notifications";
import { NotificationStatus } from "./notification-status";

const state = vi.hoisted(() => ({ notificationQueue: [] as QueuedNotification[], dismissQueuedNotification: vi.fn(), notificationsOpen: false }));
vi.mock("@/runtime/provider", () => ({ useNotifications: () => state }));
const queuedNotification = (key: string, level: QueuedNotification["notification"]["level"] = "info"): QueuedNotification => ({ key, notification: { title: key, body: "Detail", source: "runtime", level, sent_at: 1 } });
beforeEach(() => {
  vi.useFakeTimers();
  state.dismissQueuedNotification.mockReset();
  state.notificationsOpen = false;
  state.notificationQueue = [queuedNotification("First")];
  vi.stubGlobal("matchMedia", () => ({ matches: false }));
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

it("shows oldest first without interrupting or restarting it when new notifications arrive", () => {
  const view = render(<NotificationStatus />);
  act(() => vi.advanceTimersByTime(4000));
  state.notificationQueue = [queuedNotification("Second"), ...state.notificationQueue];
  view.rerender(<NotificationStatus />);
  expect(screen.getByRole("status")).toHaveTextContent("First — Detail");
  expect(screen.queryByText("Second")).toBeNull();
  act(() => vi.advanceTimersByTime(2000));
  act(() => vi.runOnlyPendingTimers());
  expect(state.dismissQueuedNotification).toHaveBeenCalledExactlyOnceWith("First");
  state.notificationQueue = [queuedNotification("Second")];
  view.rerender(<NotificationStatus />);
  expect(screen.getByRole("status")).toHaveTextContent("Second — Detail");
});

it.each(["info", "success", "warning", "error"] as const)("expires %s after the fade without requiring interaction", (level) => {
  state.notificationQueue = [queuedNotification("Notice", level)];
  render(<NotificationStatus />);
  const text = screen.getByText("Notice — Detail");
  text.style.transitionProperty = "opacity";
  text.style.transitionDuration = "120ms";
  act(() => vi.advanceTimersByTime(6000));
  expect(text).toHaveAttribute("data-closing", "true");
  expect(state.dismissQueuedNotification).not.toHaveBeenCalled();
  const event = new Event("transitionend", { bubbles: true });
  Object.defineProperty(event, "propertyName", { value: "opacity" });
  fireEvent(text, event);
  expect(state.dismissQueuedNotification).toHaveBeenCalledExactlyOnceWith("Notice");
});

it("keeps the reserved line empty when idle and hides summaries while the popup is open", () => {
  state.notificationQueue = [];
  const view = render(<NotificationStatus />);
  expect(screen.getByRole("status")).toBeEmptyDOMElement();
  state.notificationQueue = [queuedNotification("Notice")];
  state.notificationsOpen = true;
  view.rerender(<NotificationStatus />);
  expect(screen.getByText("Notice — Detail").parentElement).toHaveAttribute("aria-hidden", "true");
});
