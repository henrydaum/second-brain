/**
 * @vitest-environment jsdom
 *
 * What is pinned here is the part that is invisible when it breaks.
 *
 * The panel's three modes are one element wearing three class lists, and the
 * reason for that — a React reparent remounts, and remounting a widget reloads
 * the document inside it — is a rule nothing enforces. So the first test walks
 * the modes and asserts the *same node* survives, which is the only way to
 * notice if somebody later moves the fullscreen branch into its own subtree.
 *
 * The rest is the tolerance arithmetic, which is pure and easy to get subtly
 * wrong: a clamp that lets the composer be squeezed fails on a window size
 * nobody tests at.
 */

import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, expect, it, vi } from "vitest";

import { WidgetPanel, clampWidth } from "@/components/widget-panel";
import { modeFor, sharedMode } from "@/runtime/widget-mode";
import type { WidgetMode } from "@/runtime/widget-mode";

// The binding lives in the provider now — the agent and a conversation switch
// both change it, so the panel is no longer the only thing that decides what is
// in it. These tests are about the panel's geometry and mount nothing that
// could open a connection, so the one hook it reaches for is stubbed.
vi.mock("@/runtime/domains", () => ({
  useWidget: () => ({ widgetBinding: null, chooseWidget: () => {} }),
}));

vi.mock("@/components/assistant-ui/tooltip-icon-button", () => ({
  TooltipIconButton: ({ tooltip, children, ...rest }: {
    tooltip: string; children: React.ReactNode;
  }) => <button aria-label={tooltip} {...rest}>{children}</button>,
}));

// Auto-cleanup is off — the suite runs without `globals`, so every file
// tears its own render down. See `sidebar-files.dom.test.tsx`.
afterEach(() => { cleanup(); localStorage.clear(); });

const SPACER = '[data-slot="widget-spacer"]';

/** Closing without leaving fullscreen first, which the header's X no longer
 *  does but every other caller still can. */
const CloseInFull = () => {
  const [open, setOpen] = useState(true);
  return (
    <WidgetPanel open={open} mode="full" onClose={() => setOpen(false)} onToggleFull={() => {}} />
  );
};

const Harness = ({ start = "side" as WidgetMode }) => {
  const [mode, setMode] = useState<WidgetMode>(start);
  const [open, setOpen] = useState(true);
  return (
    <WidgetPanel
      open={open}
      mode={mode}
      onClose={() => setOpen(false)}
      onToggleFull={() => setMode((m) => (m === "full" ? start : "full"))}
    />
  );
};

it("keeps one element across every mode, so a widget is never remounted", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  const panel = screen.getByRole("complementary", { name: "Widget" });
  expect(panel.dataset.mode).toBe("side");

  await user.click(screen.getByLabelText("Fullscreen"));
  expect(screen.getByRole("complementary", { name: "Widget" })).toBe(panel);
  expect(panel.dataset.mode).toBe("full");

  await user.click(screen.getByLabelText("Exit fullscreen"));
  expect(screen.getByRole("complementary", { name: "Widget" })).toBe(panel);
  expect(panel.dataset.mode).toBe("side");
});

it("offers a way out in fullscreen, which is the mode that hides every other one", async () => {
  const user = userEvent.setup();
  render(<Harness start="pinned" />);
  await user.click(screen.getByLabelText("Fullscreen"));

  // The header, and both of its controls, survive the takeover.
  expect(screen.getByLabelText("Exit fullscreen")).toBeVisible();
  expect(screen.getByLabelText("Hide widget")).toBeVisible();

  // Escape steps back rather than closing: it undoes the enlargement, and the
  // panel is still there to be read.
  await user.keyboard("{Escape}");
  expect(screen.getByRole("complementary", { name: "Widget" }).dataset.mode).toBe("pinned");
});

it("closes from the header, and stays mounted so the widget keeps its state", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  const panel = screen.getByRole("complementary", { name: "Widget" });
  await user.click(screen.getByLabelText("Hide widget"));
  expect(panel.isConnected).toBe(true);
  expect(panel).toHaveAttribute("inert");
});

it("closes from fullscreen in the two steps it looks like", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await user.click(screen.getByLabelText("Fullscreen"));
  const panel = screen.getByRole("complementary", { name: "Widget" });

  // One width running from the whole window to nothing covers several times
  // the distance of an ordinary close in the same 220ms, so it reads as a snap.
  // Pressing X does what a person would do with the two buttons: leave
  // fullscreen first, at the speed leaving fullscreen always takes...
  await user.click(screen.getByLabelText("Hide widget"));
  expect(panel.dataset.mode).toBe("side");
  expect(panel.style.width).toBe("384px");

  // ...and only then put the panel away, at the speed closing always takes.
  await waitFor(() => expect(panel.style.width).toBe("0px"));
  expect(panel).toHaveAttribute("inert");
  expect(panel.isConnected).toBe(true);
});

it("hides a fullscreen panel that is closed, since no width can do it", async () => {
  const user = userEvent.setup();
  // Closing while `full` is reachable without the two-step path — a mode
  // change from elsewhere, or a phone, where leaving fullscreen is immediate.
  render(<CloseInFull />);
  const panel = screen.getByRole("complementary", { name: "Widget" });
  await user.click(screen.getByLabelText("Hide widget"));

  // `100%` is the whole window whether or not anybody wants to see it, so a
  // closed fullscreen panel left visible is an inert sheet over the app that
  // swallows its own controls and hands the click to the session bar beneath.
  await waitFor(() => expect(panel.classList.contains("hidden")).toBe(true));
  expect(panel).toHaveAttribute("inert");
});

it("holds the chat's width open while fullscreen, so nothing reflows behind the layer", async () => {
  const user = userEvent.setup();
  const { container } = render(<Harness />);
  expect(container.querySelector(SPACER)).toBeNull();

  await user.click(screen.getByLabelText("Fullscreen"));
  const spacer = container.querySelector<HTMLElement>(SPACER);
  expect(spacer?.style.width).toBe("384px");

  // Still held through the exit animation: the panel is a layer until it has
  // finished shrinking, and dropping it back into the row at the moment the
  // mode changes is what made the thread reflow for the whole 220ms.
  await user.click(screen.getByLabelText("Exit fullscreen"));
  expect(container.querySelector<HTMLElement>(SPACER)?.style.width).toBe("384px");
  await waitFor(() => expect(container.querySelector(SPACER)).toBeNull());
});

it("only offers a resize handle where there is a size to set", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  expect(screen.getByRole("separator", { name: "Resize the widget panel" })).toBeVisible();

  // Pinned is a fixed split and fullscreen is the window: neither is draggable,
  // and a handle for a size nobody can change is a control that lies. A fresh
  // render rather than a rerender, since `start` seeds the harness's state and
  // re-passing it changes nothing.
  cleanup();
  render(<Harness start="pinned" />);
  expect(screen.queryByRole("separator")).toBeNull();
  await user.click(screen.getByLabelText("Fullscreen"));
  expect(screen.queryByRole("separator")).toBeNull();
});

it("never lets the panel squeeze the composer, at any window size", () => {
  // Roomy: the panel's own ceiling is what stops it.
  expect(clampWidth(9999, 1920)).toBe(720);
  // Tight: the thread's floor is, and it wins.
  expect(clampWidth(9999, 1000)).toBe(400);
  // Tighter than both floors together: the panel's minimum is the last word,
  // because a panel too narrow to use is still better than one that cannot be
  // dragged back.
  expect(clampWidth(9999, 700)).toBe(280);
  expect(clampWidth(10, 1920)).toBe(280);
});

it("falls back to this window's shared mode for a mode from another device", () => {
  // `full` is a statement about wanting the widget to itself, so it crosses.
  expect(modeFor("full", true)).toBe("full");
  expect(modeFor("full", false)).toBe("full");
  // The shared modes do not: a phone's split means nothing on a desktop.
  expect(modeFor("pinned", true)).toBe("side");
  expect(modeFor("side", false)).toBe("pinned");
  expect(modeFor(null, true)).toBe("side");
  expect(modeFor(null, false)).toBe("pinned");
  expect(modeFor("nonsense", true)).toBe(sharedMode(true));
});
