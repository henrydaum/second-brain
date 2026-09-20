/**
 * @vitest-environment jsdom
 *
 * One rule, and it is invisible when it breaks: **the document's life is
 * keyed on the conversation as well as the file.**
 *
 * A widget's saved state is delivered exactly once, at mount — the relay
 * resolves its `state` promise on the first announcement and never again. So
 * the only way to hand a document a different conversation's saved game is to
 * build a different document. Keyed on the path alone, switching between two
 * conversations that both show 2048 keeps the first board and claims it is the
 * second: no error, no warning, just somebody else's game.
 *
 * The companion rule is the one the sibling file pins from the other side —
 * a widget must *not* be rebuilt for anything else, so the second test holds
 * the conversation still and asserts the document survives.
 */

import "@testing-library/jest-dom/vitest";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useEffect, useImperativeHandle, type Ref } from "react";
import { afterEach, expect, it, vi } from "vitest";

import { WidgetPanel } from "@/components/widget-panel";
import type { Binding, Widget } from "@/lib/widgets";
import { getWidget } from "@/lib/widgets";
import type { WidgetFrameHandle } from "@/components/widget-frame";

const actions = vi.hoisted(() => ({
  flush: vi.fn<() => Promise<void>>().mockResolvedValue(undefined),
  choose: vi.fn<() => Promise<void>>().mockResolvedValue(undefined),
}));

vi.mock("@/lib/widgets", async importOriginal => ({
  ...await importOriginal<typeof import("@/lib/widgets")>(),
  getWidget: vi.fn(),
}));

const WIDGET: Widget = {
  name: "2048", stem: "widget_2048", tree: "bundled",
  path: "/bundled/widgets/widget_2048.html", extension: ".html",
};

/** What `useWidget` answers, swapped between renders by the tests. */
const current: { binding: Binding | null; onSourceChange?: (changed: boolean) => void } = { binding: null };

vi.mock("@/runtime/domains", () => ({
  useWidget: () => ({ widgetBinding: current.binding, chooseWidget: actions.choose }),
}));

vi.mock("@/components/assistant-ui/tooltip-icon-button", () => ({
  TooltipIconButton: ({ tooltip, children, ...rest }: {
    tooltip: string; children: React.ReactNode;
  }) => <button aria-label={tooltip} {...rest}>{children}</button>,
}));

// The picker is the thing that fetches the catalogue; here it only has to make
// the panel believe 2048 exists so a frame is rendered at all.
vi.mock("@/components/widget-picker", () => ({
  WidgetPicker: ({ onRefresh, onChoose, disabled }: {
    onRefresh: (w: Widget[]) => void;
    onChoose: (name: string | null) => void;
    disabled?: boolean;
  }) => {
    useEffect(() => { onRefresh([WIDGET]); }, [onRefresh]);
    return <div data-testid="picker">
      <button disabled={disabled} onClick={() => onChoose("clock")}>Choose clock</button>
      <button disabled={disabled} onClick={() => onChoose(null)}>Empty</button>
    </div>;
  },
}));

/** Every state a frame was built with, in order. A new entry is a new
 *  document — which is exactly what "the widget reloaded" means. */
const mounts: (string | null)[] = [];

vi.mock("@/components/widget-frame", () => ({
  WidgetFrame: ({ state, onSourceChange, ref }: {
    state: string | null;
    onSourceChange: (changed: boolean) => void;
    ref?: Ref<WidgetFrameHandle>;
  }) => {
    useImperativeHandle(ref, () => ({ flush: actions.flush }), []);
    useEffect(() => { mounts.push(state); }, []);
    useEffect(() => {
      current.onSourceChange = onSourceChange;
      onSourceChange(false);
    }, [onSourceChange]);
    return <div data-testid="frame" data-state={state ?? ""} />;
  },
}));

afterEach(() => {
  cleanup();
  localStorage.clear();
  mounts.length = 0;
  current.binding = null;
  current.onSourceChange = undefined;
  vi.mocked(getWidget).mockReset();
  actions.flush.mockReset().mockResolvedValue(undefined);
  actions.choose.mockReset().mockResolvedValue(undefined);
});

function bind(conversationId: number | null, state: string | null): Binding {
  return { name: "2048", path: WIDGET.path, tree: "bundled",
           installed: true, state, conversationId };
}

const Panel = () => (
  <WidgetPanel open mode="side" onClose={() => {}} onToggleFull={() => {}} />
);

it("rebuilds the document when the same widget belongs to another conversation", async () => {
  current.binding = bind(1, '{"score": 12}');
  const view = render(<Panel />);
  await waitFor(() => expect(mounts).toEqual(['{"score": 12}']));

  // The switch: same widget, same file, a different conversation's saved game.
  current.binding = bind(2, '{"score": 99}');
  view.rerender(<Panel />);

  await waitFor(() => expect(mounts).toEqual(['{"score": 12}', '{"score": 99}']));
});

it("does not rebuild it while the conversation stays put", async () => {
  current.binding = bind(1, '{"score": 12}');
  const view = render(<Panel />);
  await waitFor(() => expect(mounts).toHaveLength(1));

  // A re-render for any other reason — a theme change, a resize, the panel
  // opening — must not cost the person their game.
  view.rerender(<Panel />);
  view.rerender(<Panel />);

  expect(mounts).toHaveLength(1);
});

it("treats a widget picked before the first message as its own document", async () => {
  // Pending: a real document with real state, bound to no conversation yet.
  current.binding = bind(null, null);
  const view = render(<Panel />);
  await waitFor(() => expect(mounts).toEqual([null]));

  // The first message creates the conversation and adopts the pick. That is a
  // rebuild, and it is survivable by design: the state it comes back with is
  // the state it had just saved.
  current.binding = bind(7, '{"score": 12}');
  view.rerender(<Panel />);

  await waitFor(() => expect(mounts).toEqual([null, '{"score": 12}']));
});

it("refreshes the widget with its latest saved state", async () => {
  current.binding = bind(1, '{"score": 12}');
  vi.mocked(getWidget).mockResolvedValue(bind(1, '{"score": 99}'));
  render(<Panel />);
  await waitFor(() => expect(mounts).toHaveLength(1));
  expect(screen.queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();
  act(() => current.onSourceChange?.(true));
  await userEvent.click(screen.getByRole("button", { name: "Refresh" }));
  await waitFor(() => expect(mounts).toEqual(['{"score": 12}', '{"score": 99}']));
  expect(screen.queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();
});

it("keeps the running widget when reading saved state fails", async () => {
  current.binding = bind(1, '{"score": 12}');
  vi.mocked(getWidget).mockRejectedValue(new Error("Connection unavailable"));
  render(<Panel />);
  await waitFor(() => expect(mounts).toHaveLength(1));
  act(() => current.onSourceChange?.(true));
  await userEvent.click(screen.getByRole("button", { name: "Refresh" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Connection unavailable");
  expect(mounts).toHaveLength(1);
});

it("hides refresh when no widget is selected", () => {
  render(<Panel />);
  expect(screen.queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();
});

it("waits for saves before fetching refresh state and remounting", async () => {
  let finish!: () => void;
  actions.flush.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  current.binding = bind(1, "1");
  vi.mocked(getWidget).mockResolvedValue(bind(1, "2"));
  render(<Panel />);
  await waitFor(() => expect(mounts).toHaveLength(1));
  act(() => current.onSourceChange?.(true));
  await userEvent.click(screen.getByRole("button", { name: "Refresh" }));
  expect(getWidget).not.toHaveBeenCalled();
  expect(mounts).toEqual(["1"]);
  expect(screen.getByRole("button", { name: "Refresh" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Choose clock" })).toBeDisabled();
  await act(async () => finish());
  await waitFor(() => expect(mounts).toEqual(["1", "2"]));
});

it.each(["Choose clock", "Empty"])("waits for saves before %s", async (label) => {
  let finish!: () => void;
  actions.flush.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  current.binding = bind(1, "1");
  render(<Panel />);
  await waitFor(() => expect(mounts).toHaveLength(1));
  await userEvent.click(screen.getByRole("button", { name: label }));
  expect(actions.choose).not.toHaveBeenCalled();
  await act(async () => finish());
  expect(actions.choose).toHaveBeenCalledWith(label === "Empty" ? null : "clock");
});

it.each(["Refresh", "Choose clock", "Empty"])("keeps the widget after a failed save before %s", async (label) => {
  actions.flush.mockRejectedValue(new Error("Could not save"));
  current.binding = bind(1, "1");
  render(<Panel />);
  await waitFor(() => expect(mounts).toHaveLength(1));
  act(() => current.onSourceChange?.(true));
  await userEvent.click(screen.getByRole("button", { name: label }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not save");
  expect(mounts).toEqual(["1"]);
  expect(actions.choose).not.toHaveBeenCalled();
  expect(getWidget).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Choose clock" })).toBeEnabled();
});
