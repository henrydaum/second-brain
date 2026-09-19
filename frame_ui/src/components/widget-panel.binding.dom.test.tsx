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
import { cleanup, render, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";

import { WidgetPanel } from "@/components/widget-panel";
import type { Binding, Widget } from "@/lib/widgets";

const WIDGET: Widget = {
  name: "2048", stem: "widget_2048", tree: "bundled",
  path: "/bundled/widgets/widget_2048.html", extension: ".html",
};

/** What `useWidget` answers, swapped between renders by the tests. */
const current: { binding: Binding | null } = { binding: null };

vi.mock("@/runtime/domains", () => ({
  useWidget: () => ({ widgetBinding: current.binding, chooseWidget: () => {} }),
}));

vi.mock("@/components/assistant-ui/tooltip-icon-button", () => ({
  TooltipIconButton: ({ tooltip, children, ...rest }: {
    tooltip: string; children: React.ReactNode;
  }) => <button aria-label={tooltip} {...rest}>{children}</button>,
}));

// The picker is the thing that fetches the catalogue; here it only has to make
// the panel believe 2048 exists so a frame is rendered at all.
vi.mock("@/components/widget-picker", () => ({
  WidgetPicker: ({ onRefresh }: { onRefresh: (w: Widget[]) => void }) => {
    useEffect(() => { onRefresh([WIDGET]); }, [onRefresh]);
    return <div data-testid="picker" />;
  },
}));

/** Every state a frame was built with, in order. A new entry is a new
 *  document — which is exactly what "the widget reloaded" means. */
const mounts: (string | null)[] = [];

vi.mock("@/components/widget-frame", () => ({
  WidgetFrame: ({ state }: { state: string | null }) => {
    useEffect(() => { mounts.push(state); }, []);
    return <div data-testid="frame" data-state={state ?? ""} />;
  },
}));

afterEach(() => {
  cleanup();
  localStorage.clear();
  mounts.length = 0;
  current.binding = null;
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
