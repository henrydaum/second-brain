/**
 * @vitest-environment jsdom
 *
 * What is pinned here is what only a *widget* does. The bridge itself — the
 * source check, the reserved `frontend.*` family, the pending cap, the error
 * shape — belongs to `lib/html-app.ts` and is tested there, once, because it
 * now serves both surfaces that run agent-authored HTML.
 *
 * The first test is the important one, and it is about a deployment rather
 * than a behaviour: a `srcdoc` frame inherits the embedding page's CSP, and
 * production serves this app with a hash-only `script-src`. So a widget built
 * that way runs perfectly in development and is inert the moment it ships,
 * with nothing anywhere saying why. The fix is one attribute, which is exactly
 * the kind of thing a later refactor "simplifies" back.
 */

import "@testing-library/jest-dom/vitest";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

vi.mock("@/lib/client", () => ({
  sdk: vi.fn(),
  fileUrl: (path: string) => `/files?path=${encodeURIComponent(path)}`,
  RequestFailed: class extends Error { code = ""; },
}));

const { WidgetFrame } = await import("@/components/widget-frame");

// Deliberately not one of the widgets that ship: the file is never read — the
// fetch is stubbed — so naming a real one only means this goes stale the day
// that widget is renamed.
const WIDGET = {
  name: "example",
  stem: "widget_example",
  tree: "bundled",
  path: "/w/widget_example.html",
  extension: ".html",
};

afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers(); });

/** Mount, and stand in for the host page so what it is sent can be read.
 *
 *  `settle` holds the widget's source back, so a test can decide which of the
 *  two things the delivery needs — the host page, and the file — arrives
 *  first. */
async function mountFrame({ hold = false, state = null as string | null } = {}) {
  let release = () => {};
  const text = hold
    ? new Promise<string>((resolve) => { release = () => resolve("<main>hi</main>"); })
    : Promise.resolve("<main>hi</main>");
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, text: () => text }));
  render(<WidgetFrame widget={WIDGET} scheme="light" state={state} />);
  const frame = await waitFor(() => screen.getByTitle("example") as HTMLIFrameElement);
  const posted = vi.fn();
  Object.defineProperty(frame, "contentWindow", {
    value: { postMessage: posted },
    configurable: true,
  });
  return { frame, posted, release };
}

const mountMessages = (posted: ReturnType<typeof vi.fn>) =>
  posted.mock.calls.filter(([message]) =>
    message?.channel === "second-brain-html-mount-v1");

it("loads a document with its own policy rather than inheriting ours", async () => {
  const { frame } = await mountFrame();

  // The whole of the fix. `srcdoc` would inherit a hash-only `script-src` and
  // block every inline script in the widget, in production only.
  expect(frame.getAttribute("src")).toBe("/html-app-host.html");
  expect(frame.getAttribute("srcdoc")).toBeNull();

  // And the containment is unchanged by any of it: `allow-scripts` alone.
  // Adding `allow-same-origin` beside it hands the widget this page's origin
  // and with it the proxy that authenticates every /sdk call.
  expect(frame.getAttribute("sandbox")).toBe("allow-scripts");
});

it("delivers the prepared document once, however many times the frame loads", async () => {
  const { frame, posted } = await mountFrame();

  await waitFor(() => {
    frame.dispatchEvent(new Event("load"));
    expect(mountMessages(posted)).toHaveLength(1);
  });

  const [message] = mountMessages(posted)[0];
  // Ours ahead of theirs, which is what lets a widget override a token
  // deliberately and stops it doing so by accident.
  expect(message.html.indexOf("--sb-fg")).toBeLessThan(message.html.indexOf("<main>"));
  expect(message.html).toContain("<main>hi</main>");

  // `document.write` produces another load, and a widget may navigate its own
  // frame afterwards. Delivering again would replace whatever it had become
  // with the document it started as.
  frame.dispatchEvent(new Event("load"));
  frame.dispatchEvent(new Event("load"));
  expect(mountMessages(posted)).toHaveLength(1);
});

it("tells the widget its box, which is the one thing it cannot measure", async () => {
  const { frame, posted } = await mountFrame();
  frame.dispatchEvent(new Event("load"));

  // After the document, not before: a message posted while the host is still
  // parsing lands in a realm with no bridge in it, and is lost silently. That
  // is how a widget came to read `0 × 0` and keep it until somebody dragged
  // the panel.
  await waitFor(() => {
    const told = posted.mock.calls.filter(([m]) => m?.kind === "tell");
    expect(told.map(([m]) => m.what)).toEqual(
      expect.arrayContaining(["scheme", "size"]),
    );
  });
});


it("hands the widget its saved state, which is the only storage it has", async () => {
  // A frame in an opaque origin has no working browser storage, so this
  // announcement is the *only* route anything a widget kept last time can
  // reach it by. It goes out with the first size and scheme for the reason
  // those do — earlier and it lands in a realm with no bridge in it.
  const { frame, posted } = await mountFrame({ state: '{"tab":"today"}' });
  frame.dispatchEvent(new Event("load"));

  await waitFor(() => {
    const [told] = posted.mock.calls
      .map(([m]) => m)
      .filter((m) => m?.kind === "tell" && m.what === "state");
    // Parsed here rather than in the document: a corrupted blob should be one
    // console line, not a widget that throws before it draws.
    expect(told.value).toEqual({ tab: "today" });
  });
});

it("announces no state as null rather than not announcing", async () => {
  // A conversation that has never used this widget is the ordinary case, and
  // it still has to be *said*. The bridge resolves `brain.state.ready` on this
  // message, so a silence would leave every first-time widget awaiting a
  // promise that never settles — which presents as a widget that never draws.
  const { frame, posted } = await mountFrame();
  frame.dispatchEvent(new Event("load"));

  await waitFor(() => {
    const told = posted.mock.calls
      .map(([m]) => m)
      .filter((m) => m?.kind === "tell" && m.what === "state");
    expect(told).toHaveLength(1);
    expect(told[0].value).toBeNull();
  });
});

it("sends the whole stylesheet with a scheme, not only the tokens", async () => {
  // The bridge *replaces* `#sb-theme` with whatever a scheme tell carries, and
  // `announce()` fires on mount rather than only on a theme change. Sending
  // the token block alone therefore deleted `BASE` — and with it the body
  // `font-family` — a tick after every widget appeared, which is how they all
  // came to render in the browser's serif default while their colours stayed
  // perfectly right.
  const { frame, posted } = await mountFrame();
  frame.dispatchEvent(new Event("load"));

  await waitFor(() => {
    const [scheme] = posted.mock.calls
      .map(([m]) => m)
      .filter((m) => m?.kind === "tell" && m.what === "scheme");
    expect(scheme.css).toContain("--sb-font:");
    expect(scheme.css).toContain("font-family: var(--sb-font");
  });
});

it("delivers whichever arrives last, the host page or the file", async () => {
  // The failure this pins was invisible in a test and intermittent in a
  // browser, because it *was* intermittent: the iframe starts loading when
  // React renders it and the file is a fetch that resolves whenever it
  // resolves. Delivering only from the load handler meant a slow file left the
  // host sitting on "Loading HTML App…" forever, with nothing to fire again.
  const { frame, posted, release } = await mountFrame({ hold: true });

  // The host is up first, and there is nothing yet to give it.
  frame.dispatchEvent(new Event("load"));
  expect(mountMessages(posted)).toHaveLength(0);

  // The file lands afterwards, and that is what delivers it.
  release();
  await waitFor(() => expect(mountMessages(posted)).toHaveLength(1));
});

it("checks for source edits only when the kernel says the file moved", async () => {
  // The prompt is the watcher's; the *answer* is still a content comparison,
  // because a file can be saved without being edited. What must not happen is
  // either half alone: polling re-read the whole document every three seconds
  // to learn a boolean the kernel already knew, and trusting the announcement
  // without comparing would offer to reload a file somebody reverted.
  let html = "<main>original</main>";
  const fetchSource = vi.fn().mockImplementation(async () => ({ ok: true, text: async () => html }));
  vi.stubGlobal("fetch", fetchSource);
  const changed = vi.fn();
  let view!: ReturnType<typeof render>;
  const render_ = (sourceCheck: number, watch = true) => (
    <WidgetFrame widget={WIDGET} scheme="light" watchSource={watch}
                 sourceCheck={sourceCheck} onSourceChange={changed} />
  );
  await act(async () => { view = render(render_(0)); });
  const frame = screen.getByTitle("example");

  // Nothing is asked until something says to ask, and the document this frame
  // was built from is never compared with itself.
  expect(changed).not.toHaveBeenCalled();
  expect(fetchSource).toHaveBeenCalledTimes(1);

  html = "<main>edited</main>";
  await act(async () => { view.rerender(render_(1)); });
  expect(changed).toHaveBeenLastCalledWith(true);
  // Still the same document: an edit offers a reload, it does not perform one.
  expect(screen.getByTitle("example")).toBe(frame);

  // Reverted to what is running, announced again: nothing to refresh after all.
  html = "<main>original</main>";
  await act(async () => { view.rerender(render_(2)); });
  expect(changed).toHaveBeenLastCalledWith(false);

  // A re-render for any other reason costs nothing.
  fetchSource.mockClear();
  await act(async () => { view.rerender(render_(2)); });
  expect(fetchSource).not.toHaveBeenCalled();

  // A missing file or a dropped connection is not evidence of new source.
  fetchSource.mockRejectedValue(new Error("Unavailable"));
  changed.mockClear();
  await act(async () => { view.rerender(render_(3)); });
  expect(changed).not.toHaveBeenCalled();

  // A closed panel is not watching. The announcement it missed is answered
  // when it opens, because the token it compares against is the last one it
  // actually checked.
  fetchSource.mockImplementation(async () => ({ ok: true, text: async () => html }));
  await act(async () => { view.rerender(render_(3, false)); });
  fetchSource.mockClear();
  await act(async () => { view.rerender(render_(4, false)); });
  expect(fetchSource).not.toHaveBeenCalled();

  html = "<main>edited again</main>";
  await act(async () => { view.rerender(render_(4)); });
  expect(changed).toHaveBeenLastCalledWith(true);
});
