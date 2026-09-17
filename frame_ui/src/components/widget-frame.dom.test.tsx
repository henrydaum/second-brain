/**
 * @vitest-environment jsdom
 *
 * The boundary, and the two ways it is enforced.
 *
 * Everything here is about what a widget may reach, which is the half of this
 * feature that fails silently and badly. A widget that renders wrong is
 * obvious; a widget that quietly acquires the frame's authority is not, and
 * nothing in the running app would say so.
 *
 * The containment itself — no `allow-same-origin` — is a single attribute, and
 * it is asserted here precisely because it is one attribute: it would survive
 * any amount of refactoring and it would also disappear under an innocent-
 * looking "the widget needs to read its own cookies" change.
 */

import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

const sdk = vi.fn();

vi.mock("@/lib/client", () => ({
  sdk: (type: string, args: Record<string, unknown>) => sdk(type, args),
  fileUrl: (path: string) => `/files?path=${encodeURIComponent(path)}`,
  RequestFailed: class extends Error { code = ""; },
}));

const { WidgetFrame } = await import("@/components/widget-frame");

const WIDGET = {
  name: "hello",
  stem: "widget_hello",
  tree: "bundled",
  path: "/w/widget_hello.html",
  extension: ".html",
};

afterEach(() => { cleanup(); sdk.mockReset(); vi.unstubAllGlobals(); });

function mountFrame() {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: true,
    text: () => Promise.resolve("<main>hi</main>"),
  }));
  render(<WidgetFrame widget={WIDGET} scheme="light" />);
  return waitFor(() => {
    const frame = screen.getByTitle("hello") as HTMLIFrameElement;
    expect(frame.srcdoc).not.toBe("");
    return frame;
  });
}

it("contains the widget in an opaque origin", async () => {
  const frame = await mountFrame();
  // `allow-scripts` alone. Adding `allow-same-origin` beside it cancels the
  // sandbox out: the widget gets this page's origin, and with it the proxy
  // that puts a bearer token on every /sdk call — every Request Second Brain
  // has, without ever holding the credential.
  expect(frame.getAttribute("sandbox")).toBe("allow-scripts");

  // And the document is inlined rather than loaded from /files, which would
  // hand it that same origin by the other route.
  expect(frame.getAttribute("src")).toBeNull();
  expect(frame.srcdoc).toContain("<main>hi</main>");
  // Ours first, so a widget's own styles win and a widget that says nothing
  // still looks like the app.
  expect(frame.srcdoc.indexOf("--sb-fg")).toBeLessThan(
    frame.srcdoc.indexOf("<main>"),
  );
});

it("relays a Request, and answers it", async () => {
  const frame = await mountFrame();
  sdk.mockResolvedValue({ ok: true });
  const posted = vi.fn();
  Object.defineProperty(frame, "contentWindow", {
    value: { postMessage: posted },
    configurable: true,
  });

  postFrom(frame.contentWindow, {
    channel: "sb-widget-v1",
    token: tokenOf(frame),
    kind: "call",
    id: 1,
    type: "conv.list",
    args: { limit: 5 },
  });

  await waitFor(() => expect(sdk).toHaveBeenCalledWith("conv.list", { limit: 5 }));
  await waitFor(() => expect(posted).toHaveBeenCalledWith(
    expect.objectContaining({ kind: "result", id: 1, data: { ok: true } }),
    "*",
  ));
});

it("refuses the frame's own family, and anything it did not recognise", async () => {
  const frame = await mountFrame();
  const posted = vi.fn();
  Object.defineProperty(frame, "contentWindow", {
    value: { postMessage: posted },
    configurable: true,
  });
  const send = (data: Record<string, unknown>, source?: unknown) =>
    postFrom(source ?? frame.contentWindow, data);

  // Identity, attendance and approval answers are the frame's: a widget
  // answering its own approval dialog would be approving itself.
  send({
    channel: "sb-widget-v1", token: tokenOf(frame),
    kind: "call", id: 2, type: "frontend.resolve", args: {},
  });
  await waitFor(() => expect(posted).toHaveBeenCalledWith(
    expect.objectContaining({ id: 2, code: "not_permitted" }),
    "*",
  ));
  expect(sdk).not.toHaveBeenCalled();

  // A message from anywhere else is not a widget, whatever it says. The check
  // is object identity because an opaque origin reports itself as "null" and
  // therefore identifies nobody.
  send({
    channel: "sb-widget-v1", token: tokenOf(frame),
    kind: "call", id: 3, type: "proc.run", args: { command: "rm -rf /" },
  }, window);
  expect(sdk).not.toHaveBeenCalled();
});

/**
 * A message as an iframe sends one.
 *
 * `source` is defined on the event afterwards rather than passed to the
 * constructor: it is specified as a `WindowProxy`, and jsdom drops a plain
 * object handed to `MessageEvent` — which would make every one of these look
 * like a message from nobody, i.e. exactly the case the frame refuses.
 */
function postFrom(source: unknown, data: Record<string, unknown>) {
  const event = new MessageEvent("message", { data });
  Object.defineProperty(event, "source", { value: source });
  window.dispatchEvent(event);
}

/** The per-mount token the bridge was built with. Not a security boundary —
 *  the source check is — but it keeps a replaced widget from being answered by
 *  its successor's frame. */
function tokenOf(frame: HTMLIFrameElement): string {
  return /token:\s*"([^"]+)"/.exec(frame.srcdoc)?.[1] ?? "";
}
