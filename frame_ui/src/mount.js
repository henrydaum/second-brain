/**
 * Putting one widget in one box.
 *
 * A widget is an HTML document written by somebody the frame has no reason to
 * trust — the store, or the agent. So the frame does not run it; it *contains*
 * it, and then talks to it. Everything in this file is one of those two jobs.
 *
 * **The containment is `sandbox="allow-scripts"`, and the thing that matters is
 * what is missing.** Adding `allow-same-origin` beside it cancels the sandbox
 * out entirely: the widget gets this page's origin back, and with it the dev
 * server that puts a bearer token on every `/sdk` call. It would then be able
 * to make any Request Second Brain has, silently, without ever holding the
 * credential. Withheld, the document loads into an *opaque origin* — its own,
 * shared with nothing — and can reach this page only by posting a message.
 *
 * Two consequences follow from the opaque origin and neither is optional:
 *
 * - `event.origin` arrives as the string `"null"`, so it identifies nobody.
 *   A widget is recognised by `event.source === frame.contentWindow` instead,
 *   which is an object identity and cannot be spoofed by a message.
 * - A reply must be posted with `"*"` as its target, because an opaque origin
 *   has no name to address. That is safe only *because* the source check above
 *   has already decided who is being answered.
 *
 * **The bridge grants no authority of its own.** It relays Requests through the
 * same `call()` the frame uses, so the kernel classifies them exactly as it
 * would anything else and an unsafe one still raises a dialog. What it
 * withholds is the `frontend.*` family: identity, attendance and approval
 * answers belong to the frame, which holds the stream — a widget answering its
 * own approval dialog would be approving itself.
 */

import { call } from "./client.js";
import { themeCss, THEME_STYLE_ID } from "./theme.js";

const CHANNEL = "sb-widget-v1";
const MOUNT_CHANNEL = "sb-widget-mount-v1";

/** How many Requests one widget may have in flight. A runaway loop is a bug
 *  worth reporting to its author rather than a reason to flood the kernel. */
const MAX_PENDING = 64;

/** Request families a widget may never reach. */
const RESERVED = ["frontend."];

/**
 * The widget's document, prepared: our theme, then the bridge, then its own
 * markup.
 *
 * Order is the whole of it. The theme goes into `<head>` *first* so the
 * widget's own styles come after and win — a widget that wants to override a
 * token may, and one that says nothing inherits the app's look rather than the
 * browser's defaults. The bridge goes in before any of the author's scripts,
 * because a widget calling `brain` at the top of its first script would
 * otherwise find nothing there.
 *
 * `DOMParser` rather than string surgery: a widget's markup is somebody else's
 * code, and its `<head>` may be malformed or missing entirely. The parser
 * builds the document the browser would have built, so there is always a head
 * to put these in.
 */
export function widgetDocument(html, { token, scheme = "light" }) {
  const doc = new DOMParser().parseFromString(html, "text/html");

  const style = doc.createElement("style");
  // Named, because the scheme changes and this element is what has to change
  // with it. See the bridge's `scheme` branch.
  style.id = THEME_STYLE_ID;
  style.textContent = themeCss(scheme);

  const script = doc.createElement("script");
  script.textContent = bridgeSource(token);

  doc.head.prepend(script);
  doc.head.prepend(style);
  return "<!doctype html>\n" + doc.documentElement.outerHTML;
}

/**
 * What a widget author types: `brain.call`, `brain.on`, `brain.size`,
 * `brain.scheme`.
 *
 * Written as a string because it has to run inside the widget's document
 * rather than this one — nothing is shared across the boundary, so the only
 * way to put a function there is to send its source.
 *
 * The token is the other half of the source check. The frame knows which
 * window it is talking to; the *widget* has no equivalent way to tell that a
 * message came from its own frame rather than from some other page holding a
 * reference to it, so it is handed a per-mount random value and ignores
 * anything arriving without it.
 */
function bridgeSource(token) {
  const channel = JSON.stringify(CHANNEL);
  const secret = JSON.stringify(token);
  return [
    "(() => {",
    "  const channel = " + channel + ";",
    "  const token = " + secret + ";",
    "  const pending = new Map();",
    "  const listeners = new Map();",
    "  let next = 0;",
    '  const state = { size: { width: 0, height: 0 }, scheme: "light" };',
    "",
    '  window.addEventListener("message", (event) => {',
    "    const m = event.data;",
    "    if (event.source !== parent || !m || m.channel !== channel || m.token !== token) return;",
    '    if (m.kind === "result") {',
    "      const job = pending.get(m.id);",
    "      if (!job) return;",
    "      pending.delete(m.id);",
    "      if (m.error) job.reject(Object.assign(new Error(m.error.message), m.error));",
    "      else job.resolve(m.data);",
    "      return;",
    "    }",
    '    if (m.kind === "size" || m.kind === "scheme") {',
    "      state[m.kind] = m.value;",
    // The tokens are values in a stylesheet, so *telling* a widget the scheme
    // changed is not enough: something has to replace the sheet. It cannot be
    // the frame — a widget document has an opaque origin and the frame cannot
    // reach into it — so the frame sends the new text and the bridge swaps it
    // in. Swapping the text of one element rather than remounting is what lets
    // a widget change palette with its state, its scroll position and its open
    // stream all intact.
    '    if (m.kind === "scheme" && typeof m.css === "string") {',
    "      const sheet = document.getElementById(" + JSON.stringify(THEME_STYLE_ID) + ");",
    "      if (sheet) sheet.textContent = m.css;",
    "    }",
    "      for (const fn of listeners.get(m.kind) || []) {",
    "        try { fn(m.value); } catch (error) { console.error(error); }",
    "      }",
    "    }",
    "  });",
    "",
    "  window.brain = Object.freeze({",
    "    call(type, args = {}) {",
    "      return new Promise((resolve, reject) => {",
    "        const id = ++next;",
    "        pending.set(id, { resolve, reject });",
    '        try { parent.postMessage({ channel, token, kind: "call", id, type, args }, "*"); }',
    "        catch (error) { pending.delete(id); reject(error); }",
    "      });",
    "    },",
    "    on(kind, fn) {",
    "      if (!listeners.has(kind)) listeners.set(kind, new Set());",
    "      listeners.get(kind).add(fn);",
    "      return () => listeners.get(kind).delete(fn);",
    "    },",
    "    get size() { return state.size; },",
    "    get scheme() { return state.scheme; },",
    "  });",
    "})();",
  ].join("\n");
}

/**
 * Mount one widget into one element, and answer with a handle.
 *
 * `slot` is an ordinary element the frame has already sized and positioned.
 * The widget has no say in either: an iframe is a box the *parent* styles,
 * which is what makes expanding, collapsing, moving and resizing a widget the
 * frame's business rather than something every widget has to implement.
 *
 * Which is also the one real limit worth stating out loud. A widget cannot draw
 * outside its box — a menu overhanging the next slot, a dialog over the whole
 * app. It can draw anything it likes *inside* it. Surfaces that must escape are
 * the frame's own furniture, drawn at the top level, and a widget asks for one
 * rather than rendering it.
 */
export function mountWidget(slot, widget, { html, scheme = "light" }) {
  const token = crypto.randomUUID();
  const source = widgetDocument(html, { token, scheme });

  const frame = document.createElement("iframe");
  frame.title = widget.name;
  // A real URL with its own CSP. `public/widget-host.html` says why this is
  // not `srcdoc`.
  frame.src = "/widget-host.html";
  frame.sandbox = "allow-scripts";
  frame.referrerPolicy = "no-referrer";
  // Nothing has asked for hardware, and a widget that wants some should have to
  // go through the frame rather than simply take it.
  frame.allow = "camera 'none'; microphone 'none'; geolocation 'none'";
  frame.style.cssText = "display:block;width:100%;height:100%;border:0;";

  let live = true;
  let delivered = false;
  const pending = new Set();

  /** Post to the widget, if it is still the widget we mounted. */
  const send = (message) => {
    const target = frame.contentWindow;
    // "*" because an opaque origin has no name to address. Safe only because
    // nothing is ever *received* without the source check in `receive`.
    if (live && target) target.postMessage({ channel: CHANNEL, token, ...message }, "*");
  };

  const receive = (event) => {
    const m = event.data;
    if (!live || event.source !== frame.contentWindow || event.origin !== "null" ||
        !m || m.channel !== CHANNEL || m.token !== token || m.kind !== "call" ||
        !Number.isSafeInteger(m.id) || m.id < 1 || pending.has(m.id)) return;

    const fail = (message, code) =>
      send({ kind: "result", id: m.id, error: { message, code } });

    if (typeof m.type !== "string" ||
        !/^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$/.test(m.type) ||
        !m.args || typeof m.args !== "object" || Array.isArray(m.args)) {
      return fail("Expected a request type and an arguments object.", "invalid_request");
    }
    if (RESERVED.some((prefix) => m.type.startsWith(prefix))) {
      return fail("That request belongs to the frame, not to a widget.", "reserved_request");
    }
    if (pending.size >= MAX_PENDING) {
      return fail("Too many requests in flight.", "too_many_requests");
    }

    pending.add(m.id);
    call(m.type, m.args).then(
      (data) => send({ kind: "result", id: m.id, data }),
      (error) => send({ kind: "result", id: m.id, error: {
        message: error?.message || "Request failed.",
        code: error?.code || "",
        status: error?.status || 0,
      } }),
    ).finally(() => pending.delete(m.id));
  };

  // An iframe gets no resize event for its *own* element, so a widget that
  // re-lays-out when its box changes has to be told. This is the only way it
  // can find out.
  const sizes = new ResizeObserver(([entry]) => {
    const box = entry.contentRect;
    send({ kind: "size", value: {
      width: Math.round(box.width), height: Math.round(box.height),
    } });
  });

  frame.addEventListener("load", () => {
    // `document.close()` inside the host fires a second load. Deliver once:
    // after that the frame holds the widget, and a later delivery would be
    // writing over a running document.
    if (delivered) return;
    delivered = true;
    frame.contentWindow?.postMessage({ channel: MOUNT_CHANNEL, html: source }, "*");
    sizes.observe(frame);
    send({ kind: "scheme", value: scheme, css: themeCss(scheme) });
  });

  window.addEventListener("message", receive);
  slot.replaceChildren(frame);

  return {
    widget,
    /** Tell the widget the frame changed scheme, and hand it the stylesheet
     *  that goes with it — a widget cannot build one and the frame cannot reach
     *  in to install it. */
    setScheme: (next) =>
      send({ kind: "scheme", value: next, css: themeCss(next) }),
    /** Take the widget down. Stops new calls and the delivery of late results;
     *  it cannot undo work the kernel has already accepted. */
    unmount() {
      live = false;
      sizes.disconnect();
      window.removeEventListener("message", receive);
      frame.remove();
    },
  };
}
