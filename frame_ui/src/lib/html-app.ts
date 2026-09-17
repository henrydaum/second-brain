/**
 * The one bridge between agent-authored HTML and the kernel.
 *
 * **Two surfaces, one `brain`.** An HTML file previewed in the file viewer and
 * a widget mounted in the panel are the same kind of thing — a document
 * somebody else wrote, running in an opaque origin, needing a way to ask the
 * kernel for something. They were briefly two implementations with two channel
 * names and two error shapes, which meant an agent writing HTML met a
 * different `brain` depending on where its file happened to land. One relay,
 * and the differences between the two surfaces are arguments to it.
 *
 * What a widget adds is a stylesheet (so it inherits the app's look, which a
 * standalone preview should not) and being *told its box* — a widget is
 * resized by a panel it cannot see, where a preview is sized by the page.
 */

import { RequestFailed, sdk } from "@/lib/client";

const CHANNEL = "second-brain-html-v1";

/** What the host can tell a document about itself, unprompted. */
export type Announcement = "size" | "scheme";

/**
 * Install before author scripts, without putting credentials in the document.
 *
 * `styles` is prepended ahead of the bridge and therefore ahead of everything
 * the author wrote, which is the order that lets a widget override a token
 * deliberately and stops it doing so by accident.
 */
export function appDocument(
  html: string,
  token: string,
  options: { styles?: string } = {},
): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  if (options.styles) {
    const sheet = doc.createElement("style");
    // Named so the host can swap it when the scheme changes — replacing the
    // whole document would reload the widget and lose whatever it was holding.
    sheet.id = "sb-theme";
    sheet.textContent = options.styles;
    doc.head.prepend(sheet);
  }
  const script = doc.createElement("script");
  script.textContent = `(() => {
    const channel = ${JSON.stringify(CHANNEL)};
    const token = ${JSON.stringify(token)};
    const pending = new Map();
    const listeners = { size: [], scheme: [] };
    const state = { size: { width: 0, height: 0 }, scheme: "light" };
    let next = 0;
    window.addEventListener("message", event => {
      const m = event.data;
      if (event.source !== parent || !m || m.channel !== channel ||
          m.token !== token) return;
      if (m.kind === "tell" && listeners[m.what]) {
        state[m.what] = m.value;
        // The tokens for the new scheme arrive with it: the host swaps the
        // sheet it put in rather than rebuilding the document, because a
        // rebuild would reload the widget for a colour change.
        if (m.what === "scheme" && typeof m.css === "string") {
          const sheet = document.getElementById("sb-theme");
          if (sheet) sheet.textContent = m.css;
        }
        for (const fn of listeners[m.what].slice()) {
          try { fn(m.value); } catch (error) { console.error(error); }
        }
        return;
      }
      if (m.kind !== "result") return;
      const job = pending.get(m.id);
      if (!job) return;
      pending.delete(m.id);
      if (m.error) job.reject(Object.assign(new Error(m.error.message), m.error));
      else job.resolve(m.data);
    });
    window.brain = Object.freeze({
      call(type, args = {}) {
        return new Promise((resolve, reject) => {
          const id = ++next;
          pending.set(id, { resolve, reject });
          try { parent.postMessage({ channel, token, kind: "call", id, type, args }, "*"); }
          catch (error) { pending.delete(id); reject(error); }
        });
      },
      on(what, fn) {
        if (!listeners[what] || typeof fn !== "function") return () => {};
        listeners[what].push(fn);
        return () => {
          listeners[what] = listeners[what].filter(other => other !== fn);
        };
      },
      get size() { return state.size; },
      get scheme() { return state.scheme; },
    });
  })();`;
  doc.head.prepend(script);
  return "<!doctype html>\n" + doc.documentElement.outerHTML;
}

/** Source and per-document token bind requests to this preview, including
 * across navigation. Cleanup prevents new calls and delivery of late results;
 * it cannot undo work already accepted by the kernel. */
export function attachAppRelay(frame: HTMLIFrameElement, token: string): () => void {
  let active = true;
  const pending = new Set<number>();
  const receive = (event: MessageEvent) => {
    const m = event.data;
    if (!active || event.source !== frame.contentWindow || event.origin !== "null" ||
        !m || m.channel !== CHANNEL || m.token !== token || m.kind !== "call" ||
        !Number.isSafeInteger(m.id) || m.id < 1 || pending.has(m.id)) return;
    const source = frame.contentWindow;
    const reply = (payload: object) => {
      if (active && source === frame.contentWindow) source?.postMessage({
        channel: CHANNEL, token, kind: "result", id: m.id, ...payload,
      }, "*"); // An opaque origin cannot be named as targetOrigin.
    };
    if (typeof m.type !== "string" || !/^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$/.test(m.type) ||
        !m.args || typeof m.args !== "object" || Array.isArray(m.args)) {
      reply({ error: { message: "Expected a request type and an arguments object.", code: "invalid_request" } });
      return;
    }
    // Approval answers and frontend identity/lifecycle belong to the host UI.
    if (m.type.startsWith("frontend.")) {
      reply({ error: { message: "Frontend control requests are reserved for the host UI.", code: "reserved_request" } });
      return;
    }
    if (pending.size >= 64) {
      reply({ error: { message: "Too many pending App requests.", code: "too_many_requests" } });
      return;
    }
    pending.add(m.id);
    void sdk(m.type, m.args).then(
      data => reply({ data }),
      error => reply({ error: {
        message: error instanceof Error ? error.message : "Request failed.",
        ...(error instanceof RequestFailed ? { code: error.code, status: error.status, type: error.type } : {}),
      } }),
    ).finally(() => pending.delete(m.id));
  };
  window.addEventListener("message", receive);
  return () => { active = false; window.removeEventListener("message", receive); };
}


/**
 * Tell a document something about itself.
 *
 * **A widget does not measure its own box.** What it can see is its iframe,
 * which is right by accident today and wrong the moment the same file is
 * mounted somewhere else — so the host states the size and the scheme, and the
 * document is told rather than asked.
 *
 * The token goes out with it for the same reason it comes back on a call: a
 * document that has been replaced must not be answered, or told, on behalf of
 * its successor.
 */
export function tellApp(
  frame: HTMLIFrameElement,
  token: string,
  what: Announcement,
  value: unknown,
  extra: Record<string, unknown> = {},
): void {
  // "*" as the target, because an opaque origin has no name to address. Safe
  // here for the reason it is safe in the relay: the window being posted to is
  // one we hold a reference to, not one that asked us for something.
  frame.contentWindow?.postMessage(
    { channel: CHANNEL, token, kind: "tell", what, value, ...extra },
    "*",
  );
}
