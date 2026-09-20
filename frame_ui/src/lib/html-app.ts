/**
 * The one bridge between agent-authored HTML and the kernel.
 *
 * **One `brain`, however many surfaces.** A widget mounted in the panel is a
 * document somebody else wrote, running in an opaque origin, needing a way to
 * ask the kernel for something. The file viewer used to run agent-authored
 * HTML the same way and is the reason this is a module rather than a function
 * inside `widget-frame.tsx`: two implementations meant two channel names and
 * two error shapes, so an agent writing HTML met a different `brain` depending
 * on where its file happened to land. That surface is gone — widgets replaced
 * it — and the relay stays parameterised, because the differences between
 * surfaces belong in arguments rather than in a second copy.
 *
 * What a widget passes is a stylesheet (so it inherits the app's look) and
 * being *told its box* — a widget is resized by a panel it cannot see.
 */

import { RequestFailed, sdk } from "@/lib/client";

const CHANNEL = "second-brain-html-v1";

/** What the host can tell a document about itself, unprompted. */
export type Announcement = "size" | "scheme" | "state";

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
    const listeners = { size: [], scheme: [], state: [] };
    const state = { size: { width: 0, height: 0 }, scheme: "light", state: null };
    let next = 0;
    // How long a burst of saves is collapsed into one write. A widget saving
    // on every keystroke would otherwise be one database write and one bus
    // emit per character; half a second is short enough that a reload right
    // after a change keeps it, and long enough that typing costs one write.
    const SAVE_AFTER = 500;
    let saveTimer = 0;
    // Saved state arrives once, as an announcement, and it cannot arrive any
    // sooner: the host writes this document synchronously and its scripts run
    // after that returns, so the first message lands a tick later. A widget
    // reading state at the top of its first script would therefore always read
    // null — including the widget that had something saved. So there is a
    // promise to wait on, resolved by that first announcement and resolved
    // immediately for anything that asks afterwards.
    let markReady;
    const ready = new Promise(resolve => { markReady = resolve; });
    window.addEventListener("message", event => {
      const m = event.data;
      if (event.source !== parent || !m || m.channel !== channel ||
          m.token !== token) return;
      if (m.kind === "tell" && listeners[m.what]) {
        state[m.what] = m.value;
        // Idempotent: a second resolve is ignored, so a host that announced
        // twice would not hand anybody a second, different answer.
        if (m.what === "state") markReady(m.value);
        // The stylesheet for the new scheme arrives with it: the host swaps
        // the sheet it put in rather than rebuilding the document, because a
        // rebuild would reload the widget for a colour change. Note this
        // *replaces* the sheet, so what arrives has to be the whole of it and
        // not only the tokens that changed.
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
    const call = (type, args = {}) => new Promise((resolve, reject) => {
      const id = ++next;
      pending.set(id, { resolve, reject });
      try { parent.postMessage({ channel, token, kind: "call", id, type, args }, "*"); }
      catch (error) { pending.delete(id); reject(error); }
    });
    window.brain = Object.freeze({
      call,
      /*
        Saved state, which is the only storage a widget has.

        Browser storage is not an alternative here: this document loads into an
        opaque origin, where site storage either throws or is partitioned to
        something that comes back empty. So without this there is nowhere at
        all for a widget to keep anything across a reload, a conversation
        switch or a restart.

        It is opt-in, and it has to be — nothing can snapshot a document's
        scroll position, half-typed fields and open connections on its behalf.
        A widget that wants to come back as it was says what that means.

        Reading is local, because the host delivers the saved value at mount;
        writing is debounced and stores against this widget name in its
        conversation, capped at 64 KB. Anything larger belongs in a file of the
        widget's own, with the path kept here.
      */
      state: Object.freeze({
        // Resolves with whatever was saved, or null for a conversation that
        // has never used this widget. Wait on it once, at start-up.
        ready,
        // The same value, synchronously, for reading again later — after the
        // promise has resolved it is always current, because nothing but this
        // document writes it.
        get() { return state.state; },
        set(value) {
          state.state = value;
          clearTimeout(saveTimer);
          saveTimer = setTimeout(() => {
            call("widget.state_set", { value }).catch(error => {
              // Reported rather than thrown: a save that could not happen must
              // not take down a widget that is working.
              console.error("Could not save widget state", error);
            });
          }, SAVE_AFTER);
        },
      }),
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
export function attachAppRelay(frame: HTMLIFrameElement, token: string, widgetName?: string): () => void {
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
    // Save against the document that sent this call, even if the selection
    // has changed while its last request was in flight.
    const args = m.type === "widget.state_set" && widgetName
      ? { ...m.args, name: widgetName }
      : m.args;
    void sdk(m.type, args).then(
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
