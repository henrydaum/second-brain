import { RequestFailed, sdk } from "@/lib/client";

const CHANNEL = "second-brain-html-v1";

/** Install before author scripts, without putting credentials in the document. */
export function appDocument(html: string, token: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  const script = doc.createElement("script");
  script.textContent = `(() => {
    const channel = ${JSON.stringify(CHANNEL)};
    const token = ${JSON.stringify(token)};
    const pending = new Map();
    let next = 0;
    window.addEventListener("message", event => {
      const m = event.data;
      if (event.source !== parent || !m || m.channel !== channel ||
          m.token !== token || m.kind !== "result") return;
      const job = pending.get(m.id);
      if (!job) return;
      pending.delete(m.id);
      if (m.error) job.reject(Object.assign(new Error(m.error.message), m.error));
      else job.resolve(m.data);
    });
    window.brain = Object.freeze({ call(type, args = {}) {
      return new Promise((resolve, reject) => {
        const id = ++next;
        pending.set(id, { resolve, reject });
        try { parent.postMessage({ channel, token, kind: "call", id, type, args }, "*"); }
        catch (error) { pending.delete(id); reject(error); }
      });
    }});
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
