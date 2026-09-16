/**
 * A blank page that proves the bridge.
 *
 * This is the whole app for now, and it exists to answer one question before
 * anything is built on top: does the browser reach Second Brain, with a token
 * it accepts, on a session it will speak about. So it does exactly two things —
 * one Request out (`conv.list`, read-only) and one stream in — and prints what
 * came back. Nothing here is a UI decision worth keeping; the client in
 * `client.js` is.
 */

import "./style.css";
import { THREAD, call, openStream } from "./client.js";

const root = document.getElementById("root");
root.innerHTML = `
  <h1>Second Brain</h1>
  <dl>
    <dt>Thread</dt><dd>${THREAD}</dd>
    <dt>Request</dt><dd id="request">checking…</dd>
    <dt>Stream</dt><dd id="stream">connecting…</dd>
  </dl>
  <pre id="frames" class="empty">Renders from the kernel will appear here.</pre>
`;

const say = (id, text, state) => {
  const node = document.getElementById(id);
  node.textContent = text;
  node.className = state || "";
};

// Out: one read-only Request. A 401 here is the token, a 404 is the proxy, and
// anything else is the server having an opinion — all three worth telling apart
// on a page whose only job is to connect.
call("conv.list", { limit: 5 })
  .then((data) => {
    const rows = Array.isArray(data) ? data : (data?.conversations ?? []);
    say("request", `ok — ${rows.length} conversation(s)`, "ok");
  })
  .catch((error) => say("request", `${error.message} [${error.code}]`, "bad"));

// In: the render stream. Opening it also declares this session attended, which
// is what lets an unsafe Request raise a dialog instead of being refused.
const frames = document.getElementById("frames");
const seen = [];

openStream(
  (frame) => {
    seen.unshift(`${frame.kind}  ${JSON.stringify(frame.payload)}`);
    seen.length = Math.min(seen.length, 50);
    frames.className = "";
    frames.textContent = seen.join("\n");
  },
  (state) =>
    state === "open"
      ? say("stream", "open", "ok")
      : say("stream", "reconnecting…", "bad"),
);
