/**
 * Finding widgets, and reading one.
 *
 * Both halves are Second Brain's, because the browser has no filesystem. A
 * widget is a file in one of the kernel's trees — shipped with the app,
 * installed from the store, or written by the agent — and this page can neither
 * list a directory nor open a file. So "what widgets are there" is a Request
 * and "what is in this one" is an HTTP route, and there is no third way.
 *
 * That is also why `widgets/` is a root in the kernel's `trees.py` rather than
 * a folder this app globs: the kernel routing it is the only reason the UI can
 * ever learn what exists.
 */

import { fileUrl, sdk } from "@/lib/client";

/**
 * One widget as the kernel describes it.
 *
 * `shadowed` is a list of *paths* the kernel hid behind this one — bundled
 * beats installed beats workspace, so a draft in the workspace does not
 * silently replace what the store put there. It is reported rather than
 * dropped, because "my widget is not showing up" is otherwise unanswerable
 * from in here.
 */
export type Widget = {
  name: string;
  stem: string;
  tree: string;
  path: string;
  extension: string;
  shadowed?: string[];
};

/**
 * What a conversation is showing, as the kernel holds it.
 *
 * `name` is null when it holds none, which is the ordinary state — every
 * conversation starts there. `installed` is false when the name outlived its
 * file; the binding is kept deliberately, so a reinstall finds the
 * conversation still pointing at it, and this is what lets the panel say so
 * rather than looking merely empty.
 *
 * `state` is the widget's own saved JSON, as a string. Nothing out here parses
 * it — it is handed to the document at mount and is meaningless to anyone
 * else.
 *
 * `conversationId` is what makes this a binding rather than a name, and the
 * panel keys the document's life on it. The same widget open in two
 * conversations is two documents with two saved games, and state is delivered
 * **once**, at mount — so a frame keyed on the file alone would go on showing
 * the first board while claiming to be the second. Null while the session is
 * between conversations, which is its own distinct life: a widget picked
 * before the first message is a real document with real state, and sending
 * that message rebuilds it against the conversation it just created.
 */
export type Binding = {
  name: string | null;
  path: string;
  tree: string;
  installed: boolean;
  state: string | null;
  conversationId: number | null;
};

/** Every widget installed, in the kernel's own precedence order. */
export function listWidgets(): Promise<Widget[]> {
  return sdk<Widget[]>("widget.list", {});
}

/**
 * Which widget this conversation holds.
 *
 * **The binding lives in the kernel, not in this browser.** It was
 * `localStorage`, which made the panel a property of the window you happened
 * to open it in: the same conversation showed different things on a laptop and
 * a phone, and the agent could not know what the person was looking at. On the
 * conversation row it follows the conversation, survives a restart, and is
 * readable by the turn that wants to update it.
 */
export async function getWidget(): Promise<Binding> {
  // Mapped rather than cast: the wire says `conversation_id` and the rest of
  // this app says `conversationId`, and a cast would have left the one field
  // the panel keys on quietly undefined — which reads as "no conversation" and
  // so never rebuilds the frame.
  const wire = await sdk<Omit<Binding, "conversationId"> & {
    conversation_id?: number | null;
  }>("widget.get", {});
  return { ...wire, conversationId: wire.conversation_id ?? null };
}

/** Show a widget beside this conversation, or `null` to show none. */
export function setWidget(name: string | null): Promise<unknown> {
  return sdk("widget.set", { name });
}

/**
 * Save a widget's own state against this conversation.
 *
 * Called by the relay on the document's behalf. Capped at 64 KB kernel-side,
 * where the refusal says what to do instead.
 */
export function setWidgetState(value: unknown): Promise<unknown> {
  return sdk("widget.state_set", { value });
}

/**
 * One widget's source.
 *
 * The text is fetched rather than pointed at, because the frame has to
 * *prepare* the document before a browser sees it: the theme and the bridge go
 * in ahead of the widget's own markup, which cannot happen if the iframe loads
 * the file itself. It is also what keeps the frame in `srcdoc`, and therefore
 * in an opaque origin — pointing an iframe at `/files` would hand the widget
 * this page's origin and the proxy's bearer token with it.
 */
export async function readWidget(widget: Widget): Promise<string> {
  const response = await fetch(fileUrl(widget.path));
  if (!response.ok) {
    throw new Error(`Could not read ${widget.name} (${response.status})`);
  }
  return response.text();
}
