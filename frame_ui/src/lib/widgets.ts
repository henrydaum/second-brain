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

/** Every widget installed, in the kernel's own precedence order. */
export function listWidgets(): Promise<Widget[]> {
  return sdk<Widget[]>("plugin.list", { source: "widgets" });
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
