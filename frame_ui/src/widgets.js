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

import { call, serverUrl } from "./client.js";

/**
 * Every widget installed, in the kernel's own precedence order.
 *
 * A row is `{name, stem, tree, path, extension}`, plus `shadowed` when a
 * same-named file in a lower-precedence tree was hidden — bundled beats
 * installed beats workspace, so a draft in the workspace does not silently
 * replace what the store put there. The hidden file is reported rather than
 * dropped, because "my widget is not showing up" is otherwise unanswerable
 * from in here.
 */
export function listWidgets() {
  return call("plugin.list", { source: "widgets" });
}

/**
 * One widget's source.
 *
 * `GET /files?path=` is the kernel's byte route, and the same `fs.read` policy
 * applies — a refusal here is the kernel refusing, not the page failing. The
 * text is fetched rather than pointed at, because the frame has to *prepare*
 * the document before a browser sees it: our theme and the bridge go in first,
 * which cannot happen if the iframe loads the file itself.
 */
export async function readWidget(widget) {
  // `encodeURIComponent`, deliberately not `searchParams.set`. The latter is
  // form encoding, where a space becomes `+`; the server reads this with
  // `unquote`, which turns `%20` into a space and leaves a `+` exactly as it
  // is. So `Z:\My Code\…` arrives as `Z:\My+Code\…` and comes back 404,
  // on every machine whose paths have a space in them and no other.
  const url = serverUrl("/files");
  url.search += `&path=${encodeURIComponent(widget.path)}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`could not read ${widget.name} (${response.status})`);
  }
  return response.text();
}
