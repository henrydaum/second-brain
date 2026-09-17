/**
 * What a panel is handed, and the whole of what it may touch.
 *
 * **A panel is a widget that happens to run in this page.** That is the design,
 * stated as strongly as it can be: the object below has the same shape as the
 * `brain` object a widget gets over `postMessage` — `call`, `on`, `scheme`,
 * `size` — and a panel that used only these could be moved into an iframe
 * unchanged. Nothing about a panel is allowed to be *convenient* in a way a
 * widget could not be, because the moment one is, "panel" stops meaning
 * "contained differently" and starts meaning "exempt".
 *
 * The reason to be this strict is that nothing here is enforced by the
 * machine. A widget's contract holds because an iframe *can only* post a
 * message; a panel's holds because the code says so. So the wall is elsewhere:
 * `scripts/check-panels.mjs` walks the imports under `src/panels/` and fails on
 * anything reaching into the frame. It stands in for the boundary a panel does
 * not have, and it is the reason this file is worth reading before writing a
 * panel — if you find yourself wanting `frame.js`, the answer is a wider host,
 * not an import.
 *
 * **Extra authority is declared, never assumed.** A panel that needs something
 * a widget cannot have — submitting to the conversation, answering an approval
 * — says so in the registry with `needs`, and the extra appears on the host as
 * a named member. Declaring is what makes the difference visible in one table
 * instead of buried in a component, and it keeps the default honest: a panel
 * that declares nothing is handed exactly what a widget is handed.
 */

import { call } from "../client.js";

/** The kinds a host announces, and the only ones. Identical to the widget
 *  bridge's, which is the point — see the module note. */
export const HOST_KINDS = ["size", "scheme"];

/**
 * Build the host for one panel, over one element.
 *
 * `size` is observed rather than asked for, the same way the frame observes a
 * widget's iframe: a panel is resized by drawers opening, dividers moving and
 * layouts changing, none of which it is consulted about.
 */
function createHost(element, { scheme, needs = [], extras = {} }) {
  const listeners = new Map();
  const state = { scheme, size: { width: 0, height: 0 } };

  const announce = (kind, value) => {
    state[kind] = value;
    for (const fn of listeners.get(kind) || []) {
      try { fn(value); } catch (error) { console.error(error); }
    }
  };

  const sizes = new ResizeObserver(([entry]) => {
    const box = entry.contentRect;
    announce("size", {
      width: Math.round(box.width), height: Math.round(box.height),
    });
  });
  sizes.observe(element);

  const host = {
    /** One Request to the kernel. The same route a widget's `brain.call`
     *  travels, minus the hop across the boundary. */
    call,
    on(kind, fn) {
      if (!listeners.has(kind)) listeners.set(kind, new Set());
      listeners.get(kind).add(fn);
      return () => listeners.get(kind).delete(fn);
    },
    get scheme() { return state.scheme; },
    get size() { return state.size; },
  };

  // Declared extras, and nothing else. An undeclared name is not quietly
  // withheld — it is absent, so a panel reaching for one fails at the line that
  // reached rather than three screens later.
  for (const name of needs) {
    if (extras[name]) host[name] = extras[name];
  }

  return {
    host,
    setScheme: (next) => announce("scheme", next),
    stop: () => sizes.disconnect(),
  };
}

/**
 * Mount one panel into one element, and answer with the handle the frame uses
 * for a widget.
 *
 * The handle is deliberately the same shape `mountWidget` returns, so `fill()`
 * forks once — on which kind of thing this is — and never again.
 *
 * Loading is `await`ed here rather than at start-up because the registry's
 * `load` is a dynamic import: a panel carrying React costs nothing at all until
 * somebody puts it in a slot.
 */
export async function mountPanel(element, panel, { scheme = "light", extras = {} } = {}) {
  const module = await panel.load();
  const create = module.default;
  if (typeof create !== "function") {
    throw new Error(`panel "${panel.id}" has no default export`);
  }

  const wiring = createHost(element, { scheme, needs: panel.needs, extras });
  const instance = create(wiring.host, element) || {};

  return {
    panel,
    setScheme: (next) => wiring.setScheme(next),
    unmount() {
      wiring.stop();
      try { instance.destroy?.(); } catch (error) { console.error(error); }
      element.replaceChildren();
    },
  };
}
