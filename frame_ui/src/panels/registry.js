/**
 * The built-in panels: chat, settings, the file drawer, and whatever else is
 * too much of the app to be a widget.
 *
 * **This table is to a panel what `plugin.list {source:"widgets"}` is to a
 * widget** — the one place the frame learns what exists. It is static because
 * panels ship with the app rather than being installed, and that is the only
 * difference the frame is allowed to care about: the picker concatenates this
 * with the widget catalog and asks one question.
 *
 * `id` carries the `sb:` prefix, and it is not decoration. A stored arrangement
 * is a slot name against a plain string, so without a namespace a widget called
 * `chat` would shadow the real one — silently, and eventually, because somebody
 * will write one. The prefix is also what makes an orphaned assignment readable
 * when a panel is renamed.
 *
 * `needs` is the declaration described in `host.js`: the authority this panel
 * has beyond what a widget gets. An empty list is the honest default and most
 * panels should keep it — a file viewer reads files, which is `fs.read` and
 * nothing more.
 *
 * `load` is a dynamic import on purpose. A panel that brings React brings it
 * only when somebody puts it in a slot, so a window holding nothing but widgets
 * pays for none of it.
 */

/** The prefix that tells a stored assignment apart from a widget's name. */
export const PANEL_PREFIX = "sb:";

export const PANELS = [
  {
    id: "sb:about",
    name: "About",
    hint: "What this frame is, and what it is holding",
    /** Nothing. It reads the kernel the way any widget would. */
    needs: [],
    load: () => import("./about.js"),
  },
];

export function isPanelId(name) {
  return typeof name === "string" && name.startsWith(PANEL_PREFIX);
}

export function panelById(id) {
  return PANELS.find((panel) => panel.id === id) || null;
}

/**
 * The panels as catalog rows, shaped like the widget rows beside them.
 *
 * The picker renders `label` over `name` and `tree` underneath, so a panel
 * reads as "About / built in" while a widget reads as "notes / workspace" —
 * one list, and the *kind* is visible without being a separate section. Making
 * them look alike here is what keeps every later render site from having to
 * ask which sort of thing it is holding.
 */
export function panelRows() {
  return PANELS.map((panel) => ({
    name: panel.id,
    label: panel.name,
    tree: "built in",
    hint: panel.hint,
  }));
}
