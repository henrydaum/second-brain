/**
 * The four arrangements, as one table.
 *
 * A layout is a list of slots and the grid that holds them, and nothing else —
 * no widget, no size, no state. That separation is the reason the table can be
 * this small: everything that varies between two people running the same layout
 * (which widget is where, how wide the drawers are, whether they are open) is
 * config, and lives in `config.js`.
 *
 * Slot **names are shared across layouts on purpose**. `main` in the trifold
 * and `main` in the single-slot layout are the same word, so a layout change is
 * not automatically a rearrangement — but assignments are still stored per
 * layout (see `config.js`), because the *same* word in a narrow drawer and in a
 * full window does not necessarily want the same widget.
 *
 * The grid is written here rather than in CSS because two of the four differ
 * only in their track lists, and a stylesheet with four nearly-identical
 * blocks is four places to keep one hairline in step. `--sb-hair` is that
 * hairline and `--sb-l`/`--sb-r`/`--sb-b` are the drawer extents, all of them
 * registered in `style.css` so they can be animated.
 */

const COLS_TRIFOLD =
  "var(--sb-l) var(--sb-hair) minmax(0, 1fr) var(--sb-hair) var(--sb-r)";

export const LAYOUTS = [
  {
    id: "single",
    name: "Single",
    chrome: "main",
    hint: "One slot, the whole window.",
    slots: ["main"],
    drawers: [],
    resize: [],
    // Each entry is one child of the grid: a slot, or a divider a person can
    // see and sometimes drag.
    cells: [{ slot: "main", area: "1 / 1 / -1 / -1" }],
    columns: "minmax(0, 1fr)",
    rows: "minmax(0, 1fr)",
  },
  {
    id: "split",
    name: "Split",
    chrome: "right",
    hint: "Two slots, with a divider you can move.",
    slots: ["left", "right"],
    drawers: [],
    // The one movable line. The trifold's dividers are deliberately fixed: a
    // drawer already has two states worth having, and a drag handle on a panel
    // that slides away is two gestures competing for one pixel.
    resize: ["split"],
    cells: [
      { slot: "left", area: "1 / 1 / 2 / 2" },
      { divider: "split", axis: "x", drag: "split", area: "1 / 2 / 2 / 3" },
      { slot: "right", area: "1 / 3 / 2 / 4" },
    ],
    columns: "var(--sb-split) var(--sb-hair) minmax(0, 1fr)",
    rows: "minmax(0, 1fr)",
  },
  {
    id: "trifold",
    name: "Trifold",
    chrome: "main",
    hint: "A centre with a drawer either side.",
    slots: ["left", "main", "right"],
    drawers: ["left", "right"],
    resize: [],
    cells: [
      { slot: "left", area: "1 / 1 / 2 / 2", drawer: "left" },
      { divider: "left", axis: "x", drag: "left", area: "1 / 2 / 2 / 3" },
      { slot: "main", area: "1 / 3 / 2 / 4" },
      { divider: "right", axis: "x", drag: "right", area: "1 / 4 / 2 / 5" },
      { slot: "right", area: "1 / 5 / 2 / 6", drawer: "right" },
    ],
    columns: COLS_TRIFOLD,
    rows: "minmax(0, 1fr)",
  },
  {
    id: "deck",
    name: "Trifold + deck",
    chrome: "main",
    hint: "The trifold, over a full-width slot for output.",
    slots: ["left", "main", "right", "deck"],
    drawers: ["left", "right"],
    // The deck resizes and the side drawers still do not, which is the same
    // argument read twice: a drawer has two states worth having and a drag
    // handle competing with them is two gestures for one pixel, while the deck
    // has no button and no two states — how much of it you want is the only
    // question there is about it.
    resize: ["deck"],
    cells: [
      { slot: "left", area: "1 / 1 / 2 / 2", drawer: "left" },
      { divider: "left", axis: "x", drag: "left", area: "1 / 2 / 2 / 3" },
      { slot: "main", area: "1 / 3 / 2 / 4" },
      { divider: "right", axis: "x", drag: "right", area: "1 / 4 / 2 / 5" },
      { slot: "right", area: "1 / 5 / 2 / 6", drawer: "right" },
      { divider: "deck", axis: "y", drag: "deck", area: "2 / 1 / 3 / -1" },
      { slot: "deck", area: "3 / 1 / 4 / -1" },
    ],
    columns: COLS_TRIFOLD,
    rows: "minmax(0, 1fr) var(--sb-hair) var(--sb-deck)",
  },
];

/**
 * `chrome` names the slot the frame's own editing controls live in the corner
 * of — the Edit toggle, and the layout chooser stacked above it.
 *
 * It is the **main content area**, not the frame's bottom-right corner, and the
 * difference is the whole point. Button bars belong to side panels, so the main
 * area provably has none; parking the frame's controls there means they can
 * cover widget content and never another control. The alternative was a gutter
 * reserved in whichever bar shared the corner, which is a hole in somebody's
 * panel to make room for something that is not in it.
 *
 * It is per layout rather than derived because the answer is a fact about the
 * arrangement, and reading it back off the DOM would mean measuring.
 */
export const DEFAULT_LAYOUT = "single";

export function layoutById(id) {
  return LAYOUTS.find((layout) => layout.id === id) || LAYOUTS[0];
}

/** Human names for the slots, used only by the editing chrome. */
export const SLOT_LABELS = {
  main: "Centre",
  left: "Left",
  right: "Right",
  deck: "Deck",
};
