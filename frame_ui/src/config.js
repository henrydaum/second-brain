/**
 * What the person arranged: which layout, what is in each slot, how wide the
 * drawers are and whether they are open.
 *
 * **It is in `localStorage`, and that is a decision rather than the easy
 * option.** The obvious alternative is a kernel setting — `config.write`
 * exists, the kernel has user-scoped settings, and an arrangement that
 * followed you to another browser is plainly nicer. Two things say not yet.
 * A `config.write` from this page is an UNSAFE Request unless the writer owns
 * the setting, so dragging a divider would raise an approval dialog; and half
 * of what is here (a drawer being open, a divider's position) is a fact about
 * *this window*, which is exactly what per-browser storage is for. When the
 * HTTP frontend declares the settings — the same arrangement the timekeeper's
 * `scheduled_jobs` has — the assignments half can move and this file becomes
 * the local cache in front of it.
 *
 * Writes are **serialized through one path** and the store is synchronous, so
 * there is no way for a slower earlier save to land on top of a newer one.
 * That is worth stating because it stops being free the moment any of this
 * moves to `config.write`: an async save needs its own ordering, and a failure
 * needs to be visible rather than swallowed the way a local one can be.
 *
 * Reads are total: a missing, corrupt or half-written value answers with the
 * default rather than throwing, because a stored preference is never worth a
 * blank page. Storage can also be absent entirely (private windows, blocked
 * site data), so every access is guarded.
 */

import { DEFAULT_LAYOUT, layoutById } from "./layouts.js";
import { DEFAULT_THEME, THEMES } from "./theme.js";

const KEY = "sb.frame.v1";

const DEFAULTS = {
  layout: DEFAULT_LAYOUT,
  /**
   * Light, dark, or whatever the machine is set to.
   *
   * **Three states, not two**, for the reason the old UI gives: "system" is a
   * genuine setting rather than a synonym for light — one tracks the machine
   * as it changes through the day and the other does not, and collapsing them
   * into a toggle loses that silently.
   *
   * It lives here beside the layout rather than under its own key because it
   * is the same kind of fact: something this person arranged about this window.
   * The pre-paint script in `index.html` reads it out of the same blob, which
   * is why the shape of this file is now load-bearing for something outside it.
   */
  theme: DEFAULT_THEME,
  /** `{ "<slot>": "<widget name>" }`, keyed by slot name **across all four
   *  layouts** rather than per layout. That is what lets a layout change be a
   *  rearrangement rather than a reload: the frame keeps one element per slot
   *  name for the life of the page, so `main` holding the same widget in the
   *  single and trifold layouts means switching between them moves boxes
   *  around a running widget instead of tearing it down and building it
   *  again. */
  slots: {},
  /** Drawer extents, in pixels, and the split as a percentage of the window. */
  sizes: { left: 280, right: 320, deck: 220, split: 50 },
  /** Which drawers are open. Absent means open. */
  closed: { left: false, right: false },
  /**
   * The labelled buttons on a **side panel's** edges:
   * `{ "<slot>": { header: [entry], footer: [entry] } }`, an entry being
   * `{ id, widget }`.
   *
   * `widget` is a widget's name, or `WILDCARD` for a button that asks which
   * widget when it is pressed rather than when it is placed. There are none by
   * default and there is no default anywhere — a bar with no buttons takes no
   * room at all, which is what keeps the sleek arrangement the arrangement you
   * get unless you asked for something else.
   *
   * Keyed by slot name for the same reason `slots` is: a bar belongs to the
   * box, so it follows the box across a layout change. Only a slot the current
   * layout makes a *drawer* draws them, which is `frame.js`'s call rather than
   * this file's — the storage is per slot and the eligibility is per layout.
   */
  buttons: {},
  /**
   * The icon rail, bottom left of the frame: `[{ id, widget }]`.
   *
   * **The rail belongs to the frame, the bars belong to a panel**, and that is
   * the whole difference between them. A side panel has edges to spare and room
   * for a word; the single and split layouts have neither, and the thing they
   * need is a couple of ways to summon a tool without giving up any of the
   * window to say so. So this one is a corner rather than a row, it is icons
   * rather than labels, and it is there in every layout because the frame is.
   */
  rail: [],
};

/** A button that picks its widget when pressed. Stored in the same field as a
 *  widget name, because from everywhere else's point of view it is one — the
 *  only code that knows the difference is what happens on the press. */
export const WILDCARD = "*";

/** Per bar, and for the rail. Three fits without the bar becoming the slot;
 *  more is a menu, and a menu is a widget's job rather than the frame's. */
export const MAX_BUTTONS = 3;

/** Bounds, so a stored number from an older build or a narrower window can
 *  never leave a panel unreachable. */
const LIMITS = {
  left: [180, 560],
  right: [180, 560],
  // The deck's floor is **0 here and measured at the drag**: dragging it right
  // down leaves just its header bar showing, and how tall that is depends on
  // whether there are buttons in it and how they wrapped. A number written down
  // in this table would be a guess about a bar that has not been laid out yet.
  // `frame.js::deckFloor` is the real bound.
  deck: [0, 640],
  split: [15, 85],
};

export function clampSize(key, value) {
  const [min, max] = LIMITS[key] || [0, Infinity];
  const number = Number(value);
  if (!Number.isFinite(number)) return DEFAULTS.sizes[key];
  return Math.min(max, Math.max(min, number));
}

function read() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const stored = JSON.parse(raw);
    return stored && typeof stored === "object" ? stored : null;
  } catch {
    return null;
  }
}

export function loadConfig() {
  const stored = read() || {};
  const sizes = { ...DEFAULTS.sizes, ...(stored.sizes || {}) };
  for (const key of Object.keys(sizes)) sizes[key] = clampSize(key, sizes[key]);
  return {
    // `layoutById` falls back, so a layout id this build no longer has — an
    // older arrangement, a hand-edited value — opens the single slot instead
    // of an empty window.
    layout: layoutById(stored.layout || DEFAULTS.layout).id,
    theme: THEMES.includes(stored.theme) ? stored.theme : DEFAULTS.theme,
    slots: { ...(stored.slots || {}) },
    sizes,
    closed: { ...DEFAULTS.closed, ...(stored.closed || {}) },
    buttons: readButtons(stored.buttons),
    rail: readEntries(stored.rail),
  };
}

/**
 * Store the arrangement. Answers whether it landed.
 *
 * The return value is the point: a preference that cannot be stored is still
 * honoured for this session, so nothing breaks — and that is exactly why a
 * silent failure is the wrong shape. The person finds out at the next page
 * load, having arranged four slots twice. The frame says so at the time.
 *
 * Storage is synchronous and there is one writer, so writes cannot interleave
 * and a slower earlier save cannot land on top of a newer one. That stops being
 * free the moment any of this moves to `config.write`: an async save needs its
 * own ordering, which is a reason to move the assignments and leave the
 * window-local half here.
 */
/**
 * The button bars, taken back off disk defensively.
 *
 * Stored structure is the one thing here with real shape to it — two named
 * bars of entries, each with an id — so it is the one thing worth rebuilding
 * rather than spreading. A hand-edited or half-written value costs the bar,
 * never the page.
 */
function readButtons(stored) {
  const buttons = {};
  for (const [slot, bars] of Object.entries(stored || {})) {
    if (!bars || typeof bars !== "object") continue;
    const kept = {};
    for (const place of ["header", "footer"]) kept[place] = readEntries(bars[place]);
    if (kept.header.length || kept.footer.length) buttons[slot] = kept;
  }
  return buttons;
}

/** One list of buttons, however it was stored. */
function readEntries(stored) {
  return (Array.isArray(stored) ? stored : [])
    .filter((entry) => entry && typeof entry.widget === "string")
    .slice(0, MAX_BUTTONS)
    .map((entry) => ({ id: String(entry.id || newId()), widget: entry.widget }));
}

const newId = () => `b${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;

/** One bar's entries. Always an array, so a caller never branches on absence. */
export function barButtons(config, slot, place) {
  return config.buttons[slot]?.[place] || [];
}

export function addButton(config, slot, place, widget = WILDCARD) {
  const bars = config.buttons[slot] || (config.buttons[slot] = { header: [], footer: [] });
  const entries = bars[place] || (bars[place] = []);
  if (entries.length >= MAX_BUTTONS) return true;
  entries.push({ id: newId(), widget });
  return saveConfig(config);
}

export function setButtonWidget(config, slot, place, id, widget) {
  const entry = barButtons(config, slot, place).find((row) => row.id === id);
  if (entry) entry.widget = widget;
  return saveConfig(config);
}

/* The rail's three, which are the same operations against one flat list. It is
 * a separate list rather than a bar under a reserved slot name, because a
 * pseudo-slot is a lie that every reader of `buttons` then has to know about. */

export function addRailButton(config, widget = WILDCARD) {
  if (config.rail.length >= MAX_BUTTONS) return true;
  config.rail.push({ id: newId(), widget });
  return saveConfig(config);
}

export function setRailWidget(config, id, widget) {
  const entry = config.rail.find((row) => row.id === id);
  if (entry) entry.widget = widget;
  return saveConfig(config);
}

export function removeRailButton(config, id) {
  config.rail = config.rail.filter((row) => row.id !== id);
  return saveConfig(config);
}

export function removeButton(config, slot, place, id) {
  const bars = config.buttons[slot];
  if (!bars) return true;
  bars[place] = (bars[place] || []).filter((row) => row.id !== id);
  if (!bars.header.length && !bars.footer.length) delete config.buttons[slot];
  return saveConfig(config);
}

export function saveConfig(config) {
  try {
    localStorage.setItem(KEY, JSON.stringify(config));
    return true;
  } catch {
    return false;
  }
}

/** The widget assigned to one slot, or `null` for empty. */
export function assignment(config, slot) {
  return config.slots[slot] || null;
}

export function assign(config, slot, widgetName) {
  if (widgetName) config.slots[slot] = widgetName;
  else delete config.slots[slot];
  return saveConfig(config);
}
