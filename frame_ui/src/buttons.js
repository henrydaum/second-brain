/**
 * The two kinds of button, and the widget window they open.
 *
 * This is the second way a widget reaches the screen, and it is worth being
 * clear about how it differs from the first. A slot widget is *placed*: it is
 * there, it holds the room it was given, and it stays mounted across every
 * rearrangement. A button widget is *summoned*: it takes most of the window
 * while it is wanted and is gone when it is not — a settings screen, a file
 * browser, a viewer. So one is furniture and the other is a tool, and the same
 * widget file can be either.
 *
 * **There are no buttons by default and there is no default bar.** An empty bar
 * takes no room — it is not an empty strip, it is nothing — which is what keeps
 * the arrangement you get out of the box the one with no chrome in it at all.
 *
 * **A wildcard button decides its widget when it is pressed** rather than when
 * it is placed. That is the whole of the difference — everything else about it
 * is an ordinary button — and it is stored in the same field as a widget name
 * (`WILDCARD`), so nothing between here and `localStorage` has to know which
 * kind it is holding.
 *
 * ## Bars and the rail
 *
 * A **bar** is a side panel's, top or bottom, and its buttons are *labelled*.
 * They exist only where a layout has drawers — the trifold and the deck — and
 * they are one per row, because a drawer is 280px wide and two half-width
 * buttons in it are two things nobody can read.
 *
 * The **rail** is the frame's, bottom left, and its buttons are *icons*. The
 * single and split layouts have no side panel to put a bar on and no room to
 * spare for words, and a corner costs nothing: three icons stacked in the
 * bottom-left of the window. It belongs to the frame rather than to any slot —
 * a corner is not a panel's to give — and it appears in exactly the layouts the
 * bars do not, which is the complement rule stated at `iconRail`. The icon is
 * generic for now; a widget declaring the one it wants is the obvious next
 * step, and it changes this file and nothing else.
 */

import { MAX_BUTTONS, WILDCARD } from "./config.js";
import { chooseWidget } from "./editing.js";

/**
 * One bar, built from scratch each time.
 *
 * Rebuilding is free here in a way it is emphatically not for a slot: a bar
 * holds buttons, and a button holds nothing. The window a button opened is a
 * frame-level element and does not go with it.
 */
export function buttonBar(slot, place, entries, editing, handlers) {
  const bar = document.createElement("div");
  bar.className = `bar bar-${place}`;
  bar.dataset.place = place;
  // An empty bar in the DOM is still a row of the slot's grid, so it has to be
  // hidden rather than merely empty, or every slot pays a few pixels for
  // buttons nobody added. `allowed` is the layout's answer, not the bar's: the
  // same `left` slot carries buttons in the trifold and none in the split.
  bar.hidden = !handlers.allowed || (!editing && entries.length === 0);
  if (!handlers.allowed) return bar;

  for (const entry of entries) {
    bar.append(editing
      ? editableButton(entry, handlers)
      : liveButton(entry, handlers));
  }
  if (editing && entries.length < MAX_BUTTONS) bar.append(addControl(place, handlers));
  return bar;
}

/** What the person presses when the arrangement is settled. */
function liveButton(entry, { onPress }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "bar-btn";
  button.textContent = label(entry.widget);
  button.title = entry.widget === WILDCARD
    ? "Choose a widget to open"
    : `Open ${entry.widget}`;
  button.addEventListener("click", () => onPress(entry, button));
  return button;
}

/** The same button while the arrangement is being edited: press it to say what
 *  it opens, and there is an × to take it away. */
function editableButton(entry, { onPick, onRemove }) {
  const wrap = document.createElement("span");
  wrap.className = "bar-edit";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "bar-btn bar-btn-edit";
  button.textContent = label(entry.widget);
  button.title = "Choose what this button opens";
  button.addEventListener("click", () => onPick(entry, button));

  wrap.append(button, removeControl(() => onRemove(entry)));
  return wrap;
}

function addControl(place, { onAdd }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "bar-add";
  button.textContent = "+ Button";
  button.title = `Add a button to this panel's ${place}`;
  button.addEventListener("click", () => onAdd(place));
  return button;
}

function removeControl(onRemove) {
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "bar-remove";
  remove.textContent = "×";
  remove.title = "Remove this button";
  remove.setAttribute("aria-label", "Remove this button");
  remove.addEventListener("click", onRemove);
  return remove;
}

/** A wildcard says what it does; anything else says what it opens. */
export function label(widget) {
  return widget === WILDCARD ? "Widgets…" : widget;
}

/* -------------------------------------------------------------------- rail */

/**
 * The icon rail: up to three, stacked, bottom left of the frame.
 *
 * Frame furniture, like the drawer toggles and the Edit button, and it behaves
 * like them — it fades back until the pointer is near it, and it is drawn over
 * whichever slot happens to be underneath rather than belonging to it.
 *
 * **It is exactly the complement of the bars**: a layout with side panels puts
 * its buttons on their edges, a layout without one gets the rail, and no layout
 * has both. That is one rule rather than two lists, and it says the real thing
 * — these are two answers to *where do buttons go here*, and a window that
 * offered both would be asking the person to choose between them every time.
 *
 * In editing mode each icon grows an × and an empty rail grows a `+`, which is
 * the only time the rail is visible while holding nothing. Out of editing mode
 * an empty rail is not there at all.
 */
export function iconRail(entries, editing, handlers) {
  const rail = document.createElement("div");
  rail.className = "rail";
  rail.hidden = !handlers.allowed || (!editing && entries.length === 0);
  if (!handlers.allowed) return rail;

  for (const entry of entries) {
    const slot = document.createElement("span");
    slot.className = "rail-item";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "rail-btn";
    button.innerHTML = widgetIcon();
    button.title = editing
      ? `${label(entry.widget)} — choose what this button opens`
      : entry.widget === WILDCARD ? "Choose a widget to open" : `Open ${entry.widget}`;
    button.setAttribute("aria-label", button.title);
    button.addEventListener("click", () => editing
      ? handlers.onPick(entry, button)
      : handlers.onPress(entry, button));

    slot.append(button);
    if (editing) slot.append(removeControl(() => handlers.onRemove(entry)));
    rail.append(slot);
  }

  if (editing && entries.length < MAX_BUTTONS) {
    const add = document.createElement("button");
    add.type = "button";
    add.className = "rail-btn rail-add";
    add.textContent = "+";
    add.title = "Add a button to the rail";
    add.setAttribute("aria-label", add.title);
    add.addEventListener("click", () => handlers.onAdd());
    rail.append(add);
  }
  return rail;
}

/**
 * The generic icon, until a widget can ask for its own.
 *
 * One glyph for every button is deliberately a placeholder and the `title` is
 * doing the real work meanwhile: a rail of three identical icons is legible on
 * hover and by position, which is enough for three. It stops being enough at
 * the point a widget can declare something better, which is the point this
 * function takes an argument.
 */
function widgetIcon() {
  return `<svg viewBox="0 0 24 24" aria-hidden="true">
    <rect x="4" y="4" width="7" height="7" rx="1.5"/>
    <rect x="13" y="4" width="7" height="7" rx="1.5"/>
    <rect x="4" y="13" width="7" height="7" rx="1.5"/>
    <rect x="13" y="13" width="7" height="7" rx="1.5"/></svg>`;
}

/* ----------------------------------------------------------------- pressing */

/**
 * Press a button: open its widget, or ask which one first.
 *
 * The wildcard's menu and the editing picker are the same list from the same
 * call, which is deliberate — "the widgets you can put in a slot" and "the
 * widgets a wildcard can open" are one question, and two lists that could
 * disagree would disagree.
 */
export function pressButton(frame, entry, anchor, { widgets, open }) {
  if (entry.widget !== WILDCARD) return open(entry.widget, anchor);
  chooseWidget(frame, anchor, {
    widgets,
    current: null,
    empty: false,
    onPick: (name) => name && open(name, anchor),
  });
}

/* ------------------------------------------------------------------ windows */

/**
 * A summoned widget, over most of the frame.
 *
 * **Large on purpose.** What these buttons are for is settings, a file browser,
 * a viewer — things that want room and want it briefly. A small panel by the
 * button would be a worse version of the drawer that is already there, and the
 * drawer is the right answer for anything worth keeping on screen. So the
 * window takes the frame less a margin, and the margin is what says it is
 * temporary.
 *
 * It is centred rather than anchored for the same reason: at this size the
 * button it came from tells you nothing about where it will be, and a window
 * that lands somewhere different each time is a window you have to look for.
 *
 * One at a time, and the scrim is what makes that visible as well as
 * dismissible — two of these would be two things the Escape key could mean.
 * The widget is mounted on open and unmounted on close, so closing takes its
 * state with it. That is the honest reading of a window you summoned, and it is
 * also the only version that cannot leak: a hidden widget still holding a
 * stream would be a box with no way back to it.
 */
export function openWidgetWindow(frame, anchor, name, { mount, onClose }) {
  closeWindow(frame);

  const scrim = document.createElement("div");
  scrim.className = "widget-scrim";

  const win = document.createElement("div");
  win.className = "widget-window";
  win.setAttribute("role", "dialog");
  win.setAttribute("aria-modal", "true");
  win.setAttribute("aria-label", name);

  const head = document.createElement("header");
  head.className = "widget-window-head";
  const title = document.createElement("span");
  title.textContent = name;
  const shut = document.createElement("button");
  shut.type = "button";
  shut.className = "widget-window-close";
  shut.textContent = "×";
  shut.title = "Close";
  shut.setAttribute("aria-label", "Close");
  head.append(title, shut);

  const body = document.createElement("div");
  body.className = "widget-window-body";
  win.append(head, body);
  scrim.append(win);
  frame.append(scrim);

  const mounted = mount(body);

  const close = () => {
    mounted?.unmount();
    scrim.remove();
    document.removeEventListener("keydown", escape, true);
    anchor?.focus?.();
    onClose?.();
  };
  const escape = (event) => {
    if (event.key === "Escape") { event.stopPropagation(); close(); }
  };
  shut.addEventListener("click", close);
  // On the scrim itself, not on the document: a click inside the window must
  // not close it, and a widget's own clicks never reach here anyway — they are
  // in another document.
  scrim.addEventListener("pointerdown", (event) => {
    if (event.target === scrim) close();
  });
  document.addEventListener("keydown", escape, true);
  shut.focus();

  scrim.close = close;
  return close;
}

export function closeWindow(frame) {
  frame.querySelector(".widget-scrim")?.close?.();
}
