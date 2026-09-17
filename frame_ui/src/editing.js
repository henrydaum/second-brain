/**
 * Editing mode: the one place the arrangement can be changed.
 *
 * Out of editing mode the frame is only hairlines — no headers, no handles, no
 * labels, nothing between four widgets but a half-pixel rule. That is the whole
 * look, and it is only affordable because *every* control that would otherwise
 * live permanently in a corner of a slot lives in here instead. So this file
 * holds all of it: the layout chooser, the per-slot chrome, the widget picker,
 * and the button that turns the mode on.
 *
 * **The chrome is drawn by the frame, not by the slots.** A widget is a
 * separate document in a sandboxed iframe and cannot draw over its neighbours,
 * so a picker that overhangs a 280px drawer has to be a top-level element —
 * `openPicker` appends to the frame and positions against the button that
 * opened it. This is the first of the overhanging surfaces the frame owes its
 * widgets; menus and dialogs will be more of the same.
 *
 * The layout thumbnails are drawn rather than described, because four
 * arrangements are faster to recognise than to read — and they are built from
 * the same `cells` the real grid uses, so a layout cannot come to be drawn as
 * something other than what it is.
 */

import { SLOT_LABELS } from "./layouts.js";

/* ------------------------------------------------------------ the mode itself */

/** The toggle, bottom right of every layout. The only control that is visible
 *  when editing is off. */
export function editButton(editing, onToggle) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "edit-btn";
  button.title = "Edit the layout";
  button.setAttribute("aria-label", "Edit the layout");
  button.setAttribute("aria-pressed", String(editing));
  button.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M4 20h4.5L19 9.5a2.1 2.1 0 0 0-3-3L5.5 17V20z"/>
    <path d="M14.5 7.5l2 2"/></svg><span>Edit</span>`;
  button.addEventListener("click", onToggle);
  return button;
}

/* --------------------------------------------------------------- the layouts */

/** The layout chooser: four thumbnails, one of them current. */
/**
 * Light, dark, or the machine's answer.
 *
 * A three-way radio group rather than a two-state toggle, because "System" is
 * a real answer and the one most people are already on — `theme.js` and the
 * old UI's `lib/theme.ts` both say why at more length.
 *
 * It updates its own checked state rather than being redrawn, because it is
 * the only thing that ever changes the preference: the OS flipping under
 * "System" changes the *palette* and leaves the answer to this question
 * exactly where it was. A bar that redrew on a scheme change would move the
 * selection to whichever of light or dark the machine had landed on, which is
 * the one reading of it that is wrong.
 */
export function themeBar(current, onPick) {
  const bar = document.createElement("div");
  bar.className = "theme-bar";
  bar.setAttribute("role", "radiogroup");
  bar.setAttribute("aria-label", "Appearance");

  const label = document.createElement("span");
  label.className = "layout-bar-label";
  label.textContent = "Appearance";
  bar.append(label);

  const group = document.createElement("div");
  group.className = "theme-group";
  for (const option of THEME_OPTIONS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "theme-choice";
    button.dataset.theme = option.id;
    button.title = option.hint;
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(option.id === current));
    button.innerHTML = option.icon;
    const name = document.createElement("span");
    name.textContent = option.name;
    button.append(name);
    button.addEventListener("click", () => {
      for (const sibling of group.children) {
        sibling.setAttribute("aria-checked", String(sibling === button));
      }
      onPick(option.id);
    });
    group.append(button);
  }
  bar.append(group);
  return bar;
}

/** Sun, moon, monitor — the old UI's three, drawn as strokes so they inherit
 *  the button's colour the way every other glyph in the frame does. */
const THEME_OPTIONS = [
  {
    id: "system", name: "System", hint: "Follow this machine's setting",
    icon: `<svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="4" width="18" height="12" rx="2"/>
      <path d="M8 20h8M12 16v4"/></svg>`,
  },
  {
    id: "light", name: "Light", hint: "Always light",
    icon: `<svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="4"/>
      <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4"/></svg>`,
  },
  {
    id: "dark", name: "Dark", hint: "Always dark",
    icon: `<svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z"/></svg>`,
  },
];

export function layoutBar(layouts, current, onPick) {
  const bar = document.createElement("div");
  bar.className = "layout-bar";
  bar.setAttribute("role", "radiogroup");
  bar.setAttribute("aria-label", "Layout");

  const label = document.createElement("span");
  label.className = "layout-bar-label";
  label.textContent = "Layout";
  bar.append(label);

  for (const layout of layouts) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "layout-choice";
    button.title = layout.hint;
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(layout.id === current));
    button.append(thumbnail(layout));
    const name = document.createElement("span");
    name.textContent = layout.name;
    button.append(name);
    button.addEventListener("click", () => onPick(layout.id));
    bar.append(button);
  }
  return bar;
}

/**
 * A layout, drawn small.
 *
 * Built from `cells` rather than from a hand-drawn picture per layout: the
 * grid areas are already the shape, so reusing them is both shorter and the
 * only version that cannot drift from what the layout actually does. Drawer
 * tracks get a fixed fraction of the thumbnail; their real width is a pixel
 * count that means nothing at this size.
 */
function thumbnail(layout) {
  const grid = document.createElement("span");
  grid.className = "layout-thumb";
  grid.style.gridTemplateColumns = layout.columns
    .replaceAll("var(--sb-l)", "1.1fr")
    .replaceAll("var(--sb-r)", "1.1fr")
    .replaceAll("var(--sb-split)", "1.4fr")
    .replaceAll("var(--sb-hair)", "1px")
    .replaceAll("minmax(0, 1fr)", "2.4fr");
  grid.style.gridTemplateRows = layout.rows
    .replaceAll("var(--sb-deck)", "0.9fr")
    .replaceAll("var(--sb-hair)", "1px")
    .replaceAll("minmax(0, 1fr)", "2.6fr");

  for (const cell of layout.cells) {
    const box = document.createElement("span");
    box.className = cell.slot ? "layout-thumb-slot" : "layout-thumb-hair";
    box.style.gridArea = cell.area;
    grid.append(box);
  }
  return grid;
}

/* ----------------------------------------------------------------- the slots */

/**
 * One slot's widget control, always present in the DOM and shown only while
 * editing.
 *
 * It is built once with the slot element and never rebuilt, for the reason the
 * frame's own docstring gives: the slot element is not allowed to be replaced,
 * because the iframe inside it would reload. So the chrome is a sibling of the
 * widget's box that CSS hides, rather than something added when the mode turns
 * on.
 *
 * **It sits in the middle of the slot**, which is the one thing about it worth
 * explaining. It was a bar across the top, and the top is now the header bar's
 * — a slot has two edges a person can put buttons on, and a control parked on
 * one of them is in the way of the thing being configured. The centre belongs
 * to nothing else and reads as "this slot" rather than "the top of this slot".
 *
 * There is no clear button beside it. Emptying a slot is picking `Empty` from
 * the same list every other choice comes from, and a second way to do one thing
 * is a second thing to find.
 */
export function slotChrome(slot, widgetName, { onChoose }) {
  const chrome = document.createElement("div");
  chrome.className = "slot-chrome";

  const label = document.createElement("span");
  label.className = "slot-label";
  label.textContent = SLOT_LABELS[slot] || slot;

  const name = document.createElement("button");
  name.type = "button";
  name.className = "slot-name";
  name.title = "Choose a widget for this slot";
  name.textContent = widgetName || "Empty";
  name.addEventListener("click", () => onChoose(name));

  chrome.append(label, name);
  return chrome;
}

/* ---------------------------------------------------------------- the picker */

/**
 * The widget list, anchored to whatever opened it.
 *
 * **One list, three callers**: a slot choosing what it holds, a button choosing
 * what it opens, and a wildcard button choosing at the moment it is pressed.
 * They differ by one option — only a slot can be `Empty` — and by nothing else,
 * which is the point: "which widgets are there" must have one answer, or the
 * arrangement offers one set and the wildcard offers another.
 *
 * One at a time: opening a second closes the first, because two lists on screen
 * make it ambiguous which slot a choice lands in. It closes on a pick, on
 * Escape, and on a click anywhere else.
 *
 * **"Anywhere else" needs a scrim**, and that is the whole of why one is here. A
 * document listener cannot see a click that lands in an *iframe*: the event
 * belongs to that other document and never reaches this one. Most of this
 * window is iframes, so a wildcard menu opened over a widget stayed open when
 * you clicked the widget — the one click most likely to mean "no, not that". A
 * transparent element over the frame catches it in the only document that can.
 *
 * The scrim is also what makes the capture-phase listener unnecessary, though
 * both are kept: a click on another slot's button would otherwise close this
 * list and reopen it in the same tick.
 */
export function chooseWidget(frame, anchor,
    { widgets, current, empty = true, wildcard = false, onPick }) {
  closeChooser(frame);

  const scrim = document.createElement("div");
  scrim.className = "popover-scrim";

  const popover = document.createElement("div");
  popover.className = "popover";
  popover.setAttribute("role", "listbox");
  popover.setAttribute("aria-label", "Widgets");

  const close = () => {
    popover.remove();
    scrim.remove();
    document.removeEventListener("pointerdown", outside, true);
    document.removeEventListener("keydown", escape, true);
  };
  const outside = (event) => {
    if (!popover.contains(event.target) && event.target !== anchor) close();
  };
  const escape = (event) => {
    if (event.key === "Escape") { close(); anchor.focus(); }
  };

  // `Empty` and `Wildcard` are the same slot in the list and never both
  // present: one is a slot saying it holds nothing, the other a button saying
  // it decides later. Both answer `null`, which the caller reads in its own
  // terms — the list does not need to know which question it is answering.
  if (empty) {
    popover.append(option("Empty", null, current === null, () => { onPick(null); close(); }));
  } else if (wildcard) {
    popover.append(option("Wildcard", "Ask which widget when pressed",
      current === null, () => { onPick(null); close(); }));
  }
  if (!widgets.length) {
    const empty = document.createElement("p");
    empty.className = "popover-empty";
    empty.textContent = "No widgets are installed.";
    popover.append(empty);
  }
  for (const widget of widgets) {
    popover.append(option(
      // A built-in panel carries a written name; a widget is known by its file.
      widget.label || widget.name,
      // `shadowed` is a *list of paths* the kernel hid behind this one, not a
      // name — checked against the handler rather than guessed from the field.
      widget.shadowed?.length
        ? `${widget.tree} · shadows ${widget.shadowed.length}`
        : widget.tree,
      widget.name === current,
      () => { onPick(widget.name); close(); },
    ));
  }

  scrim.addEventListener("pointerdown", close);
  frame.append(scrim, popover);
  popover.close = close;
  place(popover, anchor, frame);
  document.addEventListener("pointerdown", outside, true);
  document.addEventListener("keydown", escape, true);
  popover.querySelector("button")?.focus();
  return close;
}

/** Shut whatever list is open, if any. */
export function closeChooser(frame) {
  const open = frame.querySelector(".popover");
  if (open?.close) open.close();
  else open?.remove();
  frame.querySelector(".popover-scrim")?.remove();
}

function option(name, detail, selected, onPick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "popover-option";
  button.setAttribute("role", "option");
  button.setAttribute("aria-selected", String(selected));

  const title = document.createElement("span");
  title.className = "popover-name";
  title.textContent = name;
  button.append(title);
  if (detail) {
    const sub = document.createElement("span");
    sub.className = "popover-detail";
    sub.textContent = detail;
    button.append(sub);
  }
  button.addEventListener("click", onPick);
  return button;
}

/**
 * Put the popover under its anchor and inside the window.
 *
 * Coordinates are relative to the frame, which is the positioned ancestor. The
 * clamp matters more than it looks: a picker opened from the right-hand
 * drawer's button is anchored within a few hundred pixels of the window edge,
 * so without it the list is drawn half off screen.
 */
function place(popover, anchor, frame) {
  const box = anchor.getBoundingClientRect();
  const within = frame.getBoundingClientRect();
  const width = popover.offsetWidth;
  const height = popover.offsetHeight;
  const gap = 6;

  let left = box.left - within.left;
  left = Math.min(left, within.width - width - gap);
  left = Math.max(gap, left);

  let top = box.bottom - within.top + gap;
  if (top + height > within.height - gap) {
    top = Math.max(gap, box.top - within.top - height - gap);
  }
  popover.style.left = `${Math.round(left)}px`;
  popover.style.top = `${Math.round(top)}px`;
}
