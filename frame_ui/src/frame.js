/**
 * The frame: slots, and what goes in them.
 *
 * Four layouts — one slot filling the window, two side by side, a trifold, and
 * the trifold over a full-width deck — and a widget in any slot of any of them.
 * What the frame owns has not changed since there was one slot, and it is worth
 * restating because it is what keeps the widget contract small:
 *
 * **The boxes.** An iframe is sized and positioned by its parent, so expanding,
 * collapsing, moving and swapping a widget all happen out here, with nothing
 * required of the widget but that it lay itself out at the size it is given. A
 * drawer sliding shut is a grid track going to zero; the widget inside is told
 * its new size and is otherwise not consulted.
 *
 * **Anything that must escape a box.** A widget cannot draw over its
 * neighbours; that is the price of the containment and it is not negotiable
 * from the inside. Menus, dialogs and tooltips that overhang belong to the
 * frame, which draws them at the top level — the editing chrome in
 * `editing.js` is the first of them.
 *
 * **A slot element is created once and never moved**, which is the single most
 * load-bearing thing in this file. Reparenting an iframe reloads the document
 * inside it: a widget holding a half-typed message, a scroll position or an
 * open stream loses all of it, silently, and the symptom is a layout switch
 * that "flickers". So the frame keeps one element per slot *name* for the life
 * of the page, and a layout change rewrites the grid, each slot's `grid-area`
 * and which slots are shown — never the tree. Everything else the person can do
 * here (open a drawer, drag the divider, enter editing mode) is likewise CSS on
 * elements that stay exactly where they were. The only thing that mounts or
 * unmounts a widget is assigning one to a slot.
 *
 * That is also why `config.slots` is keyed by slot name rather than by layout
 * and slot: `main` is one box across all four arrangements, so moving between
 * them moves boxes around a running widget.
 *
 * One more rule, unchanged: **the frame states the scheme.** A widget asking
 * the operating system would be right until somebody picks a theme in the app,
 * at which point one box disagrees with the rest.
 */

import {
  addButton, addRailButton, assign, assignment, barButtons, clampSize,
  loadConfig, removeButton, removeRailButton, saveConfig, setButtonWidget,
  setRailWidget, WILDCARD,
} from "./config.js";
import {
  buttonBar, closeWindow, iconRail, openWidgetWindow, pressButton,
} from "./buttons.js";
import {
  chooseWidget, closeChooser, editButton, layoutBar, slotChrome, themeBar,
} from "./editing.js";
import { LAYOUTS, layoutById } from "./layouts.js";
import { mountWidget } from "./mount.js";
import { mountPanel } from "./panels/host.js";
import { isPanelId, panelById, panelRows } from "./panels/registry.js";
import { applyTheme, resolveScheme } from "./theme.js";
import { listWidgets, readWidget } from "./widgets.js";

/** Every slot name any layout uses. One element each, for the life of the
 *  page — including the ones the current layout does not show, which keeps a
 *  widget alive across a switch away and back. */
const ALL_SLOTS = [...new Set(LAYOUTS.flatMap((layout) => layout.slots))];

export async function startFrame(root) {
  const config = loadConfig();
  /** The palette on screen: the preference resolved against the machine. */
  let scheme = resolveScheme(config.theme);
  applyTheme(scheme);
  /** The pooled slot elements, by name. Built once, never reparented. */
  const panels = new Map();
  /** Mounted widgets by slot name. */
  const mounts = new Map();
  /** The widget catalog, refreshed whenever the picker opens. */
  let catalog = [];
  /**
   * Everything that can go in a slot: the built-in panels, then the installed
   * widgets.
   *
   * One list, because from the person's side it is one question. Panels first
   * because they are the app's own and a stable set; widgets after, in the
   * kernel's own precedence order. Nothing downstream of here asks which kind
   * it is holding except the two places that must — `fill` and `popUp`, which
   * fork on the `sb:` prefix and nowhere else.
   */
  const offer = () => [...panelRows(), ...catalog];
  let editing = false;

  const frame = document.createElement("div");
  frame.className = "frame";
  root.replaceChildren(frame);

  for (const slot of ALL_SLOTS) {
    const panel = slotElement(slot);
    panels.set(slot, panel);
    frame.append(panel);
  }

  const banner = problemBanner();
  let problem = null;
  try {
    catalog = await listWidgets();
  } catch (error) {
    problem = explain(error);
  }

  arrange();
  for (const slot of ALL_SLOTS) fill(slot);

  // The machine changing its mind only moves the app while the preference is
  // "system". An explicit choice is not something the OS gets to overrule, and
  // leaving this unguarded would let it — at dusk, silently.
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.(
    "change", () => { if (config.theme === "system") applyScheme(); });

  return {
    frame,
    /** The live mounts, by slot name — what a render stream will need when
     *  there is one to distribute. */
    mounts,
  };

  /* --------------------------------------------------------------- arrange */

  /**
   * Put the current layout on screen.
   *
   * Slots are placed and shown or hidden; the furniture — dividers, drawer
   * buttons, the editing bar — is rebuilt, because none of it holds state worth
   * keeping. Nothing here touches a widget.
   */
  function arrange() {
    const layout = layoutById(config.layout);
    frame.dataset.layout = layout.id;
    frame.style.gridTemplateColumns = layout.columns;
    frame.style.gridTemplateRows = layout.rows;

    for (const element of [...frame.children]) {
      if (!element.classList.contains("slot")) element.remove();
    }
    // The frame's own controls live *inside* a slot now, so sweeping the
    // frame's children no longer reaches them — and a layout change would
    // otherwise leave the previous main area holding an Edit button of its own,
    // one more on every switch. Furniture is cleared by what it is, not by
    // where it happens to be parked.
    for (const old of frame.querySelectorAll(".edit-btn, .edit-stack")) old.remove();

    for (const [slot, panel] of panels) {
      const cell = layout.cells.find((entry) => entry.slot === slot);
      panel.hidden = !cell;
      if (!cell) continue;
      panel.style.gridArea = cell.area;
      if (cell.drawer) panel.dataset.drawer = cell.drawer;
      else delete panel.dataset.drawer;
      fillDrawerRow(panel, cell.drawer);
    }

    for (const cell of layout.cells) {
      if (!cell.slot) frame.append(dividerElement(layout, cell));
    }
    // Only the *show* buttons float, and only while their panel is away. A
    // drawer that is open carries its own hide button in row one, where it
    // covers nothing — see `fillDrawerRow`.
    for (const side of layout.drawers) frame.append(showButton(side));
    frame.append(rail());
    chromeCorner().append(editButton(editing, () => setEditing(!editing)));
    frame.append(banner);
    say(problem);

    applySizes();
    applyEditing();
  }

  /**
   * A slot: a header bar, the widget, a footer bar, and the editing chrome over
   * the middle of all three.
   *
   * `slot-inner` exists for the drawers and for nothing else — it is what
   * slides. See `applySizes`.
   */
  function slotElement(slot) {
    const element = document.createElement("section");
    element.className = "slot";
    element.dataset.slot = slot;

    const inner = document.createElement("div");
    inner.className = "slot-inner";
    const body = document.createElement("div");
    body.className = "slot-body";
    // Row one is the drawer's own, and it is empty for every slot that is not
    // one. See `drawerRow`.
    const drawerRow = document.createElement("div");
    drawerRow.className = "drawer-row";
    drawerRow.hidden = true;
    inner.append(drawerRow, bar(slot, "header"), body, bar(slot, "footer"));

    // The widget's box and the editing chrome are siblings: the chrome has to
    // stay clickable while the iframe under it is inert, and an overlay *in*
    // the iframe's element would be inside the box it is labelling.
    element.append(inner, slotChrome(slot, shown(assignment(config, slot)), {
      onChoose: (anchor) => choose(slot, anchor),
    }));
    return element;
  }

  /* --------------------------------------------------------------- buttons */

  /**
   * One of a slot's two button bars, rebuilt in place.
   *
   * Rebuilding is free for a bar and forbidden for a slot, which is the whole
   * difference between this and `slotElement`: a bar holds buttons and a button
   * holds nothing, while a slot holds an iframe that reloads the moment it
   * moves.
   */
  function bar(slot, place) {
    return buttonBar(slot, place, barButtons(config, slot, place), editing, {
      // **Only a side panel carries labelled buttons**, which is a question
      // about the layout rather than about the slot: `left` is a drawer in the
      // trifold and half the window in the split, and a bar of words across the
      // top of half the window is a toolbar nobody asked for. The stored
      // buttons stay put either way — a split is a place they are not drawn,
      // not a place they are deleted.
      allowed: layoutById(config.layout).drawers.includes(slot),
      onPress: (entry, anchor) => press(entry, anchor),
      onAdd: () => {
        remember(addButton(config, slot, place));
        redrawBar(slot, place);
      },
      onPick: (entry, anchor) => chooseWidget(frame, anchor, {
        widgets: offer(),
        current: entry.widget === WILDCARD ? null : entry.widget,
        empty: false,
        wildcard: true,
        onPick: (name) => {
          remember(setButtonWidget(config, slot, place, entry.id, name || WILDCARD));
          redrawBar(slot, place);
        },
      }),
      onRemove: (entry) => {
        remember(removeButton(config, slot, place, entry.id));
        redrawBar(slot, place);
      },
    });
  }

  function redrawBar(slot, place) {
    const old = panels.get(slot)?.querySelector(`.bar-${place}`);
    old?.replaceWith(bar(slot, place));
    // A bar that just gained a row can be taller than the deck it sits in, and
    // the floor is a fact about the bar. Re-checking here is what keeps
    // "dragged all the way down" meaning the same thing afterwards as it did
    // before.
    if (slot === "deck" && place === "header") {
      const floor = deckFloor();
      if (config.sizes.deck < floor) {
        config.sizes.deck = floor;
        applySizes();
      }
    }
  }

  function redrawBars() {
    for (const slot of ALL_SLOTS) {
      redrawBar(slot, "header");
      redrawBar(slot, "footer");
    }
  }

  /**
   * The icon rail, bottom left. The frame's own, so it is rebuilt with the rest
   * of the furniture and belongs to no slot — but drawn only in the layouts
   * that have no side panel to carry a bar instead.
   */
  function rail() {
    return iconRail(config.rail, editing, {
      // Only where there are no side panels to put a bar on — see `iconRail`.
      allowed: layoutById(config.layout).drawers.length === 0,
      onPress: (entry, anchor) => press(entry, anchor),
      onAdd: () => {
        remember(addRailButton(config));
        redrawRail();
      },
      onPick: (entry, anchor) => chooseWidget(frame, anchor, {
        widgets: offer(),
        current: entry.widget === WILDCARD ? null : entry.widget,
        empty: false,
        wildcard: true,
        onPick: (name) => {
          remember(setRailWidget(config, entry.id, name || WILDCARD));
          redrawRail();
        },
      }),
      onRemove: (entry) => {
        remember(removeRailButton(config, entry.id));
        redrawRail();
      },
    });
  }

  function redrawRail() {
    frame.querySelector(".rail")?.replaceWith(rail());
  }

  /** Pressing any button, from a bar or from the rail. One path, because a
   *  button is a button — only where it was drawn differs. */
  function press(entry, anchor) {
    pressButton(frame, entry, anchor, {
      widgets: offer(),
      open: (name, from) => popUp(name, from),
    });
  }

  /** Summon a widget over the frame. The other way a widget reaches the screen;
   *  `buttons.js` says how it differs from being placed in a slot. */
  function popUp(name, anchor) {
    const panel = isPanelId(name) ? panelById(name) : null;
    const widget = panel ? null : catalog.find((row) => row.name === name);
    if (!panel && !widget) return say(`Nothing named "${name}" is installed.`);
    openWidgetWindow(frame, anchor, panel ? panel.name : name, {
      mount: (body) => {
        // Mounting is asynchronous and the window is already on screen, so the
        // handle arrives after `openWidgetWindow` has returned. It is parked on
        // the element the window will unmount from.
        const holder = { unmount: () => holder.mounted?.unmount() };
        const arriving = panel
          ? mountPanel(body, panel, { scheme })
          : readWidget(widget)
              .then((html) => mountWidget(body, widget, { html, scheme }));
        arriving.then(
          (mounted) => { holder.mounted = mounted; },
          (error) => note(body, explain(error)),
        );
        return holder;
      },
    });
  }

  function dividerElement(layout, cell) {
    const element = document.createElement("div");
    element.className = `hair hair-${cell.axis}`;
    element.style.gridArea = cell.area;
    if (layout.resize.includes(cell.drag)) {
      element.classList.add("hair-drag");
      element.dataset.drag = cell.drag;
      element.setAttribute("role", "separator");
      element.setAttribute("aria-label", "Resize panels");
      element.addEventListener("pointerdown", (event) => startDrag(event, cell));
    }
    return element;
  }

  /**
   * Row one of a drawer: its own hide button, against the inner edge.
   *
   * **This is where the toggle lives while there is a panel to live in**, and
   * it is a real row rather than something floating over one — so it covers
   * nothing, and the button bar below it needs no gutter cut out of it for
   * something that is not in it.
   *
   * Against the *inner* edge (the left drawer's button on its right, the
   * right drawer's on its left) because that is the edge the panel collapses
   * towards: the control sits where the panel is about to go.
   */
  function fillDrawerRow(panel, side) {
    const row = panel.querySelector(".drawer-row");
    row.hidden = !side;
    row.replaceChildren();
    if (!side) return;
    row.dataset.side = side;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "drawer-hide";
    button.innerHTML = collapseIcon(side);
    button.title = `Hide the ${side} panel`;
    button.setAttribute("aria-label", button.title);
    button.setAttribute("aria-expanded", "true");
    button.addEventListener("click", () => toggleDrawer(side));
    row.append(button);
  }

  /**
   * The floating half: what a hidden panel leaves behind.
   *
   * It is at the frame's own edge, so the drawer slides *over* it on the way
   * in — which is the whole gesture, and why it needs no separate dismissal.
   * It is hidden while the panel is open rather than merely covered, because a
   * control underneath another control is still reachable by keyboard.
   */
  function showButton(side) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `drawer-btn drawer-${side}`;
    button.innerHTML = drawerIcon(side);
    button.title = `Show the ${side} panel`;
    button.setAttribute("aria-label", button.title);
    button.setAttribute("aria-expanded", "false");
    button.hidden = !config.closed[side];
    button.addEventListener("click", () => toggleDrawer(side));
    return button;
  }

  /** Where the frame's own controls go: the main area's bottom-right, which is
   *  the one corner no button bar can reach. See `chrome` in layouts.js. */
  function chromeCorner() {
    return panels.get(layoutById(config.layout).chrome) || frame;
  }

  /* ----------------------------------------------------------------- slots */

  /** Put the assigned widget in one slot, or leave it blank. The only place a
   *  widget is mounted or taken down. */
  async function fill(slot) {
    const panel = panels.get(slot);
    if (!panel) return;
    const body = panel.querySelector(".slot-body");
    const name = assignment(config, slot);

    mounts.get(slot)?.unmount();
    mounts.delete(slot);
    body.replaceChildren();
    panel.dataset.filled = name ? "yes" : "no";
    if (!name) return;

    // The one fork. A panel is constructed here in the page; a widget is
    // prepared and written into an iframe. Both answer with the same handle, so
    // nothing past this point knows the difference.
    try {
      if (isPanelId(name)) {
        const panel = panelById(name);
        if (!panel) return note(body, `No built-in panel named "${name}".`);
        mounts.set(slot, await mountPanel(body, panel, { scheme }));
        return;
      }
      const widget = catalog.find((row) => row.name === name);
      if (!widget) {
        // An assignment outliving its widget is ordinary — the store
        // uninstalled it, or the agent renamed a file. Saying so beats a blank
        // box, which is indistinguishable from a slot nobody has filled in.
        return note(body, `No widget named "${name}" is installed.`);
      }
      const html = await readWidget(widget);
      mounts.set(slot, mountWidget(body, widget, { html, scheme }));
    } catch (error) {
      note(body, explain(error));
    }
  }

  function set(slot, name) {
    remember(assign(config, slot, name));
    fill(slot);
    const label = panels.get(slot)?.querySelector(".slot-name");
    if (label) label.textContent = shown(name);
  }

  /** What a stored assignment is *called*. A widget is known by its file, so
   *  the name is the name; a panel is stored under its namespaced id and has a
   *  written one, and `sb:about` in the slot control is an implementation
   *  detail leaking into the one place the person is choosing. */
  function shown(name) {
    if (!name) return "Empty";
    return (isPanelId(name) && panelById(name)?.name) || name;
  }

  /** The widget picker, drawn by the frame because it overhangs its slot. */
  async function choose(slot, anchor) {
    try {
      // Refreshed on opening rather than kept from start-up: the agent may have
      // authored one since the page loaded, and a picker that cannot see it
      // looks exactly like an authoring tool that did not work.
      catalog = await listWidgets();
    } catch {
      /* Keep the previous catalog; the picker is still worth showing. */
    }
    chooseWidget(frame, anchor, {
      widgets: offer(),
      current: assignment(config, slot),
      onPick: (name) => set(slot, name),
    });
  }

  /* --------------------------------------------------------- size and mode */

  function applySizes() {
    const { left, right, deck, split } = config.sizes;
    frame.style.setProperty("--sb-l", `${config.closed.left ? 0 : left}px`);
    frame.style.setProperty("--sb-r", `${config.closed.right ? 0 : right}px`);
    frame.style.setProperty("--sb-deck", `${deck}px`);
    frame.style.setProperty("--sb-split", `${split}%`);
    /*
     * A drawer **slides**, it does not shrink.
     *
     * The track is what animates, but the panel inside keeps the width it has
     * when open — `--sb-panel-w` — and is anchored to the edge the drawer
     * closes towards, so the track narrowing moves the contents off the side of
     * the window instead of reflowing them. The difference is the whole look of
     * it: reflowing re-wraps every line of text on every animation frame, which
     * reads as the panel being squeezed rather than put away, and it makes the
     * widget inside do real layout work sixty times a second for an effect
     * nobody wanted.
     *
     * It is also why the widget is never told the intermediate sizes: its box
     * is a constant width throughout, so the `ResizeObserver` in `mount.js`
     * stays quiet and the document inside does not lay out at all.
     */
    for (const side of ["left", "right"]) {
      const panel = frame.querySelector(`.slot[data-drawer="${side}"]`);
      if (!panel) continue;
      panel.toggleAttribute("data-closed", Boolean(config.closed[side]));
      panel.style.setProperty("--sb-panel-w", `${config.sizes[side]}px`);
    }
  }

  function toggleDrawer(side) {
    config.closed[side] = !config.closed[side];
    remember(saveConfig(config));
    applySizes();
    const button = frame.querySelector(`.drawer-${side}`);
    if (button) button.hidden = !config.closed[side];
  }

  /**
   * Choose light, dark, or the machine's answer.
   *
   * The frame states the scheme and every widget is told; a widget must never
   * ask the OS itself, or one box out of fourteen disagrees the moment somebody
   * picks a theme here. `theme.js` says the rest.
   */
  function setTheme(next) {
    if (next === config.theme) return;
    config.theme = next;
    remember(saveConfig(config));
    applyScheme();
  }

  /** Resolve the preference and push it everywhere it is drawn. */
  function applyScheme() {
    scheme = resolveScheme(config.theme);
    applyTheme(scheme);
    for (const mounted of mounts.values()) mounted.setScheme(scheme);
  }

  function setEditing(next) {
    editing = next;
    applyEditing();
  }

  function applyEditing() {
    frame.dataset.editing = String(editing);
    // Editing forces both drawers open (see the stylesheet), so the button that
    // says one is away would be describing something untrue.
    for (const side of ["left", "right"]) {
      const shown = frame.querySelector(`.drawer-${side}`);
      if (shown) shown.hidden = editing || !config.closed[side];
    }
    // A summoned window belongs to the settled arrangement: it was opened by a
    // button that may not exist a moment from now.
    closeWindow(frame);
    closeChooser(frame);
    redrawBars();
    redrawRail();
    frame.querySelector(".edit-btn")?.setAttribute("aria-pressed", String(editing));
    frame.querySelector(".edit-stack")?.remove();
    if (!editing) return;
    // Appearance above Layout above Edit, in one corner of one slot. A stack
    // rather than two absolutely-placed bars, because the only thing that knows
    // how tall the layout chooser is — which changes with the window, since it
    // wraps — is the layout chooser.
    const stack = document.createElement("div");
    stack.className = "edit-stack";
    stack.append(themeBar(config.theme, setTheme));
    stack.append(layoutBar(LAYOUTS, config.layout, (id) => {
      if (id === config.layout) return;
      config.layout = id;
      remember(saveConfig(config));
      arrange();
    }));
    chromeCorner().append(stack);
  }

  /**
   * What is wrong, if anything. One line, at the bottom, over everything.
   *
   * There is nowhere else for it to go: the frame has no status bar by
   * construction, and a slot is a widget's and not the frame's to write in.
   */
  function say(message) {
    problem = message || null;
    banner.textContent = problem || "";
    banner.hidden = !problem;
  }

  /**
   * A save that did not land.
   *
   * Storage can refuse — a private window, blocked site data, a full quota —
   * and the arrangement still *works*, which is the trap: everything looks
   * right until the next page load brings back an arrangement from before.
   * Saying so at the time is the difference between a known limitation and an
   * hour of rearranging slots twice.
   */
  function remember(stored) {
    if (stored) {
      if (problem?.startsWith("Could not save")) say(null);
      return;
    }
    say("Could not save the layout — this browser is refusing to store it, " +
      "so the arrangement will not come back on a reload.");
  }

  /* ------------------------------------------------------------------ drag */

  /**
   * How far down the deck may be dragged: to its header bar, and no further.
   *
   * **Measured rather than declared.** The bar's height depends on whether
   * there are buttons in it and on how many rows they wrapped into, which is
   * known only after layout — so a constant in the limits table would be a
   * guess that is wrong the moment somebody adds a third button. A bar with no
   * buttons is `hidden` and has no height at all, and there the floor is the
   * grab handle instead: a deck dragged to nothing leaves its divider flush
   * against the bottom of the window, which is a thing you cannot get hold of
   * to drag back.
   */
  function deckFloor() {
    const bar = panels.get("deck")?.querySelector(".bar-header");
    const height = bar && !bar.hidden ? bar.getBoundingClientRect().height : 0;
    return Math.max(24, Math.round(height));
  }

  /**
   * Dragging a divider, with the widgets switched off for the duration.
   *
   * Pointer events over an iframe go to the *document inside it*, so a drag
   * that crosses one stops dead halfway. `data-dragging` makes every slot inert
   * until the pointer is released, which is also what stops the cursor
   * flickering between the divider's and whatever the widget asked for.
   */
  function startDrag(event, cell) {
    if (editing) return;
    event.preventDefault();
    const key = cell.drag;
    frame.dataset.dragging = cell.axis;
    event.currentTarget.setPointerCapture?.(event.pointerId);

    const onMove = (moved) => {
      const box = frame.getBoundingClientRect();
      const value = key === "split"
        ? ((moved.clientX - box.left) / box.width) * 100
        : cell.axis === "y"
          ? box.bottom - moved.clientY
          : key === "right" ? box.right - moved.clientX : moved.clientX - box.left;
      config.sizes[key] = key === "deck"
        ? Math.max(deckFloor(), clampSize(key, value))
        : clampSize(key, value);
      applySizes();
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      delete frame.dataset.dragging;
      remember(saveConfig(config));
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }
}

/* ------------------------------------------------------------------ pieces */

function note(body, message) {
  const p = document.createElement("p");
  p.className = "slot-note";
  p.textContent = message;
  body.replaceChildren(p);
}

function problemBanner() {
  const element = document.createElement("p");
  element.className = "frame-problem";
  element.hidden = true;
  return element;
}

/** In-drawer: push this panel away, in the direction it goes. */
function collapseIcon(side) {
  const chevron = side === "left" ? "M14 7l-5 5 5 5" : "M10 7l5 5-5 5";
  return `<svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="${chevron}"/></svg>`;
}

/** Floating: there is a panel over here. */
function drawerIcon(side) {
  const bar = side === "left" ? "M9 4.5v15" : "M15 4.5v15";
  return `<svg viewBox="0 0 24 24" aria-hidden="true">
    <rect x="3.5" y="4.5" width="17" height="15" rx="3"/>
    <path d="${bar}"/></svg>`;
}

/**
 * Why nothing is showing, in terms of the thing to go and fix.
 *
 * A bare "unauthorized" is the least useful true statement available here: the
 * page holds no credential, so it cannot be the cause — the dev server adds
 * one. Naming which half is the difference between a two-second fix and an
 * afternoon.
 */
function explain(error) {
  if (error?.status === 401) {
    return "unauthorized — the dev server is not sending the API token. " +
      "Restart it; it reads Second Brain's config.json.";
  }
  if (error?.status === 404) {
    return "not found — is the proxy configured? Check vite.config.js.";
  }
  return `${error?.message || error}${error?.code ? ` [${error.code}]` : ""}`;
}
