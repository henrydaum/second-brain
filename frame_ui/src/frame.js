/**
 * The frame: slots, and what goes in them.
 *
 * This is the smallest one there will be — a single slot filling the window,
 * holding a single widget. It is deliberately the whole app for now, because
 * everything that is hard about the arrangement is already decided at one slot:
 * how a widget's document is prepared, what contains it, what it is told, and
 * what it may ask for. Layouts — a chat with drawers either side, a desktop and
 * a mobile variant of each — are arrangement on top of a boundary that either
 * works or does not. Getting the boundary right at N=1 is cheap; discovering it
 * is wrong at N=5 is not.
 *
 * Two things the frame owns and a widget never will:
 *
 * **The boxes.** An iframe is sized and positioned by its parent, so expanding,
 * collapsing, moving and swapping a widget all happen out here, with nothing
 * required of the widget but that it lay itself out at the size it is given.
 *
 * **Anything that must escape a box.** A widget cannot draw over its
 * neighbours; that is the price of the containment and it is not negotiable
 * from the inside. Menus, dialogs and tooltips that overhang belong to the
 * frame, which draws them at the top level — and mostly they are ordinary
 * furniture the agent never needs to design.
 */

import { THREAD } from "./client.js";
import { mountWidget } from "./mount.js";
import { applyTheme, preferredScheme } from "./theme.js";
import { listWidgets, readWidget } from "./widgets.js";

/** The widget the single-slot layout shows until anything can choose. */
const DEFAULT_WIDGET = "hello";

export async function startFrame(root) {
  const scheme = preferredScheme();
  applyTheme(scheme);

  root.innerHTML = `
    <div class="frame">
      <header class="frame-bar">
        <span class="frame-title">Second Brain</span>
        <span class="frame-thread">${THREAD}</span>
        <span class="frame-status" id="status">loading…</span>
      </header>
      <div class="slot" id="slot"></div>
    </div>`;

  const status = root.querySelector("#status");
  const slot = root.querySelector("#slot");

  let widgets;
  try {
    widgets = await listWidgets();
  } catch (error) {
    return fail(status, slot, explain(error));
  }
  if (!widgets?.length) {
    return fail(status, slot,
      "No widgets are installed. There should be one in bundled/widgets/.");
  }

  const widget = widgets.find((row) => row.name === DEFAULT_WIDGET) || widgets[0];
  let html;
  try {
    html = await readWidget(widget);
  } catch (error) {
    return fail(status, slot, error.message);
  }

  const mounted = mountWidget(slot, widget, { html, scheme });
  status.textContent = `${widget.name} · ${widget.tree}`;
  status.className = "frame-status ok";

  // The operating system's preference is only the *starting* answer, and the
  // frame is what tells a widget which it is on — a widget asking the OS
  // directly would be right until somebody picks a theme in the app, at which
  // point one box disagrees with the rest.
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.(
    "change", (event) => {
      const next = event.matches ? "dark" : "light";
      applyTheme(next);
      mounted.setScheme(next);
    });

  return mounted;
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

function fail(status, slot, message) {
  status.textContent = "not running";
  status.className = "frame-status bad";
  slot.innerHTML = `<p class="frame-problem"></p>`;
  slot.querySelector("p").textContent = message;
  return null;
}
