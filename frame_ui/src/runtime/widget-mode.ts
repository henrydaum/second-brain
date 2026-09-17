/**
 * Where the widget is: beside the chat, pinned above it, or the whole window.
 *
 * **One mode with three values rather than two features.** A side panel on a
 * desktop and a pinned pane on a phone are the same thing — the widget sharing
 * the window with the conversation — and fullscreen is the same thing on both:
 * the widget having the window to itself. Treating them as one enum is what
 * makes the expand button mean one thing everywhere, and it is why the panel
 * can be one element that changes its CSS instead of three that replace each
 * other.
 *
 * The breakpoint decides which two values are reachable, not what they mean:
 *
 * | | shared | alone |
 * |---|---|---|
 * | Desktop | `side` — a drawer you drag | `full` |
 * | Mobile | `pinned` — a fixed split above the chat | `full` |
 *
 * **Mobile starts shared, not fullscreen.** You pressed "show widget", not
 * "take over": opening fullscreen answers a question nobody asked, and it
 * discards the conversation that almost certainly prompted it. Starting shared
 * on both also means one story — start beside the chat, expand when the widget
 * is the point — rather than two defaults to explain.
 *
 * **The mode is sticky, and that is what serves both ways of using a widget.**
 * Someone running a finished widget as an app wants it full and wants that
 * remembered; someone building one wants it beside the chat that is helping
 * them build it. Remembering the mode gives each of them theirs after one
 * press, which is better than a preference nobody would find.
 *
 * Whether it is *open* is deliberately not remembered. A panel that reopens
 * itself on every reload is a panel deciding what you came here to do.
 */

import { useCallback, useState } from "react";

import { MD_QUERY, useMediaQuery } from "@/lib/media";

export type WidgetMode = "side" | "pinned" | "full";

const KEY = "second-brain:widget-mode";

/** What "shared with the chat" means at each width. */
export function sharedMode(isDesktop: boolean): WidgetMode {
  return isDesktop ? "side" : "pinned";
}

/**
 * The stored mode, if it is one this window can honour.
 *
 * A phone's `pinned` is meaningless on a desktop and vice versa, so a mode
 * from another device falls back to this one's shared mode rather than being
 * translated. `full` is the only value that crosses, which is the one that
 * should: it is a statement about wanting the widget to itself, and that
 * survives changing device.
 */
export function modeFor(stored: string | null, isDesktop: boolean): WidgetMode {
  if (stored === "full") return "full";
  return sharedMode(isDesktop);
}

function readMode(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    // Private windows and blocked site data both throw. The default is a fine
    // answer; a crash on first render is not.
    return null;
  }
}

function writeMode(mode: WidgetMode) {
  try {
    localStorage.setItem(KEY, mode);
  } catch {
    /* Refused. The mode still works, it just will not come back. */
  }
}

/**
 * The panel's state, and the three things that can change it.
 *
 * It is a hook rather than a provider because exactly two components need it
 * and they are siblings under `App`. When something *else* wants to open the
 * panel — a tool result, a render frame naming a widget — that is the moment
 * for a provider, and not before.
 */
export function useWidgetPanel() {
  const isDesktop = useMediaQuery(MD_QUERY);
  const [open, setOpen] = useState(false);
  const [stored, setStored] = useState(readMode);

  // Resolved on every render rather than stored resolved: the breakpoint can
  // change under a window being dragged between displays, and a `pinned` panel
  // left over from the narrow layout would be a mode the desktop cannot draw.
  const mode = modeFor(stored, isDesktop);

  const setMode = useCallback((next: WidgetMode) => {
    setStored(next);
    writeMode(next);
  }, []);

  return {
    open,
    mode,
    isDesktop,
    /** The session bar's button: show the panel, or put it away. */
    toggle: useCallback(() => setOpen((value) => !value), []),
    /** The header's X. Closing never changes the mode — it is where the panel
     *  goes when it is out, not whether it is. */
    close: useCallback(() => setOpen(false), []),
    /** The header's expand control: alone, or back to sharing. */
    toggleFull: useCallback(
      () => setMode(mode === "full" ? sharedMode(isDesktop) : "full"),
      [mode, isDesktop, setMode],
    ),
  };
}
