/**
 * Light, dark, or whatever the machine is set to.
 *
 * The palette for both has been in `index.css` from the beginning, along with
 * a `dark:` variant and a scattering of `dark:` classes across a dozen
 * components. None of it could ever fire: nothing put `.dark` on the document,
 * so the app was permanently light and half its stylesheet was unreachable.
 * This is the missing half — the part that decides.
 *
 * **Three states, not two.** "System" is the default everywhere this app is
 * trying to feel like, and it is a genuinely different setting from "light":
 * one tracks the machine as it changes through the day, the other does not.
 * Collapsing them into a toggle loses that, and loses it silently.
 *
 * The class is also applied by an inline script in `index.html`, before the
 * bundle loads. That is not a duplicate of this: React's first paint is far too
 * late to prevent a white flash on a dark desktop, and the flash is the whole
 * thing people notice.
 */

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";

export type Theme = "system" | "light" | "dark";

/** Shared with the pre-paint script in `index.html`. Changing it here without
 *  changing it there brings the flash back. */
export const THEME_KEY = "second-brain:theme";

const DARK_QUERY = "(prefers-color-scheme: dark)";
const THEME_CHANGE_EVENT = "second-brain:theme-change";

/** The stored preference, or "system" for anything unrecognised — including a
 *  browser that refuses `localStorage` entirely, which throws rather than
 *  answering null. */
function readTheme(): Theme {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark" || stored === "system") {
      return stored;
    }
  } catch {
    // Private mode, or storage disabled. Not a reason to fail to render.
  }
  return "system";
}

/** What "system" actually resolves to right now. */
function resolveTheme(theme: Theme): "light" | "dark" {
  if (theme !== "system") return theme;
  return typeof window.matchMedia === "function" &&
    window.matchMedia(DARK_QUERY).matches
    ? "dark"
    : "light";
}

function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle(
    "dark",
    resolveTheme(theme) === "dark",
  );
}

/**
 * Which of the two palettes is on screen, for something that only wants to know.
 *
 * **Read-only, and that is the whole point of it existing beside `useTheme`.**
 * `useTheme` owns the preference: it writes `localStorage` and toggles the
 * class from an effect *on every mount*. That is right for the one control that
 * changes the theme and wrong for anything that merely reads it — a transcript
 * with twenty code blocks in it would take twenty copies of that machinery, and
 * write the preference back to storage twenty times on first paint.
 *
 * `useSyncExternalStore` for the reason `useMediaQuery` gives: the value is read
 * during render rather than corrected in an effect afterwards, so a code block
 * paints in the right palette the first time instead of flashing the other one.
 */
export function useResolvedTheme(): "light" | "dark" {
  return useSyncExternalStore(
    useCallback((onChange: () => void) => {
      // Both sources, because either can move the answer: an explicit choice
      // made elsewhere in the app, and the OS flipping under "system".
      window.addEventListener(THEME_CHANGE_EVENT, onChange);
      const media =
        typeof window.matchMedia === "function"
          ? window.matchMedia(DARK_QUERY)
          : null;
      media?.addEventListener("change", onChange);
      return () => {
        window.removeEventListener(THEME_CHANGE_EVENT, onChange);
        media?.removeEventListener("change", onChange);
      };
    }, []),
    () => resolveTheme(readTheme()),
    () => "light",
  );
}

/**
 * The current theme and a way to change it.
 *
 * The media listener is only attached while the preference is "system" — an
 * explicit choice is not something the OS gets to overrule, and leaving the
 * listener attached would let it.
 */
export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(readTheme);

  // Settings has separate desktop and mobile layouts which are both mounted
  // and hidden responsively. Keep their controls in lockstep when either one
  // changes the preference, including across a viewport resize.
  useEffect(() => {
    const onThemeChange = (event: Event) => {
      const next = (event as CustomEvent<Theme>).detail;
      if (next === "system" || next === "light" || next === "dark") {
        setThemeState(next);
      }
    };
    window.addEventListener(THEME_CHANGE_EVENT, onThemeChange);
    return () => window.removeEventListener(THEME_CHANGE_EVENT, onThemeChange);
  }, []);

  useEffect(() => {
    applyTheme(theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // The preference will not survive a reload. The theme still applies.
    }

    if (theme !== "system") return;
    if (typeof window.matchMedia !== "function") return;
    const media = window.matchMedia(DARK_QUERY);
    const onChange = () => applyTheme("system");
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    window.dispatchEvent(
      new CustomEvent<Theme>(THEME_CHANGE_EVENT, { detail: next }),
    );
  }, []);

  return { theme, setTheme, resolved: resolveTheme(theme) };
}
