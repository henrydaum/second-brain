/**
 * A CSS media query, readable from React.
 *
 * Layout is Tailwind's job almost everywhere in this app, and it should stay
 * that way — a breakpoint expressed twice is a breakpoint that will eventually
 * disagree with itself. This exists for the one thing CSS cannot do: the
 * conversations sidebar *unmounts* its list rather than hiding it, and whether
 * that is the right thing to do depends on whether the sidebar is currently a
 * collapsible rail or an overlay drawer.
 *
 * `useSyncExternalStore` rather than `useState` + an effect, because it reads
 * the match during render instead of after it — which is the difference between
 * the correct layout on the first paint and a visible correction on the second.
 */

import { useCallback, useSyncExternalStore } from "react";

export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      const media = window.matchMedia(query);
      media.addEventListener("change", onChange);
      return () => media.removeEventListener("change", onChange);
    },
    [query],
  );

  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    // Server snapshot. Nothing renders this app on a server, but the argument
    // is required and "assume small" is the safer of the two guesses.
    () => false,
  );
}

/** Tailwind's `md`. Named so the value lives beside the query that uses it
 *  rather than being repeated as a magic number. */
export const MD_QUERY = "(min-width: 48rem)";
/** Tailwind's `xl`. Files remain an overlay below this width so the files
 * panel and conversation rail cannot squeeze the thread on tablets. */
export const XL_QUERY = "(min-width: 80rem)";
export const FINE_POINTER_QUERY = "(pointer: fine)";
