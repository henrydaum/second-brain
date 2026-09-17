/**
 * Where you were in a file, so that opening it again puts you back.
 *
 * The viewer remounts on every open — keyed on the path, deliberately, so that
 * stepping between files never shows one file's bytes under another's name —
 * and a remount starts at the top. That is right for a picture and wrong for
 * everything you *read*: closing a long note to check something and coming back
 * to the first line is the kind of small loss that makes a viewer feel like it
 * is fighting you.
 *
 * **Two levels of key, not one composite string.** A file remembers a place per
 * *variant* — a note scrolled halfway in Preview is nowhere near halfway in
 * Source, and the same file inline in the transcript is a different box again —
 * while a ledger row naming a file has to be able to drop everything about it
 * at once. Nesting the map is what makes both cheap. See `forgetFile`.
 *
 * Deliberately not `localStorage`: this is where you were a moment ago, not a
 * preference, and it has no business outliving the tab.
 */

import { useCallback } from "react";

/** How many files to keep a place for. Generous, because each entry is a path
 *  and a number, and a session that opens more than fifty files has long since
 *  stopped caring about the first one. */
const CAP = 50;

/** path → variant → `scrollTop`. */
const places = new Map<string, Map<string, number>>();

export function rememberPlace(
  path: string,
  variant: string,
  top: number,
): void {
  if (!Number.isFinite(top)) return;

  const known = places.get(path);
  if (known) {
    known.set(variant, top);
    // Re-inserted to mark it most recently used; `Map` keeps insertion order,
    // which is the whole of the eviction policy below. Same trick as `texts`
    // in `lib/files.ts`.
    places.delete(path);
    places.set(path, known);
  } else {
    places.set(path, new Map([[variant, top]]));
  }

  for (const oldest of places.keys()) {
    if (places.size <= CAP) break;
    places.delete(oldest);
  }
}

/** Where this variant was left, or the top — which is the right answer both
 *  for a file never opened and for one whose place has been evicted. */
export function recallPlace(path: string, variant: string): number {
  return places.get(path)?.get(variant) ?? 0;
}

/** Forget a file entirely, in every variant. Called when a ledger row says the
 *  file changed: an offset into a document that has since been rewritten points
 *  at whatever happens to be there now, which is worse than the top. */
export function forgetPlaces(path: string): void {
  places.delete(path);
}

/**
 * A ref for the scrolling box of a viewer, which restores and records its place.
 *
 * A callback ref rather than an effect, because the restore has to happen the
 * moment the node exists — an effect runs a frame later, which is one frame of
 * the top of the document before it jumps.
 *
 * **Every place is recorded as it happens, and none on the way out.** Reading
 * `scrollTop` in the cleanup is the obvious way to do this and it is wrong
 * twice over. Once because it is unnecessary — a scroll the reader performed
 * has already fired a `scroll` event, and so has one the browser performed on
 * their behalf. And once because it is actively harmful: React swaps a
 * component's children *before* it detaches that component's ref, so by the
 * time a cleanup runs on a switch from Preview to Source, the box already holds
 * the other rendering and `scrollTop` has been clamped to its height. The write
 * that looked like insurance is the one that loses the place.
 *
 * **The other subtlety is the echo.** Setting `scrollTop` fires a `scroll`
 * event of its own, and if the content has not finished laying out — a note
 * whose images are still arriving — the browser clamps that assignment to the
 * height so far. Recording that echo would overwrite the good place with the
 * clamped one, with no getting it back. So nothing is recorded until a frame
 * has passed and the restore has settled.
 */
export function useRememberedScroll(path: string, variant: string) {
  return useCallback(
    (node: HTMLElement | null) => {
      if (!node) return;

      const top = recallPlace(path, variant);
      if (top > 0) node.scrollTop = top;

      let settled = false;
      const frame = requestAnimationFrame(() => {
        settled = true;
      });

      const onScroll = () => {
        if (settled) rememberPlace(path, variant, node.scrollTop);
      };
      node.addEventListener("scroll", onScroll, { passive: true });

      // React 19 takes a cleanup from a ref callback, which is what lets this
      // be one function rather than a ref plus an effect that shadows it.
      return () => {
        cancelAnimationFrame(frame);
        node.removeEventListener("scroll", onScroll);
      };
    },
    [path, variant],
  );
}
