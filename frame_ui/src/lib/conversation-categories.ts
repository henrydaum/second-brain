import type { CategoryCount, Conversation } from "@/lib/conversations";

export type ConversationFilter =
  | { type: "all" }
  | { type: "category"; category: string | null };

type ConversationFilterOption = {
  filter: ConversationFilter;
  label: string;
  count: number;
};

export const ALL_CONVERSATIONS_FILTER: ConversationFilter = { type: "all" };
export const MAIN_CONVERSATIONS_FILTER: ConversationFilter = {
  type: "category",
  category: null,
};

const BUILT_IN_CATEGORIES = ["Subagent", "Scheduled"];

/** Blank is the server's representation of an ordinary, person-owned chat. */
export function conversationCategory(category?: string | null): string | null {
  const trimmed = category?.trim() ?? "";
  return trimmed || null;
}

export function categoryLabel(category: string | null): string {
  return category ?? "Main";
}

export function filtersEqual(
  left: ConversationFilter,
  right: ConversationFilter,
): boolean {
  return (
    left.type === right.type &&
    (left.type === "all" ||
      (right.type === "category" && left.category === right.category))
  );
}

export function filterIncludes(
  filter: ConversationFilter,
  conversation: Conversation,
): boolean {
  return (
    filter.type === "all" ||
    conversationCategory(conversation.category) === filter.category
  );
}

/**
 * Every named category present, in the order the UI shows them.
 *
 * Built-ins first in their declared order, then everything else alphabetically.
 * Extracted because two things need to agree about it: the filter list, and
 * `categoryHues` — which assigns colour by position, so a disagreement here
 * would be a pill whose dot is a different colour than its own entry in the
 * menu.
 *
 * `null` — the ordinary, person-owned chat — is deliberately not in here. It is
 * not a tag, and it wears a neutral grey rather than a generated hue.
 */
export function orderedCategories(counts: CategoryCount[]): string[] {
  const present = new Set<string>();
  for (const { category } of counts) {
    const name = conversationCategory(category);
    if (name !== null) present.add(name);
  }

  return [
    ...BUILT_IN_CATEGORIES.filter((category) => present.has(category)),
    ...[...present]
      .filter((category) => !BUILT_IN_CATEGORIES.includes(category))
      .sort((left, right) => left.localeCompare(right)),
  ];
}

/**
 * The filter menu, built from the server's own tally.
 *
 * **Counted over the table rather than over what is loaded**, which is the
 * whole reason `conv.list` answers counts at all: the sidebar holds a page, and
 * a menu built from a page reports "3 Subagent" while showing a window of
 * three — and omits entirely any category that did not happen to appear in it.
 */
export function conversationFilterOptions(
  counts: CategoryCount[],
): ConversationFilterOption[] {
  const byName = new Map<string | null, number>();
  for (const { category, count } of counts) {
    const name = conversationCategory(category);
    byName.set(name, (byName.get(name) ?? 0) + count);
  }
  const total = [...byName.values()].reduce((sum, count) => sum + count, 0);

  return [
    { filter: ALL_CONVERSATIONS_FILTER, label: "All conversations", count: total },
    {
      filter: MAIN_CONVERSATIONS_FILTER,
      label: "Main",
      count: byName.get(null) ?? 0,
    },
    ...orderedCategories(counts).map((category) => ({
      filter: { type: "category", category } as ConversationFilter,
      label: category,
      count: byName.get(category) ?? 0,
    })),
  ];
}

/* ── Colour ───────────────────────────────────────────────────────────── */

/**
 * The golden angle, in degrees.
 *
 * **Placing each tag this far around the wheel from the last is what keeps them
 * apart at every count, not just at one.** It is the irrational rotation, so the
 * sequence never repeats and — this is the useful part — *every prefix of it* is
 * near-optimally spread. Two tags land 137° apart, three fall roughly 137/85/137,
 * six are never closer than 30°. A fixed `360 / n` would do as well for one
 * chosen `n` and re-colour every tag the moment the count changed.
 */
const GOLDEN_ANGLE = 137.50776405003785;

/** Where the first tag sits. Nothing depends on the value; it decides only
 *  which colour "Subagent" happens to be. */
const FIRST_HUE = 25;

/**
 * A hue for each category, chosen so no two are close.
 *
 * **Assigned across the set rather than derived from each name, and it has to
 * be.** This used to hash the name — stable, needed no palette, and could not
 * possibly work: hashing scatters names uniformly, so how close two of them land
 * is luck. `Subagent` drew 270° and `Scheduled` 264°, six degrees apart and
 * indistinguishable. That was not a bad pair of names, it was the birthday
 * problem: for *n* independently hashed tags the expected gap between the
 * closest two is about `360 / n²`, so ten tags are already down to ~4° and a
 * collision is near-certain. No choice of hash function improves that.
 *
 * The cost is that a colour is a function of *position*, so tags cannot be
 * coloured in isolation and inserting one can shift the ones after it. The
 * built-ins are pinned to the front of the order, which is what keeps `Subagent`
 * and `Scheduled` fixed forever; a new plugin category sorting alphabetically
 * ahead of another will re-colour it. That is the trade this makes on purpose —
 * distinctness is the thing being asked for, and it is not available at the same
 * time as per-name independence.
 *
 * Hues do eventually crowd, as they must: 30 tags sit about 4° apart at the
 * closest. There is no arrangement of *n* colours on a wheel that avoids it, and
 * this is the arrangement that puts it off longest.
 */
export function categoryHues(categories: string[]): Map<string, number> {
  const hues = new Map<string, number>();
  categories.forEach((category, index) => {
    // Rounded: the CSS takes any number, but whole degrees keep the inline
    // styles readable and cost nothing — the sequence stays several degrees
    // apart well past any plausible tag count.
    hues.set(category, Math.round((FIRST_HUE + index * GOLDEN_ANGLE) % 360));
  });
  return hues;
}

/**
 * The hue to draw a category in.
 *
 * The fallback is for a category the map has never heard of, which should not
 * happen — the map and every call site are built from the same conversation
 * list. It exists because a dot with no colour resolves to `oklch(… 0)`, a red
 * that looks deliberate; landing on the same colour as the first tag is at
 * least visibly a fallback rather than a claim.
 */
export function hueOf(hues: Map<string, number>, category: string): number {
  return hues.get(category) ?? FIRST_HUE;
}
