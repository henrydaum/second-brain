/**
 * Tag colours, and the reason they are not hashed.
 *
 * The bug that prompted this file: `Subagent` and `Scheduled` came out six
 * degrees apart and looked identical. That was not an unlucky pair of names —
 * it is what hashing each name independently does, and the tests below pin the
 * property that replaced it rather than the two strings that exposed it.
 */

import { describe, expect, it } from "vitest";

import {
  categoryHues,
  hueOf,
  orderedCategories,
} from "@/lib/conversation-categories";

type CategoryCount = import("@/lib/conversations").CategoryCount;

/** The server's tally, written the way it reads: one argument per
 *  conversation, folded into the `{category, count}` rows `conv.list` answers. */
const inCategory = (...categories: (string | null)[]): CategoryCount[] => {
  const counts = new Map<string | null, number>();
  for (const category of categories) {
    counts.set(category, (counts.get(category) ?? 0) + 1);
  }
  return [...counts].map(([category, count]) => ({ category, count }));
};

/** The closest any two hues sit, going the short way round the wheel. */
function closestPair(hues: number[]): number {
  let closest = 360;
  for (let i = 0; i < hues.length; i += 1) {
    for (let j = i + 1; j < hues.length; j += 1) {
      const apart = Math.abs(hues[i] - hues[j]);
      closest = Math.min(closest, apart, 360 - apart);
    }
  }
  return closest;
}

const huesFor = (...categories: string[]) => [
  ...categoryHues(categories).values(),
];

describe("the bug", () => {
  it("puts the two built-ins nowhere near each other", () => {
    // They used to be 270 and 264.
    expect(closestPair(huesFor("Subagent", "Scheduled"))).toBeGreaterThan(90);
  });
});

describe("no two tags share a colour", () => {
  it("keeps a realistic set clearly apart", () => {
    const hues = huesFor(
      "Subagent",
      "Scheduled",
      "Archive",
      "Email",
      "Inbox",
      "Notes",
    );

    // 30 degrees is about where two chips stop reading as the same colour.
    expect(closestPair(hues)).toBeGreaterThanOrEqual(30);
  });

  it("degrades gradually rather than collapsing", () => {
    // The point of the golden angle: *every* prefix is spread, so there is no
    // count at which this suddenly stops working. Hashing hit 0 degrees at ten
    // tags; this is still legible at thirty.
    const many = Array.from({ length: 30 }, (_, index) => `tag-${index}`);

    expect(closestPair(huesFor(...many))).toBeGreaterThan(5);
  });

  it("never assigns the same hue twice", () => {
    const many = Array.from({ length: 50 }, (_, index) => `tag-${index}`);
    const hues = huesFor(...many);

    expect(new Set(hues).size).toBe(hues.length);
  });
});

describe("what stays put", () => {
  it("does not move the built-ins when plugin categories appear", () => {
    // The stability that matters: `Subagent` and `Scheduled` sort to the front,
    // so their colours are fixed for good. Tags after them can shift, which is
    // the price of assigning across the set rather than per name.
    const before = categoryHues(orderedCategories(inCategory("Subagent", "Scheduled")));
    const after = categoryHues(
      orderedCategories(inCategory("Subagent", "Scheduled", "Aardvark", "Zebra")),
    );

    expect(after.get("Subagent")).toBe(before.get("Subagent"));
    expect(after.get("Scheduled")).toBe(before.get("Scheduled"));
  });

  it("gives one category the same hue however often it appears", () => {
    const hues = categoryHues(
      orderedCategories(inCategory("Email", "Email", "Email")),
    );

    expect(hues.size).toBe(1);
  });
});

describe("the order colours are assigned in", () => {
  it("puts built-ins first, in their declared order, then the rest by name", () => {
    // The filter menu and the hue map read this same function, so a
    // disagreement here would be a pill whose dot is a different colour than
    // its own entry in the menu.
    expect(
      orderedCategories(
        inCategory("Zebra", "Scheduled", "Aardvark", "Subagent"),
      ),
    ).toEqual(["Subagent", "Scheduled", "Aardvark", "Zebra"]);
  });

  it("leaves out the ordinary chats", () => {
    // `null` is not a tag. It wears a neutral grey and takes no hue, so
    // including it here would shift every real tag along by one.
    expect(orderedCategories(inCategory(null, "Email", null))).toEqual([
      "Email",
    ]);
  });

  it("ignores a category that is only whitespace", () => {
    expect(orderedCategories(inCategory("   ", "Email"))).toEqual(["Email"]);
  });
});

describe("hueOf", () => {
  it("answers from the map", () => {
    const hues = categoryHues(["Subagent", "Scheduled"]);

    expect(hueOf(hues, "Scheduled")).toBe(hues.get("Scheduled"));
  });

  it("falls back rather than resolving to a red that looks deliberate", () => {
    // Unreachable in practice — the map and every call site come from one
    // conversation list — but `oklch(… 0)` is a confident-looking red, and a
    // fallback should not look like an answer.
    expect(hueOf(new Map(), "Never seen")).toBeGreaterThan(0);
  });
});
