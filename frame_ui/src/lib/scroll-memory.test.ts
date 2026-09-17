import { describe, expect, it } from "vitest";

import {
  forgetPlaces,
  recallPlace,
  rememberPlace,
} from "@/lib/scroll-memory";

describe("scroll memory", () => {
  it("gives a file back the place it was left", () => {
    rememberPlace("/srv/vault/plan.md", "full:preview", 820);
    expect(recallPlace("/srv/vault/plan.md", "full:preview")).toBe(820);
  });

  it("starts at the top for a file it has never seen", () => {
    expect(recallPlace("/srv/vault/unseen.md", "full:preview")).toBe(0);
  });

  it("keeps a place per variant, because they are different boxes", () => {
    // Halfway down a rendering is nowhere near halfway down its source, and
    // the inline copy in the transcript is a third height again.
    rememberPlace("/srv/notes.md", "full:preview", 400);
    rememberPlace("/srv/notes.md", "full:source", 1300);
    rememberPlace("/srv/notes.md", "inline:preview", 60);

    expect(recallPlace("/srv/notes.md", "full:preview")).toBe(400);
    expect(recallPlace("/srv/notes.md", "full:source")).toBe(1300);
    expect(recallPlace("/srv/notes.md", "inline:preview")).toBe(60);
  });

  it("forgets a file in every variant at once", () => {
    // What `forgetFile` calls when a ledger row says the file changed: an
    // offset into a document that has since been rewritten points at whatever
    // happens to be there now, which is worse than the top.
    rememberPlace("/srv/rewritten.md", "full:preview", 500);
    rememberPlace("/srv/rewritten.md", "full:source", 900);

    forgetPlaces("/srv/rewritten.md");

    expect(recallPlace("/srv/rewritten.md", "full:preview")).toBe(0);
    expect(recallPlace("/srv/rewritten.md", "full:source")).toBe(0);
  });

  it("drops the least recently used file once it is holding enough", () => {
    for (let i = 0; i < 60; i++) {
      rememberPlace(`/srv/bulk/f${i}.md`, "full:preview", i + 1);
    }
    // Fifty kept, oldest out first.
    expect(recallPlace("/srv/bulk/f0.md", "full:preview")).toBe(0);
    expect(recallPlace("/srv/bulk/f59.md", "full:preview")).toBe(60);
  });

  it("ignores a position that is not a number", () => {
    // `scrollTop` off a detached node can be `NaN`, and a `NaN` written here
    // would be handed straight back to `scrollTop` on the next open.
    rememberPlace("/srv/odd.md", "full:preview", 120);
    rememberPlace("/srv/odd.md", "full:preview", Number.NaN);
    expect(recallPlace("/srv/odd.md", "full:preview")).toBe(120);
  });
});
