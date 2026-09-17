import { describe, expect, it, vi } from "vitest";

// This module reaches `@/lib/files`, which imports the client for `fileUrl` —
// and that module reads `window.location` as it loads. Nothing under test here
// needs a DOM, so the client is stubbed rather than the suite moved into jsdom.
// Same trick, and the same reason, as `files.test.ts`.
vi.mock("@/lib/client", () => ({
  sdk: () => Promise.resolve("unknown"),
  authHeaders: () => ({}),
  fileUrl: (path: string) => `http://host/files?path=${encodeURIComponent(path)}`,
}));

import { hostPathFor, splitFrontMatter } from "@/lib/markdown";

describe("splitFrontMatter", () => {
  it("lifts a YAML block off the top and leaves the document", () => {
    const { front, body } = splitFrontMatter(
      "---\ntitle: Weekly plan\nstatus: draft\n---\n# Monday\n\nWrite it down.\n",
    );

    expect(front).toEqual([
      { key: "title", value: "Weekly plan" },
      { key: "status", value: "draft" },
    ]);
    expect(body).toBe("# Monday\n\nWrite it down.\n");
  });

  it("gathers a block list under the key it belongs to", () => {
    const { front } = splitFrontMatter(
      "---\ntags:\n  - work\n  - urgent\n---\nbody\n",
    );
    expect(front).toEqual([{ key: "tags", value: "work, urgent" }]);
  });

  it("unquotes a value without pretending to be YAML", () => {
    const { front } = splitFrontMatter('---\ntitle: "Q3: the reckoning"\n---\n');
    expect(front).toEqual([{ key: "title", value: "Q3: the reckoning" }]);
  });

  it("keeps a line it cannot read rather than dropping it", () => {
    const { front } = splitFrontMatter(
      "---\ntitle: Plan\n{ this is not a pair }\n---\n",
    );
    expect(front).toEqual([
      { key: "title", value: "Plan" },
      { key: "", value: "{ this is not a pair }" },
    ]);
  });

  it("ends the block at the first closing fence, not the last", () => {
    const { front, body } = splitFrontMatter(
      "---\ntitle: One\n---\nText.\n\n---\n\nMore.\n",
    );
    expect(front).toEqual([{ key: "title", value: "One" }]);
    expect(body).toBe("Text.\n\n---\n\nMore.\n");
  });

  it("takes `...` as a close, which YAML allows", () => {
    const { front, body } = splitFrontMatter("---\ntitle: One\n...\nText.\n");
    expect(front).toEqual([{ key: "title", value: "One" }]);
    expect(body).toBe("Text.\n");
  });

  it("leaves a document that merely opens with a rule alone", () => {
    // The failure worth avoiding: fences alone cannot tell YAML from a thematic
    // break, and swallowing a chunk of the document is invisible where a stray
    // `---` is not. So a block with no `key: value` line in it is not one.
    const text = "---\n\nSome opening line.\n\n---\n\nMore.\n";
    expect(splitFrontMatter(text)).toEqual({ front: null, body: text });
  });

  it("says there was none when the file just starts", () => {
    const text = "# Heading\n\nProse.\n";
    expect(splitFrontMatter(text)).toEqual({ front: null, body: text });
  });
});

describe("hostPathFor", () => {
  const note = "/srv/vault/daily/2026-08-16.md";

  it("resolves a link to a neighbouring file", () => {
    expect(hostPathFor(note, "../plan.md")).toBe("/srv/vault/plan.md");
    expect(hostPathFor(note, "attachments/chart.png")).toBe(
      "/srv/vault/daily/attachments/chart.png",
    );
  });

  it("decodes the escaping a Markdown link uses for a path", () => {
    expect(hostPathFor(note, "my%20note.md")).toBe(
      "/srv/vault/daily/my note.md",
    );
  });

  it("drops a fragment and a query, which are not part of the name", () => {
    expect(hostPathFor(note, "plan.md#monday")).toBe(
      "/srv/vault/daily/plan.md",
    );
  });

  it("refuses anything already addressed somewhere else", () => {
    // Null is what keeps these on the browser's own handling — and it is the
    // only thing standing between `javascript:` and a `/files` URL built out
    // of it.
    expect(hostPathFor(note, "https://example.com/x")).toBeNull();
    expect(hostPathFor(note, "mailto:someone@example.com")).toBeNull();
    expect(hostPathFor(note, "javascript:alert(1)")).toBeNull();
    expect(hostPathFor(note, "//example.com/x")).toBeNull();
    expect(hostPathFor(note, "#monday")).toBeNull();
    expect(hostPathFor(note, "  ")).toBeNull();
  });

  it("reads a drive letter as a drive, not as a scheme", () => {
    expect(hostPathFor(note, "C:\\notes\\plan.md")).toBe("C:\\notes\\plan.md");
  });
});
