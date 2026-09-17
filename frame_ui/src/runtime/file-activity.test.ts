import { describe, expect, it } from "vitest";

import type { FileEffect, FileEvent } from "@/lib/ledger";
import type { Turn } from "@/runtime/store";
import type { Part } from "@/runtime/store";
import {
  bindByTime,
  collapse,
  countOf,
  entriesOf,
  fileTurns,
  sameFileTurns,
  toSections,
  UNATTRIBUTED,
  withStoreAttachments,
} from "@/runtime/file-activity";

let nextRow = 0;

describe("durable shared-file projection", () => {
  it("recovers only confirmed show_files output and notices completion", () => {
    const base: Turn = { id: "reply", role: "assistant", running: false, aborted: false, createdAt: 1000,
      parts: [{ kind: "tool", callId: "show", name: "show_files", status: "started", isCommand: false, summary: "", args: { paths: ["/a.png", "/a.png", "relative.png"] } }] };
    const before = fileTurns([base]);
    expect(before[0].parts).toEqual([]);
    const success: Turn = { ...base, parts: [{ ...base.parts[0], kind: "tool", callId: "show", name: "show_files", status: "finished", isCommand: false, ok: true, summary: "Showed 1 file." }] };
    expect(sameFileTurns(before, [success])).toBe(false);
    expect(fileTurns([success])[0].parts[0]).toMatchObject({ kind: "files", paths: ["/a.png"] });
    expect(sameFileTurns(fileTurns([success]), [success])).toBe(true);
    for (const override of [{ ok: false }, { name: "read_file" }, { summary: "" }]) {
      const failed = { ...success, parts: success.parts.map((part) => ({ ...part, ...override })) } as Turn;
      expect(fileTurns([failed])[0].parts).toEqual([]);
    }
  });
});

function event(
  path: string,
  effect: FileEffect,
  ts: number,
  extra: Partial<FileEvent> = {},
): FileEvent {
  return { rowId: ++nextRow, ts, path, effect, viaShell: false, ...extra };
}

function turn(
  id: string,
  role: Turn["role"],
  createdAt?: number,
  parts: Part[] = [],
): Turn {
  return {
    id,
    role,
    parts,
    running: false,
    aborted: false,
    ...(createdAt === undefined ? {} : { createdAt }),
  };
}

describe("bindByTime", () => {
  const turns = [
    turn("u1", "user", 1000),
    turn("a1", "assistant", 1100),
    turn("u2", "user", 2000),
    turn("a2", "assistant", 2100),
  ];

  it("puts an event on the assistant turn that was running", () => {
    const bound = bindByTime(
      [event("/late", "wrote", 2500), event("/early", "wrote", 1500)],
      turns,
    );
    expect(bound.get("a2")?.map((e) => e.path)).toEqual(["/late"]);
    expect(bound.get("a1")?.map((e) => e.path)).toEqual(["/early"]);
  });

  it("keeps an event on the previous answer when it lands between turns", () => {
    // Between the person's next question and the first row of the reply to it,
    // the previous turn is still the one that was running.
    const bound = bindByTime([event("/between", "wrote", 2050)], turns);
    expect(bound.get("a1")?.map((e) => e.path)).toEqual(["/between"]);
  });

  it("does not attribute anything to a user turn", () => {
    const bound = bindByTime([event("/x", "wrote", 1050)], turns);
    expect(bound.has("u1")).toBe(false);
  });

  it("collects events older than the first reply as unattributed", () => {
    const bound = bindByTime([event("/ancient", "wrote", 500)], turns);
    expect(bound.get(UNATTRIBUTED)?.map((e) => e.path)).toEqual(["/ancient"]);
  });

  it("ignores turns with no time, rather than dating them to zero", () => {
    // A stored row without a timestamp cannot be a boundary — treating a
    // missing time as 0 would swallow every event in the conversation.
    const bound = bindByTime(
      [event("/x", "wrote", 1500)],
      [turn("a0", "assistant"), turn("a1", "assistant", 1100)],
    );
    expect(bound.get("a1")?.map((e) => e.path)).toEqual(["/x"]);
  });
});

describe("collapse", () => {
  it("makes one entry per file, counting the rest", () => {
    const events = [
      event("/notes.md", "wrote", 300),
      event("/notes.md", "wrote", 200),
      event("/notes.md", "wrote", 100),
    ];
    const { touched } = collapse(events);
    expect(touched).toHaveLength(1);
    expect(touched[0]).toMatchObject({ edits: 3, effect: "wrote", ts: 300 });
  });

  it("takes the newest event as the current state", () => {
    // Newest first is the order `ledger.read` answers in, so the first event
    // seen for a path is what became of it.
    const { touched } = collapse([
      event("/tmp.md", "deleted", 300),
      event("/tmp.md", "wrote", 100),
    ]);
    expect(touched[0]).toMatchObject({ effect: "deleted", gone: true });
  });

  it("keeps shown and edited apart for a file that is both", () => {
    const { shown, touched } = collapse([
      event("/chart.png", "shown", 300),
      event("/chart.png", "wrote", 200),
    ]);
    expect(shown.map((e) => e.path)).toEqual(["/chart.png"]);
    expect(touched.map((e) => e.path)).toEqual(["/chart.png"]);
  });

  it("folds a rename into one entry naming where it came from", () => {
    const row = ++nextRow;
    const { touched } = collapse([
      { rowId: row, ts: 100, path: "/old.md", effect: "moved-from", viaShell: false },
      { rowId: row, ts: 100, path: "/new.md", effect: "moved-to", viaShell: false },
    ]);
    expect(touched).toHaveLength(1);
    expect(touched[0]).toMatchObject({
      path: "/new.md",
      effect: "moved-to",
      movedFrom: "/old.md",
      gone: false,
    });
  });

  it("keeps a move whose destination is missing as a gone file", () => {
    // Half a rename is still a path that stopped existing.
    const { touched } = collapse([event("/old.md", "moved-from", 100)]);
    expect(touched[0]).toMatchObject({ gone: true });
  });

  it("carries the shell flag and command through", () => {
    const { touched } = collapse([
      event("/build", "deleted", 100, { viaShell: true, command: "rm -rf build" }),
    ]);
    expect(touched[0]).toMatchObject({
      viaShell: true,
      command: "rm -rf build",
      gone: true,
    });
  });
});

describe("toSections", () => {
  const turns = [
    turn("u1", "user", 1000),
    turn("a1", "assistant", 1100),
    turn("u2", "user", 2000),
    turn("a2", "assistant", 2100),
  ];

  it("reads the way the transcript does: oldest first, loose events last", () => {
    const bound = bindByTime(
      [
        event("/c", "wrote", 2500),
        event("/b", "wrote", 1500),
        event("/a", "wrote", 500),
      ],
      turns,
    );
    expect(toSections(bound, turns).map((s) => s.turnId)).toEqual([
      "a1",
      "a2",
      UNATTRIBUTED,
    ]);
  });

  it("leaves out turns that touched nothing", () => {
    const bound = bindByTime([event("/only", "wrote", 2500)], turns);
    const sections = toSections(bound, turns);
    expect(sections.map((s) => s.turnId)).toEqual(["a2"]);
    expect(sections[0].at).toBe(2100);
  });

  it("counts shown and edited together", () => {
    const bound = bindByTime(
      [event("/chart.png", "shown", 2500), event("/notes.md", "wrote", 2400)],
      turns,
    );
    expect(countOf(toSections(bound, turns)[0])).toBe(2);
  });

  it("counts a file the agent wrote and then showed you once", () => {
    const bound = bindByTime(
      [event("/chart.png", "shown", 2500), event("/chart.png", "wrote", 2400)],
      turns,
    );
    expect(countOf(toSections(bound, turns)[0])).toBe(1);
  });

  it("makes no sections at all out of nothing", () => {
    expect(toSections(new Map(), turns)).toEqual([]);
  });
});

describe("entriesOf", () => {
  it("gives one row per file, in the order things happened", () => {
    const section = {
      turnId: "a1",
      ...collapse([
        event("/chart.png", "shown", 300),
        event("/notes.md", "wrote", 100),
      ]),
    };
    expect(entriesOf(section).map((e) => e.path)).toEqual([
      "/notes.md",
      "/chart.png",
    ]);
  });

  it("lets the change take the row when a file was written and shown", () => {
    // Two rows with the same name, one tagged and one not, reads as a bug.
    const section = {
      turnId: "a1",
      ...collapse([
        event("/chart.png", "shown", 300),
        event("/chart.png", "wrote", 100),
      ]),
    };
    const entries = entriesOf(section);
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({ path: "/chart.png", effect: "wrote" });
  });
});

describe("withStoreAttachments", () => {
  const files = (...paths: string[]): Part => ({ kind: "files", paths });

  it("stands in for a file the ledger has not caught up with", () => {
    const turns = [turn("a1", "assistant", 1000, [files("/chart.png")])];
    const merged = withStoreAttachments(new Map(), turns);
    expect(merged.get("a1")).toEqual([
      expect.objectContaining({ path: "/chart.png", effect: "shown" }),
    ]);
  });

  it("stands aside once the ledger names the file — even on another turn", () => {
    // The bug this exists to prevent: the frame lands on whichever turn was
    // open, the ledger row on whichever turn its poll caught, and a per-turn
    // merge then lists one file twice in two different sections.
    const turns = [
      turn("a1", "assistant", 1000),
      turn("a2", "assistant", 2000, [files("/chart.png")]),
    ];
    const bound = new Map([["a1", [event("/chart.png", "shown", 1500)]]]);
    const merged = withStoreAttachments(bound, turns);
    expect(merged.get("a2")).toEqual([
      expect.objectContaining({ path: "/chart.png", effect: "shown" }),
    ]);
    expect(merged.get("a1")).toBeUndefined();
  });

  it("leaves the person's own attachments alone", () => {
    const sent: Part = { kind: "files", paths: ["/note.wav"], sent: true };
    const merged = withStoreAttachments(new Map(), [
      turn("a1", "assistant", 1000, [sent]),
    ]);
    expect(merged.size).toBe(0);
  });

  it("ignores user turns", () => {
    const merged = withStoreAttachments(new Map(), [
      turn("u1", "user", 1000, [files("/theirs.png")]),
    ]);
    expect(merged.size).toBe(0);
  });
});

/**
 * The projection that keeps a streamed reply out of the file panel's
 * derivation. Its whole job is to answer "no, nothing changed" cheaply and
 * often — and to be exactly right the rest of the time, since a missed change
 * is a file that never appears.
 */
describe("fileTurns and sameFileTurns", () => {
  const text = (body: string): Part => ({
    kind: "text",
    streamId: "s1",
    text: body,
    done: false,
  });
  const files = (...paths: string[]): Part => ({ kind: "files", paths });

  it("keeps only assistant turns and their agent file parts", () => {
    const projected = fileTurns([
      turn("u1", "user", 1000, [
        { kind: "files", paths: ["/theirs.png"], sent: true },
      ]),
      turn("a1", "assistant", 1100, [text("hello"), files("/chart.png")]),
    ]);

    expect(projected).toHaveLength(1);
    expect(projected[0].id).toBe("a1");
    expect(projected[0].parts).toEqual([files("/chart.png")]);
  });

  it("calls another token the same conversation", () => {
    const before = fileTurns([turn("a1", "assistant", 1100, [text("hel")])]);
    const after = [turn("a1", "assistant", 1100, [text("hello there")])];
    expect(sameFileTurns(before, after)).toBe(true);
  });

  it("notices a file the agent has just shown", () => {
    const before = fileTurns([turn("a1", "assistant", 1100, [text("hi")])]);
    const after = [
      turn("a1", "assistant", 1100, [text("hi"), files("/chart.png")]),
    ];
    expect(sameFileTurns(before, after)).toBe(false);
  });

  it("notices a new turn, a dropped turn and a re-read conversation", () => {
    const one = fileTurns([turn("a1", "assistant", 1100)]);
    expect(sameFileTurns(one, [])).toBe(false);
    expect(
      sameFileTurns(one, [
        turn("a1", "assistant", 1100),
        turn("a2", "assistant", 2100),
      ]),
    ).toBe(false);
    // A history read mints new ids for the same prose.
    expect(sameFileTurns(one, [turn("stored-7", "assistant", 1100)])).toBe(
      false,
    );
  });

  it("notices a file part that changed its paths", () => {
    const before = fileTurns([
      turn("a1", "assistant", 1100, [files("/one.png")]),
    ]);
    expect(
      sameFileTurns(before, [
        turn("a1", "assistant", 1100, [files("/two.png")]),
      ]),
    ).toBe(false);
  });

  it("ignores a user turn being patched with its stored attachments", () => {
    // `hydrateSentAttachments` rewrites a user turn after the fact. Nothing in
    // this panel reads those, so it must not invalidate the derivation.
    const before = fileTurns([
      turn("u1", "user", 1000),
      turn("a1", "assistant", 1100),
    ]);
    const after = [
      turn("u1", "user", 1000, [
        { kind: "files", paths: ["/cache/theirs.png"], sent: true },
      ]),
      turn("a1", "assistant", 1100),
    ];
    expect(sameFileTurns(before, after)).toBe(true);
  });
});
