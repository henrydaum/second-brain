// @vitest-environment jsdom

import { describe, expect, it } from "vitest";

import { toTurns, type StoredMessage } from "@/lib/history";

const user = (over: Partial<StoredMessage> = {}): StoredMessage => ({
  id: 1,
  role: "user",
  content: "Compare these",
  tool_call_id: null,
  tool_name: null,
  attachments: [],
  ...over,
});

describe("stored user attachments", () => {
  it("also preserves explicit assistant output attachments without prose", () => {
    const stored = [user({ role: "assistant", content: "", attachments: [{ path: "/output.png" }] })];
    expect(toTurns(stored)[0].parts).toEqual([expect.objectContaining({ kind: "files", paths: ["/output.png"] })]);
    expect(toTurns(stored)).toEqual(toTurns(stored));
  });

  it("preserves returned attachments on stored tool results", () => {
    const stored = [
      user({ role: "assistant", content: JSON.stringify({ tool_calls: [{ id: "search", function: { name: "hybrid_search", arguments: "{}" } }] }) }),
      user({ id: 2, role: "tool", tool_call_id: "search", content: "Results found", attachments: [{ path: "/result.png" }, { path: "/result.pdf" }] }),
    ];
    expect(toTurns(stored)[0].parts.at(-1)).toMatchObject({ kind: "files", paths: ["/result.png", "/result.pdf"] });
  });

  it("keeps prose clean and turns attachment records into their own part", () => {
    const [turn] = toTurns([
      user({
        attachments: [
          {
            path: "/workspace/attachments/1_chart.png",
            file_name: "chart.png",
            modality: "image",
            extension: "png",
          },
          {
            path: "/workspace/attachments/1_notes.pdf",
            file_name: "notes.pdf",
            modality: "text",
            extension: "pdf",
          },
        ],
      }),
    ]);

    expect(turn?.parts).toEqual([
      {
        kind: "files",
        paths: [
          "/workspace/attachments/1_chart.png",
          "/workspace/attachments/1_notes.pdf",
        ],
        sent: true,
        attachments: [
          {
            path: "/workspace/attachments/1_chart.png",
            fileName: "chart.png",
            modality: "image",
            extension: "png",
          },
          {
            path: "/workspace/attachments/1_notes.pdf",
            fileName: "notes.pdf",
            modality: "text",
            extension: "pdf",
          },
        ],
      },
      {
        kind: "text",
        streamId: "stored-1",
        text: "Compare these",
        done: true,
      },
    ]);
  });

  it("keeps an attachment-only message even when its content is empty", () => {
    const turns = toTurns([
      user({
        content: "",
        attachments: [
          {
            path: "/workspace/voice.wav",
            file_name: "voice.wav",
            modality: "audio",
            extension: "wav",
          },
        ],
      }),
    ]);

    expect(turns).toHaveLength(1);
    expect(turns[0]?.parts).toHaveLength(1);
    expect(turns[0]?.parts[0]).toMatchObject({ kind: "files", sent: true });
  });

  it("leaves pre-column pointer prose untouched instead of guessing", () => {
    const pointer =
      "[Attached image file: old.png (cached at /workspace/old.png)]";
    const [turn] = toTurns([user({ content: pointer })]);

    expect(turn?.parts).toEqual([
      { kind: "text", streamId: "stored-1", text: pointer, done: true },
    ]);
  });
});

it("reconstructs interrupted logical turns by turn_id, not by user row boundaries", () => {
  const rows = [
    user({ id: 1, role: "assistant", turn_id: "a", content: "Before" }),
    user({ id: 2, turn_id: "a", content: "Interrupt" }),
    user({ id: 3, role: "assistant", turn_id: "a", content: "After" }),
    user({ id: 4, role: "assistant", turn_id: "b", content: "Next logical turn" }),
  ];
  const turns = toTurns(rows);
  expect(turns).toHaveLength(4);
  expect(turns[0]).toMatchObject({ turnId: "a", continues: true });
  expect(turns[2]).toMatchObject({ turnId: "a" });
  expect(turns[2].continues).toBeUndefined();
  expect(turns[3].turnId).toBe("b");
  expect(toTurns(rows)).toEqual(turns);
});

describe("stored message authorship", () => {
  it("does not render kernel-authored user-role rows as the person's words", () => {
    const turns = toTurns([
      user({
        content: "[Conversation summary from earlier] not a person",
        author: "compaction",
      }),
      user({ id: 2, content: "Actually from the person", author: null }),
    ]);

    expect(turns).toHaveLength(1);
    expect(turns[0]).toMatchObject({ id: "stored-2", role: "user" });
  });

  it("keeps old rows whose author field is absent", () => {
    const turns = toTurns([user({ content: "An old real message" })]);

    expect(turns).toHaveLength(1);
    expect(turns[0]?.role).toBe("user");
  });
});

describe("compaction markers", () => {
  const marker = (over: Partial<StoredMessage> = {}): StoredMessage => ({
    id: 7,
    role: "system",
    content: JSON.stringify({
      __second_brain_compaction__: true,
      summary: "User asked about X. Agent edited foo.py:42.",
      tail_count: 2,
      // Two different instants on purpose: the packer's own and the row's, so
      // the assertions below can say which one was read.
      created_at: 1786732595.25,
    }),
    tool_call_id: null,
    tool_name: null,
    timestamp: 1786732596.5,
    ...over,
  });

  it("shows the seam a compaction left, carrying the summary and its time", () => {
    const [turn] = toTurns([marker()]);

    expect(turn).toMatchObject({ id: "stored-7", role: "system" });
    // Seconds on the wire, milliseconds in a `Date` — the trap this whole file
    // is careful about.
    expect(turn?.createdAt).toBe(1786732596500);
    expect(turn?.parts).toEqual([
      {
        kind: "text",
        streamId: "stored-7",
        text: "User asked about X. Agent edited foo.py:42.",
        done: true,
      },
    ]);
  });

  it("falls back to the packed time when the row carries no timestamp", () => {
    const [turn] = toTurns([marker({ timestamp: null })]);

    expect(turn?.createdAt).toBe(1786732595250);
  });

  it("still hides the state machine's own system rows", () => {
    const turns = toTurns([
      {
        id: 7,
        role: "system",
        content: JSON.stringify({
          __second_brain_state_machine__: true,
          state: { phase: "idle" },
        }),
        tool_call_id: null,
        tool_name: null,
      },
    ]);

    expect(turns).toEqual([]);
  });

  it("does not let one assistant turn straddle the line", () => {
    const assistant = (id: number, text: string): StoredMessage => ({
      id,
      role: "assistant",
      content: JSON.stringify({ content: text }),
      tool_call_id: null,
      tool_name: null,
    });
    const turns = toTurns([
      assistant(1, "Before"),
      marker(),
      assistant(8, "After"),
    ]);

    expect(turns.map((turn) => turn.role)).toEqual([
      "assistant",
      "system",
      "assistant",
    ]);
    expect(turns[2]?.id).toBe("stored-8");
  });
});
