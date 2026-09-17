import { describe, expect, it } from "vitest";

import { fromThreadMessageLike } from "@assistant-ui/react";

import {
  COMPACTED,
  convertMessage,
  SENT_ATTACHMENTS,
  SENT_AT,
} from "@/runtime/convert";
import type { Turn } from "@/runtime/store";

describe("user attachment conversion", () => {
  it("places files in message metadata above prose instead of rendering data text", () => {
    const turn: Turn = {
      id: "u1",
      role: "user",
      parts: [
        {
          kind: "files",
          paths: ["/workspace/chart.png"],
          sent: true,
          attachments: [
            {
              path: "/workspace/chart.png",
              fileName: "chart.png",
              modality: "image",
              extension: "png",
            },
          ],
        },
        { kind: "text", streamId: "u1", text: "Compare this", done: true },
      ],
      running: false,
      aborted: false,
    };

    const converted = convertMessage(turn);
    expect(converted.content).toEqual([{ type: "text", text: "Compare this" }]);
    expect(converted.metadata?.custom?.[SENT_ATTACHMENTS]).toEqual([
      {
        path: "/workspace/chart.png",
        fileName: "chart.png",
        modality: "image",
        extension: "png",
      },
    ]);
  });
});

describe("agent attachment conversion", () => {
  it("keeps a shown file between the text parts surrounding its frame", () => {
    const turn: Turn = {
      id: "a1",
      role: "assistant",
      parts: [
        { kind: "text", streamId: "s1", text: "Before", done: true },
        { kind: "files", paths: ["/workspace/chart.png"] },
        { kind: "text", streamId: "s2", text: "After", done: true },
      ],
      running: false,
      aborted: false,
    };

    expect(convertMessage(turn).content).toMatchObject([
      { type: "text", text: "Before" },
      {
        type: "data",
        name: "agentFiles",
        data: { paths: ["/workspace/chart.png"] },
      },
      { type: "text", text: "After" },
    ]);
  });
});

describe("compaction marker conversion", () => {
  const turn: Turn = {
    id: "stored-7",
    role: "system",
    parts: [
      { kind: "text", streamId: "stored-7", text: "What was said", done: true },
    ],
    running: false,
    aborted: false,
    createdAt: 1786732595340,
  };

  it("travels as a system message with the one text part that role allows", () => {
    const converted = convertMessage(turn);

    expect(converted.role).toBe("system");
    expect(converted.content).toEqual([{ type: "text", text: "What was said" }]);
    expect(converted.metadata?.custom?.[COMPACTED]).toBe(true);
    expect(converted.metadata?.custom?.[SENT_AT]).toBe(1786732595340);
  });

  it("survives assistant-ui's own conversion, which rejects any other shape", () => {
    expect(() =>
      fromThreadMessageLike(convertMessage(turn), "stored-7", {
        type: "complete",
        reason: "stop",
      }),
    ).not.toThrow();
  });

  it("carries no status, which that role refuses", () => {
    expect(convertMessage(turn)).not.toHaveProperty("status");
  });
});
