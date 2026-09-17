import { describe, expect, it } from "vitest";

import { activityFor, type ActivityInput } from "@/lib/activity";

/** A turn in progress whose last part is a sentence being written. */
const streaming: ActivityInput = {
  threadRunning: true,
  isLast: true,
  messageStatus: "running",
  lastPart: { type: "text", text: "Looking that up" },
};

describe("activityFor", () => {
  it("gives the cursor the turn while prose is arriving", () => {
    expect(activityFor(streaming)).toBe("streaming");
  });

  it("says Working when the turn ends on a tool call", () => {
    expect(
      activityFor({ ...streaming, lastPart: { type: "tool-call" } }),
    ).toBe("working");
  });

  it("says Working before the turn has produced anything", () => {
    expect(activityFor({ ...streaming, lastPart: undefined })).toBe("working");
  });

  /** The screenshot this rule came from: the model printed a blank line
   *  between two tool calls, which left a text part with nothing in it. The
   *  dot pulsed beside it while "Working" pulsed below. */
  it.each([" ", "", "\n\n", "  \t "])(
    "says Working, not streaming, for text that is only %j",
    (text) => {
      expect(activityFor({ ...streaming, lastPart: { type: "text", text } })).toBe(
        "working",
      );
    },
  );

  it("draws nothing once the server gives the turn back", () => {
    expect(activityFor({ ...streaming, threadRunning: false })).toBe("none");
  });

  it("draws nothing on a message that is not running", () => {
    expect(activityFor({ ...streaming, messageStatus: "complete" })).toBe("none");
    expect(activityFor({ ...streaming, messageStatus: undefined })).toBe("none");
  });

  /** Two turns that both believe they are running get one indicator between
   *  them, at the bottom, rather than one each. */
  it("draws nothing above the last message", () => {
    expect(activityFor({ ...streaming, isLast: false })).toBe("none");
  });
});
