import { describe, expect, it } from "vitest";

import {
  forgetStagedPath,
  rememberStagedPath,
  stagedPath,
} from "@/runtime/staged-attachments";

describe("staged attachment paths", () => {
  it("shares an uploaded scratch path until the attachment is removed", () => {
    rememberStagedPath("attachment-1", "/tmp/report.pdf");
    expect(stagedPath("attachment-1")).toBe("/tmp/report.pdf");

    forgetStagedPath("attachment-1");
    expect(stagedPath("attachment-1")).toBeUndefined();
  });
});
