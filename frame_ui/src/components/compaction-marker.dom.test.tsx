/** @vitest-environment jsdom */

import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { CompactionRule } from "@/components/compaction-marker";

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal("ResizeObserver", ResizeObserverStub);

describe("CompactionRule", () => {
  /** The tooltip is the answer to a hover, and a hover is not available to
   *  everyone. What the line means has to be readable without one. */
  it("says what the line means to a reader who cannot hover", () => {
    render(<CompactionRule />);

    expect(
      screen.getByText(/only see a summary of everything above this line/i),
    ).toBeInTheDocument();
  });
});
