// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";

import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import {
  forgetStagedPath,
  rememberStagedPath,
  useStagedPath,
} from "@/runtime/staged-attachments";

const Probe = ({ id }: { id: string }) => (
  <span>{useStagedPath(id) ?? "not uploaded"}</span>
);

afterEach(() => {
  forgetStagedPath("attachment-1");
  cleanup();
});

describe("staged attachment path subscription", () => {
  it("makes a non-image tile viewable when its upload finishes", () => {
    render(<Probe id="attachment-1" />);
    expect(screen.getByText("not uploaded")).toBeInTheDocument();

    act(() => rememberStagedPath("attachment-1", "/tmp/voice-note.wav"));

    expect(screen.getByText("/tmp/voice-note.wav")).toBeInTheDocument();
  });
});
