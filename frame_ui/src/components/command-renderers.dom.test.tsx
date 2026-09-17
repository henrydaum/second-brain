/** @vitest-environment jsdom */

import "@testing-library/jest-dom/vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CommandMarkdown } from "@/components/command-renderers";

const fence = (language: string, body: string) =>
  ["```" + language, body, "```"].join("\n");

describe("Settings command tables", () => {
  it("isolates a wide table in its own horizontal scroller", () => {
    render(
      <CommandMarkdown text={"| First | Second |\n| --- | --- |\n| one | two |"} />,
    );

    const table = screen.getByRole("table");
    expect(table).toHaveClass("w-max", "max-w-none");
    expect(table.parentElement).toHaveClass(
      "w-full",
      "max-w-full",
      "overflow-x-auto",
      "[contain:inline-size]",
    );
  });
});

// Queries are scoped to each render's own container rather than to `screen`:
// this project runs vitest without RTL's global auto-cleanup, so every render
// in a file shares one `document.body`.
describe("fenced code", () => {
  it("highlights a known language and names it in the header", () => {
    const { container } = render(
      <CommandMarkdown text={fence("python", "x = 1")} />,
    );

    expect(within(container).getByText("python")).toBeInTheDocument();
    // Prism splits the source into per-token spans; an unhighlighted block
    // would leave `x = 1` as a single text node under `code`.
    const code = container.querySelector("code");
    expect(code?.querySelectorAll("span").length ?? 0).toBeGreaterThan(1);
  });

  it("still frames a language Prism has no grammar for", () => {
    const { container } = render(
      <CommandMarkdown text={fence("brainfuck", "+++.")} />,
    );

    // The header and its copy button are the point: the block is complete,
    // only the colours are missing.
    expect(within(container).getByText("brainfuck")).toBeInTheDocument();
    expect(
      within(container).getByRole("button", { name: "Copy code" }),
    ).toBeInTheDocument();
    expect(container.querySelector("code")).toHaveTextContent("+++.");
  });

  it("copies the source, not the rendered tokens", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    const source = "const a = 1;\nconst b = 2;";
    const { container } = render(
      <CommandMarkdown text={fence("javascript", source)} />,
    );

    await userEvent.click(
      within(container).getByRole("button", { name: "Copy code" }),
    );
    // The fence's own trailing newline rides along, which is what you want
    // pasting into a file — and it is the source rather than the highlighted
    // markup, which is the thing actually being pinned here.
    expect(writeText).toHaveBeenCalledWith(`${source}\n`);
  });
});
