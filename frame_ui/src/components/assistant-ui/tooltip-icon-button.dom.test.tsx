/** @vitest-environment jsdom */

import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal("ResizeObserver", ResizeObserverStub);

describe("TooltipIconButton", () => {
  it("releases pointer focus but preserves keyboard focus", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(
      <TooltipIconButton tooltip="Files" onClick={onClick}>
        F
      </TooltipIconButton>,
    );

    const button = screen.getByRole("button", { name: "Files" });
    await user.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
    expect(button).not.toHaveFocus();

    button.focus();
    await user.keyboard("{Enter}");
    expect(onClick).toHaveBeenCalledTimes(2);
    expect(button).toHaveFocus();
  });
});
