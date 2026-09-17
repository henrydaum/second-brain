/** @vitest-environment jsdom */

import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

describe("Sheet", () => {
  it("traps the mobile surface and restores focus after Escape", async () => {
    const user = userEvent.setup();
    render(
      <Sheet>
        <SheetTrigger>Open files</SheetTrigger>
        <SheetContent>
          <SheetTitle>Files</SheetTitle>
          <SheetClose>Done</SheetClose>
        </SheetContent>
      </Sheet>,
    );

    const trigger = screen.getByRole("button", { name: "Open files" });
    await user.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "Files" });
    expect(dialog).toBeVisible();
    expect(dialog).toHaveFocus();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Files" })).toBeNull();
    expect(trigger).toHaveFocus();
  });
});
