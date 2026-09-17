// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";

/**
 * The dialog, in a DOM.
 *
 * The reducer underneath is covered without one, and that covers most of this
 * surface — but not the part that matters most about it. **Escape and the
 * corner button must reach a real cancel**, and whether they do is a fact about
 * Radix's dismissal wiring, not about any function written here. The rule those
 * two affordances bend is the rule this whole surface exists to enforce, so
 * "probably fine" is not a good enough answer about them.
 *
 * The provider is stubbed rather than rendered: what is under test is the
 * dialog's contract with it — which callback, with which id — and standing up a
 * real one would put an SSE connection between the test and the assertion.
 */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { InputRequestDialog } from "@/components/input-request-dialog";
import type { InputRequest } from "@/lib/input-requests";
import * as provider from "@/runtime/provider";

afterEach(cleanup);

const ask = (fields: Partial<InputRequest> = {}): InputRequest => ({
  id: "approve_1",
  title: "Run a shell command",
  body: "rm -rf /tmp/x",
  ...fields,
});

/** Stand in for the provider, and hand back the two callbacks to assert on. */
function mount(queue: InputRequest[]) {
  const resolve = vi.fn().mockResolvedValue(undefined);
  const cancelInputRequest = vi.fn().mockResolvedValue(undefined);
  vi.spyOn(provider, "useApprovals").mockReturnValue({
    inputRequests: queue,
    resolve,
    cancelInputRequest,
  });
  render(<InputRequestDialog />);
  return { resolve, cancelInputRequest, user: userEvent.setup() };
}

describe("answering", () => {
  it("draws one button per option, labelled not valued", () => {
    // The rule that puts `always:api.brave.com` on somebody's button when it
    // is got backwards.
    mount([
      ask({
        type: "string",
        enum: ["allow", "always:api.brave.com", "deny"],
        enum_labels: ["Allow once", "Always allow api.brave.com", "Deny"],
      }),
    ]);

    expect(
      screen.getAllByRole("button").map((button) => button.textContent),
    ).toEqual([
      "Cancel this request",
      "Allow once",
      "Always allow api.brave.com",
      "Deny",
    ]);
  });

  it("sends the value, with the id it was asked under", async () => {
    const { resolve, user } = mount([ask()]);

    await user.click(screen.getByRole("button", { name: "Allow" }));

    expect(resolve).toHaveBeenCalledWith("approve_1", true);
  });

  it("stacks options that are sentences rather than labels", () => {
    // A row of these ran off the side of the dialog and took the options at the
    // end of it out of reach. `tool_ask_question` offers exactly this shape.
    mount([
      ask({
        type: "string",
        enum: ["conversation", "file"],
        enum_labels: [
          "Notes from one of my conversations — you'll tell me which one",
          "Notes from a file on disk — you'll give me the path",
        ],
      }),
    ]);

    const options = screen.getByRole("button", { name: /conversations/ })
      .parentElement;
    expect(options?.dataset.layout).toBe("stacked");
  });

  it("keeps short options in a row", () => {
    mount([ask()]);

    expect(
      screen.getByRole("button", { name: "Allow" }).parentElement?.dataset.layout,
    ).toBe("row");
  });

  it("shows the body in full", () => {
    // It carries the arguments and who is asking, which is the entire basis on
    // which anybody can answer.
    mount([ask()]);

    expect(screen.getByText("rm -rf /tmp/x")).toBeDefined();
  });

  it("reserves the close button's touch target beside the title", () => {
    mount([ask({ title: "A deliberately long request title" })]);

    const title = screen.getByRole("heading", {
      name: "A deliberately long request title",
    });
    expect(title).toHaveClass("px-12", "sm:ps-0", "sm:pe-12");
    expect(title.parentElement).not.toHaveClass("pe-12");
    expect(
      screen.getByRole("button", { name: "Cancel this request" }),
    ).toHaveClass("size-8");
  });

  it("takes free text when there is nothing to press", async () => {
    const { resolve, user } = mount([ask({ type: "string" })]);

    await user.type(screen.getByRole("textbox"), "the answer");
    await user.click(screen.getByRole("button", { name: "Answer" }));

    expect(resolve).toHaveBeenCalledWith("approve_1", "the answer");
  });

  it("stops a second press answering the next question", async () => {
    // The POST completes only when the *original* blocked Request finishes,
    // which can take a while.
    const { resolve, user } = mount([ask()]);

    const allow = screen.getByRole("button", { name: "Allow" });
    await user.click(allow);
    await user.click(allow);

    expect(resolve).toHaveBeenCalledTimes(1);
  });
});

describe("backing out", () => {
  it("cancels on Escape rather than merely closing", async () => {
    // **The distinction this surface is about.** Cancelling settles the
    // question on the server, so the turn unblocks in the safe direction;
    // hiding the panel would leave the agent waiting on something nobody can
    // see, which is what the old "no escape" rule was written against.
    const { cancelInputRequest, resolve, user } = mount([ask()]);

    await user.keyboard("{Escape}");

    expect(cancelInputRequest).toHaveBeenCalledWith("approve_1");
    expect(resolve).not.toHaveBeenCalled();
  });

  it("cancels from the corner button too", async () => {
    const { cancelInputRequest, user } = mount([ask()]);

    await user.click(screen.getByRole("button", { name: "Cancel this request" }));

    expect(cancelInputRequest).toHaveBeenCalledWith("approve_1");
  });

  it("ignores a press on the backdrop", () => {
    // The one route out that carries no intent — it is the press people make
    // by accident, and an accident must not settle anything.
    //
    // `fireEvent`, not `user.click`: a modal dialog puts `pointer-events: none`
    // on everything behind it, so `user-event` refuses to dispatch at all and
    // would pass this test without ever reaching the handler under test. This
    // aims the press straight at the overlay, which is what `onPointerDownOutside`
    // is there to decline.
    const { cancelInputRequest, resolve } = mount([ask()]);
    const overlay = document.querySelector('[data-slot="dialog-overlay"]');

    expect(overlay).not.toBeNull();
    fireEvent.pointerDown(overlay!);

    expect(cancelInputRequest).not.toHaveBeenCalled();
    expect(resolve).not.toHaveBeenCalled();
    expect(screen.getByText("Run a shell command")).toBeDefined();
  });

  it("does not cancel behind an answer already on its way", async () => {
    // Cancelling after the answer went out would settle whatever came *next*.
    const { cancelInputRequest, user } = mount([ask()]);

    await user.click(screen.getByRole("button", { name: "Allow" }));
    await user.keyboard("{Escape}");

    expect(cancelInputRequest).not.toHaveBeenCalled();
  });
});

describe("the queue", () => {
  it("shows the head, and says how many wait behind it", () => {
    // Without the count, a second blocked call looks like the first failing to
    // close.
    mount([ask(), ask({ id: "approve_2", title: "Write a setting" })]);

    expect(screen.getByText("Run a shell command")).toBeDefined();
    expect(screen.getByText("1 more question after this one.")).toBeDefined();
  });

  it("says nothing about a queue of one", () => {
    mount([ask()]);

    expect(screen.queryByText(/more question/)).toBeNull();
  });

  it("draws nothing at all when nothing is waiting", () => {
    mount([]);

    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("answers a question that could not be named", async () => {
    // `id: null` is the shape a kernel too old to describe what it holds
    // produces. It must reach `resolve` as null, never as a placeholder the
    // server would stringify into an id matching nothing.
    const { resolve, user } = mount([ask({ id: null })]);

    await user.click(screen.getByRole("button", { name: "Deny" }));

    expect(resolve).toHaveBeenCalledWith(null, false);
  });
});
