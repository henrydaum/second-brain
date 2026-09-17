/** @vitest-environment jsdom */

import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { FormFieldPayload } from "@/lib/events";

const say = vi.fn(async () => true);
const dismissCommand = vi.fn();
let form: FormFieldPayload | undefined;

vi.mock("@/runtime/provider", () => ({
  useSession: () => ({ state: { command: undefined, form }, say, dismissCommand }),
  useApprovals: () => ({ inputRequests: [] }),
}));

const { CommandPanel } = await import("@/components/command-panel");

/** The step Settings shows when adding a language model: one backend to pick,
 *  and it is already the field's default. */
const backendStep = (): FormFieldPayload => ({
  field: { name: "backend", default: "LiteLLMBackend" },
  collected: { model_name: "add" },
  display: {
    prompt: "Choose how Second Brain should connect to this model.",
    assist: "Select an option.",
    choices: [{ value: "LiteLLMBackend" }],
    allow_back: true,
  },
});

beforeEach(() => {
  vi.clearAllMocks();
  form = backendStep();
});
afterEach(cleanup);

it("offers path picking for text and JSON fields, with valid JSON array defaults", () => {
  form = { field: { name: "fs_writable_dirs", default: ["/work"] }, display: { prompt: "Directories", input_mode: "json" } };
  render(<CommandPanel />);
  expect(screen.getByRole("textbox")).toHaveValue('[\n  "/work"\n]');
  expect(screen.getByRole("button", { name: "Choose path" })).toBeEnabled();
});

it("does not offer path picking for numeric fields", () => {
  form = { field: { name: "count" }, display: { prompt: "Count", input_mode: "number" } };
  render(<CommandPanel />);
  expect(screen.queryByRole("button", { name: "Choose path" })).not.toBeInTheDocument();
});

it("uses a multiline field for long text settings", () => {
  form = { field: { name: "subagent_prompt", default: "A long prompt" }, display: { prompt: "Prompt", input_mode: "text" } };
  render(<CommandPanel />);
  const field = screen.getByRole("textbox");
  expect(field.tagName).toBe("TEXTAREA");
  expect(field).toHaveAttribute("rows", "1");
});

describe("CommandPanel choices", () => {
  it("sends the pre-selected default when it is picked again", async () => {
    const user = userEvent.setup();
    render(<CommandPanel />);

    const choice = screen.getByRole("radio", { name: "LiteLLMBackend" });
    expect(choice).toBeChecked();
    // No Continue button on a choice step — picking is what continues, so
    // picking the already-checked choice has to be what sends it.
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull();

    await user.click(screen.getByText("LiteLLMBackend"));

    expect(say).toHaveBeenCalledTimes(1);
    expect(say).toHaveBeenCalledWith("LiteLLMBackend");
  });

  it("sends a choice once, not twice, when the selection changes", async () => {
    form = {
      ...backendStep(),
      display: {
        ...backendStep().display!,
        choices: [{ value: "LiteLLMBackend" }, { value: "OllamaBackend" }],
      },
    };
    const user = userEvent.setup();
    render(<CommandPanel />);

    await user.click(screen.getByText("OllamaBackend"));

    expect(say).toHaveBeenCalledTimes(1);
    expect(say).toHaveBeenCalledWith("OllamaBackend");
  });

  it("submits the selected choice on Enter", async () => {
    const user = userEvent.setup();
    render(<CommandPanel />);

    screen.getByRole("radio", { name: "LiteLLMBackend" }).focus();
    await user.keyboard("{Enter}");

    expect(say).toHaveBeenCalledWith("LiteLLMBackend");
  });
});
