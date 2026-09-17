/** @vitest-environment jsdom */

import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const setModel = vi.fn();
const openSettings = vi.fn();
const say = vi.fn();

// Mutable so a test can put the composer mid-turn without a second mock.
let typing = false;

vi.mock("@/runtime/provider", () => ({
  useSettings: () => ({ openSettings }),
  useSession: () => ({ say, state: { typing } }),
  useModels: () => ({
    models: [
      { model_name: "anthropic/sonnet-4.6", loaded: true },
      { model_name: "openrouter/sonnet-4.6", loaded: false },
    ],
    modelName: "anthropic/sonnet-4.6",
    agentProfile: "researcher",
    modelsLoading: false,
    modelsFailure: false,
    switchingModel: false,
    setModel,
  }),
}));

const { ModelSelector, compactModelName } = await import(
  "@/components/model-selector"
);

beforeEach(() => {
  vi.clearAllMocks();
  typing = false;
});
afterEach(cleanup);

describe("compactModelName", () => {
  it.each([
    ["anthropic/sonnet-4.6", "Sonnet 4.6"],
    ["openai/gpt-5.4", "GPT 5.4"],
    ["minimax/minimax-m3", "MiniMax M3"],
    ["deepseek-v3.2", "DeepSeek V3.2"],
    ["openai/gpt-oss-120b", "GPT OSS 120B"],
    ["claude_opus", "Claude Opus"],
    ["gateway/team/model-name", "Model Name"],
    ["gateway/team/", "Team"],
    ["", "Select model"],
  ])("humanizes %s", (input, expected) => {
    expect(compactModelName(input)).toBe(expected);
  });
});

describe("ModelSelector", () => {
  it("keeps exact provider-prefixed IDs in the menu", async () => {
    const user = userEvent.setup();
    render(<ModelSelector />);

    expect(screen.getByRole("button", { name: "Model: anthropic/sonnet-4.6" }))
      .toHaveTextContent("Sonnet 4.6");
    await user.click(screen.getByRole("button", { name: /Model:/ }));

    expect(screen.getByRole("menuitemradio", { name: "anthropic/sonnet-4.6" }))
      .toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("menuitemradio", { name: "openrouter/sonnet-4.6" }))
      .toBeInTheDocument();
    expect(screen.getByText("Agent profile:").parentElement)
      .toHaveTextContent("researcher");
  });

  it("selects through the SDK state action", async () => {
    const user = userEvent.setup();
    render(<ModelSelector />);
    await user.click(screen.getByRole("button", { name: /Model:/ }));
    await user.click(
      screen.getByRole("menuitemradio", { name: "openrouter/sonnet-4.6" }),
    );
    expect(setModel).toHaveBeenCalledWith("openrouter/sonnet-4.6");
  });
});

describe("the configure link", () => {
  const openPanel = async () => {
    const user = userEvent.setup();
    render(<ModelSelector />);
    await user.click(screen.getByRole("button", { name: /Model:/ }));
    return user;
  };

  const clickConfigure = async (user: ReturnType<typeof userEvent.setup>) =>
    user.click(screen.getByRole("menuitem", { name: "Configure this model" }));

  it("opens the selected model's parameters, not just the section", async () => {
    // Opening the section alone is one click from the composer and one more
    // from there — which is what the panel already offered. `/llm <model> edit`
    // fills the two steps before the one worth arriving at, so the form lands
    // on the field list: every setting this profile has, each configured
    // parameter with its value, and Add a parameter.
    const user = await openPanel();
    await clickConfigure(user);

    expect(openSettings).toHaveBeenCalledWith("plugins");
    expect(say).toHaveBeenCalledWith("/llm 'anthropic/sonnet-4.6' edit");
  });

  it("quotes the model name for the kernel's parser", async () => {
    // Model names are the user's own text and routinely carry `/`. The kernel
    // lexes a command line with `shlex`, so the name is a single quoted
    // argument rather than something it might split.
    const user = await openPanel();
    await clickConfigure(user);

    const [line] = say.mock.calls[0] as [string];
    expect(line).toBe("/llm 'anthropic/sonnet-4.6' edit");
  });

  it("closes the panel on the way", async () => {
    const user = await openPanel();
    await clickConfigure(user);

    // Settings opens over the composer, so a panel left standing would be
    // behind it and still open when it closes.
    await waitFor(() =>
      expect(
        screen.queryByRole("menuitem", { name: "Configure this model" }),
      ).not.toBeInTheDocument(),
    );
  });

  it("stays out of the way while a turn is running", async () => {
    // Every way back out of Settings submits `/cancel` when something is
    // running, and that lands on the turn. Ending a turn is a decision for the
    // composer's Stop button, not something to arrive at by following a link
    // about a parameter.
    typing = true;
    const user = await openPanel();
    const item = screen.getByRole("menuitem", { name: "Configure this model" });

    expect(item).toHaveAttribute("data-disabled");
    await user.click(item);
    expect(say).not.toHaveBeenCalled();
    expect(openSettings).not.toHaveBeenCalled();
  });

  it("offers no reasoning control of its own", async () => {
    // The panel used to carry one: four values written into `reasoning_effort`,
    // which guessed the parameter's name, whether this model takes it, and
    // what its values are. Settings shows what the backend actually reports.
    await openPanel();

    expect(screen.queryByText(/Reasoning/)).not.toBeInTheDocument();
  });
});
