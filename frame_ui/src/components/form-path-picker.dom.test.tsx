/** @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";
import { useState } from "react";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { FormPathPicker } from "@/components/form-path-picker";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";

const mocks = vi.hoisted(() => ({ submit: vi.fn(), chat: vi.fn(), sdk: vi.fn() }));
vi.mock("@/lib/client", () => ({ sdk: mocks.sdk, fileUrl: (path: string) => path }));
vi.mock("@assistant-ui/react", () => ({ useAui: () => ({ composer: () => ({ setText: mocks.chat }) }) }));
vi.mock("@/runtime/file-activity-provider", () => ({ useFileActivity: () => ({ view: vi.fn(), viewing: null }) }));

const identity = {};
function Harness({ mode = "text", initial = "", step = identity }: { mode?: string; initial?: string; step?: object }) {
  const [value, setValue] = useState(initial);
  return <Dialog open><DialogContent aria-describedby={undefined}>
    <DialogTitle>Settings</DialogTitle>
    <form onSubmit={(event) => { event.preventDefault(); mocks.submit(value); }}>
      <label htmlFor="value">Value</label>
      <textarea id="value" value={value} onChange={(event) => setValue(event.target.value)} />
      <FormPathPicker inputId="value" value={value} onChange={setValue} mode={mode} identity={step} disabled={false} />
      <button type="submit">Continue</button>
    </form>
  </DialogContent></Dialog>;
}
beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
  mocks.sdk.mockImplementation(async (type: string, args: { path?: string }) => {
    if (type === "paths.get") return "/data";
    if (type === "config.read") return [];
    if (type === "fs.stat") return { path: args.path, is_dir: true };
    if (type === "fs.list") return [
      { path: `${args.path}/notes`, name: "notes", is_dir: true },
      { path: `${args.path}/file.txt`, name: "file.txt", is_dir: false },
    ];
    return null;
  });
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("selects a folder into a text field without submitting Settings or mentioning it in chat", async () => {
  const user = userEvent.setup();
  render(<Harness initial="Read THIS please" />);
  const field = screen.getByLabelText("Value") as HTMLTextAreaElement;
  field.focus(); field.setSelectionRange(5, 9);
  await user.click(screen.getByRole("button", { name: "Choose path" }));
  await user.click(await screen.findByRole("button", { name: "Select notes" }));
  await waitFor(() => expect(field).toHaveFocus());
  expect(field).toHaveValue("Read /data/notes please");
  expect(mocks.submit).not.toHaveBeenCalled();
  expect(mocks.chat).not.toHaveBeenCalled();
  expect(screen.getByRole("dialog", { name: "Settings" })).toBeInTheDocument();
});

it("navigates with Enter inside the picker and selects the current folder into a JSON list", async () => {
  const user = userEvent.setup();
  render(<Harness mode="json" initial={'["/existing"]'} />);
  await user.click(screen.getByRole("button", { name: "Choose path" }));
  await screen.findByRole("button", { name: "Select notes" });
  await user.click(screen.getByRole("button", { name: "Edit folder path" }));
  await user.clear(screen.getByLabelText("Host folder path"));
  await user.type(screen.getByLabelText("Host folder path"), "/work{Enter}");
  await waitFor(() => expect(screen.getByRole("button", { name: "Use this folder" })).toBeEnabled());
  await user.click(screen.getByRole("button", { name: "Use this folder" }));
  await waitFor(() => expect(screen.getByLabelText("Value")).toHaveFocus());
  expect(JSON.parse((screen.getByLabelText("Value") as HTMLTextAreaElement).value)).toEqual(["/existing", "/work"]);
  expect(mocks.submit).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "Choose path" }));
  await user.click(await screen.findByRole("button", { name: "Select file.txt" }));
  expect(JSON.parse((screen.getByLabelText("Value") as HTMLTextAreaElement).value)).toEqual(["/existing", "/work", "/work/file.txt"]);
});

it("cancels only the picker and preserves the original field", async () => {
  const user = userEvent.setup();
  render(<Harness initial="keep this" />);
  await user.click(screen.getByRole("button", { name: "Choose path" }));
  await screen.findByRole("button", { name: "Select notes" });
  await user.keyboard("{Escape}");
  await waitFor(() => expect(screen.queryByRole("dialog", { name: "Choose a path" })).not.toBeInTheDocument());
  expect(screen.getByRole("dialog", { name: "Settings" })).toBeInTheDocument();
  expect(screen.getByLabelText("Value")).toHaveValue("keep this");
  expect(mocks.submit).not.toHaveBeenCalled();
});

it("closes a picker when the form step changes", async () => {
  const user = userEvent.setup();
  const { rerender } = render(<Harness />);
  await user.click(screen.getByRole("button", { name: "Choose path" }));
  await screen.findByRole("button", { name: "Select notes" });
  rerender(<Harness step={{}} />);
  await waitFor(() => expect(screen.queryByRole("dialog", { name: "Choose a path" })).not.toBeInTheDocument());
  expect(screen.getByLabelText("Value")).toHaveValue("");
});
