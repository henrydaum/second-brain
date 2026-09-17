/** @vitest-environment jsdom */
import "@testing-library/jest-dom/vitest";
import { useState } from "react";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { FileExplorerDialog } from "@/components/file-explorer-dialog";
import { FileViewerDialog } from "@/components/file-viewer-dialog";

const mocks = vi.hoisted(() => ({
  sdk: vi.fn(), setText: vi.fn(), view: vi.fn(), forgetFile: vi.fn(), forgetThumbnail: vi.fn(),
  draft: "", activity: {} as Record<string, unknown>, openExplorer: vi.fn(),
}));
vi.mock("@/lib/client", () => ({ sdk: mocks.sdk, fileUrl: (path: string) => path }));
vi.mock("@assistant-ui/react", () => ({ useAui: () => ({ composer: () => ({
  getState: () => ({ text: mocks.draft, attachments: ["existing attachment"] }), setText: mocks.setText,
}) }) }));
vi.mock("@/runtime/file-activity-provider", () => ({ useFileActivity: () => mocks.activity }));
vi.mock("@/runtime/file-explorer-provider", () => ({ useFileExplorer: () => ({ openExplorer: mocks.openExplorer }) }));
vi.mock("@/lib/files", async (original) => ({ ...await original<object>(), forgetFile: mocks.forgetFile }));
vi.mock("@/lib/thumbnails", () => ({ forgetThumbnail: mocks.forgetThumbnail }));
vi.mock("@/components/file-view", () => ({ FileView: ({ path }: { path: string }) => <div data-slot="file-view" tabIndex={0}>{path}</div> }));
vi.mock("@/components/markdown-mode", () => ({ MarkdownModePicker: () => null }));


const entry = (name: string, is_dir = false, root = "/data") => ({ name, path: `${root}/${name}`, is_dir, size: 0, mtime: 0 });
const initial = [entry("b.txt"), entry("notes", true), entry("a.txt")];
function Harness({ target }: { target?: { path: string; request: number } }) {
  const [open, setOpen] = useState(true);
  const [viewing, setViewing] = useState<{ paths: string[]; index: number; source?: "explorer" } | null>(null);
  mocks.activity = {
    viewing,
    view: (paths: string[], index: number, source?: "explorer") => { mocks.view(paths, index, source); setViewing({ paths, index, source }); },
    stepView: (by: number) => setViewing((value) => value && ({ ...value, index: (value.index + by + value.paths.length) % value.paths.length })),
    closeView: () => setViewing(null),
  };
  return <>
    <button onClick={() => setOpen(true)}>Open explorer</button>
    <button onClick={() => { setOpen(false); setViewing({ paths: ["/data/a.txt", "/data/b.txt"], index: 0 }); }}>Preview from chat</button>
    <textarea data-slot="chat-composer-input" defaultValue="Draft" />
    <FileExplorerDialog open={open} onOpenChange={setOpen} target={target} />
    {viewing && <FileViewerDialog />}
  </>;
}
beforeEach(() => {
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
  vi.clearAllMocks();
  mocks.draft = "";
  mocks.sdk.mockImplementation(async (type: string, args: { path?: string; name?: string; key?: string }) => {
    if (type === "paths.get") return args.name === "project" ? "/kernel" : "/data";
    if (type === "config.read") return args.key === "sync_directories" ? ["/sync", "/work"] : ["/work"];
    if (type === "fs.stat") return { path: args.path, is_dir: true };
    return args.path === "/data" ? initial : [entry("child.txt", false, args.path)];
  });
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("discovers shortcuts, navigates and filters without per-file requests", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await screen.findByRole("button", { name: "notes" });
  expect(mocks.sdk).toHaveBeenCalledWith("fs.list", { path: "/data", details: true });
  expect(mocks.sdk).not.toHaveBeenCalledWith("fs.stat", expect.anything());
  await user.click(screen.getByRole("button", { name: "Folder shortcuts" }));
  expect(screen.getByRole("menuitem", { name: "Kernel root: /kernel" })).toBeInTheDocument();
  expect(screen.getByRole("menuitem", { name: "Sync: /sync" })).toBeInTheDocument();
  expect(screen.getAllByRole("menuitem", { name: /\/work/ })).toHaveLength(1);
  await user.keyboard("{Escape}");
  await user.type(screen.getByRole("textbox", { name: /^Search in / }), "A.");
  expect(screen.queryByRole("button", { name: "b.txt" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "a.txt" })).toBeInTheDocument();
  await user.clear(screen.getByRole("textbox", { name: /^Search in / }));
  await user.click(screen.getByRole("button", { name: "Folder shortcuts" }));
  await user.click(screen.getByRole("menuitem", { name: /\/work$/ }));
  await screen.findByRole("button", { name: "child.txt" });
  expect(mocks.sdk).toHaveBeenCalledWith("fs.stat", { path: "/work" });
  expect(document.querySelector('[aria-current="location"]')).toHaveAttribute("title", "/work");
});

it.each(["", "Please read this", "Please read this\n"])("mentions into draft %j and focuses chat", async (draft) => {
  mocks.draft = draft;
  const user = userEvent.setup();
  render(<Harness />);
  await user.click(await screen.findByRole("button", { name: "Actions for a.txt" }));
  await user.click(screen.getByRole("menuitem", { name: "Mention in chat" }));
  expect(mocks.setText).toHaveBeenCalledWith(`${draft}${draft && !draft.endsWith("\n") ? "\n" : ""}"/data/a.txt"\n`);
  await waitFor(() => expect(document.querySelector("textarea")).toHaveFocus());
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("previews visible files, pages them, and returns with Escape and focus intact", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  const file = await screen.findByRole("button", { name: "a.txt" });
  await user.click(file);
  expect(mocks.view).toHaveBeenCalledWith(["/data/a.txt", "/data/b.txt"], 0, "explorer");
  expect(screen.queryByRole("button", { name: "Open containing folder" })).not.toBeInTheDocument();
  expect(mocks.forgetFile).toHaveBeenCalledWith("/data/b.txt");
  await user.click(screen.getByRole("button", { name: "Next file" }));
  expect(screen.getByRole("dialog", { name: "b.txt" })).toBeInTheDocument();
  await user.keyboard("{Escape}");
  await waitFor(() => expect(screen.queryByRole("dialog", { name: "b.txt" })).not.toBeInTheDocument());
  expect(screen.getByRole("dialog", { name: "File explorer" })).toBeInTheDocument();
  await waitFor(() => expect(file).toHaveFocus());
  await user.keyboard("{Escape}");
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
});

it("retains the last directory and filter on reopening and refreshes configuration", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await user.click(await screen.findByRole("button", { name: "notes" }));
  await screen.findByRole("button", { name: "child.txt" });
  await user.type(screen.getByRole("textbox", { name: /^Search in / }), "child");
  await user.keyboard("{Escape}");
  await user.click(screen.getByRole("button", { name: "Open explorer" }));
  await screen.findByRole("button", { name: "child.txt" });
  expect(document.querySelector('[aria-current="location"]')).toHaveAttribute("title", "/data/notes");
  expect(screen.getByRole("textbox", { name: /^Search in / })).toHaveValue("child");
  expect(mocks.sdk.mock.calls.filter(([type]) => type === "config.read")).toHaveLength(4);
});

it("keeps the previous listing on failed navigation and retries the failed target", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await screen.findByRole("button", { name: "a.txt" });
  mocks.sdk.mockRejectedValueOnce(new Error("Access denied"));
  await user.click(screen.getByRole("button", { name: "notes" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Access denied");
  expect(screen.getByRole("button", { name: "a.txt" })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Retry" }));
  await screen.findByRole("button", { name: "child.txt" });
});

it("ignores stale listings when a newer navigation completes", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await screen.findByRole("button", { name: "notes" });
  let resolve!: (entries: typeof initial) => void;
  mocks.sdk.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
  await user.click(screen.getByRole("button", { name: "notes" }));
  await user.click(screen.getByRole("button", { name: "Edit folder path" }));
  fireEvent.change(screen.getByLabelText("Host folder path"), { target: { value: "/new" } });
  await user.keyboard("{Enter}");
  await screen.findByRole("button", { name: "child.txt" });
  await act(async () => resolve([entry("stale.txt")]));
  expect(document.querySelector('[aria-current="location"]')).toHaveAttribute("title", "/new");
  expect(screen.queryByRole("button", { name: "stale.txt" })).not.toBeInTheDocument();
});

it("handles absent writable folders and rejects relative paths without a request", async () => {
  mocks.sdk.mockImplementation(async (type: string) => type === "paths.get" ? "/data" : type === "config.read" ? null : []);
  const user = userEvent.setup();
  render(<Harness />);
  await screen.findByText("This folder is empty.");
  const count = mocks.sdk.mock.calls.length;
  await user.click(screen.getByRole("button", { name: "Edit folder path" }));
  fireEvent.change(screen.getByLabelText("Host folder path"), { target: { value: "relative" } });
  await user.keyboard("{Enter}");
  expect(await screen.findByRole("alert")).toHaveTextContent("absolute path");
  expect(mocks.sdk).toHaveBeenCalledTimes(count);
});

it("keeps browsing available when shortcut discovery fails", async () => {
  mocks.sdk.mockImplementation(async (type: string) => {
    if (type === "config.read") throw new Error("Unavailable");
    return type === "paths.get" ? "/data" : initial;
  });
  render(<Harness />);
  await screen.findByRole("button", { name: "a.txt" });
  expect(screen.getByText(/Some folder shortcuts/)).toBeInTheDocument();
});

it("rejects a manually entered file without listing it", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await screen.findByRole("button", { name: "a.txt" });
  mocks.sdk.mockResolvedValueOnce({ path: "/data/a.txt", is_dir: false });
  await user.click(screen.getByRole("button", { name: "Edit folder path" }));
  fireEvent.change(screen.getByLabelText("Host folder path"), { target: { value: "/data/a.txt" } });
  await user.keyboard("{Enter}");
  expect(await screen.findByRole("alert")).toHaveTextContent("This path is a file");
  expect(mocks.sdk).not.toHaveBeenCalledWith("fs.list", { path: "/data/a.txt", details: true });
});

it("discards a pending navigation after closure", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await screen.findByRole("button", { name: "notes" });
  let resolve!: (entries: typeof initial) => void;
  mocks.sdk.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
  await user.click(screen.getByRole("button", { name: "notes" }));
  await user.keyboard("{Escape}");
  await act(async () => resolve([entry("stale.txt")]));
  await user.click(screen.getByRole("button", { name: "Open explorer" }));
  await screen.findByRole("button", { name: "a.txt" });
  expect(document.querySelector('[aria-current="location"]')).toHaveAttribute("title", "/data");
  expect(screen.queryByRole("button", { name: "stale.txt" })).not.toBeInTheDocument();
});

it("copies the full path through the menu and reports clipboard failures", async () => {
  const user = userEvent.setup();
  const write = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue();
  render(<Harness />);
  await user.click(await screen.findByRole("button", { name: "Actions for a.txt" }));
  await user.click(screen.getByRole("menuitem", { name: "Copy path" }));
  expect(write).toHaveBeenCalledWith("/data/a.txt");
  expect((await screen.findAllByText("Path copied")).some((element) => element.classList.contains("sr-only"))).toBe(true);
  write.mockRejectedValueOnce(new Error("Denied"));
  await user.click(screen.getByRole("button", { name: "Actions for a.txt" }));
  await user.click(screen.getByRole("menuitem", { name: "Copy path" }));
  expect((await screen.findAllByText(/Could not copy path/)).some((element) => element.classList.contains("sr-only"))).toBe(true);
  write.mockRestore();
});

it("reveals a requested file, clears filtering, and refreshes the folder navigated to afterwards", async () => {
  const user = userEvent.setup();
  const { rerender } = render(<Harness />);
  await screen.findByRole("button", { name: "a.txt" });
  await user.type(screen.getByRole("textbox", { name: /^Search in / }), "no matches");
  rerender(<Harness target={{ path: "/work/child.txt", request: 1 }} />);
  const file = await screen.findByRole("button", { name: "child.txt" });
  await waitFor(() => expect(file).toHaveFocus());
  expect(document.querySelector('[aria-current="location"]')).toHaveAttribute("title", "/work");
  expect(screen.getByRole("textbox", { name: /^Search in / })).toHaveValue("");
  expect(file.closest("li")).toHaveClass("ring-2");
  await user.click(screen.getByRole("button", { name: "Folder shortcuts" }));
  await user.click(screen.getByRole("menuitem", { name: /\/data$/ }));
  await screen.findByRole("button", { name: "a.txt" });
  await user.click(screen.getByRole("button", { name: "Refresh folder" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "a.txt" })).toBeEnabled());
  expect(document.querySelector('[aria-current="location"]')).toHaveAttribute("title", "/data");
});

it("keeps the explorer open when Escape dismisses the actions menu", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await user.click(await screen.findByRole("button", { name: "Actions for a.txt" }));
  await user.keyboard("{Escape}");
  expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  expect(screen.getByRole("dialog", { name: "File explorer" })).toBeInTheDocument();
});

it("mentions a folder without navigating into it", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await user.click(await screen.findByRole("button", { name: "Actions for notes" }));
  await user.click(screen.getByRole("menuitem", { name: "Mention in chat" }));
  expect(mocks.setText).toHaveBeenCalledWith('"/data/notes"\n');
  expect(mocks.sdk).not.toHaveBeenCalledWith("fs.list", { path: "/data/notes", details: true });
  await waitFor(() => expect(document.querySelector("textarea")).toHaveFocus());
});

it("navigates breadcrumbs and goes back across folder and shortcut changes", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await user.click(await screen.findByRole("button", { name: "notes" }));
  await screen.findByRole("button", { name: "child.txt" });
  await user.click(screen.getByRole("button", { name: "data" }));
  await screen.findByRole("button", { name: "a.txt" });
  await user.click(screen.getByRole("button", { name: "Back" }));
  await screen.findByRole("button", { name: "child.txt" });
  expect(document.querySelector('[aria-current="location"]')).toHaveAttribute("title", "/data/notes");
  await user.click(screen.getByRole("button", { name: "Folder shortcuts" }));
  await user.click(screen.getByRole("menuitem", { name: /\/work$/ }));
  await waitFor(() => expect(document.querySelector('[aria-current="location"]')).toHaveAttribute("title", "/work"));
  await user.click(screen.getByRole("button", { name: "Refresh folder" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Back" })).toBeEnabled());
  await user.click(screen.getByRole("button", { name: "Back" }));
  await waitFor(() => expect(document.querySelector('[aria-current="location"]')).toHaveAttribute("title", "/data/notes"));
  await user.click(screen.getByRole("button", { name: "Back" }));
  await screen.findByRole("button", { name: "a.txt" });
  expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();
});

it("does not consume Back history when its request fails", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await user.click(await screen.findByRole("button", { name: "notes" }));
  await screen.findByRole("button", { name: "child.txt" });
  mocks.sdk.mockRejectedValueOnce(new Error("Unavailable"));
  await user.click(screen.getByRole("button", { name: "Back" }));
  await screen.findByRole("alert");
  expect(document.querySelector('[aria-current="location"]')).toHaveAttribute("title", "/data/notes");
  await user.click(screen.getByRole("button", { name: "Retry" }));
  await screen.findByRole("button", { name: "a.txt" });
  expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();
});

it("reveals the currently viewed file after paging", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await screen.findByRole("button", { name: "a.txt" });
  await user.click(screen.getByRole("button", { name: "Close" }));
  await user.click(screen.getByRole("button", { name: "Preview from chat" }));
  await user.click(screen.getByRole("button", { name: "Next file" }));
  await user.click(screen.getByRole("button", { name: "Open containing folder" }));
  expect(mocks.openExplorer).toHaveBeenCalledWith("/data/b.txt");
  expect(screen.queryByRole("dialog", { name: "b.txt" })).not.toBeInTheDocument();
});

it("walks Forward history and clears it only after a successful new destination", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await user.click(await screen.findByRole("button", { name: "notes" }));
  await screen.findByRole("button", { name: "child.txt" });
  await user.click(screen.getByRole("button", { name: "Back" }));
  await screen.findByRole("button", { name: "a.txt" });
  await user.click(screen.getByRole("button", { name: "Forward" }));
  await screen.findByRole("button", { name: "child.txt" });
  expect(screen.getByRole("button", { name: "Forward" })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Back" }));
  await screen.findByRole("button", { name: "a.txt" });
  mocks.sdk.mockRejectedValueOnce(new Error("Unavailable"));
  await user.click(screen.getByRole("button", { name: "Forward" }));
  await screen.findByRole("alert");
  expect(screen.getByRole("button", { name: "Forward" })).toBeEnabled();
  await user.click(screen.getByRole("button", { name: "Folder shortcuts" }));
  await user.click(screen.getByRole("menuitem", { name: /\/work$/ }));
  await screen.findByRole("button", { name: "child.txt" });
  expect(screen.getByRole("button", { name: "Forward" })).toBeDisabled();
});

it("switches between breadcrumbs and the selected path, with Escape cancelling only editing", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await screen.findByRole("button", { name: "a.txt" });
  expect(screen.queryByLabelText("Host folder path")).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Edit folder path" }));
  const input = screen.getByLabelText("Host folder path") as HTMLInputElement;
  expect(input).toHaveValue("/data");
  expect(input.selectionStart).toBe(0);
  expect(input.selectionEnd).toBe(5);
  await user.type(input, "/discarded");
  await user.keyboard("{Escape}");
  expect(screen.queryByLabelText("Host folder path")).not.toBeInTheDocument();
  expect(screen.getByRole("dialog", { name: "File explorer" })).toBeInTheDocument();
  expect(document.querySelector('[aria-current="location"]')).toHaveAttribute("title", "/data");
  await user.click(screen.getByRole("button", { name: "Edit folder path" }));
  await user.clear(screen.getByLabelText("Host folder path"));
  await user.type(screen.getByLabelText("Host folder path"), "/work{Enter}");
  await screen.findByRole("button", { name: "child.txt" });
  expect(screen.queryByLabelText("Host folder path")).not.toBeInTheDocument();
  expect(screen.getByRole("textbox", { name: "Search in work" })).toBeInTheDocument();
});

it("restores the list position after mentioning a file and reopening", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await screen.findByRole("button", { name: "a.txt" });
  const list = document.querySelector('[data-slot="explorer-list"]') as HTMLElement;
  list.scrollTop = 850;
  fireEvent.scroll(list);
  await user.click(screen.getByRole("button", { name: "Actions for a.txt" }));
  await user.click(screen.getByRole("menuitem", { name: "Mention in chat" }));
  await user.click(screen.getByRole("button", { name: "Open explorer" }));
  await screen.findByRole("button", { name: "a.txt" });
  await waitFor(() => expect(document.querySelector('[data-slot="explorer-list"]')?.scrollTop).toBe(850));
});

it("keeps a separate scroll position for each directory", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  await screen.findByRole("button", { name: "notes" });
  const list = document.querySelector('[data-slot="explorer-list"]') as HTMLElement;
  list.scrollTop = 600;
  fireEvent.scroll(list);
  await user.click(screen.getByRole("button", { name: "notes" }));
  await screen.findByRole("button", { name: "child.txt" });
  expect(list.scrollTop).toBe(0);
  await user.click(screen.getByRole("button", { name: "Back" }));
  await screen.findByRole("button", { name: "a.txt" });
  expect(list.scrollTop).toBe(600);
});

it("places the path cursor at the end on touch devices", async () => {
  vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: true })));
  const user = userEvent.setup();
  render(<Harness />);
  await screen.findByRole("button", { name: "a.txt" });
  await user.click(screen.getByRole("button", { name: "Edit folder path" }));
  const input = screen.getByLabelText("Host folder path") as HTMLInputElement;
  expect(input.selectionStart).toBe(input.value.length);
  expect(input.selectionEnd).toBe(input.value.length);
});
