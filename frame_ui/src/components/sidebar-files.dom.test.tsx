// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { useState } from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { FileActivityContext, type FileActivity } from "@/runtime/file-activity-provider";

const newConversation = vi.fn(async () => {});
const view = vi.fn();
let desktop = true;
vi.mock("@/runtime/provider", () => ({
  useConversations: () => ({
    conversations: [{id: 1, title: "Current chat"}, {id: 2, title: "Another chat"}],
    conversationsLoaded: true, conversationId: 1,
    openConversation: vi.fn(async () => {}), newConversation,
    deleteConversation: vi.fn(), loadMoreConversations: vi.fn(),
    conversationsHasMore: false, conversationCategories: [],
    conversationFilter: {type: "category", category: null}, setConversationFilter: vi.fn(),
  }),
  useApprovals: () => ({inputRequests: []}),
  useSession: () => ({status: "open", state: {command: null, form: null, typing: false}}),
  useSettings: () => ({settingsOpen: false, setSettingsOpen: vi.fn()}),
}));
vi.mock("@/lib/media", () => ({MD_QUERY: "desktop", useMediaQuery: () => desktop}));
vi.mock("@/components/lazy-settings", () => ({preloadSettings: vi.fn(), SettingsDialog: () => null}));
vi.mock("@/components/lazy-file-explorer", () => ({preloadFileExplorer: vi.fn()}));
vi.mock("@/components/lazy-file-viewer", () => ({preloadFileViewer: vi.fn()}));
vi.mock("@/runtime/file-explorer-provider", () => ({useFileExplorer: () => ({openExplorer: vi.fn()})}));
vi.mock("@/components/file-kind-icon", () => ({FileThumbnail: () => null}));
const { ConversationSidebar } = await import("@/components/conversation-sidebar");

function Harness({empty = false, failure = null}: {empty?: boolean; failure?: string | null}) {
  const [filesOpen, setFilesOpen] = useState(false);
  const [focusRequest, setFocusRequest] = useState(0);
  const [navOpen, setNavOpen] = useState(true);
  const entries = empty ? [] : [{path: "/current.txt", ts: 1, effect: "shown" as const, gone: false, edits: 1, viaShell: false}];
  const activity = {
    filesOpen, setFilesOpen, focusRequest, focusTurn: null, clearFocus: vi.fn(),
    entries, sections: [], total: entries.length, failure,
    view, viewing: null,
  } as unknown as FileActivity;
  return <FileActivityContext value={activity}>
    <button onClick={() => {setFilesOpen(true); setFocusRequest(n => n + 1);}}>Message files</button>
    <button onClick={() => setNavOpen(true)}>Open navigation</button>
    <ConversationSidebar open={navOpen} onOpenChange={setNavOpen} />
  </FileActivityContext>;
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  desktop = true;
  vi.stubGlobal("matchMedia", vi.fn(() => ({matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn()})));
});
afterEach(() => {cleanup(); vi.unstubAllGlobals();});

it("switches only the list, retains its scroll, and keeps New chat working in Files", async () => {
  const user = userEvent.setup();
  render(<Harness />);
  const chats = screen.getByRole("navigation", {name: "Conversations"});
  chats.scrollTop = 120;
  fireEvent.scroll(chats);
  const newChat = screen.getAllByRole("button", {name: "New chat"})[0];
  await user.click(screen.getByRole("button", {name: "Files", exact: true}));
  await screen.findByText("current.txt");
  expect(chats).not.toBeVisible();
  expect(newChat).toBeVisible();
  expect(newChat).toBeEnabled();
  expect(screen.getAllByRole("button", {name: "File explorer"})[0]).toBeVisible();
  const files = document.querySelector<HTMLElement>('[data-slot="conversation-files-list"]')!;
  files.scrollTop = 75;
  fireEvent.scroll(files);
  await user.click(screen.getByRole("button", {name: "Chats", exact: true}));
  expect(chats).toBeVisible();
  expect(chats.scrollTop).toBe(120);
  await user.click(screen.getByRole("button", {name: "Files", exact: true}));
  expect(files.scrollTop).toBe(75);
  await user.click(newChat);
  expect(newConversation).toHaveBeenCalledOnce();
});

it("opens a file through the existing viewer and handles empty/error lists", async () => {
  const user = userEvent.setup();
  const result = render(<Harness />);
  await user.click(screen.getByRole("button", {name: "Files", exact: true}));
  await user.click(await screen.findByRole("button", {name: "current.txt"}));
  expect(view).toHaveBeenCalledWith(["/current.txt"], 0);
  result.rerender(<Harness empty />);
  expect(screen.getByText(/No files in this chat yet/)).toBeVisible();
  result.rerender(<Harness failure="File history could not be loaded." />);
  expect(screen.getByRole("status")).toHaveTextContent("File history could not be loaded.");
});

it("reveals a collapsed left sidebar for repeated message-file requests", async () => {
  const user = userEvent.setup();
  localStorage.setItem("second-brain:sidebar-collapsed", "true");
  render(<Harness />);
  await user.click(screen.getByRole("button", {name: "Message files"}));
  expect(await screen.findByText("current.txt")).toBeVisible();
  await user.click(screen.getAllByRole("button", {name: "Hide sidebar"})[1]);
  expect(screen.getByText("current.txt")).not.toBeVisible();
  await user.click(screen.getByRole("button", {name: "Message files"}));
  expect(screen.getByText("current.txt")).toBeVisible();
});

it("uses the left mobile sheet and dismisses it when opening a file", async () => {
  desktop = false;
  const user = userEvent.setup();
  render(<Harness />);
  await user.click(screen.getByRole("button", {name: "Files", exact: true}));
  await user.click(await screen.findByRole("button", {name: "current.txt"}));
  expect(view).toHaveBeenCalledWith(["/current.txt"], 0);
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  await user.click(screen.getByRole("button", {name: "Message files"}));
  expect(await screen.findByRole("dialog", {name: "Chats and files"})).toBeVisible();
  expect(screen.getByRole("button", {name: "Files", exact: true})).toHaveAttribute("aria-pressed", "true");
});
