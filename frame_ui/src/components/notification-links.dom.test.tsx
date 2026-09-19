// @vitest-environment jsdom

/**
 * Where a notification sends you, in a DOM.
 *
 * The reducer underneath is covered without one, and the rows themselves are
 * ordinary markup — but the two links out of the panel are not, for the same
 * reason `input-request-dialog.dom.test.tsx` needs a DOM: what is worth pinning
 * is a contract with Radix (a popover that closes, a section that survives being
 * asked for) rather than any function written here.
 *
 * The provider is stubbed rather than rendered. What is under test is which
 * callback fires with which argument, and standing up a real provider would put
 * an SSE connection between the test and the assertion.
 */

import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { NotificationPanel } from "@/components/notification-panel";
import { SettingsDialogContent } from "@/components/settings-dialog";
import type { Notification } from "@/lib/notifications";
import * as provider from "@/runtime/domains";

/**
 * The one Request these links make: "can `/config` open this setting?".
 *
 * Only `sdk` is replaced. The rest of `client.ts` — `fileUrl`, `THREAD` — is
 * loaded by the markdown renderers underneath the panel, and a bare stub would
 * take those away from a test that has nothing to say about them.
 */
const sdk = vi.fn();
vi.mock("@/lib/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/client")>()),
  sdk: (...args: unknown[]) => sdk(...args),
}));

afterEach(cleanup);

/** Browsable unless a test says otherwise: `config.read` with a `key` answers
 *  the one matching catalogue entry, or nothing for a hidden setting. */
beforeEach(() => {
  sdk.mockReset();
  sdk.mockImplementation(async (_type: string, args: { key: string }) => [
    { key: args.key },
  ]);
});

const row = (over: Partial<Notification> = {}): Notification => ({
  id: 1,
  ts: 1_786_384_521,
  title: "Settings changed",
  body: "http_port",
  source: "config",
  source_id: "core",
  level: "info",
  session_key: null,
  conversation_id: null,
  user_id: null,
  read_at: null,
  ...over,
});

const runSettingsAction = async (action: () => void | Promise<void>) => {
  await action();
  return true;
};

function stub(over: Partial<provider.SecondBrain>) {
  const spies = {
    openSettings: vi.fn(),
    openConversation: vi.fn().mockResolvedValue(undefined),
    setNotificationsOpen: vi.fn(),
    markNotificationsRead: vi.fn().mockResolvedValue(undefined),
    clearSettingsRequest: vi.fn(),
    say: vi.fn().mockResolvedValue(true),
    dismissCommand: vi.fn(),
    resolve: vi.fn().mockResolvedValue(undefined),
    cancelInputRequest: vi.fn().mockResolvedValue(undefined),
  };
  const value = {
    notifications: [],
    notificationQueue: [],
    unread: 0,
    notificationsFailure: null,
    notificationsOpen: true,
    conversationId: 5,
    commands: [],
    state: { turns: [], typing: false },
    submitting: false,
    inputRequests: [],
    settingsRequest: null,
    ...spies,
    ...over,
  } as unknown as provider.SecondBrain;
  vi.spyOn(provider, "useNotifications").mockReturnValue({
    notificationQueue: value.notificationQueue,
    notifications: value.notifications,
    unread: value.unread,
    notificationsFailure: value.notificationsFailure,
    dismissQueuedNotification: value.dismissQueuedNotification,
    markNotificationsRead: value.markNotificationsRead,
    notificationsOpen: value.notificationsOpen,
    setNotificationsOpen: value.setNotificationsOpen,
  });
  vi.spyOn(provider, "useConversations").mockReturnValue({
    conversations: value.conversations,
    conversationsLoaded: true,
    conversationId: value.conversationId,
    openConversation: value.openConversation,
    newConversation: value.newConversation,
    deleteConversation: value.deleteConversation,
    openConversationRow: null,
    renameConversation: vi.fn(),
    categoriseConversation: vi.fn(),
    conversationsHasMore: false,
    scrollbackHasMore: false,
    loadingOlderMessages: false,
    loadOlderMessages: vi.fn(),
    loadMoreConversations: vi.fn(),
    conversationCategories: [],
    conversationFilter: { type: "category", category: null } as const,
    setConversationFilter: vi.fn(),
  });
  vi.spyOn(provider, "useSettings").mockReturnValue({
    commands: value.commands,
    settingsOpen: value.settingsOpen,
    setSettingsOpen: value.setSettingsOpen,
    openSettings: value.openSettings,
    settingsRequest: value.settingsRequest,
    clearSettingsRequest: value.clearSettingsRequest,
  });
  vi.spyOn(provider, "useSession").mockReturnValue({
    status: value.status,
    state: value.state,
    submitting: value.submitting,
    say: value.say,
    report: value.report,
    dismissError: value.dismissError,
    dismissCommand: value.dismissCommand,
  });
  vi.spyOn(provider, "useApprovals").mockReturnValue({
    inputRequests: value.inputRequests,
    resolve: value.resolve,
    cancelInputRequest: value.cancelInputRequest,
  });
  return { ...spies, user: userEvent.setup() };
}

describe("the links out of a notification", () => {
  it("drills into the setting the notification named", async () => {
    // Naming the setting is what makes `/config` skip its category and plugin
    // steps and land on that setting's own page. `all` is the category that
    // matches wherever the setting actually lives.
    const { say, openSettings, user } = stub({ notifications: [row()] });
    render(<NotificationPanel />);

    await user.click(screen.getByRole("button", { name: "Open settings" }));

    await waitFor(() =>
      expect(say).toHaveBeenCalledWith("/config all http_port"),
    );
    // And the dialog goes up now rather than a round trip later — which is also
    // where a failed submit lands.
    expect(openSettings).toHaveBeenCalledWith("kernel");
  });

  it("sends the setting nowhere `/config` refuses to go", async () => {
    // `hidden` settings are announced like any other and are not in `/config`'s
    // catalogue, so the command comes back as the enum of every settable key
    // printed into the chat. `scheduled_jobs` belongs to the timekeeper and has
    // no page of its own, so the section is as far as this can honestly go.
    sdk.mockResolvedValue([]);
    const { say, openSettings, user } = stub({
      notifications: [row({ body: "scheduled_jobs" })],
    });
    render(<NotificationPanel />);

    await user.click(screen.getByRole("button", { name: "Open settings" }));

    await waitFor(() => expect(sdk).toHaveBeenCalled());
    expect(say).not.toHaveBeenCalled();
    expect(openSettings).toHaveBeenCalledWith("kernel");
  });

  it("sends a setting managed elsewhere to the section that manages it", async () => {
    // `default_llm_profile` is `/llm`'s to edit, and `/llm` lives on Packages.
    // `/config` is the one form that provably does not list it, so asking the
    // kernel about it is not worth a round trip when the answer is known.
    const { say, openSettings, user } = stub({
      notifications: [row({ body: "default_llm_profile" })],
    });
    render(<NotificationPanel />);

    await user.click(screen.getByRole("button", { name: "Open settings" }));

    expect(openSettings).toHaveBeenCalledWith("packages");
    expect(say).not.toHaveBeenCalled();
    expect(sdk).not.toHaveBeenCalled();
  });

  it("offers the section, once, when several settings changed", async () => {
    // Not a link each. The row is a line of 10px text, and the body directly
    // above already names them — so several links would repeat it in a less
    // readable form.
    const { say, openSettings, user } = stub({
      notifications: [row({ body: "http_port, http_token" })],
    });
    render(<NotificationPanel />);

    const links = screen.getAllByRole("button", { name: "Open settings" });
    expect(links).toHaveLength(1);

    await user.click(links[0]);

    // The section, with no attempt to guess which of the two you meant.
    expect(say).not.toHaveBeenCalled();
    expect(openSettings).toHaveBeenCalledWith("kernel");
  });

  it("falls back to the section when the body is not setting names", async () => {
    // `body` is a convention rather than a field. Prose must not be pasted onto
    // a command line.
    const { say, openSettings, user } = stub({
      notifications: [row({ body: "Several settings were updated." })],
    });
    render(<NotificationPanel />);

    await user.click(screen.getByRole("button", { name: "Open settings" }));

    expect(say).not.toHaveBeenCalled();
    expect(openSettings).toHaveBeenCalledWith("kernel");
  });

  it("closes the panel on the way out", async () => {
    // Otherwise the first thing you do in Settings is dismiss the popover
    // standing over it.
    const { setNotificationsOpen, user } = stub({ notifications: [row()] });
    render(<NotificationPanel />);

    await user.click(screen.getByRole("button", { name: "Open settings" }));

    expect(setNotificationsOpen).toHaveBeenCalledWith(false);
  });

  it("offers nothing for a source with nowhere to send you", () => {
    // Most notifications map to no section, and that is the honest answer — a
    // plugin registering has no settings page of its own to open.
    stub({ notifications: [row({ source: "plugin_watcher" })] });
    render(<NotificationPanel />);

    expect(screen.queryByRole("button", { name: "Open settings" })).toBeNull();
  });

  it("offers the conversation instead when there is one", async () => {
    const { openConversation, user } = stub({
      notifications: [
        row({ source: "subagents", title: "Agent finished", conversation_id: 9 }),
      ],
    });
    render(<NotificationPanel />);

    await user.click(screen.getByRole("button", { name: "Open chat" }));

    expect(openConversation).toHaveBeenCalledWith(9);
  });

  it("does not offer the conversation you are already in", () => {
    // `conversationId` is 5 in the stub. Offering to go where you are is a
    // button that does nothing and says it did something.
    stub({
      notifications: [row({ source: "subagents", conversation_id: 5 })],
    });
    render(<NotificationPanel />);

    expect(screen.queryByRole("button", { name: "Open chat" })).toBeNull();
  });
});

describe("Settings lands where it was asked to", () => {
  it("opens at the requested section rather than the default", () => {
    // Without this the link lands on "Kernel", which is the default page and
    // proves nothing. The requested page has to be one the default is not.
    stub({ settingsRequest: { page: "packages" } });
    render(
      <SettingsDialogContent
        commandActionPending={false}
        afterCurrentCommand={runSettingsAction}
      />,
    );

    expect(screen.getByRole("heading", { name: "Packages" })).toBeTruthy();
  });

  it("consumes the request instead of standing on it", () => {
    // A request that never cleared would undo the next navigation: press
    // "Kernel" and the still-standing `plugins` would put you back.
    const { clearSettingsRequest } = stub({
      settingsRequest: { page: "packages" },
    });
    render(
      <SettingsDialogContent
        commandActionPending={false}
        afterCurrentCommand={runSettingsAction}
      />,
    );

    expect(clearSettingsRequest).toHaveBeenCalled();
  });

  it("uses the default section when nothing asked for one", () => {
    stub({ settingsRequest: null });
    render(
      <SettingsDialogContent
        commandActionPending={false}
        afterCurrentCommand={runSettingsAction}
      />,
    );

    expect(screen.getByRole("heading", { name: "Kernel" })).toBeTruthy();
  });

  it("shows immediate pending feedback and blocks duplicate command starts", async () => {
    const say = vi.fn(() => new Promise<boolean>(() => undefined));
    const { user } = stub({
      say,
      commands: [{ name: "llm", description: "Choose a model" }],
      // `/llm` is on Packages, and the dialog opens on Kernel — without this the
      // card under test is not on screen to be clicked.
      settingsRequest: { page: "packages" },
    });
    render(
      <SettingsDialogContent
        commandActionPending={false}
        afterCurrentCommand={runSettingsAction}
      />,
    );

    const card = screen.getByRole("button", { name: /Language models/i });
    await user.click(card);

    expect(card.getAttribute("aria-busy")).toBe("true");
    expect((card as HTMLButtonElement).disabled).toBe(true);
    await user.click(card);
    expect(say).toHaveBeenCalledTimes(1);
  });

  it("does not render the command machine's Back acknowledgement", () => {
    stub({
      state: {
        turns: [],
        typing: false,
        suppressedCommand: null,
        suppressNextCancellationNotice: false,
        buttons: [],
        error: null,
        shownText: [],
      scrollback: { hasMore: false, oldestId: null },
        carried: {},
        streams: {},
        form: {
          name: "agent",
          field: { name: "profile" },
          display: {
            prompt: "Select an agent profile.",
            choices: [{ value: "default", label: "Default" }],
            allow_back: true,
          },
        },
        command: {
          callId: "cmd:agent:test",
          name: "agent",
          args: {},
          status: "progressed",
          outcome: ["Back.", "Useful command output"],
        },
      } as provider.SecondBrain["state"],
    });
    render(
      <SettingsDialogContent
        commandActionPending={false}
        afterCurrentCommand={runSettingsAction}
      />,
    );

    expect(screen.queryByText("Back.")).toBeNull();
    expect(screen.getByText("Useful command output")).toBeTruthy();
  });
});

/**
 * Settings and the agent share one lane.
 *
 * Opening Settings mid-turn is harmless in itself, but every way back out of it
 * submits `/cancel` when something is running, and that lands on the turn. So
 * the link refuses while the agent has it, and says why — rather than being a
 * quiet way to lose a turn you were waiting on.
 */
describe("opening settings while the agent is working", () => {
  /** Only one field of the session state decides this, and spelling out the
   *  other nine would say that they mattered. */
  const turnState = (typing: boolean) =>
    ({ turns: [], typing }) as unknown as provider.SecondBrain["state"];

  it("refuses the link and explains, rather than opening", async () => {
    const { say, openSettings, setNotificationsOpen, user } = stub({
      notifications: [row()],
      state: turnState(true),
    });
    render(<NotificationPanel />);

    const link = screen.getByRole("button", { name: "Open settings" });
    expect(link).toHaveAttribute("aria-disabled", "true");
    expect(link).toHaveAccessibleDescription(/working/i);

    await user.click(link);

    expect(openSettings).not.toHaveBeenCalled();
    expect(say).not.toHaveBeenCalled();
    // And the panel is still standing, so nothing about the click read as
    // having gone somewhere.
    expect(setNotificationsOpen).not.toHaveBeenCalledWith(false);
  });

  it("opens as usual once the turn is over", async () => {
    const { openSettings, user } = stub({
      notifications: [row()],
      state: turnState(false),
    });
    render(<NotificationPanel />);

    const link = screen.getByRole("button", { name: "Open settings" });
    expect(link).not.toHaveAttribute("aria-disabled");

    await user.click(link);
    await waitFor(() => expect(openSettings).toHaveBeenCalledWith("kernel"));
  });
});
