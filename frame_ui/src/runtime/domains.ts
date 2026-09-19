/**
 * The context objects the provider publishes, and the hooks that read them.
 *
 * **Its own module because `provider.tsx` must export only its component.**
 * Vite's Fast Refresh can keep a component module alive across an edit; a
 * module mixing a component with other runtime values it cannot, so
 * `provider.tsx` was re-evaluated on every hot update — and re-evaluating it
 * ran `createContext` again. The mounted `SecondBrainProvider` then published
 * the *old* context objects while a freshly refreshed child read the *new*
 * ones, which answer null: "useConversations outside SecondBrainProvider",
 * thrown from inside the provider's own subtree. Contexts live here, where
 * nothing is a component, so their identity survives an edit to the provider.
 */

import { createContext, use, type Context } from "react";

import type { SettingsPageId } from "@/components/settings-structure";
import type { Command } from "@/lib/commands";
import type { CategoryCount, Conversation } from "@/lib/conversations";
import type { ConversationFilter } from "@/lib/conversation-categories";
import type { StreamStatus } from "@/lib/events";
import type { InputRequest } from "@/lib/input-requests";
// `Notification` deliberately shadows the DOM global of that name here. Ours is
// a row in the kernel's table; the browser's is a desktop popup this app does
// not use, and leaving the global reachable under the same spelling is how a
// missing import turns into a type error nobody can read.
import type { Notification } from "@/lib/notifications";
import type { Binding } from "@/lib/widgets";
import type { QueuedNotification } from "@/runtime/notifications";
import type { State } from "@/runtime/store";

/* ── The context the non-chat surfaces read ─────────────────────────── */

export type SecondBrain = {
  status: StreamStatus;
  state: State;
  /** A chat submission accepted by the UI but not yet acknowledged by the
   *  server's turn-lifecycle stream. */
  submitting: boolean;
  /** The server's own command catalogue, organized by Settings. */
  commands: Command[];
  /** Every conversation this user owns, newest first. */
  conversations: Conversation[];
  /** Whether `conversations` has been read yet. An empty list means nothing
   *  until this is true. */
  conversationsLoaded: boolean;
  /** The open conversation itself, read alongside its scrollback rather than
   *  looked up in `conversations` — which holds one page of one category and
   *  need not contain it. */
  openConversationRow: Conversation | null;
  /** Whether another page exists behind what is shown. */
  conversationsHasMore: boolean;
  /** Fetch it and append. */
  loadMoreConversations: () => Promise<void>;
  /**
   * Whether the open conversation continues *above* the scrollback on screen.
   *
   * The sidebar's paging one row down is the same shape and exists for
   * convenience; this one is not optional. `conv.read` answers with a page
   * because a transcript grows without limit — compaction shrinks what the
   * model sees and deletes nothing — so there is no size at which the whole
   * thing can be asked for.
   */
  scrollbackHasMore: boolean;
  /** Whether a page of older messages is in flight. */
  loadingOlderMessages: boolean;
  /** Fetch the page above and prepend it. */
  loadOlderMessages: () => Promise<void>;
  /** Every category that exists, with how many are in it — counted by the
   *  server over the whole table, not over the page it sent. */
  conversationCategories: CategoryCount[];
  /** Which slice the sidebar is showing. Changing it is a Request, not a
   *  predicate: the server does the filtering. */
  conversationFilter: ConversationFilter;
  setConversationFilter: (filter: ConversationFilter) => void;
  /** Rename the open conversation. */
  renameConversation: (id: number, title: string) => Promise<void>;
  /** File it under a category, or `null` for Main. */
  categoriseConversation: (id: number, category: string | null) => Promise<void>;
  /** The one the session is currently pointing at. */
  conversationId: number | null;
  /** Point the session at another conversation and show it. */
  openConversation: (id: number) => Promise<void>;
  /** Start a fresh conversation and switch to it — or stay where you are, when
   *  the conversation on screen has never been used. */
  newConversation: () => Promise<void>;
  /** Delete one. **Unsafe** — the server raises an approval dialog, which
   *  arrives on the event stream while this is still in flight. */
  deleteConversation: (id: number) => Promise<void>;
  /**
   * Questions the kernel is blocking on, head first.
   *
   * Not part of `state`: a pending question belongs to the *session*, which
   * outlives any one conversation, and living in the conversation store is
   * what used to make a page reload throw one away.
   */
  inputRequests: InputRequest[];
  /** Answer one, by the id it was asked under. The value goes to the server;
   *  the label is the person's business.
   *
   *  **The id is passed rather than read**, because between drawing a dialog
   *  and pressing a button another question can arrive, and "the current one"
   *  is then a different question than the one on screen. */
  resolve: (id: string | null, value: unknown) => Promise<void>;
  /**
   * Back out of one without answering it.
   *
   * **Still an answer, and the conservative one.** `frontend.cancel` in the
   * approving phase pops the question's own phase frame and settles the request
   * as cancelled, which every asker reads as the safe outcome: a sandbox
   * permission gate refuses, `ui.ask` comes back a refusal, a gated command is
   * dropped without running. So this unblocks the turn rather than walking away
   * from it — the distinction the dialog's "no dismissal" rule is really about.
   */
  cancelInputRequest: (id: string | null) => Promise<void>;
  /** Send a line of text as if typed — how form steps and quick replies are
   *  answered, since both are plain submissions. */
  say: (text: string) => Promise<boolean>;
  /** Put something in the error banner. For the surfaces that are not Requests
   *  and so have nowhere else to fail — a refused microphone, say. */
  report: (error: unknown) => void;
  dismissError: () => void;
  /** Put a finished command's panel away. */
  dismissCommand: () => void;
  /** Configured LLM profiles and the global default model. */
  models: LlmProfile[];
  modelName: string | null;
  agentProfile: string;
  modelsLoading: boolean;
  modelsFailure: boolean;
  switchingModel: boolean;
  setModel: (modelName: string) => Promise<void>;

  /**
   * What the system has told you.
   *
   * **`notificationQueue` and `notifications` are two sets, not two views of one.**
   * Transient progress enters the queue but is never stored, so it is in the first and
   * not the second; anything from before this page connected is in the second
   * and never was in the first. See `runtime/notifications.ts`.
   *
   * Here rather than in the store for the same reason `inputRequests` is: a
   * notification belongs to the session, and most of them are not about the open
   * conversation at all.
   */
  notificationQueue: QueuedNotification[];
  /** The persisted ones, newest first. Backfilled on boot, kept current by the
   *  stream. */
  notifications: Notification[];
  /** How many are still unread — what the bell's dot is drawn from. */
  unread: number;
  /** Why the panel is empty, when the reason is not "nothing happened". */
  notificationsFailure: string | null;
  /** Remove a completed status message without marking its notification read. */
  dismissQueuedNotification: (key: string) => void;
  /** Settle everything held. What opening the panel does. */
  markNotificationsRead: () => Promise<void>;
  notificationsOpen: boolean;
  setNotificationsOpen: (open: boolean) => void;

  settingsOpen: boolean;
  setSettingsOpen: (open: boolean) => void;
  /**
   * Open Settings, optionally at a particular section.
   *
   * For the surfaces that know *why* they are sending you there — a "Settings
   * changed" notification knows the change was configuration — as opposed to the
   * gear button, which knows nothing and lands on the default page.
   */
  openSettings: (page?: SettingsPageId) => void;
  /** The section Settings was asked to open at, until it has. **One-shot**: the
   *  dialog consumes it and clears it, so navigating away afterwards is not
   *  fought by a request that never expired. */
  settingsRequest: { page: SettingsPageId } | null;
  clearSettingsRequest: () => void;
  securityMode: "lockdown" | "ask" | "yolo";
  setSecurityMode: (mode: "lockdown" | "ask" | "yolo") => Promise<void>;
  /** The widget bound to the open conversation, or null while the first read
   *  is in flight. A bound `name` of null means the conversation holds none,
   *  which is the ordinary state and different from not yet knowing. */
  widgetBinding: Binding | null;
  /** Bind a widget to the open conversation, or `null` to bind none. */
  chooseWidget: (name: string | null) => void;
};

export type LlmProfile = {
  model_name: string;
  loaded?: boolean;
};

/**
 * The provider's whole surface, in one type.
 *
 * **A description, not a context.** Nothing subscribes to all of this at once
 * — every consumer takes one of the domain slices below — and publishing it as
 * a context as well meant building a thirty-field object, on every change to
 * any of them, for no reader. The type stays because it is the single place
 * that says what this provider offers, and each domain is a `Pick` of it.
 */
export type SessionDomain = Pick<
  SecondBrain,
  | "status"
  | "state"
  | "submitting"
  | "say"
  | "report"
  | "dismissError"
  | "dismissCommand"
>;
export type ModelDomain = Pick<
  SecondBrain,
  | "models"
  | "modelName"
  | "agentProfile"
  | "modelsLoading"
  | "modelsFailure"
  | "switchingModel"
  | "setModel"
>;
export type ConversationDomain = Pick<
  SecondBrain,
  | "conversations"
  | "conversationsLoaded"
  | "conversationId"
  | "openConversation"
  | "newConversation"
  | "deleteConversation"
  | "openConversationRow"
  | "renameConversation"
  | "categoriseConversation"
  | "conversationsHasMore"
  | "loadMoreConversations"
  | "scrollbackHasMore"
  | "loadingOlderMessages"
  | "loadOlderMessages"
  | "conversationCategories"
  | "conversationFilter"
  | "setConversationFilter"
>;
export type ApprovalDomain = Pick<
  SecondBrain,
  "inputRequests" | "resolve" | "cancelInputRequest"
>;
export type NotificationDomain = Pick<
  SecondBrain,
  | "notificationQueue"
  | "notifications"
  | "unread"
  | "notificationsFailure"
  | "dismissQueuedNotification"
  | "markNotificationsRead"
  | "notificationsOpen"
  | "setNotificationsOpen"
>;
export type SettingsDomain = Pick<
  SecondBrain,
  | "commands"
  | "settingsOpen"
  | "setSettingsOpen"
  | "openSettings"
  | "settingsRequest"
  | "clearSettingsRequest"
>;
export type SecurityDomain = Pick<SecondBrain, "securityMode" | "setSecurityMode">;
/**
 * The widget beside this conversation.
 *
 * A domain of its own because the panel is not the only thing that decides
 * what is in it any more. The agent sets one with `sdk.widget.set`, switching
 * conversations swaps it, and a second window of the same conversation moves
 * with both — so the binding belongs where the frames already arrive rather
 * than inside the component that draws it.
 */
export type WidgetDomain = Pick<SecondBrain, "widgetBinding" | "chooseWidget">;

export const SessionContext = createContext<SessionDomain | null>(null);
export const ModelContext = createContext<ModelDomain | null>(null);
export const ConversationContext = createContext<ConversationDomain | null>(null);
export const ApprovalContext = createContext<ApprovalDomain | null>(null);
export const NotificationContext = createContext<NotificationDomain | null>(null);
export const SettingsContext = createContext<SettingsDomain | null>(null);
export const SecurityContext = createContext<SecurityDomain | null>(null);
export const WidgetContext = createContext<WidgetDomain | null>(null);

function useDomain<T>(context: Context<T | null>, name: string): T {
  const value = use(context);
  if (value === null) throw new Error(`${name} outside SecondBrainProvider`);
  return value;
}

export const useSession = () => useDomain(SessionContext, "useSession");
export const useModels = () => useDomain(ModelContext, "useModels");
export const useConversations = () =>
  useDomain(ConversationContext, "useConversations");
export const useApprovals = () => useDomain(ApprovalContext, "useApprovals");
export const useNotifications = () =>
  useDomain(NotificationContext, "useNotifications");
export const useSettings = () => useDomain(SettingsContext, "useSettings");
export const useSecurity = () => useDomain(SecurityContext, "useSecurity");
export const useWidget = () => useDomain(WidgetContext, "useWidget");
