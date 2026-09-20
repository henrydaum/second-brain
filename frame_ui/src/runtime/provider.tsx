/**
 * Where the stream, the store and assistant-ui meet.
 *
 * `ExternalStoreRuntime` is the right runtime for this server because it does
 * not own the conversation. It renders whatever array of messages it is handed
 * and calls back when the person does something — which is exactly our
 * relationship with the kernel: the server is the source of truth, the event
 * stream is the only thing that moves it, and this app is a projection. The
 * other runtimes all want to own a request/response cycle we do not have.
 *
 * Everything the chat window itself cannot express — the dialog for a question
 * the kernel is blocking on, the form panel, the connection status — is
 * published through `SecondBrainContext` rather than through the runtime,
 * because assistant-ui has no concept of them.
 *
 * Those questions get their own reducer rather than a slot on the conversation
 * store, and that is the point rather than tidiness: they belong to the
 * *session*, which outlives any one conversation, and living in the
 * conversation store meant a history read threw a live one away. See
 * `runtime/input-requests.ts`.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type PropsWithChildren,
} from "react";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
  type AppendMessage,
  type QueueItemState,
} from "@assistant-ui/react";

// Type-only, deliberately. The sections are the settings dialog's own
// vocabulary and belong beside it; naming one here has to cost nothing at
// runtime, and an erased import is how the dependency stays one-way in the
// bundle even though the *name* points the other way.
import type { SettingsPageId } from "@/components/settings-structure";
import { RequestFailed, sdk } from "@/lib/client";
import { listCommands, looksLikeCommand, type Command } from "@/lib/commands";
import {
  CONVERSATION_PAGE,
  listConversations,
  setConversationCategory,
  setConversationTitle,
  type CategoryCount,
  type Conversation,
  type LoadResult,
  conversationLoadError,
} from "@/lib/conversations";
import {
  ALL_CONVERSATIONS_FILTER,
  MAIN_CONVERSATIONS_FILTER,
  type ConversationFilter,
} from "@/lib/conversation-categories";
import { connect, type StreamStatus } from "@/lib/events";
import { readConversation } from "@/lib/history";
import { isPendingInput } from "@/lib/input-requests";
// `Notification` deliberately shadows the DOM global of that name here. Ours is
// a row in the kernel's table; the browser's is a desktop popup this app does
// not use, and leaving the global reachable under the same spelling is how a
// missing import turns into a type error nobody can read.
import { listNotifications, markRead } from "@/lib/notifications";
import { attachmentSubmitArgs, extensionOf } from "@/lib/upload";
import { getWidget, setWidget, type Binding } from "@/lib/widgets";
import { attachmentAdapter } from "@/runtime/attachment-adapter";
import { convertMessage } from "@/runtime/convert";
import {
  ApprovalContext,
  ConversationContext,
  ModelContext,
  NotificationContext,
  SecurityContext,
  SessionContext,
  SettingsContext,
  WidgetContext,
  type ApprovalDomain,
  type ConversationDomain,
  type LlmProfile,
  type ModelDomain,
  type NotificationDomain,
  type SecurityDomain,
  type SessionDomain,
  type SettingsDomain,
  type WidgetDomain,
} from "@/runtime/domains";
import {
  initialInputRequests,
  reduceInputRequests,
  unseenRequest,
} from "@/runtime/input-requests";
import {
  highestId,
  initialNotifications,
  reduceNotifications,
  unreadCount,
} from "@/runtime/notifications";
import {
  initialState,
  reduce,
  type FilesPart,
  type MessageAttachment,
} from "@/runtime/store";
import { stagedPath } from "@/runtime/staged-attachments";

/** The queue adapter's two lanes, which are always empty — see `queue` in the
 *  provider. One frozen array rather than a fresh literal per render, so the
 *  runtime's own memos over it never see a changed identity. */
const NO_QUEUED_MESSAGES: readonly QueueItemState[] = [];

/**
 * One profile as `config.read` returns it, which is not what `llm.list` returns.
 *
 * `llm.list` answers "which profiles exist"; the extra params are only in the
 * setting. Every other field is declared unknown rather than omitted because
 * this shape is written back whole — see `setReasoningEffort` — and a narrower
 * type would invite dropping the fields it does not name.
 */
type LlmProfileConfig = {
  [field: string]: unknown;
};



/* ── The provider ───────────────────────────────────────────────────── */

/** How often an idle page re-reads the conversation list. See the effect that
 *  uses it for why a minute is the right number and why a poll is here at all. */
const IDLE_REFRESH_MS = 60_000;

/** A missing SSE lifecycle acknowledgement must not leave Stop up forever.
 * This is a backstop, not the normal path; a healthy stream clears the local
 * submission state almost immediately. */
const SUBMIT_ACK_RECONCILE_MS = 10_000;

/** The kernel caps `conv.list`'s `limit` at 200. Refreshing everything shown
 *  stops there; past four pages a refresh re-reads the front of the list and
 *  the rest keeps whatever it last had, which is what it would have anyway. */
const MOST_CONVERSATIONS_AT_ONCE = 200;

const FILTER_KEY = "second-brain:conversation-filter";

/**
 * What to send as `category` for a filter.
 *
 * `undefined` and `null` are different questions on the wire and both are
 * needed: nothing at all means every conversation, and Main is asked for
 * explicitly. See `listConversations`.
 */
function filterCategory(filter: ConversationFilter): string | null | undefined {
  return filter.type === "all" ? undefined : filter.category;
}

/** The remembered filter, or the default. Read at startup rather than in an
 *  effect, so the first read goes out for the right slice. */
function readConversationFilter(): ConversationFilter {
  try {
    const stored = JSON.parse(localStorage.getItem(FILTER_KEY) ?? "null") as {
      type?: unknown;
      category?: unknown;
    } | null;
    if (stored?.type === "all") return ALL_CONVERSATIONS_FILTER;
    if (stored?.type === "category") {
      if (stored.category === null) return MAIN_CONVERSATIONS_FILTER;
      if (typeof stored.category === "string" && stored.category.trim()) {
        return { type: "category", category: stored.category };
      }
    }
  } catch {
    // A malformed or unavailable preference is just the default view.
  }
  return MAIN_CONVERSATIONS_FILTER;
}

function writeConversationFilter(filter: ConversationFilter): void {
  try {
    localStorage.setItem(FILTER_KEY, JSON.stringify(filter));
  } catch {
    // Filtering still works for this visit when storage is unavailable.
  }
}

export function SecondBrainProvider({ children }: PropsWithChildren) {
  const [state, dispatch] = useReducer(reduce, initialState);
  const [submitting, setSubmitting] = useState(false);
  const [inputRequests, askDispatch] = useReducer(
    reduceInputRequests,
    initialInputRequests,
  );
  const [notifications, notifyDispatch] = useReducer(
    reduceNotifications,
    initialNotifications,
  );
  const [status, setStatus] = useState<StreamStatus>("connecting");
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  /**
   * The widget this conversation is showing.
   *
   * Read once on connect and maintained by the `widget` frame from then on.
   * Both halves are needed: a fresh page has missed every announcement ever
   * made, and a page with only the read goes stale the first time the agent,
   * another window, or a conversation switch changes it.
   */
  const [widgetBinding, setWidgetBinding] = useState<Binding | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsRequest, setSettingsRequest] = useState<{
    page: SettingsPageId;
  } | null>(null);
  const [securityMode, setSecurityModeState] = useState<
    "lockdown" | "ask" | "yolo"
  >("ask");
  const [models, setModels] = useState<LlmProfile[]>([]);
  const [modelName, setModelName] = useState<string | null>(null);
  const [agentProfile, setAgentProfile] = useState("default");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsFailure, setModelsFailure] = useState(false);
  const [switchingModel, setSwitchingModel] = useState(false);
  /** The profiles setting as it was last read. A ref rather than state because
   *  nothing renders it — it exists so switching model can show that profile's
   *  effort without another round trip. The write path re-reads regardless. */
  const llmProfilesRef = useRef<Record<string, LlmProfileConfig>>({});

  // `isLoading` covers the gap between "the page is up" and "scrollback is on
  // screen", so the thread does not flash an empty-conversation welcome at
  // someone who has a conversation.
  const [loading, setLoading] = useState(true);

  // Read once at boot. The catalogue only changes when a package is installed
  // or removed, which is rare enough that re-reading on every keystroke would
  // be a Request per character for no benefit.
  const [commands, setCommands] = useState<Command[]>([]);

  const [conversations, setConversations] = useState<Conversation[]>([]);
  /** Whether the list has been asked for yet. Until it has, an empty sidebar
   *  means "not yet" rather than "none", and only one of those is worth
   *  saying — see `loadCatalogue`. */
  const [conversationsLoaded, setConversationsLoaded] = useState(false);
  /** The open conversation's notification mode, as `conv.read` last reported
   *  it. Not on the `Conversation` row — the kernel derives it from the state
   *  machine — so it is held beside the id rather than looked up in the list. */
  /**
   * The open conversation's own row, from `conv.read` rather than from the
   * list.
   *
   * **The list is one page of one category, so it cannot be asked this.** File
   * the open conversation under a category the current filter excludes and it
   * leaves the list — and looking it up there meant the header lost the
   * conversation the moment you used the header to move it.
   */
  const [openConversationRow, setOpenConversationRow] =
    useState<Conversation | null>(null);
  const [conversationsHasMore, setConversationsHasMore] = useState(false);
  const [conversationCategories, setConversationCategories] = useState<
    CategoryCount[]
  >([]);
  /**
   * Which slice the sidebar is showing.
   *
   * **Here rather than in the sidebar, because the server does the filtering
   * now.** Picking a category is a different Request, not a different
   * predicate over rows already held — which is the whole fix: the rows for a
   * quiet category were never in the 50 most recent to begin with.
   */
  const [conversationFilter, setConversationFilterState] =
    useState<ConversationFilter>(readConversationFilter);
  /** Readable from callbacks that must not be rebuilt when it changes — the
   *  same trick `commandsRef` plays. */
  const conversationFilterRef = useRef(conversationFilter);
  /**
   * The rows on screen, readable from a callback without being a dependency of
   * one — and their count *is* the next page's offset, so there is no separate
   * cursor to keep in step with the list.
   */
  const conversationsRef = useRef<Conversation[]>([]);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const conversationIdRef = useRef<number | null>(null);
  /**
   * A message is on its way into a session that holds no conversation, so the
   * next one announced is the one that message just made.
   *
   * **Set before the submit, because the announcement beats the answer.** The
   * server creates the conversation on the way in and emits
   * `SESSION_CONVERSATION_CHANGED`; that frame reaches the stream in a
   * fraction of the round trip `adoptConversation` needs to ask for the same
   * id. So the frame arrives while we are still bound to nothing, and the only
   * way to tell "the conversation I am talking into" from "somebody pointed me
   * at a different one" is to have said so beforehand. See the `conversation`
   * branch of the frame handler for what it buys.
   *
   * Only an ordinary message sets it. A slash command and a form step travel
   * the same path and can *load* a conversation — `/conversations` does
   * exactly that — where the scrollback is the whole point of the switch.
   */
  const creatingConversationRef = useRef(false);
  const renderRevision = useRef(0);
  conversationIdRef.current = conversationId;
  const [loadingOlderMessages, setLoadingOlderMessages] = useState(false);
  /** Guards against two pages of the same cursor being in flight at once — an
   *  intersection observer will happily fire again before the first returns. */
  const loadingOlderRef = useRef(false);
  /**
   * The paging cursor, readable from a callback without being a dependency of
   * one — the same trick `conversationsRef` plays.
   *
   * It *lives* in the reducer, beside the turns it describes, because it was
   * provider state briefly and that was a mistake: six sites replace the
   * scrollback and every one of them had to remember to update the cursor too.
   * One did not. Now the `history` action carries it and forgetting is a type
   * error.
   */
  const scrollbackRef = useRef(state.scrollback);
  scrollbackRef.current = state.scrollback;
  /** Whether a command is part-way through asking its questions, readable from
   *  the send callbacks — which are deliberately dependency-free. */
  const formRef = useRef(state.form);
  formRef.current = state.form;
  const scrollbackHasMore = state.scrollback.hasMore;
  conversationsRef.current = conversations;
  /**
   * `refreshConversations`, reachable from callbacks declared above it.
   *
   * `adoptConversation` runs the moment a message creates a conversation and
   * wants the list re-read, but it sits far above the refresher — and a
   * dependency array is evaluated while rendering, so naming it there would
   * read a `const` that does not exist yet.
   */
  const refreshConversationsRef = useRef<(() => Promise<void>) | null>(null);

  /**
   * The catalogue, readable from a callback without becoming a dependency of
   * one.
   *
   * `say` and `onNew` need it to tell a slash command from a message that
   * merely opens with a slash, but every callback the runtime adapter is built
   * from is deliberately dependency-free so the adapter keeps one identity for
   * the life of the page. A ref is how a value can be current without being a
   * reason to rebuild.
   */
  const commandsRef = useRef<Command[]>([]);
  commandsRef.current = commands;

  /**
   * Turn a thrown Request into the error banner.
   *
   * **Nothing that reaches the server may reject.** assistant-ui calls several
   * of these from an event handler, and an unhandled rejection there unmounts
   * the tree — the symptom is the whole page going white, with the actual cause
   * (a refusal, a dropped connection) never shown. A failed Request is ordinary
   * news and belongs in the banner, not in a crash.
   *
   * Declared here, above its callers, rather than beside the other actions: a
   * `useCallback` that lists it as a dependency reads it while the component
   * body is still running, so anything using it has to come after it.
   */
  const report = useCallback((error: unknown) => {
    const failed = error instanceof RequestFailed;
    // **Saying no is not an error.** A declined Request is the answer the
    // person just gave in the dialog; putting "denied: conv.delete" in a banner
    // afterwards is approval fallout escaping the popup — the one thing this
    // surface is supposed to keep contained — and it reads as a malfunction
    // rather than as their own decision being carried out.
    if (failed && error.isDeclined) return;
    dispatch({
      type: "frame",
      frame: {
        kind: "error",
        payload: {
          message: failed
            ? error.message
            : error instanceof Error
              ? error.message
              : String(error),
          code: failed ? error.code : "",
          // The banner reads this to explain a dead session rather than
          // repeating the kernel's wording at somebody who cannot act on it.
          details: failed && error.isSessionTaken ? "session_taken" : undefined,
        },
      },
    });
  }, []);

  /**
   * The two lists the chrome is built from.
   *
   * Together because they fail together: both are ordinary Requests, so a
   * session the server will not act on takes out both at once — which is what
   * makes empty Settings a *symptom* rather than a bug in that surface.
   * Fetching them in one place means one retry brings back both.
   */
  const loadCatalogue = useCallback(async () => {
    try {
      setCommands(await listCommands());
      const page = await listConversations({
        category: filterCategory(conversationFilterRef.current),
      });
      setConversations(page.items);
      setConversationsHasMore(page.hasMore);
      setConversationCategories(page.categories);
    } catch (error) {
      // Reported rather than thrown: a chat window with no sidebar is still a
      // chat window, and the banner says why it is empty.
      report(error);
    } finally {
      // In `finally` rather than after the read: a failed read has still
      // answered the sidebar's question. "We asked and got nothing" and "we
      // have not asked yet" are different sentences, and only the first one is
      // allowed to say "No conversations yet."
      setConversationsLoaded(true);
    }
  }, [report]);

  /**
   * Open the stream, then boot.
   *
   * **Order is the whole point.** Opening `/events` is what declares that
   * somebody is watching; attendance decides whether an unsafe Request raises a
   * dialog or is refused outright. A boot that POSTed first would get silent
   * refusals for anything interesting, so the stream goes up before the first
   * Request goes out.
   *
   * Empty dependencies: one stream for the life of the page. A second
   * `GET /events` on the same thread *replaces* the first, so re-running this
   * effect would quietly steal the connection from itself.
   */
  useEffect(() => {
    const close = connect((frame) => {
      renderRevision.current += 1;
      // This is the server's acknowledgement of the optimistic submit state.
      // `typing` is normally first; the other turn frames are fallbacks for an
      // older or interrupted transport that omitted it.
      if (
        frame.kind === "typing" ||
        frame.kind === "turn_activity" ||
        frame.kind === "stream_delta" ||
        (frame.kind === "tool_status" && frame.payload.kind !== "command")
      ) {
        setSubmitting(false);
      }
      // **Fanned out here, not inside the reducer.** A question the kernel is
      // blocking on belongs to the session; routing it through the
      // conversation store is what let a history read discard one. Keeping the
      // split at the doorway means nothing downstream has to remember it.
      if (frame.kind === "approval") {
        askDispatch({ type: "raised", request: frame.payload });
        return;
      }
      if (frame.kind === "approval_settled") {
        askDispatch({ type: "settled", id: frame.payload.request_id });
        return;
      }
      // Same shape of decision, same reason. A notification is the system
      // speaking, not the agent, and usually not about the conversation on
      // screen at all — so it must not reach the store, whose next history read
      // would drop it.
      if (frame.kind === "notification") {
        notifyDispatch({
          type: "raised",
          notification: frame.payload,
          // Generated here rather than in the reducer so the reducer stays a
          // pure function of its arguments. A transient notification has no
          // `notification_id` to key a list on; this is what it gets instead.
          key: crypto.randomUUID(),
        });
        return;
      }
      /*
        Which widget the panel holds. Routed here rather than into the store
        for the same reason a notification is: it is not part of the
        conversation's history, and the next history read would drop it.
        It arrives on a conversation switch as well as on a deliberate change,
        which is what makes the panel follow the conversation without the
        `conversation` branch below having to know widgets exist.
      */
      if (frame.kind === "widget") {
        setWidgetBinding({
          name: frame.payload.name ?? null,
          path: frame.payload.path ?? "",
          tree: frame.payload.tree ?? "",
          installed: frame.payload.installed ?? false,
          state: frame.payload.state ?? null,
          conversationId: frame.payload.conversation_id ?? null,
        });
        return;
      }
      if (frame.kind === "conversation") {
        const id = frame.payload.conversation_id;
        if (id === null) {
          conversationIdRef.current = null;
          creatingConversationRef.current = false;
          setConversationId(null);
          setOpenConversationRow(null);
          dispatch({ type: "history", turns: [], hasMore: false, oldestId: null });
          return;
        }
        if (id !== conversationIdRef.current) {
          /*
            **A conversation our own message just made is not a switch.**

            The server creates one from the first message sent into a session
            and announces it immediately, so this frame lands in the middle of
            the turn that message started — typically between `typing: true`
            and the model's first word. Reading the scrollback back is what a
            switch means, and `history` resets everything transient with it:
            the running turn, and the `typing` flag that puts Stop in the
            composer and the Thinking line in the transcript. The turn then
            drew nothing at all until the agent spoke, because
            `turn_activity` only fires on a phase *change* and the phase had
            not changed — so the model's entire first think went unaccounted
            for, uninterruptible, and for a page that had already drawn the
            indicator, visibly taken back.

            There is also nothing to read. The only thing in that conversation
            is the line we echoed locally on the way out. So only the row is
            fetched, for the title the header reads, and the transcript is
            left alone.
          */
          // The null is part of the claim, not a restatement of the `if`
          // above: a claim outlives its message whenever `adoptConversation`
          // wins the race, and a switch arriving later — from another session,
          // or the sidebar — must not inherit it.
          const ours =
            creatingConversationRef.current && conversationIdRef.current === null;
          creatingConversationRef.current = false;
          conversationIdRef.current = id;
          setConversationId(id);
          void readConversation(id, ours ? { limit: 0 } : {}).then((read) => {
            if (conversationIdRef.current !== id) return;
            if (!ours) {
              dispatch({ type: "history", turns: read.turns,
                         hasMore: read.hasMore, oldestId: read.oldestId });
            }
            setOpenConversationRow(read.conversation);
          }).catch((error) => report(error));
          if (ours) void refreshConversationsRef.current?.();
        }
        return;
      }
      dispatch({ type: "frame", frame });
    }, setStatus);

    // `cancelled` guards the async boot below: React can unmount between the
    // await and the dispatch, and a dispatch after unmount is a wasted render
    // at best and a stale conversation at worst.
    let cancelled = false;

    void (async () => {
      try {
        const session = await sdk<{
          conversation_id?: number | null;
          mode?: "lockdown" | "ask" | "yolo" | null;
          busy?: boolean | null;
          turn_id?: string | null;
        } | null>(
          "session.get",
          { details: true },
        );
        if (!cancelled) setSecurityModeState(session?.mode ?? "ask");

        // A session holds no conversation until somebody sends a message, and
        // that is now the ordinary resting state rather than a problem to fix
        // on the way in. Booting used to create one here, because a session
        // with no conversation could not be talked to — every submit came back
        // "No conversation loaded. Try /new." That refusal is gone: the server
        // creates the conversation from the first message, titled with it.
        //
        // So a fresh page load binds nothing and shows the empty composer. It
        // is also what stopped the sidebar filling with untouched rows, one per
        // page load, since this ran before the person had typed anything.
        const bound = session?.conversation_id ?? null;

        if (bound !== null) {
          const read = await readConversation(bound);
          if (!cancelled) {
            dispatch({ type: "history", turns: read.turns,
                       hasMore: read.hasMore, oldestId: read.oldestId });
            setConversationId(bound);
            setOpenConversationRow(read.conversation);
          }
        }

        // **The turn a reload landed in the middle of.**
        //
        // `typing` only ever moves on a `typing` frame, and the one that opened
        // the running turn was sent before this page existed — so a reload
        // mid-turn came up believing the agent was idle. That is not a cosmetic
        // wrong indicator: `isRunning` is what puts Stop in the composer, so the
        // one control that ends a turn disappeared exactly when it was wanted,
        // and reloading became a way to lose the ability to interrupt.
        //
        // The server has always answered this — `session.get` returns `busy`,
        // read here off the response boot already makes. **After the history
        // dispatch**, which resets everything transient and would otherwise wipe
        // it; and dispatched as the frame it stands in for, so the store opens a
        // turn for whatever arrives next exactly as the real frame would have.
        if (!cancelled && (session?.busy || session?.turn_id)) {
          dispatch(session.turn_id
            ? { type: "resumeTurn", turnId: session.turn_id }
            : { type: "frame", frame: { kind: "typing", payload: true } });
        }

        // After the conversation, not before: neither Settings nor the sidebar
        // is useful until there is somewhere to run a command, and scrollback
        // is what the person is actually waiting to see.
        if (!cancelled) await loadCatalogue();

        // Pending questions are reconciled by the effect below, which runs on
        // every stream open rather than once here — see `reconcile`.
      } catch (error) {
        // A boot that cannot read history is still a usable chat window, so
        // this reports and carries on rather than refusing to start.
        console.error("second brain: could not read history", error);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      close();
    };
  }, []);

  /**
   * Put the session back on the conversation that is on screen.
   *
   * **A restart can leave the two disagreeing, and nothing says so.** The
   * transcript, the sidebar and the composer all go on looking right while
   * every `frontend.submit` answers "No conversation loaded. Try /new." — a
   * session with no conversation cannot be talked to, and that state is only
   * visible by asking. This is the half of "just refresh the page" that
   * re-reading the catalogue does not cover.
   *
   * Three answers, and the ordering between them is the point:
   *
   * - **The server agrees.** Much the commonest, since a session normally
   *   comes back from `persistence` still holding its conversation. Nothing is
   *   done, and that is load-bearing: `history` resets everything transient, so
   *   re-reading on every reconnection would wipe a half-answered form or a
   *   command's output because the connection blinked.
   * - **The server names a different one.** It is the source of truth, so
   *   follow it and read the scrollback that goes with it.
   * - **The server names none.** Clear what is on screen. Another session may
   *   have taken the conversation, and reclaiming it here would undo that
   *   explicit handoff.
   */
  const resyncConversation = useCallback(async () => {
    try {
      const session = await sdk<{ conversation_id?: number | null } | null>(
        "session.get",
        { details: true },
      );
      const bound = session?.conversation_id ?? null;
      const showing = conversationIdRef.current;
      if (bound === showing) return;

      if (bound === null) {
        if (showing === null) return;
        conversationIdRef.current = null;
        setConversationId(null);
        setOpenConversationRow(null);
        dispatch({ type: "history", turns: [], hasMore: false, oldestId: null });
        return;
      }

      // Written eagerly as well as through `setConversationId`, because the
      // render that copies state into it has not happened yet and the effects
      // reading it must not see the conversation this one replaced.
      conversationIdRef.current = bound;
      setConversationId(bound);
      const read = await readConversation(bound);
      dispatch({ type: "history", turns: read.turns,
                 hasMore: read.hasMore, oldestId: read.oldestId });
      setOpenConversationRow(read.conversation);
    } catch (error) {
      report(error);
    }
  }, [report]);

  /**
   * Come back to life when the connection does.
   *
   * The stream recovers itself — see `connect`, which takes over retrying at
   * the point `EventSource` gives up — but everything fetched over `/sdk` was
   * read once at boot and is now stale or, if the boot failed, still empty.
   * Re-reading on the *transition* back to open is what turns "restart Second
   * Brain" into a fix the person sees, rather than one that also requires
   * reloading the page.
   *
   * The ref skips the first open, which boot has already covered.
   */
  const openedBefore = useRef(false);
  useEffect(() => {
    if (status !== "open") return;
    if (!openedBefore.current) {
      openedBefore.current = true;
      return;
    }
    void loadCatalogue();
    void resyncConversation();
  }, [status, loadCatalogue, resyncConversation]);

  useEffect(() => {
    if (!submitting) return;
    let cancelled = false;
    const reconcileSubmission = async () => {
      try {
        const session = await sdk<{
          busy?: boolean | null;
          turn_id?: string | null;
        } | null>("session.get", { details: true });
        if (!cancelled && !session?.busy && !session?.turn_id) {
          setSubmitting(false);
        }
      } catch {
        // The permanent connection status and ordinary Request error paths
        // already explain an unavailable backend. This check only prevents a
        // stale optimistic state and should not add a second error banner.
      }
    };
    const timer = window.setInterval(
      () => void reconcileSubmission(),
      SUBMIT_ACK_RECONCILE_MS,
    );
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [submitting]);

  /* ── Questions the kernel is blocking on ────────────────────────── */

  /**
   * Ask the server what it is still waiting for.
   *
   * **This covers the gap a stream cannot: what happened while nobody was
   * connected.** A render is an event, and events are not re-sent because you
   * asked, so a question raised before this page existed — or settled while it
   * was away — has no frame coming. `details` makes the server hand back the
   * question itself rather than its id, which is the difference between the
   * real dialog and a stand-in that can only say "something was asked".
   *
   * Once the stream is up, `approval` and `approval_settled` say the rest, so
   * this runs on connect and not on a timer. It used to poll every minute,
   * which was this client working around a protocol that could announce a
   * question and not its end.
   */
  const reconcile = useCallback(async () => {
    try {
      const pending = await sdk<unknown>("frontend.pending", { details: true });
      if (pending === null || pending === undefined) {
        askDispatch({ type: "reconciled", pending: null });
        return;
      }
      if (isPendingInput(pending)) {
        askDispatch({ type: "reconciled", pending });
        return;
      }
      // A kernel older than `details` answers the bare id, or `true` for "one
      // exists but I cannot name it". Neither describes the question, so all
      // that can be drawn is a stand-in — and `true` must never be used as an
      // id, which the server would stringify into one that matches nothing and
      // then report as already answered.
      askDispatch({
        type: "reconciled",
        pending: {
          kind: "approval",
          payload: unseenRequest(
            typeof pending === "string" && pending ? pending : null,
          ),
        },
      });
    } catch (error) {
      // Reported rather than thrown: this runs from an effect and on a timer,
      // and a failed poll is not worth tearing anything down over.
      report(error);
    }
  }, [report]);

  /** On **every** open, not just reconnections — unlike `loadCatalogue` above,
   *  whose first run boot covers. Boot cannot cover this one: the answer to
   *  "what is pending" is only meaningful once the stream is up, because the
   *  stream is what declares somebody is watching. */
  useEffect(() => {
    if (status !== "open") return;
    void reconcile();
  }, [status, reconcile]);

  const waiting = inputRequests.queue.length > 0;

  /* ── Notifications ──────────────────────────────────────────────── */

  /**
   * Fill the panel from the table.
   *
   * **The stream only ever answers "what happened since you connected."** For a
   * client that was closed while a scheduled agent ran, that is none of it — so
   * the panel needs a read, not just a subscription. Exactly the division
   * `FileActivityProvider` draws for the ledger: renders are events, the table
   * is state.
   *
   * `since_id` on the later calls is an index seek rather than a scan, and the
   * reducer merges by id — so a notification that arrived as a frame *and* comes
   * back in the read lands once.
   */
  const notificationsCursor = useRef(0);
  const backfillNotifications = useCallback(async (since: number) => {
    try {
      const rows = await listNotifications(
        since > 0 ? { since_id: since } : { limit: 50 },
      );
      notifyDispatch({ type: "backfilled", rows });
    } catch {
      // Said in the panel rather than the error banner, for the reason
      // `FileActivityProvider` gives about `ledger.read`: a kernel without the
      // Request would otherwise raise a banner on every boot, about a surface
      // that may never be opened. The status messages still work either way — they come
      // off the stream and owe this call nothing.
      notifyDispatch({
        type: "failed",
        message: "Second Brain would not hand back your notifications.",
      });
    }
  }, []);

  /**
   * On **every** open, like `reconcile` and unlike `loadCatalogue`.
   *
   * `EventSource` replays from `Last-Event-ID` and the buffer holds 500 frames
   * per session, so a short drop is covered for free — but a long one loses the
   * middle, and the middle is where the notification you actually wanted is. The
   * cursor makes the repeat calls cheap enough that covering the long case
   * costs nothing in the common one.
   */
  useEffect(() => {
    if (status !== "open") return;
    void backfillNotifications(notificationsCursor.current);
  }, [status, backfillNotifications]);

  // Kept in a ref rather than read from state inside the effect, so topping up
  // does not re-run every time a notification arrives.
  notificationsCursor.current = highestId(notifications);

  /**
   * Settle everything held — what opening the panel does.
   *
   * `before_id` rather than a list of ids: the handler filters `id <= ?`, so the
   * highest id held settles the rows behind it too, including any this client
   * never loaded. Guarded on there being one, because `mark_read` with neither
   * argument is a `400` rather than a no-op.
   *
   * Dispatched before the await, not after. The count is what the bell's dot is
   * drawn from, and a dot that lingers for a round trip after you have opened
   * the panel reads as a failure to notice you.
   */
  const markNotificationsRead = useCallback(async () => {
    const before = highestId(notifications);
    if (before === 0 || unreadCount(notifications) === 0) return;
    notifyDispatch({ type: "read", before });
    try {
      await markRead({ before_id: before });
    } catch {
      // Left as read locally. The next backfill carries the server's own
      // `read_at`, so this corrects itself rather than needing to be undone —
      // and a badge that reappeared under someone who had just cleared it would
      // be worse than one that is briefly optimistic.
    }
  }, [notifications]);

  const dismissQueuedNotification = useCallback((key: string) => {
    notifyDispatch({ type: "dismissed", key });
  }, []);

  /**
   * Open Settings, optionally at a section.
   *
   * A fresh object per call rather than a bare page id, so clicking the same
   * link twice is two requests. With the id alone, asking for `config` while
   * already showing `config` — having navigated away in between — would be a
   * state that never changed and therefore an effect that never fired.
   */
  const openSettings = useCallback((page?: SettingsPageId) => {
    if (page) setSettingsRequest({ page });
    setSettingsOpen(true);
  }, []);

  const clearSettingsRequest = useCallback(() => setSettingsRequest(null), []);

  /* ── What the person can do ─────────────────────────────────────── */

  /**
   * Claim the conversation the message about to be sent is going to create.
   *
   * The counterpart to `adoptConversation` and the half that has to happen
   * *before* the submit: the server's announcement of the new conversation
   * outruns the round trip that asks for its id, so by the time we could find
   * out, the frame that needed the answer has already been handled. See
   * `creatingConversationRef`.
   *
   * Written on every send rather than only on a claiming one, so a claim
   * cannot outlive the message that made it — a submit that fails leaves the
   * flag set, and the next thing to arrive would be adopted as ours.
   */
  const expectOwnConversation = useCallback((isCommand?: boolean) => {
    creatingConversationRef.current =
      conversationIdRef.current === null && !isCommand && formRef.current === null;
  }, []);

  /**
   * Find out which conversation the message we just sent created.
   *
   * The server makes a conversation from the first message sent into it, so
   * between the submit and this we are bound to nothing and do not know the id.
   *
   * **A backstop, not the usual route.** The `conversation` render kind carries
   * the id and normally arrives first, which is the whole reason
   * `expectOwnConversation` exists — the announcement beats this question's
   * answer. This covers what a stream cannot: a frame dropped on a connection
   * that blinked at the wrong moment. It returns immediately once the frame has
   * already bound us, and the two never disagree because the server is the only
   * source of the id.
   *
   * Not cosmetic. `FileActivityProvider` and `hydrateSentAttachments` both bail
   * on a null id, so without this the files panel stays empty and sent
   * attachments keep their optimistic names for the life of the page.
   */
  const adoptConversation = useCallback(async () => {
    if (conversationIdRef.current !== null) return;
    try {
      const session = await sdk<{ conversation_id?: number | null } | null>(
        "session.get",
        { details: true },
      );
      const bound = session?.conversation_id ?? null;
      if (bound === null || conversationIdRef.current !== null) return;
      creatingConversationRef.current = false;
      conversationIdRef.current = bound;
      setConversationId(bound);
      // The row is brand new, so it is not in the copy of the list we hold, and
      // the header reads its title from there. Through a ref because
      // `refreshConversations` is declared further down and naming it in the
      // dependency array would read it before it exists.
      await refreshConversationsRef.current?.();
    } catch (error) {
      report(error);
    }
  }, [report]);

  const say = useCallback(
    async (text: string) => {
      const isCommand = looksLikeCommand(text, commandsRef.current);
      dispatch({ type: "said", text, isCommand });
      expectOwnConversation(isCommand);
      try {
        await sdk("frontend.submit", { input_kind: "text", text });
        await adoptConversation();
        return true;
      } catch (error) {
        report(error);
        return false;
      }
    },
    [adoptConversation, expectOwnConversation, report],
  );

  const resolve = useCallback(
    async (id: string | null, value: unknown) => {
      // Closed before the answer lands, deliberately: the POST completes only
      // once the *original* blocked Request finishes, which can be a while, and
      // leaving the dialog up in the meantime invites a second click.
      askDispatch({ type: "answered", id });
      try {
        // **Only a real id travels.** With none, the server answers whatever is
        // next in its own queue, which is exactly what a question we could not
        // name means. Sending a placeholder instead gets it stringified into an
        // id that matches nothing, and the answer comes back "already
        // answered" while the turn stays blocked.
        await sdk("frontend.resolve", id ? { value, request_id: id } : { value });
        // A `false` answer means there was nothing left to answer — already
        // resolved elsewhere, or timed out after 300s. That is a stale dialog,
        // not an error, and closing it is the whole response.
      } catch (error) {
        report(error);
      }
    },
    [report],
  );

  const cancelInputRequest = useCallback(
    async (id: string | null) => {
      // Closed first, for the same reason `resolve` closes first: cancelling
      // drives the state machine, and the POST does not come back until it has.
      askDispatch({ type: "answered", id });
      // The disappearing dialog is the acknowledgement. The kernel also emits
      // a literal "Cancelled." message; mark that one protocol echo so it does
      // not become an assistant line in the conversation.
      dispatch({ type: "suppressNextCancellationNotice" });
      try {
        // **Not `resolve` with a falsy value.** That would be answering "no" to
        // a yes/no question and nonsense to a free-text one; a sandbox gate
        // typed `string` would reject `false` against its enum outright and
        // leave the frame standing. Cancelling is its own action, and the only
        // one that means "back out" regardless of what was asked.
        //
        // It names no request: the kernel cancels the frame on top of the
        // stack, which is the one being shown. If a second question arrived in
        // between, reconciliation is what corrects the drift.
        await sdk("frontend.cancel", {});
      } catch (error) {
        report(error);
      }
    },
    [report],
  );

  const dismissError = useCallback(() => dispatch({ type: "clearError" }), []);
  const dismissCommand = useCallback(
    () => dispatch({ type: "clearCommand" }),
    [],
  );

  /**
   * Whether the agent has the turn, readable without becoming a dependency.
   *
   * `syncSession` below only dispatches when the server disagrees with what is
   * on screen, and a `useCallback` that listed `state.typing` would be rebuilt
   * on every turn — taking the effects that depend on it with it, which for the
   * stream-open effect means re-asking on every frame.
   */
  const typingRef = useRef(false);
  typingRef.current = state.typing;

  /**
   * Re-read what the session itself says, for the two fields a client cannot
   * work out on its own.
   *
   * Both are ephemeral kernel state that no frame will re-announce: the
   * per-conversation security mode is cleared by a backend restart, and `busy`
   * is the answer to "does the agent have the turn *right now*", which the
   * stream only ever states at the moment it changes. One `session.get` answers
   * both, so this is one Request rather than two.
   *
   * The `typing` dispatch is guarded on disagreement rather than sent every
   * time: it is idempotent (a second `typing: true` reuses the open turn) but it
   * rebuilds the turn array, and this runs on every reconnect.
   */
  const syncSession = useCallback(async () => {
    const revision = renderRevision.current;
    const conversation = conversationIdRef.current;
    setModelsLoading(true);
    const [sessionResult, modelsResult, defaultModelResult, profileConfigResult] =
      await Promise.allSettled([
        sdk<{
          mode?: "lockdown" | "ask" | "yolo" | null;
          busy?: boolean | null;
          turn_id?: string | null;
          agent_profile?: string | null;
        } | null>("session.get", { details: true }),
        sdk<{ profiles?: LlmProfile[] } | null>("llm.list"),
        sdk<string | null>("config.read", { key: "default_llm_profile" }),
        sdk<Record<string, LlmProfileConfig> | null>("config.read", {
          key: "llm_profiles",
        }),
      ]);

    if (sessionResult.status === "fulfilled") {
      const session = sessionResult.value;
      setSecurityModeState(session?.mode ?? "ask");
      setAgentProfile(session?.agent_profile || "default");
      const busy = Boolean(session?.busy || session?.turn_id);
      if (revision === renderRevision.current && conversation === conversationIdRef.current &&
          busy !== typingRef.current) {
        dispatch(busy && session?.turn_id
          ? { type: "resumeTurn", turnId: session.turn_id }
          : { type: "frame", frame: { kind: "typing", payload: busy } });
      }
    } else {
      report(sessionResult.reason);
    }

    if (modelsResult.status === "fulfilled") {
      setModels(modelsResult.value?.profiles ?? []);
      setModelsFailure(false);
    } else {
      setModelsFailure(true);
      report(modelsResult.reason);
    }

    const defaultModel =
      defaultModelResult.status === "fulfilled"
        ? defaultModelResult.value || null
        : null;
    if (defaultModelResult.status === "fulfilled") {
      setModelName(defaultModel);
    } else {
      setModelsFailure(true);
      report(defaultModelResult.reason);
    }

    // Deliberately not `modelsFailure`: the list and the default both arrived,
    // and a panel that can still switch model is worth more than one that
    // refuses to draw because the extra params did not load.
    if (profileConfigResult.status === "fulfilled") {
      llmProfilesRef.current = profileConfigResult.value ?? {};
    } else {
      report(profileConfigResult.reason);
    }
    setModelsLoading(false);
  }, [report]);

  // On every stream open. A backend restart clears the mode, and a drop long
  // enough to outrun the replay buffer loses whichever `typing` frame moved the
  // turn — either way the composer would go on showing the pre-drop state, and
  // for `busy` that means offering Send where Stop belongs.
  useEffect(() => {
    if (status === "open") void syncSession();
  }, [status, syncSession]);

  /** Set the per-conversation mode from the dedicated composer control.
   * The kernel treats an attended switch to Ask as Lockdown's narrow escape
   * hatch; YOLO remains unsafe and therefore raises the normal approval. */
  const setSecurityMode = useCallback(
    async (mode: "lockdown" | "ask" | "yolo") => {
      try {
        const selected = await sdk<"lockdown" | "ask" | "yolo">(
          "session.set_mode",
          {
            mode,
            scope: "conversation",
          },
        );
        setSecurityModeState(selected ?? mode);
      } catch (error) {
        report(error);
      }
    },
    [report],
  );

  /**
   * Re-read the session after any command, not after three named ones.
   *
   * **The names were `mode`, `llm` and `agent`, and the list was the bug.** It
   * is a client-side guess at which commands can move the security mode, the
   * model or the agent profile — and every other route to the same state was
   * therefore invisible until the stream next opened or the page was reloaded.
   * `/config` writes settings by name, which is one such route; a command
   * installed from the store is another, and it cannot be in a list written
   * before it existed. Same shape of mistake as reading `command.list` once:
   * the app deciding in advance what the server is allowed to have done.
   *
   * Every command instead, because a command is the only thing on this surface
   * that can change any of it — ordinary chat cannot write config — and because
   * running one is rare and deliberate. The cost is one `session.get` and two
   * `config.read`s per command run, which is nothing beside the command itself.
   */
  useEffect(() => {
    if (!state.command?.name) return;
    if (state.command.status !== "finished") return;
    void syncSession();
  }, [state.command?.name, state.command?.status, syncSession]);

  /**
   * Pick up the line a compaction just drew.
   *
   * A marker is a stored row and nothing announces it: the kernel emits its
   * `session_compacted` on the internal bus, which no browser is subscribed to,
   * and `/compact`'s own result card says how much was saved rather than that
   * the conversation now has a seam in it. Without this, the transcript went on
   * looking untouched — with the agent's memory of it already gone — until the
   * conversation was next reopened.
   *
   * **The marker is merged rather than the history replaced.** `history` resets
   * everything transient, and the command's card is on screen at exactly this
   * moment; re-reading would close the panel reporting the thing that was just
   * asked for. So the newest marker is lifted out of the read and appended, and
   * the reducer ignores it if it is one already shown — which is also the
   * answer for a `/compact` that found nothing to compact.
   *
   * Automatic compaction, which the loop performs under context pressure
   * mid-turn, is not covered here: it has no command to hang off, and its
   * marker appears the next time the conversation is read.
   */
  const noteCompaction = useCallback(async () => {
    const id = conversationIdRef.current;
    if (id === null) return;
    try {
      // A small page on purpose: the marker was written moments ago, so it is
      // at the recent end by construction and a default page would be 200 rows
      // fetched to read the last one.
      const { turns } = await readConversation(id, { limit: 20 });
      const marker = [...turns]
        .reverse()
        .find((turn) => turn.role === "system");
      // Not onto somebody else's transcript: switching conversations while
      // this was in flight makes the marker belong to a conversation nobody is
      // looking at any more.
      if (!marker || conversationIdRef.current !== id) return;
      dispatch({ type: "compacted", turn: marker });
    } catch {
      // An optimistic top-up of one row. A failed read costs the line until the
      // conversation is next opened, which is where it comes from anyway.
    }
  }, []);

  useEffect(() => {
    if (state.command?.name !== "compact") return;
    if (state.command.status !== "finished") return;
    // A refused or failed command wrote no marker, so there is nothing to read
    // and the Request can be saved.
    if (state.command.ok === false) return;
    void noteCompaction();
  }, [
    state.command?.name,
    state.command?.status,
    state.command?.ok,
    noteCompaction,
  ]);

  const modelNameRef = useRef<string | null>(null);
  modelNameRef.current = modelName;
  const switchingModelRef = useRef(false);

  /** Change the same global default setting as `/llm`'s Set default action.
   * Optimistic UI keeps the compact control responsive; a failed Request
   * restores the kernel-confirmed value. */
  const setModel = useCallback(
    async (nextModel: string) => {
      if (switchingModelRef.current || nextModel === modelNameRef.current) return;
      const previous = modelNameRef.current;
      switchingModelRef.current = true;
      setSwitchingModel(true);
      setModelName(nextModel);
      try {
        await sdk<boolean>("config.write", {
          key: "default_llm_profile",
          value: nextModel,
          scope: "plugin",
        });
      } catch (error) {
        setModelName(previous);
        await syncSession();
        report(error);
      } finally {
        switchingModelRef.current = false;
        setSwitchingModel(false);
      }
    },
    [report, syncSession],
  );

  /* ── Conversations ──────────────────────────────────────────────── */

  /**
   * Re-read the list from the top, as far as it is currently shown.
   *
   * **One Request, however many pages are open.** Re-reading only the first
   * page would drop everything a person had loaded past it, on a timer; asking
   * for each page again would be a Request per page for the same reason. A
   * single read of `min(what is shown, the server's cap)` is both, and is
   * exactly right: this exists to catch retitles and reordering, and both of
   * those move rows *within* what is already on screen.
   *
   * Past the cap the read covers only the front, so the tail is kept rather
   * than replaced — otherwise a page somebody had asked for would vanish on a
   * timer. A row that moved up appears in the fresh front, so the tail drops
   * its stale copy.
   */
  const refreshConversations = useCallback(async () => {
    try {
      const held = conversationsRef.current;
      const limit = Math.min(
        Math.max(held.length, CONVERSATION_PAGE),
        MOST_CONVERSATIONS_AT_ONCE,
      );
      const page = await listConversations({
        limit,
        category: filterCategory(conversationFilterRef.current),
      });
      // **A short page is the whole list, and is therefore authoritative.**
      // The tail below is kept because a read capped at `limit` cannot speak
      // for rows past it — but a read that came back *under* its own limit was
      // not capped, so every row this user has is in it, and anything still
      // held past its end is a row the server no longer has. Keeping that tail
      // is what left a deleted conversation in the sidebar until a reload: the
      // row it had just lost was the one row the merge put back, every minute,
      // forever.
      const complete = page.items.length < limit;
      const refreshed = new Set(page.items.map((item) => item.id));
      setConversations(
        complete || held.length <= page.items.length
          ? page.items
          : [
              ...page.items,
              ...held
                .slice(page.items.length)
                .filter((item) => !refreshed.has(item.id)),
            ],
      );
      setConversationsHasMore(page.hasMore);
      setConversationCategories(page.categories);
    } catch (error) {
      report(error);
    }
  }, [report]);
  refreshConversationsRef.current = refreshConversations;

  /**
   * Re-read the command catalogue.
   *
   * **`command.list` was read exactly once, at boot.** Which was right when the
   * vocabulary was fixed, and stopped being right the moment a command could be
   * installed — after `/plugin install` the command it just added was missing
   * from Settings, missing from the composer's idea of what a slash command is,
   * and stayed missing until the page was reloaded. The install reports success
   * and the thing installed is nowhere: as far as the app is concerned it did
   * not happen.
   *
   * Kept on failure rather than emptied. A catalogue that momentarily could not
   * be read is not a session with no commands, and Settings drawn empty is the
   * symptom `loadCatalogue` above goes out of its way not to invent.
   */
  const refreshCommands = useCallback(async () => {
    try {
      setCommands(await listCommands());
    } catch (error) {
      report(error);
    }
  }, [report]);

  /**
   * Keep the header's row in step with the list.
   *
   * A conversation the first message just created is not in the copy of the
   * list held when its id was adopted, and `refreshConversations` only sets the
   * list — so without this the header goes on reading "New chat" over a
   * conversation the server has already titled from that message.
   */
  useEffect(() => {
    if (conversationId === null) return;
    const row = conversations.find((item) => item.id === conversationId);
    if (!row) return;
    setOpenConversationRow((held) =>
      held?.id === row.id && held.title === row.title ? held : row,
    );
  }, [conversationId, conversations]);

  /**
   * The next page, appended.
   *
   * Appending rather than replacing is what makes this a "load more" rather
   * than a "jump to page 2": the sidebar is a list you scroll, and a list that
   * replaced itself under you would lose the row you were reaching for.
   */
  const loadMoreConversations = useCallback(async () => {
    try {
      const held = conversationsRef.current;
      const page = await listConversations({
        offset: held.length,
        category: filterCategory(conversationFilterRef.current),
      });
      // By id, because the two reads are a moment apart and a conversation
      // that moved to the top in between would otherwise arrive twice.
      const seen = new Set(held.map((conversation) => conversation.id));
      setConversations([
        ...held,
        ...page.items.filter((item) => !seen.has(item.id)),
      ]);
      setConversationsHasMore(page.hasMore);
      setConversationCategories(page.categories);
    } catch (error) {
      report(error);
    }
  }, [report]);

  /**
   * Show a different slice.
   *
   * **The held rows are cleared first, and that is not cosmetic.** They answer
   * the old question, and every read below is written to preserve what is
   * already loaded — so leaving them would splice one category's conversations
   * into another's. Clearing is also what puts the offset back to zero, since
   * the offset *is* how many rows are held.
   */
  const setConversationFilter = useCallback(
    (filter: ConversationFilter) => {
      conversationFilterRef.current = filter;
      setConversationFilterState(filter);
      writeConversationFilter(filter);
      conversationsRef.current = [];
      setConversations([]);
      void refreshConversations();
    },
    [refreshConversations],
  );

  /**
   * Re-read the list when the agent hands the turn back.
   *
   * The sidebar was only ever refreshed by opening, creating or deleting a
   * conversation — so a list read at boot stayed frozen for the rest of the
   * session: the conversation you were actively talking in never moved to the
   * top, `updated_ago` still claimed "15 seconds ago" an hour later, and a
   * title the kernel assigned after the first exchange never arrived. Once per
   * completed turn is the right cadence: it is when any of that can have
   * changed, and it is far rarer than a frame.
   */
  const wasTyping = useRef(false);
  useEffect(() => {
    if (state.typing) {
      wasTyping.current = true;
      return;
    }
    if (!wasTyping.current) return;
    wasTyping.current = false;
    void refreshConversations();
    // The catalogue too, and for the same reason: installing a command is
    // something a turn *does*, whether it was `/plugin install` typed into the
    // composer or the agent reaching for the same tool on your behalf. The end
    // of the turn is when the new command exists.
    void refreshCommands();
  }, [state.typing, refreshConversations, refreshCommands]);

  /**
   * Ask again while nothing is happening.
   *
   * The refresh above covers everything that changes *because of something
   * done here*. Two things change without that: the store's title sweep runs
   * on its own schedule and renames a conversation minutes after the exchange
   * that earned the name, and `updated_ago` is the server's wording computed
   * when the list was read — so an untouched sidebar goes on claiming "2
   * minutes ago" an hour later. Both are only visible by asking again.
   *
   * A minute, because that is the resolution of the thing being shown: the
   * server's own wording is in minutes once anything is a minute old, so
   * asking more often would re-read the list to render the same sentence.
   *
   * **This is a poll standing in for an event**, and the event exists — the
   * kernel has a `conversation_changed` channel and the sweep emits `retitled`
   * on it — but nothing carries a bus event to a browser: `/events` carries
   * the ten render kinds and the HTTP frontend does not subscribe to anything.
   * Closing that gap is a kernel change (`sandbox/residency.py` wires bus
   * subscriptions for services and not for frontends), and this is what the
   * sidebar needs until it happens.
   *
   * Nothing is asked while a turn is running — the end of it refreshes anyway —
   * and nothing while the tab is hidden, since the answer is only worth having
   * when somebody is looking at it. Coming back to the tab is exactly such a
   * moment, so that asks straight away rather than waiting out the interval.
   */
  useEffect(() => {
    if (status !== "open" || state.typing) return;
    const refreshIfWatched = () => {
      if (!document.hidden) void refreshConversations();
    };
    /**
     * The catalogue, on coming back to the tab only — not on the timer.
     *
     * The turn-end refresh already covers a command installed *here*, so what
     * is left is one installed somewhere else: another browser, or the store on
     * the Mac Mini. That is a thing you do and then come back from, which is
     * exactly what a `visibilitychange` is. Putting it on the minute timer as
     * well would double the idle traffic to catch an install nobody is waiting
     * on — the list only has to be right by the time somebody looks at it, and
     * this fires before they can.
     */
    const refreshOnReturn = () => {
      if (document.hidden) return;
      void refreshConversations();
      void refreshCommands();
    };
    const timer = window.setInterval(refreshIfWatched, IDLE_REFRESH_MS);
    document.addEventListener("visibilitychange", refreshOnReturn);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refreshOnReturn);
    };
  }, [status, state.typing, refreshConversations, refreshCommands]);

  /**
   * Fetch the page above what is on screen and prepend it.
   *
   * The cursor is the oldest row the client actually holds, not a page number:
   * rows arrive while somebody is reading, and an offset would slide under
   * them. Same reason the ledger pages by `since_id`.
   *
   * Failure is deliberately quiet about `hasMore`. Leaving it true means the
   * affordance stays and the person can try again, where clearing it would
   * present a transient network error as the top of the conversation — a lie
   * that cannot be recovered from without reloading the page.
   */
  const loadOlderMessages = useCallback(async () => {
    const id = conversationIdRef.current;
    const before = scrollbackRef.current.oldestId;
    if (id === null || before === null) return;
    if (loadingOlderRef.current) return;
    loadingOlderRef.current = true;
    setLoadingOlderMessages(true);
    try {
      const read = await readConversation(id, { before });
      // Switching conversations mid-flight makes this page belong to one
      // nobody is looking at. Same guard `noteCompaction` makes.
      if (conversationIdRef.current !== id) return;
      dispatch({
        type: "olderTurns",
        turns: read.turns,
        // A page that carried nothing renderable still moves the cursor, or
        // the next request asks for the same rows forever. `hasMore` is
        // conditioned on there being a cursor at all, since without one there
        // is no way to ask again.
        hasMore: read.hasMore && read.oldestId !== null,
        oldestId: read.oldestId ?? scrollbackRef.current.oldestId,
      });
    } catch (error) {
      report(error);
    } finally {
      loadingOlderRef.current = false;
      setLoadingOlderMessages(false);
    }
  }, [report]);

  /**
   * Point the session at another conversation.
   *
   * Not a view change — `conv.load` re-points the *session*, so after this the
   * agent is talking about something else. The scrollback is re-read rather
   * than taken from `conv.load`'s own `history`, because `conv.read` hands back
   * whole rows and `history.ts` knows which of those are the kernel's own
   * bookkeeping and must never reach the screen.
   *
   * Requires the kernel fix that keeps a session's frontend across a load
   * (`runtime/persistence.py`). Without it, one switch hands `http:<thread>` to
   * whichever frontend last had that conversation open and every later Request
   * is refused as somebody else's.
   */
  const openConversation = useCallback(
    async (id: number) => {
      try {
        creatingConversationRef.current = false;
        const result = await sdk<LoadResult>("conv.load", { id });
        if (!result?.ok) {
          // "No such conversation." is also what a conversation this user does
          // not own looks like — the server declines to distinguish them, and
          // repeating its wording is more honest than inventing a reason.
          report(new Error(conversationLoadError(result)));
          return;
        }
        const read = await readConversation(id);
        dispatch({ type: "history", turns: read.turns,
                   hasMore: read.hasMore, oldestId: read.oldestId });
        setConversationId(id);
        setOpenConversationRow(read.conversation);
        await syncSession();
        await refreshConversations();
      } catch (error) {
        report(error);
      }
    },
    [report, refreshConversations, syncSession],
  );

  /**
   * Start a new conversation — which means letting go of the current one, not
   * making anything.
   *
   * The server creates a conversation from the first message sent into it, so
   * there is nothing to create here and nothing to reuse. Pressing New chat
   * twice over costs nothing and leaves nothing behind; this used to make a row
   * per press, which is what filled the sidebar with untouched conversations.
   *
   * **The server-side unbind is not optional.** Clearing our own state only
   * changes what is drawn; the session would still be pointing at the previous
   * conversation, and the next message would land in it.
   */
  const newConversation = useCallback(async () => {
    try {
      await sdk("frontend.submit", {
        input_kind: "action",
        action_type: "new_conversation",
      });
      creatingConversationRef.current = false;
      // Not a `setState` no-op: `history` also clears a half-answered form and
      // a finished command's panel, which is what "start over" means here.
      dispatch({ type: "history", turns: [], hasMore: false, oldestId: null });
      setConversationId(null);
      setOpenConversationRow(null);
      await refreshConversations();
    } catch (error) {
      report(error);
    }
  }, [report, refreshConversations]);

  /**
   * Delete a conversation.
   *
   * **Unsafe, and therefore not a plain call.** The server raises a real
   * approval, which arrives on the event stream while this POST is still open;
   * the modal answers it and only then does this finish. So it may sit here for
   * as long as a person takes, which is the design rather than a hang.
   */
  const deleteConversation = useCallback(
    async (id: number) => {
      try {
        await sdk("conv.delete", { id });
        // **Dropped here as well as re-read below.** The re-read is the truth,
        // but it can only speak for the rows it covers: past
        // `MOST_CONVERSATIONS_AT_ONCE` the merge keeps what is already held,
        // and a row deleted out of that tail would survive its own deletion.
        // The ref is set alongside the state because `refreshConversations`
        // reads it on the next line, before React has re-rendered.
        conversationsRef.current = conversationsRef.current.filter(
          (item) => item.id !== id,
        );
        setConversations(conversationsRef.current);
        await refreshConversations();
        // Deleting the one being read leaves the session pointing at nothing,
        // which is simply the empty-composer state now — the next message
        // starts a conversation. The server has already detached the session
        // (`_detach_deleted_conversation`); this catches our own state up so
        // the header stops naming a conversation that is gone.
        if (id === conversationId) {
          dispatch({ type: "history", turns: [], hasMore: false,
                     oldestId: null });
          setConversationId(null);
          setOpenConversationRow(null);
        }
      } catch (error) {
        report(error);
      }
    },
    [conversationId, refreshConversations, report],
  );

  /* ── Changing what a conversation *is* ──────────────────────────────
   *
   * All three are `ALWAYS_SAFE`, so none of them raises the dialog `conv.delete`
   * above does, and all three refresh the list afterwards — the sidebar draws
   * the title, the category dot and the filter counts, and none of those move
   * on their own.
   */

  const renameConversation = useCallback(
    async (id: number, title: string) => {
      try {
        await setConversationTitle(id, title);
        // The header draws from this row, and the list may not contain the
        // conversation at all — so it is updated here rather than waited for.
        if (id === conversationIdRef.current) {
          setOpenConversationRow((row) => (row ? { ...row, title } : row));
        }
        await refreshConversations();
      } catch (error) {
        report(error);
      }
    },
    [refreshConversations, report],
  );

  const categoriseConversation = useCallback(
    async (id: number, category: string | null) => {
      try {
        await setConversationCategory(id, category);
        if (id === conversationIdRef.current) {
          setOpenConversationRow((row) => (row ? { ...row, category } : row));
        }
        await refreshConversations();
      } catch (error) {
        report(error);
      }
    },
    [refreshConversations, report],
  );

  /* ── The runtime ────────────────────────────────────────────────── */

  const hydrateSentAttachments = useCallback(async (names: string[], text: string) => {
    const id = conversationIdRef.current;
    if (id === null) return;

    // The HTTP frontend acknowledges a background submit before the runtime
    // stores its row. Read only until that row appears, then patch the local
    // user turn without replacing the assistant turn that may be streaming.
    //
    // **Backing off rather than hammering.** Each attempt is a whole
    // `conv.read` — every row of the conversation, re-parsed by `toTurns` — and
    // twelve of them at a flat 150ms went out while the reply to that very
    // message was streaming, over the same tunnel. The row lands on the first
    // or second look in practice; the later attempts exist for the case where
    // the runtime is busy, and that case is not helped by asking more often.
    let wait = 150;
    for (let attempt = 0; attempt < 6; attempt++) {
      if (attempt > 0) {
        await new Promise((settle) => setTimeout(settle, wait));
        wait = Math.min(wait * 2, 1200);
      }
      try {
        // Likewise: this wants the newest user turn, and asking for a whole
        // page of a long conversation six times over — which the backoff below
        // may well do — is the cost this argument removes.
        const { turns } = await readConversation(id, { limit: 20 });
        const user = [...turns].reverse().find((turn) => turn.role === "user");
        const storedText =
          user?.parts
            .filter((part) => part.kind === "text")
            .map((part) => part.text)
            .join("\n") ?? "";
        const files = user?.parts.find(
          (part): part is FilesPart =>
            part.kind === "files" && part.sent === true,
        );
        const attachments = files?.attachments ?? [];
        const storedNames = attachments.map((file) => file.fileName);
        if (
          storedText.trim() === text &&
          storedNames.length === names.length &&
          storedNames.every((name, index) => name === names[index])
        ) {
          if (conversationIdRef.current === id) {
            dispatch({ type: "hydrateSentAttachments", attachments });
          }
          return;
        }
      } catch {
        // This is an optimistic enhancement; the normal history path reports
        // persistent read failures and a later reload can still hydrate it.
      }

      // Nothing to wait for once the conversation has moved on — the patch
      // would land on somebody else's transcript, and the check above already
      // declines to dispatch it.
      if (conversationIdRef.current !== id) return;
    }
  }, []);

  /** Every callback below is `useCallback` with no dependencies, so the adapter
   *  keeps one identity for the life of the page. They talk to the server and
   *  dispatch; neither needs to read state, and rebuilding them on every frame
   *  would make assistant-ui redo work it does not need to. */
  const onNew = useCallback(
    async (message: AppendMessage) => {
      const text = message.content
        .filter((part) => part.type === "text")
        .map((part) => part.text)
        .join("\n")
        .trim();

      // The name travels alongside the path, not derived from it. `fs.temp`
      // makes names like `frontend_http-kr6oyis8.pdf`, and that is what would
      // end up in the attachment cache — so the file the person actually chose
      // has to be named explicitly or it loses its identity on the way in.
      const files = (message.attachments ?? [])
        .map((attachment) => ({
          path: stagedPath(attachment.id),
          name: attachment.name,
          modality:
            attachment.type === "image"
              ? "image"
              : attachment.contentType?.split("/", 1)[0] || "unknown",
          extension: extensionOf(attachment.name),
        }))
        .filter(
          (
            file,
          ): file is {
            path: string;
            name: string;
            modality: string;
            extension: string;
          } => Boolean(file.path),
        );

      const attachments: MessageAttachment[] = files.map((file) => ({
        fileName: file.name,
        modality: file.modality,
        extension: file.extension,
      }));
      const isCommand =
        files.length === 0 && looksLikeCommand(text, commandsRef.current);

      dispatch({
        type: "said",
        text,
        attachments,
        // A message carrying files is a message, whatever it starts with —
        // there is no such thing as a slash command with an attachment.
        isCommand,
      });
      expectOwnConversation(isCommand);

      // Local interaction feedback must not wait for the external runtime to
      // receive and reinterpret a synthetic server frame. Keep this fact
      // separate: it begins synchronously here and a genuine lifecycle frame
      // above hands ownership back to the server.
      if (!isCommand) {
        setSubmitting(true);
      }

      try {
        if (files.length) {
          // Several files are one user action and therefore one state-machine
          // action. A sequence cannot work: the first attachment hands priority
          // to the agent, making every later attachment the wrong actor type.
          await sdk("frontend.submit", attachmentSubmitArgs(files, text));
          // **Before hydrating, not after.** This message may be the one that
          // created the conversation, and `hydrateSentAttachments` reads the
          // id on its first line and gives up when it is null — so the chips
          // would keep their optimistic names for the life of the page.
          await adoptConversation();
          void hydrateSentAttachments(
            files.map((file) => file.name),
            text,
          );
          return;
        }

        await sdk("frontend.submit", { input_kind: "text", text });
        await adoptConversation();
      } catch (error) {
        if (!isCommand) {
          setSubmitting(false);
        }
        report(error);
      }
    },
    [adoptConversation, expectOwnConversation, hydrateSentAttachments, report],
  );

  const onCancel = useCallback(async () => {
    try {
      await sdk("frontend.cancel", {});
    } catch (error) {
      report(error);
    }
  }, [report]);

  /**
   * Sending while the agent still has the turn.
   *
   * **The queue is the kernel's, and this adapter holds none of it.** A message
   * submitted mid-turn is queued server-side and drained at the next loop
   * boundary, with a notification saying so — so there is nothing for a client
   * to hold, dispatch, reorder or edit, and `items`/`steerItems` are empty
   * forever. Everything here is `onNew` under two other names.
   *
   * It is declared anyway because assistant-ui gates the *keyboard* on it:
   * `ComposerPrimitive.Input` swallows Enter outright while `isRunning` unless
   * `thread.capabilities.queue` says the runtime can take a message now
   * (`primitives/composer/ComposerInput`). Clicking Send was never gated —
   * `composer.send` checks only `isEditing`, emptiness and `isSendDisabled` —
   * so without this the button and the Enter key would disagree, which is the
   * worst of the three possible states.
   *
   * `steer` and `enqueue` are the same call because we have one lane. The
   * composer picks `steer` while a run is live and `enqueue` otherwise, and the
   * distinction assistant-ui draws — steering *interrupts*, queueing waits — is
   * not one the kernel offers: a queued message never cancels the turn it
   * arrived during. `move` and `edit` are no-ops for the same reason, and
   * nothing can call them, since the lanes they reorder are always empty.
   */
  const queue = useMemo(
    () => ({
      items: NO_QUEUED_MESSAGES,
      steerItems: NO_QUEUED_MESSAGES,
      enqueue: onNew,
      steer: onNew,
      move: () => {},
      edit: () => {},
      remove: () => {},
    }),
    [onNew],
  );

  /** Re-read the conversation from the server. The escape hatch for a long
   *  disconnect: the replay buffer holds 500 frames per session, so a client
   *  that has been away a while should re-read rather than trust the replay. */
  const onRefetchThread = useCallback(async () => {
    const session = await sdk<{ conversation_id?: number | null } | null>(
      "session.get",
      { details: true },
    );
    const id = session?.conversation_id ?? null;
    if (id === null) return;
    const read = await readConversation(id);
    dispatch({ type: "history", turns: read.turns,
               hasMore: read.hasMore, oldestId: read.oldestId });
  }, []);

  const runtime = useExternalStoreRuntime({
    messages: state.turns,
    convertMessage,
    isLoading: loading,

    // `typing` is the only end-of-turn signal the server has — `false` means the
    // *logical* turn ended, not each internal drive — so it drives this
    // directly rather than being inferred from the last message's status.
    isRunning: state.typing || submitting,

    // A pending question blocks the turn on the server side — and worse, the
    // state machine coerces plain text in that phase into the *answer*, so a
    // message typed now would be eaten by the dialog rather than reaching the
    // agent. Blocking the composer makes that visible instead of surprising.
    isSendDisabled: waiting,

    onNew,
    // Every send actually travels through here — `onNew` is unused once a queue
    // adapter is present, and both of its lanes are `onNew` itself. It stays
    // declared because the adapter type requires it and because it is the
    // honest statement of what a send does.
    queue,
    onCancel,
    onRefetchThread,

    // Quick replies from a store plugin. `buttons` carries {value, label} and
    // the value is submitted as text, same as a form choice.
    // `label` is the field assistant-ui exposes as `SuggestionState.label` and
    // therefore the one a chip can render; `text` — what this used to emit —
    // is not part of that shape and arrived as `undefined` on the other side.
    // `prompt` is what gets submitted, which is the wire's `value`.
    suggestions: useMemo(
      () =>
        state.buttons.map((button) => ({
          prompt: String(button.value),
          label: button.label ?? String(button.value),
        })),
      [state.buttons],
    ),

    adapters: { attachments: attachmentAdapter },

    // Deliberately absent: `onEdit`, `onReload` and any branch adapter. The
    // server has no regenerate and no message tree, and assistant-ui hides the
    // affordances it has no callback for — which is how the UI stays honest
    // about what this backend can actually do.
  });

  const sessionValue = useMemo<SessionDomain>(
    () => ({
      status,
      state,
      submitting,
      say,
      report,
      dismissError,
      dismissCommand,
    }),
    [
      status,
      state,
      submitting,
      say,
      report,
      dismissError,
      dismissCommand,
    ],
  );
  const modelValue = useMemo<ModelDomain>(
    () => ({
      models,
      modelName,
      agentProfile,
      modelsLoading,
      modelsFailure,
      switchingModel,
      setModel,
    }),
    [
      models,
      modelName,
      agentProfile,
      modelsLoading,
      modelsFailure,
      switchingModel,
      setModel,
    ],
  );
  const conversationValue = useMemo<ConversationDomain>(
    () => ({
      conversations,
      conversationsLoaded,
      conversationId,
      openConversation,
      newConversation,
      deleteConversation,
      openConversationRow,
      renameConversation,
      categoriseConversation,
      conversationsHasMore,
      loadMoreConversations,
      scrollbackHasMore,
      loadingOlderMessages,
      loadOlderMessages,
      conversationCategories,
      conversationFilter,
      setConversationFilter,
    }),
    [
      conversations,
      conversationsLoaded,
      conversationId,
      openConversation,
      newConversation,
      deleteConversation,
      openConversationRow,
      renameConversation,
      categoriseConversation,
      conversationsHasMore,
      loadMoreConversations,
      scrollbackHasMore,
      loadingOlderMessages,
      loadOlderMessages,
      conversationCategories,
      conversationFilter,
      setConversationFilter,
    ],
  );
  const approvalValue = useMemo<ApprovalDomain>(
    () => ({
      inputRequests: inputRequests.queue,
      resolve,
      cancelInputRequest,
    }),
    [inputRequests.queue, resolve, cancelInputRequest],
  );
  const notificationValue = useMemo<NotificationDomain>(
    () => ({
      notificationQueue: notifications.notificationQueue,
      notifications: notifications.rows,
      unread: unreadCount(notifications),
      notificationsFailure: notifications.failure,
      dismissQueuedNotification,
      markNotificationsRead,
      notificationsOpen,
      setNotificationsOpen,
    }),
    [
      notifications,
      dismissQueuedNotification,
      markNotificationsRead,
      notificationsOpen,
    ],
  );
  const settingsValue = useMemo<SettingsDomain>(
    () => ({
      commands,
      settingsOpen,
      setSettingsOpen,
      openSettings,
      settingsRequest,
      clearSettingsRequest,
    }),
    [
      commands,
      settingsOpen,
      openSettings,
      settingsRequest,
      clearSettingsRequest,
    ],
  );
  /**
   * Wait for the kernel's binding, which carries name and state together.
   * Changing only the name would mount the new widget with the old one's state.
   */
  const chooseWidget = useCallback((name: string | null) => {
    void setWidget(name).catch((error: unknown) => {
      report(error);
      void getWidget().then(setWidgetBinding, () => {});
    });
  }, [report]);

  useEffect(() => {
    let live = true;
    void getWidget().then(
      (found) => { if (live) setWidgetBinding(found); },
      // A fresh page in no conversation yet is answered rather than refused —
      // the kernel holds the pick on the session until the first message gives
      // it a row. So this is a real failure, and the panel still has to open:
      // an empty binding draws the picker, which is the one thing that can
      // recover from whatever went wrong.
      () => {
        if (live) {
          setWidgetBinding({ name: null, path: "", tree: "",
                             installed: false, state: null,
                             conversationId: null });
        }
      },
    );
    return () => { live = false; };
  }, []);

  const securityValue = useMemo<SecurityDomain>(
    () => ({ securityMode, setSecurityMode }),
    [securityMode, setSecurityMode],
  );
  const widgetValue = useMemo<WidgetDomain>(
    () => ({ widgetBinding, chooseWidget }),
    [widgetBinding, chooseWidget],
  );

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <SessionContext value={sessionValue}>
        <ModelContext value={modelValue}>
          <ConversationContext value={conversationValue}>
            <ApprovalContext value={approvalValue}>
              <NotificationContext value={notificationValue}>
                <SettingsContext value={settingsValue}>
                  <SecurityContext value={securityValue}>
                    <WidgetContext value={widgetValue}>
                      {children}
                    </WidgetContext>
                  </SecurityContext>
                </SettingsContext>
              </NotificationContext>
            </ApprovalContext>
          </ConversationContext>
        </ModelContext>
      </SessionContext>
    </AssistantRuntimeProvider>
  );
}
