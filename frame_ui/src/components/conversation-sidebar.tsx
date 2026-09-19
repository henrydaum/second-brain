/**
 * The conversations list.
 *
 * Switching here is not a view change: `conv.load` re-points the session, so
 * afterwards the agent is genuinely talking about something else. That is why
 * this is disabled mid-turn — swapping the session out from under a running
 * turn would leave the reply landing in a conversation nobody is looking at.
 *
 * Deleting raises a real approval on the server. Nothing special happens here
 * to make that work: the dialog arrives on the event stream and the modal
 * answers it, exactly as it does for anything else consequential.
 */

import {
  memo,
  Suspense,
  useRef,
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useState,
  type CSSProperties,
  type FC,
} from "react";
import {
  MessageSquarePlusIcon,
  PanelLeftCloseIcon,
  PanelLeftOpenIcon,
  SettingsIcon,
  FolderOpenIcon,
  Trash2Icon,
} from "lucide-react";

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { preloadSettings, SettingsDialog } from "@/components/lazy-settings";
import { LazyFilesDrawer, preloadFilesDrawer } from "@/components/lazy-files-drawer";
import type { FilesScrollPosition } from "@/components/files-drawer";
import { useFileActivity } from "@/runtime/file-activity-provider";
import { preloadFileExplorer } from "@/components/lazy-file-explorer";
import { useFileExplorer } from "@/runtime/file-explorer-provider";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  MAIN_CONVERSATIONS_FILTER,
  categoryHues,
  categoryLabel,
  conversationCategory,
  conversationFilterOptions,
  filtersEqual,
  hueOf,
  orderedCategories,
  type ConversationFilter,
} from "@/lib/conversation-categories";
import { conversationTitle, type Conversation } from "@/lib/conversations";
import { MD_QUERY, useMediaQuery } from "@/lib/media";
import { cn } from "@/lib/utils";
import { useApprovals, useConversations, useSession, useSettings } from "@/runtime/domains";

/** Remembered across reloads. A collapse that undoes itself every time the page
 *  loads is a preference the app keeps overruling. */
const COLLAPSED_KEY = "second-brain:sidebar-collapsed";
type CategoryColorStyle = CSSProperties & {
  "--conversation-category-hue": string;
};

/** The hue map is threaded through rather than looked up globally: a colour here
 *  is a function of which categories exist, so there is nothing to ask without
 *  the set in hand. */
function categoryColorStyle(
  category: string,
  hues: Map<string, number>,
): CategoryColorStyle {
  return { "--conversation-category-hue": String(hueOf(hues, category)) };
}

/**
 * The list itself, kept out of the streaming render.
 *
 * **Its own component because the sidebar around it reads the whole session
 * state.** That is legitimate — the rail locks itself while a turn is running,
 * while a form is open, while a question is waiting — but it means the sidebar
 * re-renders on every token of every reply, and the list is the expensive part
 * of it: a row per conversation, each with a tooltip-wrapped button. Nothing
 * here changes token by token, so `memo` and a stable set of props are enough
 * to take the whole list out of that path. Everything it draws is what it drew
 * before, prop for prop.
 */
const ConversationList = memo(function ConversationList({
  conversations,
  activeId,
  locked,
  loaded,
  showCategories,
  categoryColors,
  emptyMessage,
  hasMore,
  onOpen,
  onDelete,
  onLoadMore,
  scrollPosition,
}: {
  conversations: Conversation[];
  activeId: number | null;
  locked: boolean;
  /** Whether the list has been read yet. */
  loaded: boolean;
  showCategories: boolean;
  categoryColors: Map<string, number>;
  emptyMessage: string;
  /** Whether the server says there is another page behind this one. */
  hasMore: boolean;
  onOpen: (id: number) => void;
  onDelete: (id: number) => void;
  onLoadMore: () => void;
  scrollPosition: RefObject<number>;
}) {
  const restoreScroll = useCallback((node: HTMLElement | null) => {
    if (node) node.scrollTop = scrollPosition.current;
  }, [scrollPosition]);
  return (
    <nav aria-label="Conversations" ref={restoreScroll}
      onScroll={event => { scrollPosition.current = event.currentTarget.scrollTop; }}
      className="min-h-0 flex-1 overflow-y-auto p-2 pt-2">
      {/* Nothing at all until the list has been asked for. Boot takes a few
          Requests to get here, and for all of them the old empty message was
          telling you that you had no conversations — which is a claim, and
          usually a false one. */}
      {loaded && conversations.length === 0 && (
        <p className="text-muted-foreground px-2 py-4 text-xs">
          {emptyMessage}
        </p>
      )}

      {conversations.map((conversation) => {
        const active = conversation.id === activeId;
        const category = conversationCategory(conversation.category);
        const showCategory = showCategories && category !== null;
        return (
          <div
            key={conversation.id}
            data-slot="conversation"
            data-active={active || undefined}
            className={cn(
              "group relative flex items-center gap-1 rounded-md",
              active ? "bg-accent" : "hover:bg-accent/50",
            )}
          >
            <button
              type="button"
              disabled={locked}
              onClick={() => onOpen(conversation.id)}
              // The delete button below is positioned over this rather than
              // beside it, so a title has the whole row until something wants
              // the corner. Then — and only then — this makes way for it. It
              // used to give up that width permanently, including on a phone,
              // where the button it was making way for never appears at all.
              className={cn(
                "min-w-0 flex-1 px-2 py-1.5 text-start disabled:opacity-50",
                "pointer-fine:group-hover:pe-9 pointer-fine:group-focus-within:pe-9",
              )}
            >
              <span className="block truncate text-sm">
                {conversationTitle(conversation)}
              </span>
              {(conversation.updated_ago || showCategory) && (
                // The server's own wording, rather than a second notion of
                // "recent" computed here that could disagree with it — and
                // relative wording rather than a date, because a list read by
                // recency wants "how recent" answered without arithmetic.
                <span className="text-muted-foreground flex min-w-0 items-center gap-1.5 text-xs">
                  {conversation.updated_ago && (
                    <span className="truncate">{conversation.updated_ago}</span>
                  )}
                  {showCategory && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span
                          role="img"
                          aria-label={`Category: ${categoryLabel(category)}`}
                          className="conversation-category-dot size-2 shrink-0 rounded-full"
                          style={categoryColorStyle(category, categoryColors)}
                        />
                      </TooltipTrigger>
                      <TooltipContent variant="subtle" side="right">
                        {categoryLabel(category)}
                      </TooltipContent>
                    </Tooltip>
                  )}
                </span>
              )}
            </button>

            <TooltipIconButton
              tooltip="Delete"
              side="right"
              className={cn(
                "absolute end-1 top-1/2 size-7 -translate-y-1/2",
                // Present on hover or focus only — a destructive control
                // sitting permanently beside every row is one that gets hit
                // by accident. It stays keyboard-reachable.
                "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
                // **Nothing at all where there is no hover.** Revealed by
                // pointing, this was invisible on a phone yet still tappable,
                // which is the worst of both: an unannounced destructive
                // control under your thumb. The conversation menu in the header
                // carries Delete there instead.
                "hidden pointer-fine:inline-flex",
              )}
              disabled={locked}
              onClick={() => onDelete(conversation.id)}
            >
              <Trash2Icon className="size-3.5" />
            </TooltipIconButton>
          </div>
        );
      })}

      {/* A button rather than infinite scroll. The sidebar is also the thing
          you scroll to *find* an old conversation, and a list that grows under
          you while you are reading it is worse than one you ask to grow. */}
      {hasMore && (
        <button
          type="button"
          disabled={locked}
          onClick={onLoadMore}
          className="text-muted-foreground hover:text-foreground hover:bg-accent/50 mt-1 w-full rounded-md px-2 py-1.5 text-xs disabled:opacity-50"
        >
          Load more
        </button>
      )}
    </nav>
  );
});

type ConversationSidebarProps = {
  /** Whether the overlay drawer is showing. Only meaningful below `md`, where
   *  this is a drawer rather than an inline rail. */
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export const ConversationSidebar: FC<ConversationSidebarProps> = ({
  open,
  onOpenChange,
}) => {
  const {
    conversations,
    conversationsLoaded,
    conversationId,
    openConversation,
    newConversation,
    deleteConversation,
    conversationsHasMore,
    loadMoreConversations,
    conversationCategories,
    conversationFilter,
    setConversationFilter,
  } = useConversations();
  const { inputRequests } = useApprovals();
  const { state, status } = useSession();
  const { settingsOpen, setSettingsOpen } = useSettings();
  const [settingsMounted, setSettingsMounted] = useState(settingsOpen);
  const { openExplorer } = useFileExplorer();
  const { filesOpen, setFilesOpen, focusRequest } = useFileActivity();
  const [filesMounted, setFilesMounted] = useState(filesOpen);
  const chatScroll = useRef(0);
  const fileScroll = useRef<FilesScrollPosition>({ conversationId, top: 0, initialized: false });

  // One switch at a time. Each of these is several Requests, and a second click
  // partway through would interleave two loads into one session.
  const [busy, setBusy] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);
  const commandRunning = Boolean(
    state.command && state.command.status !== "finished",
  );
  const locked =
    busy ||
    status !== "open" ||
    state.typing ||
    state.form !== null ||
    inputRequests.length > 0 ||
    commandRunning;

  // Read lazily so the stored preference applies on the first paint rather than
  // expanding and then snapping shut.
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(COLLAPSED_KEY) === "true",
  );
  useEffect(() => {
    localStorage.setItem(COLLAPSED_KEY, String(collapsed));
  }, [collapsed]);

  /**
   * The menu, and the colours, both from the server's tally.
   *
   * **Not from `conversations`.** That is one page of one category now — the
   * server does the filtering — so a menu derived from it would list only the
   * category you are already looking at, and count only the rows you happen to
   * have loaded.
   */
  const filterOptions = useMemo(
    () => conversationFilterOptions(conversationCategories),
    [conversationCategories],
  );
  /** One assignment for the whole sidebar, so the pill, the menu dot and the
   *  dot on a row all agree about what colour a category is. */
  const categoryColors = useMemo(
    () => categoryHues(orderedCategories(conversationCategories)),
    [conversationCategories],
  );
  const selectedFilter =
    filterOptions.find((option) =>
      filtersEqual(option.filter, conversationFilter),
    ) ?? filterOptions[1];
  const selectedFilterLabel =
    selectedFilter.filter.type === "all" ? "All" : selectedFilter.label;
  const filterValue = (filter: ConversationFilter) =>
    filter.type === "all" ? "all" : `category:${filter.category ?? "main"}`;

  /**
   * The filter names a category that no longer exists.
   *
   * **A category is not a record — it is a value some conversation holds.** So
   * it stops existing the moment the last conversation leaves it, with nothing
   * deleted and nothing to delete. That is the right model, and this is the one
   * loose end it leaves: empty the category you are *filtered to* and the
   * sidebar would sit on a filter nothing can match, showing an empty list
   * indistinguishable from "you have none" while the pill quietly relabelled
   * itself Main.
   *
   * So it falls back whenever that happens, not once at startup — emptying a
   * category is something you do *while looking at it*. This settles rather
   * than loops: Main is always among the options, so the filter it moves to is
   * one the next run finds.
   *
   * An empty tally is "not asked yet" rather than "no categories", which is why
   * it is not treated as everything having disappeared.
   */
  useEffect(() => {
    if (conversationCategories.length === 0) return;
    const known = filterOptions.some((option) =>
      filtersEqual(option.filter, conversationFilter),
    );
    if (!known) setConversationFilter(MAIN_CONVERSATIONS_FILTER);
  }, [
    conversationCategories,
    filterOptions,
    conversationFilter,
    setConversationFilter,
  ]);

  /**
   * Two different components sharing one file.
   *
   * Above `md` this is an inline rail that collapses to 48px, which is the
   * behaviour it always had. Below `md` it was *also* that — a fixed 256px
   * column on a 375px phone, leaving about 119px of chat, with no way to get it
   * back except finding and pressing a collapse button inside the thing that
   * was in the way. Below `md` it is now an overlay drawer, which is what every
   * app this one is imitating does.
   *
   * `collapsed` is a rail concept and must not leak into the drawer: a person
   * who collapsed the rail on a laptop should not find an empty drawer on their
   * phone. Hence the media query — the list is unmounted rather than hidden, so
   * this is a genuine behavioural fork that CSS cannot express.
   */
  const isDesktop = useMediaQuery(MD_QUERY);
  const railCollapsed = isDesktop && collapsed;

  // A message's file-count button selects Files and reveals the left sidebar,
  // including repeated requests while Files is selected but the sidebar is shut.
  useEffect(() => {
    if (!filesOpen) return;
    setFilesMounted(true);
    setCollapsed(false);
    if (!isDesktop) onOpenChange(true);
  }, [filesOpen, focusRequest, isDesktop, onOpenChange]);

  useEffect(() => {
    if (filesOpen) setFilterOpen(false);
  }, [filesOpen]);

  useEffect(() => {
    if (railCollapsed) setFilterOpen(false);
  }, [railCollapsed]);

  // Picking a conversation on a phone means you are done with the drawer.
  const closeDrawer = useCallback(() => onOpenChange(false), [onOpenChange]);

  // Settings is a secondary chunk. Warm it after the first paint, and again on
  // intent below, so code-splitting does not turn the familiar gear into a
  // delayed first interaction.
  useEffect(() => {
    const timer = window.setTimeout(preloadSettings, 700);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (settingsOpen) setSettingsMounted(true);
  }, [settingsOpen]);

  const showSettings = () => {
    preloadSettings();
    if (!isDesktop) closeDrawer();
    setSettingsOpen(true);
  };

  const showExplorer = () => {
    preloadFileExplorer();
    if (!isDesktop) closeDrawer();
    openExplorer();
  };

  useEffect(() => {
    const name = state.form?.name ?? state.command?.name;
    if (name) setSettingsOpen(true);
  }, [state.form?.name, state.command?.name, setSettingsOpen]);

  const run = useCallback(async (work: () => Promise<void>) => {
    setBusy(true);
    try {
      await work();
    } finally {
      setBusy(false);
    }
  }, []);

  /**
   * The list's two actions, held still.
   *
   * `ConversationList` below is memoised, and a handler rebuilt on every render
   * would defeat that entirely — which matters because this component reads the
   * whole session state and therefore re-renders on every streamed token.
   */
  const openAndClose = useCallback(
    (id: number) => void run(() => openConversation(id)).then(closeDrawer),
    [run, openConversation, closeDrawer],
  );
  const removeConversation = useCallback(
    (id: number) => void run(() => deleteConversation(id)),
    [run, deleteConversation],
  );
  // Not through `run`: `busy` locks every row, and paging is not a session
  // switch — there is nothing to interleave and nothing to protect.
  const loadMore = useCallback(
    () => void loadMoreConversations(),
    [loadMoreConversations],
  );

  const startNewConversation = () => {
    setConversationFilter(MAIN_CONVERSATIONS_FILTER);
    void run(newConversation).then(closeDrawer);
  };

  /**
   * The panel itself, positioned by whoever is showing it.
   *
   * **It carries no off-canvas machinery of its own.** Below `md` it is handed
   * to `Sheet` below, which is already a fixed, scrimmed, animated overlay that
   * unmounts when closed; above `md` it is an inline rail and animates its own
   * width. Those two paths are mutually exclusive — `isDesktop` picks one — so
   * a `fixed`/`translate-x` pair here was overridden by the `md:` variants in
   * the one case and layered underneath an identical `SheetContent` in the
   * other. Two mechanisms, one job, and neither branch used both halves.
   */
  const sidebar = (
    <aside
      data-slot="conversation-sidebar"
      data-collapsed={railCollapsed || undefined}
      className={cn(
        // `w-full` below `md`, so the panel fills whatever `SheetContent` gave
        // it — including the `max-w-[85vw]` that keeps a drawer off the edge of
        // a narrow phone, which a fixed `w-64` here would have overrun.
        "sb-panel sb-divider-end bg-sidebar flex h-full w-full flex-col overflow-hidden",
        // From `md`: an inline rail, transitioning width so collapsing
        // animates rather than sliding the whole panel.
        "md:shrink-0 md:transition-[width]",
        railCollapsed ? "md:w-12" : "md:w-64",
      )}
    >
      {!isDesktop && (
        <SheetTitle className="sr-only">Chats and files</SheetTitle>
      )}
      {/* New chat remains pinned to the rail while its label is revealed. The
          drawer toggle follows the moving outer edge, matching the panel it
          opens and closes. */}
      <div className="sb-divider-bottom relative grid shrink-0 grid-cols-[2rem_minmax(0,1fr)_auto] gap-x-1 p-2 pt-12">
        {/* Match the session bar's 48px row and center its controls, including
            the larger touch targets, instead of offsetting them from the top. */}
        <div className="absolute inset-x-0 top-0 flex h-12 items-center justify-between px-2">
        {!railCollapsed && (
          <div role="group" aria-label="Sidebar view"
            className="relative isolate ml-[3px] flex h-8 items-center px-0.5 before:pointer-events-none before:absolute before:inset-0 before:-z-10 before:rounded-full before:bg-muted/60 pointer-coarse:h-12 pointer-coarse:before:inset-y-1.5">
            {([false, true] as const).map(files => (
              <button key={String(files)} type="button" aria-pressed={filesOpen === files}
                aria-controls={files ? "sidebar-files" : "sidebar-chats"}
                onPointerEnter={files ? preloadFilesDrawer : undefined}
                onFocus={files ? preloadFilesDrawer : undefined}
                onClick={() => setFilesOpen(files)}
                className="group flex h-full items-center rounded-full text-xs font-medium focus-visible:outline-2 focus-visible:outline-ring">
                {/* Paint a 36px pill with a 32px selection inside it while
                    keeping each mobile button's full 48px touch target. */}
                <span className={cn("flex items-center rounded-full px-3 py-1 transition-colors pointer-coarse:h-8",
                  filesOpen === files ? "bg-accent text-foreground" : "text-muted-foreground group-hover:text-foreground")}>
                  {files ? "Files" : "Chats"}
                </span>
              </button>
            ))}
          </div>
        )}
        {/* Two buttons, not one with a media query in JavaScript: on a phone
            this closes an overlay, on a laptop it collapses a rail. They use
            the same panel-shaped icon because the surface is the same even
            though the underlying action differs. */}
        <TooltipIconButton
          tooltip="Hide sidebar"
          side="right"
          className="ml-auto size-8 md:hidden"
          onClick={closeDrawer}
        >
          <PanelLeftCloseIcon className="size-4 translate-x-[0.5px]" />
        </TooltipIconButton>
        <TooltipIconButton
          tooltip={railCollapsed ? "Show sidebar" : "Hide sidebar"}
          side="right"
          className="ml-auto hidden size-8 md:inline-flex"
          aria-expanded={!railCollapsed}
          onClick={() => setCollapsed((value) => !value)}
        >
          {railCollapsed ? (
            <PanelLeftOpenIcon className="size-4 translate-x-[0.5px]" />
          ) : (
            <PanelLeftCloseIcon className="size-4 translate-x-[0.5px]" />
          )}
        </TooltipIconButton>
        </div>

        <TooltipIconButton
          tooltip="New chat"
          side="right"
          className="col-start-1 size-8"
          disabled={locked}
          onClick={startNewConversation}
        >
          <MessageSquarePlusIcon className="size-4" />
        </TooltipIconButton>
        <button
          type="button"
          disabled={locked}
          tabIndex={railCollapsed ? -1 : undefined}
          aria-hidden={railCollapsed || undefined}
          onClick={startNewConversation}
          className={cn(
            "text-muted-foreground hover:text-foreground col-start-2 min-w-0 truncate px-1 text-start text-sm transition-opacity disabled:opacity-50",
            railCollapsed ? "pointer-events-none opacity-0" : "opacity-100",
          )}
        >
          New chat
        </button>

        {!railCollapsed && !filesOpen && (
          <DropdownMenu open={filterOpen} onOpenChange={setFilterOpen}>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                aria-label={`Filter conversations: ${selectedFilter.label}`}
                className="group col-start-3 flex min-w-0 max-w-28 self-center items-center justify-center"
              >
                {/* Keep the 44px mobile touch target, but paint only the
                    compact inner pill. Painting the target itself turned short
                    labels such as All into circles. */}
                <span
                  className={cn(
                    "flex h-6 min-w-0 max-w-full items-center rounded-full px-2 text-xs font-medium transition-colors",
                    conversationFilter.type === "all"
                      ? "bg-primary/10 text-primary"
                      : conversationFilter.category === null
                        ? "bg-muted text-muted-foreground group-hover:text-foreground"
                        : "conversation-category-pill",
                  )}
                  style={
                    conversationFilter.type === "category" &&
                    conversationFilter.category !== null
                      ? categoryColorStyle(conversationFilter.category, categoryColors)
                      : undefined
                  }
                >
                  <span className="truncate">{selectedFilterLabel}</span>
                </span>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="end"
              sideOffset={6}
              className="w-60"
            >
              <DropdownMenuRadioGroup
                aria-label="Conversation category"
                value={filterValue(conversationFilter)}
                onValueChange={(value) => {
                  const option = filterOptions.find(
                    (candidate) => filterValue(candidate.filter) === value,
                  );
                  if (option) setConversationFilter(option.filter);
                }}
              >
                {filterOptions.map((option) => {
                const category =
                  option.filter.type === "category"
                    ? option.filter.category
                    : null;
                return (
                  <DropdownMenuRadioItem
                    key={filterValue(option.filter)}
                    value={filterValue(option.filter)}
                    aria-label={`${option.label}, ${option.count} conversations`}
                    title={option.label}
                    className="gap-2"
                  >
                    <span
                      aria-hidden
                      className={cn(
                        "size-2.5 shrink-0 rounded-full",
                        option.filter.type === "all"
                          ? "bg-primary"
                          : category === null
                            ? "bg-muted-foreground/60"
                            : "conversation-category-dot",
                      )}
                      style={
                        category === null
                          ? undefined
                          : categoryColorStyle(category, categoryColors)
                      }
                    />
                    <span className="min-w-0 flex-1 truncate">
                      {option.label}
                    </span>
                    <span className="text-muted-foreground text-xs tabular-nums">
                      {option.count}
                    </span>
                  </DropdownMenuRadioItem>
                );
                })}
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      {/* Keep both list surfaces mounted when switching so their positions stay put.
          The header (including New chat) and footer belong to neither list. */}
      <div id="sidebar-chats" hidden={railCollapsed || filesOpen}
        className={cn("min-h-0 flex-1 flex-col overflow-hidden", railCollapsed || filesOpen ? "hidden" : "flex")}>
        <ConversationList
          conversations={conversations}
          activeId={conversationId}
          locked={locked}
          loaded={conversationsLoaded}
          showCategories={conversationFilter.type === "all"}
          categoryColors={categoryColors}
          emptyMessage={
            conversations.length === 0
              ? "No conversations yet."
              : `No ${selectedFilter.label} conversations.`
          }
          hasMore={conversationsHasMore}
          onOpen={openAndClose}
          onDelete={removeConversation}
          onLoadMore={loadMore}
          scrollPosition={chatScroll}
        />
      </div>
      <div id="sidebar-files" hidden={railCollapsed || !filesOpen}
        className={cn("min-h-0 flex-1 flex-col overflow-hidden", railCollapsed || !filesOpen ? "hidden" : "flex")}>
        {filesMounted && (
          <Suspense fallback={<p className="text-muted-foreground p-4 text-xs" role="status">Loading files...</p>}>
            <LazyFilesDrawer visible={filesOpen && !railCollapsed && (isDesktop || open)}
              conversationId={conversationId} scrollPosition={fileScroll}
              onOpenFile={() => { if (!isDesktop) closeDrawer(); }} />
          </Suspense>
        )}
      </div>

      {/* `mt-auto` is what pins this to the bottom in both states — with the
          list unmounted there is nothing else to push it down. Separated by a
          rule, because it is not another conversation. */}
      <div className="sb-divider-top mt-auto grid shrink-0 grid-cols-[2rem_1fr] gap-x-1 gap-y-0 p-2 md:gap-y-1">
        <TooltipIconButton tooltip="File explorer" side="right" className="size-8"
          onPointerEnter={preloadFileExplorer} onFocus={preloadFileExplorer} onClick={showExplorer}>
          <FolderOpenIcon className="size-4" />
        </TooltipIconButton>
        <button type="button" tabIndex={railCollapsed ? -1 : undefined}
          aria-hidden={railCollapsed || undefined} onPointerEnter={preloadFileExplorer}
          onFocus={preloadFileExplorer} onClick={showExplorer}
          className={cn("text-muted-foreground hover:text-foreground min-w-0 truncate px-1 text-start text-sm transition-opacity", railCollapsed ? "pointer-events-none opacity-0" : "opacity-100")}>
          File explorer
        </button>
        <TooltipIconButton
          tooltip="Settings"
          side="right"
          className="size-8"
          onPointerEnter={preloadSettings}
          onFocus={preloadSettings}
          onClick={showSettings}
        >
          <SettingsIcon className="size-4" />
        </TooltipIconButton>
        <button
          type="button"
          tabIndex={railCollapsed ? -1 : undefined}
          aria-hidden={railCollapsed || undefined}
          onPointerEnter={preloadSettings}
          onFocus={preloadSettings}
          onClick={showSettings}
          className={cn(
            "text-muted-foreground hover:text-foreground min-w-0 truncate px-1 text-start text-sm transition-opacity",
            railCollapsed ? "pointer-events-none opacity-0" : "opacity-100",
          )}
        >
          Settings
        </button>
      </div>
    </aside>
  );

  const dialogs = <>
    {settingsMounted && <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />}
  </>;

  if (isDesktop) {
    return (
      <>
        {sidebar}
        {dialogs}
      </>
    );
  }

  return (
    <>
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent side="left" className="w-80 max-w-[85vw] border-0 pt-[env(safe-area-inset-top)]">
          {sidebar}
        </SheetContent>
      </Sheet>
      {dialogs}
    </>
  );
};
