/**
 * The chat window.
 *
 * Structurally this is the assistant-ui starter template, with everything the
 * server cannot do taken out: no regenerate, no message editing, no branch
 * picker. The model picker is backed by Second Brain's global configuration
 * SDK rather than assistant-ui's request config because this runtime is an
 * external store.
 *
 * assistant-ui's "primitives" are unstyled components that carry behaviour:
 * `ThreadPrimitive.Messages` knows how to iterate messages, `ComposerPrimitive.
 * Send` knows how to send. They read the runtime through context, which is why
 * nothing here is passed any props about the conversation.
 */

import { NotificationStatus } from "@/components/notification-status";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useReducer,
  useRef,
  useState,
  type FC,
} from "react";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CheckIcon,
  CopyIcon,
  SquareIcon,
} from "lucide-react";
import {
  ActionBarPrimitive,
  AuiIf,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiState,
} from "@assistant-ui/react";

import {
  ComposerAddAttachment,
  ComposerAttachments,
} from "@/components/assistant-ui/attachment";
import { ReplyActivity } from "@/components/reply-activity";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import {
  ToolGroupContent,
  ToolGroupRoot,
  ToolGroupTrigger,
} from "@/components/assistant-ui/tool-group";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { CompactionMarker } from "@/components/compaction-marker";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { UserMessageAttachments } from "@/components/host-file";
import { ErrorBanner } from "@/components/session-bar";
import { ModelSelector } from "@/components/model-selector";
import { SecurityModePicker } from "@/components/security-mode-picker";
import {
  TurnFilesButton,
  TurnOutcomeFiles,
} from "@/components/turn-files";
import { VoiceNoteButton } from "@/components/voice-note";
import { fullTimestamp, shortTimestamp } from "@/lib/time";
import { FINE_POINTER_QUERY, useMediaQuery } from "@/lib/media";
import { cn } from "@/lib/utils";
import { useConversations, useSession } from "@/runtime/domains";
import { AGENT_FILES, PRESENTATION, SENT_AT } from "@/runtime/convert";

/**
 * How a user message's parts are drawn.
 *
 * **A user message ending in a data part is complete, not empty.**
 * `MessagePrimitive.Parts` asks its Empty renderer to fill a message whose last
 * part is non-text, and a voice note is precisely such a message. This shared
 * one object with the assistant, whose `Empty` was `WorkingIndicator` — which
 * put "Working" inside the attachment bubble while the agent replied. The
 * assistant does not come through `Parts` at all now, and there is exactly one
 * place its indicator is placed from; nothing here may grow a second.
 */
const userMessageComponents = {
  Text: MarkdownText,
  Empty: () => null,
  tools: { Fallback: ToolFallback },
} as const;

/**
 * The top of the scrollback, and the way further up.
 *
 * `conv.read` answers with a page rather than a whole conversation, so there
 * is genuinely more above — this is what asks for it.
 *
 * **The scroll anchoring is the whole difficulty.** Prepending rows moves
 * everything already on screen down by however tall the new rows are, so
 * without correction the reader is thrown an arbitrary distance from the line
 * they were reading. Recording `scrollHeight` before the page lands and adding
 * the difference back afterwards keeps the same content under the same pixel.
 * It has to happen before the browser paints, which is what
 * `requestAnimationFrame` inside a layout-effect-shaped callback buys.
 *
 * Loading is triggered by an observer rather than a button press, but the
 * button is still there and still real: an observer that never fires — because
 * the sentinel is offscreen, because the browser lacks the API — leaves the
 * person a way up, and one that fires does not need them to press it.
 */
export const LoadOlder: FC = () => {
  const { scrollbackHasMore, loadingOlderMessages, loadOlderMessages } =
    useConversations();
  const sentinel = useRef<HTMLDivElement | null>(null);
  /** How tall the scrollback was before a page landed, or null when idle.
   *  A ref rather than state: writing it must not cause the render whose
   *  effect is about to read it. */
  const anchor = useRef<{ height: number; top: number } | null>(null);
  const turns = useAuiState((s) => s.thread.messages.length);

  /** Resolve the viewport even before its content is tall enough to scroll. */
  const viewportOf = (node: HTMLElement | null) => {
    return node?.closest<HTMLElement>('[data-slot="chat-viewport"]') ?? null;
  };

  /**
   * Put the reader back on the line they were reading.
   *
   * **In a layout effect, keyed on the turn count** — not in a callback after
   * the `await`. `loadOlderMessages` resolves when the dispatch is *scheduled*,
   * not when React has committed the new rows, so measuring there (even inside
   * `requestAnimationFrame`) can read the pre-prepend height and compute a
   * correction of zero. That produces exactly the jump this exists to prevent,
   * and only sometimes, which is the worst way for it to be wrong. A layout
   * effect runs after commit and before paint, which is the one moment the new
   * height is real and nobody has seen it yet.
   */
  useLayoutEffect(() => {
    const held = anchor.current;
    if (!held || loadingOlderMessages) return;
    anchor.current = null;
    const viewport = viewportOf(sentinel.current);
    if (!viewport) return;
    // Override CSS smooth scrolling so prepending is corrected before paint.
    viewport.scrollTo({ top: held.top + (viewport.scrollHeight - held.height), behavior: "instant" });
  }, [turns, loadingOlderMessages]);

  const loadAnchored = useCallback(async () => {
    const viewport = viewportOf(sentinel.current);
    if (viewport) {
      anchor.current = { height: viewport.scrollHeight, top: viewport.scrollTop };
    }
    await loadOlderMessages();
  }, [loadOlderMessages]);

  useEffect(() => {
    const node = sentinel.current;
    if (!node || !scrollbackHasMore || loadingOlderMessages) return;
    if (typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) void loadAnchored();
      },
      // A little ahead of the edge, so the page is on its way before the reader
      // arrives at the gap rather than after. Restoring the anchor above is
      // also what stops this cascading: the correction scrolls the viewport
      // down past everything that just arrived, which carries the sentinel back
      // out of view. Without it the sentinel stays put and fires again
      // immediately, pulling page after page in one burst.
      { root: viewportOf(node), rootMargin: "400px 0px 0px 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [scrollbackHasMore, loadingOlderMessages, loadAnchored]);

  // Keep the sentinel mounted on the final page: its ref is still needed to
  // restore the anchor after the load button disappears.
  return (
    <div
      ref={sentinel}
      className={cn("mx-auto w-full shrink-0 max-w-(--thread-max-width)", scrollbackHasMore && "py-2")}
      aria-busy={loadingOlderMessages || undefined}
    >
      {!scrollbackHasMore ? null : loadingOlderMessages ? (
        // Standing in for the turns about to arrive, so the gap fills with
        // something the shape of the answer rather than with nothing.
        <div className="flex animate-pulse flex-col gap-y-8" aria-hidden>
          <div className="ml-auto h-10 w-2/5 rounded-2xl bg-muted" />
          <div className="flex flex-col gap-2">
            <div className="h-4 w-11/12 rounded bg-muted" />
            <div className="h-4 w-4/5 rounded bg-muted" />
          </div>
        </div>
      ) : (
        <div className="flex justify-center">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void loadAnchored()}
            className="text-muted-foreground text-xs"
          >
            Load earlier messages
          </Button>
        </div>
      )}
    </div>
  );
};

export const Thread: FC = () => {
  // A conversation with nothing in it centres the composer, the way a new chat
  // does everywhere else. The zero-message geometry applies during loading as
  // well, so an empty chat's composer paints where it will remain instead of
  // jumping up from the bottom. The greeting is still separately guarded by
  // `isLoading` below, because loaded scrollback must not flash a welcome on
  // its way in.
  const centerComposer = useAuiState(
    (s) => s.thread.messages.length === 0,
  );
  const isLoading = useAuiState((s) => s.thread.isLoading);

  return (
    <ThreadPrimitive.Root
      data-empty={centerComposer}
      className="sb-thread @container relative flex h-full flex-col [--composer-bottom-space:max(1rem,env(safe-area-inset-bottom))] md:[--composer-bottom-space:1.5rem]"
      style={{
        ["--thread-max-width" as string]: "44rem",
        ["--composer-bg" as string]:
          "color-mix(in oklab, var(--color-muted) 30%, var(--color-background))",
        ["--composer-radius" as string]: "1.5rem",
      }}
    >
      <ThreadPrimitive.Viewport
        data-slot="chat-viewport"
        turnAnchor="top"
        autoScroll={false}
        scrollToBottomOnRunStart={false}
        className={cn(
          "relative flex min-h-0 flex-1 flex-col overflow-y-scroll motion-safe:pointer-fine:scroll-smooth px-4 pt-4",
          centerComposer && "justify-center",
        )}
      >
        {/* Keep the greeting's exact space while history loads, but hide its
            content. An empty conversation can then reveal it without moving
            the composer, while loaded scrollback never flashes the greeting. */}
        {centerComposer && (
          <div
            aria-hidden={isLoading || undefined}
            className={cn(
              "mx-auto mb-6 w-full max-w-(--thread-max-width) text-center",
              isLoading && "invisible",
            )}
          >
            <img
              src="/second-brain-logotype.png"
              alt="Second Brain"
              draggable={false}
              className="sb-logotype pointer-events-none block h-auto w-full select-none"
            />
          </div>
        )}

        {/* Footer height is reserved in each reply; this gap separates replies. */}
        {/* Outside the `empty:hidden` list below, and above it: this is the
            top of the conversation, and it must not be part of the run of
            messages whose spacing that container owns. */}
        <LoadOlder />

        <div className="mb-8 flex shrink-0 flex-col gap-y-5 empty:hidden">
          <ThreadPrimitive.Messages>
            {({ message }) => {
              // The third role is nobody: a compaction marker, which is in the
              // transcript without being part of the conversation. See
              // `components/compaction-marker.tsx`.
              if (message.role === "user") return <UserMessage />;
              if (message.role === "system") return <CompactionMarker />;
              return <AssistantMessage />;
            }}
          </ThreadPrimitive.Messages>
        </div>

        <ThreadPrimitive.ViewportFooter
          className={cn(
            "sb-composer-footer mx-auto flex w-full max-w-(--thread-max-width) flex-col gap-3 pb-(--composer-bottom-space)",
            !centerComposer &&
              "sticky bottom-0 mt-auto rounded-t-(--composer-radius)",
          )}
        >
          <ScrollToBottom />
          <Suggestions />
          <ErrorBanner />
          <Composer />
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
      <NotificationStatus />

    </ThreadPrimitive.Root>
  );
};

/**
 * Quick replies offered by a store plugin.
 *
 * `buttons` frames have been carried all the way from the wire, through the
 * store, into the runtime's `suggestions` — and then nothing rendered them, so
 * the whole path was inert. Nothing in the kernel emits `buttons` today, which
 * is exactly why this was easy to leave unfinished and hard to notice.
 */
const Suggestions: FC = () => (
  <AuiIf condition={(s) => s.thread.suggestions.length > 0}>
    <div className="flex flex-wrap gap-2">
      <ThreadPrimitive.Suggestions>
        {/* `send`, not the deprecated `autoSend`: a quick reply is an answer,
            so pressing it submits rather than filling the composer in. */}
        {({ suggestion }) => (
          <ThreadPrimitive.Suggestion asChild prompt={suggestion.prompt} send>
            <Button variant="outline" size="sm" className="rounded-full">
              {suggestion.label || suggestion.prompt}
            </Button>
          </ThreadPrimitive.Suggestion>
        )}
      </ThreadPrimitive.Suggestions>
    </div>
  </AuiIf>
);

const ScrollToBottom: FC = () => (
  <ThreadPrimitive.ScrollToBottom asChild>
    <TooltipIconButton
      tooltip="Scroll to bottom"
      variant="outline"
      // `p-4` used to sit here alongside the icon size, which with a working
      // `size` variant would crush the arrow into a 36px button. The variant
      // supplies the border and background now, so the class list only has to
      // say where it floats.
      className="sb-glass bg-background absolute -top-12 z-10 size-9 self-center rounded-full shadow-md disabled:invisible"
    >
      <ArrowDownIcon />
    </TooltipIconButton>
  </ThreadPrimitive.ScrollToBottom>
);

const Composer: FC = () => {
  const finePointer = useMediaQuery(FINE_POINTER_QUERY);
  const input = useRef<HTMLTextAreaElement>(null);

  /**
   * Re-measure the input when its *width* changes.
   *
   * **The autosize behind `ComposerPrimitive.Input` only recalculates on a
   * render or a window resize** (`react-textarea-autosize`: a layout effect
   * with no deps, plus a `window.resize` listener). Nothing else is watching.
   * So any change to the composer's width that is not a window resize leaves
   * the inline height it computed at the old width — too short while a panel
   * opens beside it, and stuck tall after one closes, with the jump arriving
   * later when some unrelated state finally causes a render.
   *
   * The widget panel is what surfaced this, but it is not the cause and the
   * fix does not belong there: collapsing the sidebar does the same thing, and
   * so would any future split. The composer is what has to notice its own box
   * changing.
   *
   * Width only, and compared before bumping — observing height would re-render
   * in response to the very resize it just performed, which is a loop.
   */
  const [, remeasure] = useReducer((tick: number) => tick + 1, 0);
  useEffect(() => {
    const element = input.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    let last = element.getBoundingClientRect().width;
    const observer = new ResizeObserver(([entry]) => {
      const width = entry.contentRect.width;
      if (Math.abs(width - last) < 1) return;
      last = width;
      remeasure();
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const viewport = window.visualViewport;
    if (!viewport) return;
    let frame = 0;
    const syncCaret = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        const element = input.current;
        if (element && document.activeElement === element) {
          // iOS can leave the caret's paint layer at its pre-keyboard viewport
          // coordinates. Reasserting the unchanged selection refreshes it
          // without changing the draft or moving the insertion point.
          element.setSelectionRange(element.selectionStart, element.selectionEnd);
        }
      });
    };
    viewport.addEventListener("resize", syncCaret);
    viewport.addEventListener("scroll", syncCaret);
    return () => {
      window.cancelAnimationFrame(frame);
      viewport.removeEventListener("resize", syncCaret);
      viewport.removeEventListener("scroll", syncCaret);
    };
  }, []);

  return (
      <ComposerPrimitive.Root className="relative flex w-full flex-col">
        <ComposerPrimitive.AttachmentDropzone asChild>
          <div className="sb-composer border-primary/25 data-[dragging=true]:border-ring focus-within:border-primary/60 flex w-full flex-col rounded-(--composer-radius) border bg-(--composer-bg) p-2 data-[dragging=true]:border-dashed">
            <ComposerAttachments />
            <ComposerPrimitive.Input
              ref={input}
              data-slot="chat-composer-input"
              rows={1}
              autoFocus={finePointer}
              unstable_insertNewlineOnTouchEnter
              placeholder="Message Second Brain"
              className="placeholder:text-muted-foreground mb-2 block max-h-40 min-h-10 w-full resize-none bg-transparent px-2.5 py-1 text-base leading-6 outline-none"
            />
            <div className="relative flex min-w-0 items-center gap-1">
              <div className="flex min-w-0 items-center gap-1">
                <ComposerAddAttachment />
                {/* Beside the paperclip because it produces the same thing: a
                    voice note is an attachment, not a second kind of input. */}
                <VoiceNoteButton />
                <SecurityModePicker />
              </div>
              <div className="ms-auto flex min-w-0 flex-1 items-center justify-end gap-1">
                <ModelSelector />
                <ComposerAction />
              </div>
            </div>
          </div>
        </ComposerPrimitive.AttachmentDropzone>

      </ComposerPrimitive.Root>
  );
};

/**
 * The one control in the composer's corner: Send, or Stop.
 *
 * **What decides between them is the box, not the turn.** It used to be
 * `isRunning` alone — a turn was either yours to start or the agent's to stop,
 * never both — which made the composer useless for as long as the agent was
 * working: the only thing you could do with a thought was interrupt with it.
 * The kernel has always taken a message mid-turn and queued it for the next
 * loop boundary, so the restriction was the client's own.
 *
 * So: an empty box while the agent works can only mean stop, and a box with
 * something in it can only mean send it. Typing is what changes the button,
 * which is the rule people already know from Claude Code, and it costs nothing
 * to discover — nobody types into a composer they meant to press Stop on.
 *
 * `isEmpty` counts attachments too, so a staged file with no caption is a send
 * rather than a stop. See `runtime/provider.tsx`'s `queue` for why the Enter
 * key agrees with this button.
 *
 * **Only text is actually queueable**, and that is a kernel fact rather than a
 * choice made here: the busy guard in `runtime/conversation_runtime.py` queues
 * `send_text` and refuses `send_attachment` with "Still working." So a file
 * sent mid-turn comes back as that sentence in the transcript. It is offered
 * anyway because the alternative — a Send that refuses to appear while a file
 * is staged — hides the composer's own state to prevent one legible refusal.
 */
const ComposerAction: FC = () => {
  const runtimeRunning = useAuiState((s) => s.thread.isRunning);
  // `submitting` is read directly from our provider so the control responds in
  // the click's first React commit instead of waiting for assistant-ui's
  // external-store adapter effect to copy the same fact into its runtime.
  const { submitting } = useSession();
  const running = runtimeRunning || submitting;
  const empty = useAuiState((s) => s.composer.isEmpty);

  if (running && empty) {
    return (
      <ComposerPrimitive.Cancel asChild>
        <Button
          type="button"
          variant="default"
          size="icon"
          className="size-7 rounded-full"
          aria-label="Stop generating"
        >
          <SquareIcon className="size-3.5 fill-current" />
        </Button>
      </ComposerPrimitive.Cancel>
    );
  }

  return (
    <ComposerPrimitive.Send asChild>
      <TooltipIconButton
        // Named for what actually happens: mid-turn the kernel holds the
        // message until the agent reaches a loop boundary, and says so with a
        // notification. A button promising "send" and delivering a queue is
        // the sort of small lie that makes people press it twice.
        tooltip={running ? "Queue message" : "Send message"}
        side="bottom"
        type="button"
        variant="default"
        size="icon"
        className="size-7 rounded-full"
      >
        <ArrowUpIcon className="size-4.5" />
      </TooltipIconButton>
    </ComposerPrimitive.Send>
  );
};

export const AssistantMessage: FC = () => {
  const content = useAuiState((s) => s.message.content);

  return (
    <MessagePrimitive.Root
      data-role="assistant"
      className="relative mx-auto w-full max-w-(--thread-max-width)"
    >

      <div
        className="text-foreground relative px-2 leading-relaxed wrap-break-word"

      >
        {/* Activity has one explicit owner below the ordered parts. */}
        <MessagePrimitive.GroupedParts
          indicator="never"
          groupBy={(part) => part.type === "tool-call" ||
            (part.type === "data" && part.name === AGENT_FILES) ||
            (part.type === "text" && !part.text.trim()) ? ["group-tool"] : []}
        >
          {({ part, children }) => {
            switch (part.type) {
              case "group-tool": {
                const count = part.indices.filter((index) => content[index]?.type === "tool-call").length;
                if (!count) return null;
                return (
                  <ToolGroupRoot>
                    <ToolGroupTrigger
                      count={count}
                      active={part.status.type === "running"}
                    />
                    <ToolGroupContent>{children}</ToolGroupContent>
                  </ToolGroupRoot>
                );
              }
              case "text":
                return <MarkdownText />;
              case "tool-call":
                return part.toolUI ?? <ToolFallback {...part} />;
              case "data":
                return part.name === AGENT_FILES ? null : part.dataRendererUI;
              default:
                return null;
            }
          }}
        </MessagePrimitive.GroupedParts>
        {/* After the parts rather than among them: the ledger records that a turn
            showed you a file, not where in the turn it did. See
            `components/turn-files.tsx`. */}
        <TurnOutcomeFiles />
        <MessagePrimitive.Error>
          <ErrorPrimitive.Root className="border-destructive bg-destructive/10 text-destructive mt-2 rounded-md border p-3 text-sm">
            <ErrorPrimitive.Message />
          </ErrorPrimitive.Root>
        </MessagePrimitive.Error>
        <ReplyActivity />
        <AssistantMessageFooter />
      </div>
    </MessagePrimitive.Root>
  );
};

/** Reserved in normal flow even when the controls are hidden. */
const FOOTER_HEIGHT = "h-7";

/** The single action copies the full logical reply, not only its final segment. */
function TurnCopyButton() {
  const messages = useAuiState((s) => s.thread.messages);
  const turnId = useAuiState((s) =>
    (s.message.metadata.custom[PRESENTATION] as { turnId?: string } | undefined)?.turnId ?? s.message.id);
  const [feedback, setFeedback] = useState<"idle" | "copied" | "failed">("idle");
  useEffect(() => {
    if (feedback === "idle") return;
    const timer = window.setTimeout(() => setFeedback("idle"), 3000);
    return () => window.clearTimeout(timer);
  }, [feedback]);
  const text = messages.filter((message) => message.role === "assistant" &&
    ((message.metadata.custom[PRESENTATION] as { turnId?: string } | undefined)?.turnId ?? message.id) === turnId)
    .flatMap((message) => message.parts.flatMap((part) => part.type === "text" ? [part.text] : []))
    .join("\n\n");
  return <TooltipIconButton tooltip={feedback === "failed" ? "Could not copy" : "Copy"}
    side="bottom" className="size-7" disabled={!text} onClick={async () => {
      try { await navigator.clipboard.writeText(text); setFeedback("copied"); }
      catch { setFeedback("failed"); }
    }}>
    {feedback === "copied" ? <CheckIcon className="size-3.5" /> : <CopyIcon className="size-3.5" />}
  </TooltipIconButton>;
}

/** Final row of the reply. Opacity preserves geometry on hover and focus. */
const AssistantMessageFooter: FC = () => {
  const continues = useAuiState((s) =>
    (s.message.metadata.custom[PRESENTATION] as { continues?: boolean } | undefined)?.continues);
  const running = useAuiState((s) => s.message.status?.type === "running");
  const visible = useAuiState(
    (s) =>
      s.message.status?.type !== "running" &&
      (s.message.isLast || s.message.isHovering),
  );

  if (continues) return null;
  if (running) return <div className={FOOTER_HEIGHT} aria-hidden />;
  return (
    <div
      data-slot="assistant-message-footer"
      className={cn(
        // `top-full` is the bottom of the reply; `start-2` matches the padding
        // the text itself sits behind, so the button lines up with the prose
        // rather than with the column edge.
        "mt-1 flex items-center gap-2",
        FOOTER_HEIGHT,
      )}
    >
      <div
        className={cn(
          "flex items-center gap-2 transition-opacity duration-(--sb-motion-reveal)",
          visible ? "opacity-100" : "pointer-events-none opacity-0",
          // Keyboard users get it back, since an invisible control is still in
          // the tab order and focusing it has to show what was focused.
          "focus-within:pointer-events-auto focus-within:opacity-100",
        )}
      >
        <ActionBarPrimitive.Root
          autohide="never"
          className="text-muted-foreground flex items-center gap-1"
        >
          <TurnCopyButton />
        </ActionBarPrimitive.Root>

        <MessageTime />
        {/* Draws nothing for a turn that touched no files, which is most of
            them. See `FOOTER_HEIGHT` for why nothing here may grow taller. */}
        <TurnFilesButton />
      </div>
    </div>
  );
};

/**
 * When the message was sent.
 *
 * Read from `metadata.custom`, not from assistant-ui's own `createdAt`, because
 * that field is defaulted to the present when a message arrives without one —
 * which would date every message of a re-read conversation to the page load.
 * See `runtime/convert.ts`. A turn with no known time simply shows nothing.
 */
const MessageTime: FC = () => {
  const sentAt = useAuiState((s) => {
    const value = s.message.metadata?.custom?.[SENT_AT];
    return typeof value === "number" ? value : undefined;
  });

  if (sentAt === undefined) return null;
  const moment = new Date(sentAt);
  const full = fullTimestamp(moment);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <time
          dateTime={moment.toISOString()}
          // The visible text is abbreviated by design, so the full moment is
          // what assistive technology should hear — it cannot hover, and this
          // is not a control it can focus either.
          aria-label={full}
          // Not text you would ever want to select, and the I-beam over it
          // reads as an invitation to try. `select-none` also keeps it out of
          // a drag-selection that started in the reply above.
          className="text-muted-foreground cursor-default text-[11px] tabular-nums select-none"
        >
          {shortTimestamp(moment)}
        </time>
      </TooltipTrigger>
      {/* `subtle`, and above rather than below: this is a footnote on text
          already on screen, not a control announcing what it does, and the
          buttons beside it own the louder treatment. */}
      <TooltipContent side="top" variant="subtle">
        {full}
      </TooltipContent>
    </Tooltip>
  );
};

const UserMessage: FC = () => (
  <MessagePrimitive.Root
    data-role="user"
    className="fade-in animate-in mx-auto grid w-full max-w-(--thread-max-width) auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] gap-y-2 px-2 duration-(--sb-motion-reveal) [&:where(>*)]:col-start-2"
  >
    <UserMessageAttachments />
    <div className="col-start-2 min-w-0">
      <div className="sb-user-message bg-muted text-foreground rounded-xl px-4 py-2 wrap-break-word empty:hidden">
        <MessagePrimitive.Parts components={userMessageComponents} />
      </div>
    </div>
  </MessagePrimitive.Root>
);
