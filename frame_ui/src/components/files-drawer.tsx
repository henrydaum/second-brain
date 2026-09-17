/** Current-conversation files, displayed inside the left sidebar's list area. */
import { useEffect, useLayoutEffect, useState, type FC, type RefObject } from "react";
import { TerminalIcon } from "lucide-react";

import { FileThumbnail } from "@/components/file-kind-icon";
import { preloadFileViewer } from "@/components/lazy-file-viewer";
import { nameOf } from "@/lib/files";
import { fullTimestamp } from "@/lib/time";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type FileEntry,
} from "@/runtime/file-activity";
import { useFileActivity } from "@/runtime/file-activity-provider";

/** How long a section stays ringed after being jumped to. Long enough to
 *  notice, short enough not to become part of the design. */
const FLASH_MS = 1600;

export type FilesScrollPosition = { conversationId: number | null; top: number; initialized: boolean };

export const FilesDrawer: FC<{
  visible: boolean;
  conversationId: number | null;
  scrollPosition: RefObject<FilesScrollPosition>;
  onOpenFile: () => void;
}> = ({ visible, conversationId, scrollPosition, onOpenFile }) => {
  const {
    sections,
    entries,
    total,
    failure,
    focusTurn,
    focusRequest,
    clearFocus,
    view,
  } = useFileActivity();
  const focusedPaths = new Set(
    focusTurn
      ? sections
          .filter((section) => section.turnId === focusTurn)
          .flatMap((section) => [
            ...section.shown.map((entry) => entry.path),
            ...section.touched.map((entry) => entry.path),
          ])
      : [],
  );

  // Jumping to a section, when the chip under a message asked for one. The
  // clear is on a timer rather than on the scroll finishing, because a smooth
  // scroll has no completion event worth waiting for.
  // The sidebar may mount through a mobile Sheet; observe node arrival too.
  const [body, setBody] = useState<HTMLDivElement | null>(null);
  // The owner keeps this ref across mobile Sheet unmounts and tab changes.
  useLayoutEffect(() => {
    if (scrollPosition.current.conversationId !== conversationId) {
      scrollPosition.current = { conversationId, top: 0, initialized: false };
    }
    if (!visible || !body) return;
    body.scrollTop = scrollPosition.current.top;
  }, [visible, body, conversationId, scrollPosition]);

  useEffect(() => {
    if (!visible || !body || entries.length === 0 || scrollPosition.current.initialized) return;
    scrollPosition.current.initialized = true;
    if (!focusTurn) body.scrollTop = body.scrollHeight;
    scrollPosition.current.top = body.scrollTop;
  }, [visible, body, conversationId, entries.length, focusTurn, scrollPosition]);

  useEffect(() => {
    if (!focusTurn || !visible) return;
    if (!body) return;
    const targets = [...body.querySelectorAll<HTMLElement>("[data-file-path]")]
      .filter((element) => focusedPaths.has(element.dataset.filePath ?? ""));
    if (!targets.length) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const target = targets.at(-1)!;
    const bodyRect = body.getBoundingClientRect();
    const rect = target.getBoundingClientRect();
    const delta = rect.top < bodyRect.top ? rect.top - bodyRect.top - 8
      : rect.bottom > bodyRect.bottom ? rect.bottom - bodyRect.bottom + 8 : 0;
    if (delta) body.scrollTo({ top: body.scrollTop + delta, behavior: reduced ? "instant" : "smooth" });
    const layers = targets.map((element) =>
      element.querySelector<HTMLElement>("[data-file-highlight]")!);
    const animations = layers.map((layer) => {
      if (reduced) { layer.style.opacity = "1"; return null; }
      return layer.animate([
        { opacity: 0, offset: 0, easing: "ease-in-out" },
        { opacity: 1, offset: 0.125 },
        { opacity: 1, offset: 0.6875, easing: "ease-in-out" },
        { opacity: 0, offset: 1 },
      ], { duration: FLASH_MS, easing: "linear" });
    });
    const timer = setTimeout(clearFocus, FLASH_MS);
    return () => {
      clearTimeout(timer);
      animations.forEach((animation) => animation?.cancel());
      layers.forEach((layer) => { layer.style.opacity = ""; });
    };
  }, [focusTurn, focusRequest, visible, clearFocus, sections, body]);

  return (
    <section aria-label="Files in this chat" className="flex min-h-0 flex-1 flex-col">
      <header className="flex shrink-0 items-center gap-2 px-4 pt-3 pb-1 text-xs text-muted-foreground">
        <span>Files in this chat</span>
        {total > 0 && <span className="tabular-nums">{total}</span>}
      </header>
      <div ref={setBody} data-slot="conversation-files-list"
        onScroll={event => { if (visible) scrollPosition.current.top = event.currentTarget.scrollTop; }}
        className="min-h-0 flex-1 overflow-y-auto">
        {failure ? (
          <p className="text-muted-foreground px-4 py-3 text-xs" role="status">{failure}</p>
        ) : entries.length === 0 ? (
          <p className="text-muted-foreground px-4 py-3 text-xs">
            No files in this chat yet. Files the agent shows you or writes will appear here.
          </p>
        ) : (
          <FileList entries={entries} onOpen={(paths, index) => {
            onOpenFile();
            view(paths, index);
          }} />
        )}
      </div>
    </section>
  );
};

const FileList: FC<{
  entries: FileEntry[];
  onOpen: (paths: string[], index: number) => void;
}> = ({ entries, onOpen }) => {
  // One list for the arrows to walk, in the order the section draws them, so
  // "next" in the viewer means what it looks like it means.
  const openable = entries
    .filter((entry) => !entry.gone)
    .map((entry) => entry.path);

  const open = (entry: FileEntry) => {
    const index = openable.indexOf(entry.path);
    if (index >= 0) onOpen(openable, index);
  };

  return (
    <section className="px-2 py-2">
      {/* The turn, and nothing else. No count beside it: the rows underneath
          are the count, and no "shown"/"changed" headings either — a row that
          was changed says so on itself, and one that was not is a row the
          agent showed you. */}
      <ul className="flex flex-col">
        {entries.map((entry) => (
          <Row
            key={entry.path}
            entry={entry}
            onOpen={open}
          />
        ))}
      </ul>
    </section>
  );
};

/** What a section is called. `UNATTRIBUTED` gets a name that says what it is
 *  rather than a time it does not have — see the constant's own note. */
/**
 * One file.
 *
 * **A file that is gone is not a button.** Its path is a record of what
 * happened, not a promise it is still there, and offering to open something
 * that can only answer 404 is an invitation to a dead end. It stays in the list
 * — it is half the point of the list — struck through and inert.
 */
const Row: FC<{
  entry: FileEntry;
  onOpen: (entry: FileEntry) => void;
}> = ({ entry, onOpen }) => {
  const inside = (
    <>
      {/* Any image still on disk gets its own picture, whether the agent showed
          it to you or merely wrote it — a thumbnail identifies a file faster
          than its name does, and there is no reason the two cases should look
          different. A file that is gone gets no `path`, so nothing is fetched
          for it and the icon stands alone. */}
      <FileThumbnail
        name={entry.path}
        path={entry.gone ? undefined : entry.path}
        // The last thing that happened to this file, which is what makes a
        // cached thumbnail safe: the agent editing a picture changes the
        // timestamp, and a changed timestamp is a different cache key.
        version={entry.ts}
        className="size-9 rounded"
        iconClassName="size-4"
      />

      <span className="flex min-w-0 flex-1 flex-col text-start">
        <span
          className={cn(
            "truncate text-xs font-medium",
            entry.gone && "text-muted-foreground line-through",
          )}
        >
          {nameOf(entry.path)}
        </span>
        <Badges entry={entry} />
      </span>
    </>
  );

  const className =
    "relative isolate flex w-full items-center gap-2 rounded-md px-1 py-1.5 text-start";

  // Where the file lives is not in this list at all — not as a line, and not
  // as a tooltip either. The full host path is three times the width of the
  // panel, and the viewer already puts it under the filename the moment you
  // open one. A row is a name, a picture and what happened to it.
  return (
    <li>
      {entry.gone ? (
        <div className={className}>{inside}</div>
      ) : (
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              data-file-path={entry.path}
              type="button"
              onClick={() => onOpen(entry)}
              onPointerEnter={preloadFileViewer}
              onFocus={preloadFileViewer}
              className={cn(
                className,
                "hover:bg-accent focus-visible:bg-accent",
              )}
            >
              <span data-file-highlight aria-hidden className="pointer-events-none absolute inset-0 -z-10 rounded-md bg-accent opacity-0" />
              {inside}
            </button>
          </TooltipTrigger>
          <TooltipContent side="right" variant="subtle">
            {fullTimestamp(new Date(entry.ts))}
          </TooltipContent>
        </Tooltip>
      )}
    </li>
  );
};

const Badges: FC<{ entry: FileEntry }> = ({ entry }) => {
  const words: string[] = [];
  switch (entry.effect) {
    case "shown":
      break;
    case "deleted":
      words.push("deleted");
      break;
    case "moved-from":
      words.push("moved away");
      break;
    case "moved-to":
      words.push(
        entry.movedFrom ? `moved from ${nameOf(entry.movedFrom)}` : "moved",
      );
      break;
    case "wrote":
      words.push(entry.edits > 1 ? `edited ×${entry.edits}` : "edited");
      break;
  }

  if (!words.length && !entry.viaShell) return null;

  return (
    <span className="text-muted-foreground mt-0.5 flex items-center gap-1.5 text-[11px]">
      {words.map((word) => (
        <span key={word} className="bg-muted rounded px-1 py-px">
          {word}
        </span>
      ))}
      {/* A weaker claim than the rest, so it says so. These paths were read out
          of a command line rather than serviced by the kernel. */}
      {entry.viaShell && (
        <span
          className="inline-flex items-center gap-0.5"
          title={entry.command ?? "Read from a shell command line"}
        >
          <TerminalIcon className="size-3" aria-hidden />
          shell
        </span>
      )}
    </span>
  );
};
