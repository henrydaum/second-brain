/**
 * One file, big enough to actually look at.
 *
 * The drawer is a list and stays one — at 24rem a chart is a thumbnail and a
 * table is unreadable. Opening a file therefore takes over the middle of the
 * screen, and brings its neighbours with it: the arrows step through the files
 * of the section it was opened from, because "the next one" is the thing you
 * want after looking at a file, and going back to the list for it is a click
 * per file.
 */

import { useEffect, useRef, type FC } from "react";
import { ChevronLeftIcon, ChevronRightIcon, DownloadIcon, FolderOpenIcon } from "lucide-react";

import { FileView } from "@/components/file-view";
import { MarkdownModePicker } from "@/components/markdown-mode";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { fileUrl } from "@/lib/client";
import { guessKind, nameOf } from "@/lib/files";
import { useFileActivity } from "@/runtime/file-activity-provider";
import { useFileExplorer } from "@/runtime/file-explorer-provider";

export const FileViewerDialog: FC = () => {
  const { viewing, stepView, closeView } = useFileActivity();
  const { openExplorer } = useFileExplorer();
  const revealing = useRef(false);

  /**
   * Arrow keys, bound to the window rather than to the dialog: the focus after
   * opening sits on whatever Radix chose, and a viewer you cannot page through
   * without first clicking it is a viewer that feels broken.
   *
   * **Unless the arrows already mean something.** A focused `<video>` or
   * `<audio>` scrubs with the same two keys, so paging the viewer as well made
   * nudging back five seconds jump to the next file instead. Whatever has focus
   * has first claim on a key it uses.
   */
  useEffect(() => {
    if (!viewing || viewing.paths.length < 2) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const focused = document.activeElement;
      if (focused instanceof HTMLMediaElement) return;
      if (focused instanceof HTMLElement && focused.isContentEditable) return;
      if (focused instanceof HTMLInputElement) return;
      // A wide table scrolls sideways with these same two keys now that its box
      // can hold focus. Same rule as the media elements above: whatever has
      // focus has first claim on a key it uses.
      if (
        focused instanceof HTMLElement &&
        focused.dataset.slot === "file-view" &&
        focused.scrollWidth > focused.clientWidth
      ) {
        return;
      }
      stepView(event.key === "ArrowLeft" ? -1 : 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [viewing, stepView]);

  if (!viewing) return null;

  const path = viewing.paths[viewing.index];
  if (path === undefined) return null;

  const many = viewing.paths.length > 1;

  return (
    <Dialog open onOpenChange={(open) => !open && closeView()}>
      <DialogContent
        className="sb-glass sb-glass-sheet h-[min(92dvh,52rem)] min-w-0 w-[calc(100vw-1rem)] max-w-none grid-rows-[auto_minmax(0,1fr)_auto] gap-3 overflow-hidden p-3 sm:w-full sm:max-w-4xl sm:p-4"
        overlayClassName="bg-black/50"
        onCloseAutoFocus={(event) => { if (revealing.current) event.preventDefault(); }}
        /**
         * Focus the document itself where there is one, so the arrow keys
         * scroll it without a click first.
         *
         * The frame usually is not there yet — the bytes are still in flight
         * and what is mounted is the skeleton — in which case the dialog takes
         * focus as it always did, and the frame claims it a moment later when
         * it mounts. This branch is for the other case: a file already in the
         * text cache renders before Radix gets here, and without it the dialog
         * would take focus straight back off the document.
         */
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          const content = event.currentTarget as HTMLElement | null;
          const document_ = content?.querySelector<HTMLElement>(
            '[data-slot="file-view"][tabindex]',
          );
          (document_ ?? content)?.focus();
        }}
      >
        <DialogHeader className="min-w-0 pe-12">
          <DialogTitle className="truncate text-base" title={path}>
            {nameOf(path)}
          </DialogTitle>
          <DialogDescription
            className="overflow-hidden text-left text-xs text-ellipsis whitespace-nowrap [direction:rtl]"
            title={path}
            aria-label={path}
          >
            {path}
          </DialogDescription>
        </DialogHeader>

        {/* The file and its arrows on one row, so paging does not move the
            thing you are looking at. */}
        <div className="relative flex min-h-0 min-w-0 items-center sm:gap-2">
          {many && (
            <TooltipIconButton
              tooltip="Previous file"
              side="right"
              className="bg-background/40! text-foreground/60! hover:bg-background/40! hover:text-foreground/60! active:bg-background/55! absolute start-1 z-10 size-8 rounded-full shadow-sm backdrop-blur-sm sm:static sm:bg-transparent! sm:text-foreground! sm:hover:bg-accent! sm:hover:text-accent-foreground! sm:shadow-none sm:backdrop-blur-none"
              onClick={() => stepView(-1)}
            >
              <ChevronLeftIcon className="size-4" />
            </TooltipIconButton>
          )}
          {/**
           * The stage, and its size is fixed on purpose.
           *
           * Every file that lands here is a different shape, and each one takes
           * a moment to arrive — so a box that sized itself to its contents
           * spent that moment as a sliver and then sprang open, moving the
           * dialog, the arrows and the download link out from under the
           * pointer. Deciding the dimensions up front costs some empty space
           * around a small image and buys a viewer that never moves: the
           * spinner is already in the middle of the box the picture will fill.
           */}
          <div className="flex min-h-0 min-w-0 flex-1 self-stretch items-center justify-center overflow-hidden">
            {/* Keyed on the path so switching files remounts rather than
                showing the previous file's bytes under the new one's name. */}
            <FileView key={path} path={path} size="full" />
          </div>
          {many && (
            <TooltipIconButton
              tooltip="Next file"
              side="left"
              className="bg-background/40! text-foreground/60! hover:bg-background/40! hover:text-foreground/60! active:bg-background/55! absolute end-1 z-10 size-8 rounded-full shadow-sm backdrop-blur-sm sm:static sm:bg-transparent! sm:text-foreground! sm:hover:bg-accent! sm:hover:text-accent-foreground! sm:shadow-none sm:backdrop-blur-none"
              onClick={() => stepView(1)}
            >
              <ChevronRightIcon className="size-4" />
            </TooltipIconButton>
          )}
        </div>

        <div className="text-muted-foreground flex min-w-0 items-center justify-between gap-3 text-xs">
          {/* The footer is where a control belongs that would otherwise want a
              band of its own above the file, costing a strip of height on every
              note whether anybody touches it or not. `guessKind` rather than
              `kindOf` because Markdown is settled by extension in both — it is
              the one kind whose immediate answer and asked answer cannot
              differ, so there is nothing to wait for. */}
          <div className="flex min-w-0 items-center gap-3">
            {viewing.source !== "explorer" && <TooltipIconButton tooltip="Open containing folder" side="top" className="size-8" onClick={() => {
              revealing.current = true;
              closeView();
              openExplorer(path);
            }}><FolderOpenIcon className="size-4" /></TooltipIconButton>}
            {guessKind(path) === "markdown" && <MarkdownModePicker />}
            {many && (
              <span className="shrink-0">
                {viewing.index + 1} of {viewing.paths.length}
              </span>
            )}
          </div>
          <a
            href={fileUrl(path)}
            download={nameOf(path)}
            className="hover:text-foreground inline-flex shrink-0 items-center gap-1.5"
          >
            <DownloadIcon className="size-3.5" aria-hidden />
            Download
          </a>
        </div>
      </DialogContent>
    </Dialog>
  );
};
