import { type PropsWithChildren, useEffect, useState, type FC } from "react";
import { AlertCircleIcon, MicIcon, PlusIcon, XIcon } from "lucide-react";
import { Suspense } from "react";
import {
  AttachmentPrimitive,
  ComposerPrimitive,
  useAuiState,
  useAui,
} from "@assistant-ui/react";
import { useShallow } from "zustand/shallow";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  LazyFileView,
  preloadFileView,
} from "@/components/lazy-file-view";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { FileKindIcon } from "@/components/file-kind-icon";
import { cn } from "@/lib/utils";
import { useStagedPath } from "@/runtime/staged-attachments";

const useFileSrc = (file: File | undefined) => {
  const [src, setSrc] = useState<string | undefined>(undefined);

  useEffect(() => {
    if (!file) {
      setSrc(undefined);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setSrc(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  return src;
};

const useAttachmentSrc = () => {
  const { file, src } = useAuiState(
    useShallow((s): { file?: File; src?: string } => {
      if (s.attachment.type !== "image") return {};
      if (s.attachment.file) return { file: s.attachment.file };
      const src = s.attachment.content?.filter((c) => c.type === "image")[0]
        ?.image;
      if (!src) return {};
      return { src };
    }),
  );

  return useFileSrc(file) ?? src;
};

type AttachmentPreviewProps = {
  src: string;
};

const AttachmentPreview: FC<AttachmentPreviewProps> = ({ src }) => {
  const [isLoaded, setIsLoaded] = useState(false);
  return (
    <img
      src={src}
      alt="Attachment preview"
      className={cn(
        "block h-auto max-h-[80vh] w-auto max-w-full object-contain",
        isLoaded
          ? "aui-attachment-preview-image-loaded"
          : "aui-attachment-preview-image-loading invisible",
      )}
      onLoad={() => setIsLoaded(true)}
    />
  );
};

const AttachmentPreviewDialog: FC<
  PropsWithChildren<{ hostPath?: string }>
> = ({ children, hostPath }) => {
  const src = useAttachmentSrc();

  if (!src && !hostPath) return children;

  return (
    <Dialog>
      <DialogTrigger
        className="aui-attachment-preview-trigger cursor-pointer transition-colors sm:hover:bg-accent/50"
        asChild
      >
        {children}
      </DialogTrigger>
      <DialogContent
        className={cn(
          "aui-attachment-preview-dialog-content [&>button]:bg-foreground/60 [&>button_svg]:text-background [&>button:hover_svg]:text-destructive p-2 sm:max-w-3xl [&>button]:rounded-full [&>button]:p-1 [&>button]:opacity-100 [&>button]:ring-0!",
          hostPath &&
            "h-[min(88dvh,48rem)] w-[calc(100vw-1rem)] max-w-4xl grid-rows-[minmax(0,1fr)]",
        )}
      >
        <DialogTitle className="aui-sr-only sr-only">
          Attachment preview
        </DialogTitle>
        <div
          className={cn(
            "aui-attachment-preview bg-background relative mx-auto flex w-full items-center justify-center overflow-hidden",
            hostPath ? "min-h-0 flex-1 self-stretch" : "max-h-[80dvh]",
          )}
        >
          {hostPath ? (
            <Suspense
              fallback={
                <span className="text-muted-foreground text-xs">
                  Loading preview…
                </span>
              }
            >
              <LazyFileView path={hostPath} size="full" />
            </Suspense>
          ) : (
            <AttachmentPreview src={src!} />
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

const AttachmentThumb: FC<{ compact: boolean }> = ({ compact }) => {
  const src = useAttachmentSrc();
  // A voice note has no thumbnail and never will, so it gets its own icon
  // rather than looking like a document that failed to render.
  const isAudio = useAuiState((s) =>
    Boolean(s.attachment.contentType?.startsWith("audio/")),
  );
  const fileName = useAuiState((s) => s.attachment.name);

  return (
    <Avatar
      className={cn(
        "aui-attachment-tile-avatar",
        compact && src
          ? "absolute inset-0 h-full w-full rounded-none sm:static"
          : compact
            ? "size-6 shrink-0 rounded-none bg-transparent sm:h-full sm:w-full"
            : "h-full w-full rounded-none",
      )}
    >
      <AvatarImage
        src={src}
        alt="Attachment preview"
        className="aui-attachment-tile-image object-cover"
      />
      <AvatarFallback>
        {isAudio ? (
          <MicIcon
            className={cn(
              "aui-attachment-tile-fallback-icon text-muted-foreground",
              compact ? "size-5 sm:size-8" : "size-8",
            )}
          />
        ) : (
          <FileKindIcon
            path={fileName}
            className={cn(
              "aui-attachment-tile-fallback-icon text-muted-foreground",
              compact ? "size-5 sm:size-8" : "size-8",
            )}
          />
        )}
      </AvatarFallback>
    </Avatar>
  );
};

/**
 * What the tile is doing, drawn over the top of it.
 *
 * An attachment is not instant here — the bytes go to the host a window at a
 * time — and it can fail, because a write is a Request like any other. Without
 * this the tile looks identical whether the file is on its way, arrived, or
 * refused, which is the difference between a slow app and a broken one.
 */
const AttachmentProgress: FC = () => {
  const status = useAuiState((s) => s.attachment.status);

  if (status.type === "running") {
    return (
      <div className="absolute inset-0 flex items-center justify-center bg-black/45 text-[10px] font-medium text-white tabular-nums">
        {Math.round((status.progress ?? 0) * 100)}%
      </div>
    );
  }

  if (status.type === "incomplete" && status.reason === "error") {
    return (
      <div className="border-destructive bg-destructive/15 absolute inset-0 flex items-center justify-center rounded-md border">
        <AlertCircleIcon className="text-destructive size-5" />
      </div>
    );
  }

  return null;
};

/** The tooltip carries the failure's actual wording. A red tile says something
 *  went wrong; only this says what. */
const AttachmentLabel: FC = () => {
  const status = useAuiState((s) => s.attachment.status);
  const failed =
    status.type === "incomplete" && status.reason === "error"
      ? (status.message ?? "Could not attach this file.")
      : null;

  return (
    <>
      <AttachmentPrimitive.Name />
      {failed && <span className="text-destructive block text-xs">{failed}</span>}
    </>
  );
};

const AttachmentUI: FC = () => {
  const aui = useAui();
  const isComposer = aui.attachment.source !== "message";

  const isImage = useAuiState((s) => s.attachment.type === "image");
  const attachmentId = useAuiState((s) => s.attachment.id);
  const uploadedPath = useStagedPath(attachmentId);
  const hostPath = isComposer ? uploadedPath : undefined;
  const typeLabel = useAuiState((s) => {
    const type = s.attachment.type;
    switch (type) {
      case "image":
        return "Image";
      case "document":
        return "Document";
      case "file":
        return "File";
      default:
        return type;
    }
  });

  return (
    <Tooltip>
      <AttachmentPrimitive.Root
        className={cn(
          "aui-attachment-root relative shrink-0",
          isComposer &&
            "flex h-12 w-24 items-stretch overflow-hidden rounded-full border bg-muted sm:block sm:h-auto sm:w-auto sm:overflow-visible sm:rounded-none sm:border-0 sm:bg-transparent",
          isComposer && isImage && "bg-transparent",
          isImage &&
            !isComposer &&
            "aui-attachment-root-message only:*:first:size-24",
        )}
      >
        <AttachmentPreviewDialog hostPath={hostPath}>
          <TooltipTrigger asChild>
            <button
              type="button"
              // `relative`, so the progress and error overlays have something
              // to sit inside.
              className={cn(
                "aui-attachment-tile bg-muted relative cursor-pointer overflow-hidden transition-opacity sm:hover:opacity-75",
                isComposer
                  ? cn(
                      "h-full w-full sm:size-14 sm:block sm:rounded-md sm:border sm:p-0",
                      isImage
                        ? "p-0"
                        : "flex items-center ps-3 pe-11 text-start",
                    )
                  : "size-14 rounded-md border",
              )}
              aria-label={
                isComposer
                  ? `View ${typeLabel.toLowerCase()} attachment`
                  : `${typeLabel} attachment`
              }
              onPointerEnter={hostPath ? preloadFileView : undefined}
              onFocus={hostPath ? preloadFileView : undefined}
            >
              <AttachmentThumb compact={isComposer} />
              <AttachmentProgress />
            </button>
          </TooltipTrigger>
        </AttachmentPreviewDialog>
        {isComposer && <AttachmentRemove />}
      </AttachmentPrimitive.Root>
      <TooltipContent side="top">
        <AttachmentLabel />
      </TooltipContent>
    </Tooltip>
  );
};

const AttachmentRemove: FC = () => {
  return (
    <AttachmentPrimitive.Remove asChild>
      <TooltipIconButton
        tooltip="Remove file"
        className="aui-attachment-tile-remove group text-muted-foreground absolute end-0.5 top-1/2 z-10 size-11 -translate-y-1/2 rounded-full bg-transparent shadow-none sm:end-1.5 sm:top-1.5 sm:size-3.5 sm:translate-y-0 sm:bg-white/60 sm:shadow-sm sm:hover:bg-white/90! sm:hover:[&_svg]:text-destructive"
        side="top"
      >
        <span className="aui-attachment-remove-surface bg-background/85 flex size-9 items-center justify-center rounded-full shadow-sm backdrop-blur-[2px] group-active:bg-background/95 sm:size-full sm:bg-transparent sm:shadow-none sm:backdrop-blur-none">
          <XIcon className="aui-attachment-remove-icon size-4 text-black/70 sm:size-3 dark:stroke-[2.5px]" />
        </span>
      </TooltipIconButton>
    </AttachmentPrimitive.Remove>
  );
};

export const ComposerAttachments: FC = () => {
  return (
    <div className="grid min-w-0 grid-rows-[0fr] opacity-0 transition-[grid-template-rows,opacity,margin] duration-200 ease-out has-[.aui-attachment-root]:mb-2 has-[.aui-attachment-root]:grid-rows-[1fr] has-[.aui-attachment-root]:opacity-100">
      <div className="min-h-0 overflow-hidden">
        <div className="aui-composer-attachments flex min-w-0 w-full flex-row items-center gap-2 overflow-x-auto px-2 pt-2 pb-1">
          <ComposerPrimitive.Attachments>
            {() => <AttachmentUI />}
          </ComposerPrimitive.Attachments>
        </div>
      </div>
    </div>
  );
};

export const ComposerAddAttachment: FC = () => {
  return (
    <ComposerPrimitive.AddAttachment asChild>
      <TooltipIconButton
        tooltip="Add Attachment"
        side="bottom"
        variant="ghost"
        size="icon"
        className="aui-composer-add-attachment hover:bg-muted-foreground/15 dark:border-muted-foreground/15 dark:hover:bg-muted-foreground/30 size-7 rounded-full p-1 text-xs font-semibold"
        aria-label="Add Attachment"
      >
        <PlusIcon className="aui-attachment-add-icon size-4.5 stroke-[1.5px]" />
      </TooltipIconButton>
    </ComposerPrimitive.AddAttachment>
  );
};
