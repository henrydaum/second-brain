/** Files a user message carried, above the prose rather than inside it. */

import type { FC } from "react";
import { useAuiState } from "@assistant-ui/react";

import { FileThumbnail } from "@/components/file-kind-icon";
import { preloadFileViewer } from "@/components/lazy-file-viewer";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { SENT_ATTACHMENTS } from "@/runtime/convert";
import { useFileActivity } from "@/runtime/file-activity-provider";
import type { MessageAttachment } from "@/runtime/store";

function messageAttachments(value: unknown): MessageAttachment[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (item): item is MessageAttachment =>
      item !== null &&
      typeof item === "object" &&
      typeof (item as MessageAttachment).fileName === "string",
  );
}

export const UserMessageAttachments: FC = () => {
  const { view } = useFileActivity();
  const stored = useAuiState(
    (state) => state.message.metadata.custom[SENT_ATTACHMENTS],
  );
  const attachments = messageAttachments(stored);
  if (!attachments.length) return null;

  const paths = attachments.flatMap((attachment) =>
    attachment.path ? [attachment.path] : [],
  );

  return (
    <div className="col-span-full col-start-1 row-start-1 flex max-w-full flex-wrap justify-end gap-2">
      {attachments.map((attachment, position) => {
        const path = attachment.path;
        const tile = <AttachmentTile attachment={attachment} />;
        if (!path) {
          return (
            <Tooltip key={`${attachment.fileName}-${position}`}>
              <TooltipTrigger asChild>
                {/* The name is the only accessible one this has: there is no
                    button here to carry an `aria-label`, since a file with no
                    stored path yet has nothing to open. */}
                <div>
                  {tile}
                  <span className="sr-only">{attachment.fileName}</span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="top">{attachment.fileName}</TooltipContent>
            </Tooltip>
          );
        }
        const index = paths.indexOf(path);
        return (
          <Tooltip key={`${path}-${position}`}>
            <TooltipTrigger asChild>
              <button
                type="button"
                aria-label={`View ${attachment.fileName}`}
                onClick={() => view(paths, index)}
                onPointerEnter={preloadFileViewer}
                onFocus={preloadFileViewer}
                className="focus-visible:ring-ring rounded-md outline-none transition-opacity hover:opacity-80 focus-visible:ring-2"
              >
                {tile}
              </button>
            </TooltipTrigger>
            <TooltipContent side="top">{attachment.fileName}</TooltipContent>
          </Tooltip>
        );
      })}
    </div>
  );
};

const AttachmentTile: FC<{ attachment: MessageAttachment }> = ({
  attachment,
}) => (
  <FileThumbnail
    name={attachment.fileName}
    path={attachment.path}
    // The stored record says what the file is. That beats reading the
    // extension, which is all `FileThumbnail` can do on its own.
    image={attachment.modality === "image"}
    className="size-14 rounded-md shadow-sm"
    iconClassName="size-5"
  />
);
