/** One combined outcome group inside each reply, above its footer. */
import { useState } from "react";
import { FilesIcon, FileIcon, Maximize2Icon } from "lucide-react";
import { useAuiState } from "@assistant-ui/react";
import { preloadFileViewer } from "@/components/lazy-file-viewer";
import { Button } from "@/components/ui/button";
import { fileUrl } from "@/lib/client";
import { guessKind, nameOf } from "@/lib/files";
import { cn } from "@/lib/utils";
import { countOf } from "@/runtime/file-activity";
import { useFileActivity } from "@/runtime/file-activity-provider";
import { AGENT_FILES, PRESENTATION } from "@/runtime/convert";

type AgentFilesData = { paths?: unknown; id?: string };
/** Live frame ownership and attributed tool edits share the same reply and count. */
export function TurnOutcomeFiles() {
  const id = useAuiState((s) => s.message.id);
  const running = useAuiState((s) => s.message.status?.type === "running");
  const continues = useAuiState((s) =>
    (s.message.metadata.custom[PRESENTATION] as { continues?: boolean } | undefined)?.continues);
  const parts = useAuiState((s) => s.message.parts);
  const shared = parts.flatMap((part) =>
    part.type === "data" && part.name === AGENT_FILES
      ? ((part.data as AgentFilesData).paths as string[] ?? []) : []);
  const { sectionFor, recoveredFor } = useFileActivity();
  const section = sectionFor(id);
  const paths = [...new Set([
    ...shared, ...recoveredFor(id),
    ...(section ? [...section.shown, ...section.touched].map((entry) => entry.path) : []),
  ])].sort((a, b) => a.localeCompare(b));
  // Both sources remain drawer-only until this reply is complete.
  return running || continues ? null : <AttachmentGroup paths={paths} />;
}

export function AttachmentGroup({ paths }: { paths: string[] }) {
  const [expanded, setExpanded] = useState(false);
  const { fileFor, view } = useFileActivity();
  const unique = [...new Set(paths)];
  if (!unique.length) return null;
  const visible = expanded ? unique : unique.slice(0, 4);
  const multiple = unique.length > 1;
  return (
    <div data-slot="attachment-group" className="my-3 min-w-0">
      <div className={cn("grid min-w-0 gap-3", multiple && "sm:grid-cols-2")}>
        {visible.map((path) => {
          const entry = fileFor(path);
          return <AttachmentTile key={path} path={path} removed={entry?.gone ?? false}
            version={entry?.ts} compact={multiple}
            onOpen={() => view(unique, unique.indexOf(path))} />;
        })}
      </div>
      {unique.length > 4 && (
        <Button variant="ghost" size="sm" className="mt-2 text-muted-foreground"
          aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
          {expanded ? "Show fewer" : `Show ${unique.length - 4} more`}
        </Button>
      )}
    </div>
  );
}

function AttachmentTile({ path, removed, version, compact, onOpen }: {
  path: string; removed: boolean; version?: number; compact: boolean; onOpen: () => void;
}) {
  const image = guessKind(path) === "image";
  const [failed, setFailed] = useState<string | null>(null);
  const src = fileUrl(path) + (version === undefined ? "" : `&v=${encodeURIComponent(version)}`);
  const unavailable = failed === src;
  if (removed) return (
    <div data-slot="removed-file" className="text-muted-foreground flex items-center gap-2 rounded-lg border px-3 py-2 text-xs">
      <FileIcon className="size-4 shrink-0" aria-hidden />
      <span className="min-w-0 break-all">{nameOf(path)}<span className="block">File removed</span></span>
    </div>
  );
  return (
    <button type="button" data-slot="attachment-tile" data-path={path}
      onClick={onOpen} onPointerEnter={preloadFileViewer} onFocus={preloadFileViewer}
      aria-label={`Open ${nameOf(path)}`}
      className={cn("group min-w-0 rounded-lg text-start outline-none focus-visible:ring-2 focus-visible:ring-ring",
        image ? "w-full" : "hover:bg-accent flex items-center gap-3 border p-3 self-start")}>
      {image && !unavailable ? (
        <span key={src} className={cn("bg-muted/20 relative isolate block overflow-hidden rounded-lg border",
          compact ? "h-52 w-full" : "w-fit max-w-full")}>
          <span aria-hidden="true" className="sb-image-ambient" style={{ backgroundImage: `url(${JSON.stringify(src)})` }} />
          <img src={src} alt={nameOf(path)} onError={() => setFailed(src)}
            className={cn("relative block object-contain",
              compact ? "h-full w-full" : "max-h-96 max-w-full")} />
        </span>
      ) : <FileIcon className="text-muted-foreground size-5 shrink-0" aria-hidden />}
      <span className={cn("text-muted-foreground flex min-w-0 items-center gap-1.5 text-xs",
        image && "mt-1.5 px-1")}>
        {image && <Maximize2Icon className="size-3 shrink-0" aria-hidden />}
        <span className="min-w-0 break-all">{nameOf(path)}
          {unavailable && <span className="block text-[11px]">Preview unavailable · Open file</span>}
        </span>
      </span>
    </button>
  );
}

export function TurnFilesButton() {
  const id = useAuiState((s) => s.message.id);
  const { sectionFor, openFilesAt } = useFileActivity();
  const section = sectionFor(id);
  const count = section ? countOf(section) : 0;
  if (!count) return null;
  return (
    <Button variant="ghost" size="xs" onClick={() => openFilesAt(id)}
      className="text-muted-foreground gap-1.5">
      <FilesIcon aria-hidden />{count} {count === 1 ? "file" : "files"}
    </Button>
  );
}
