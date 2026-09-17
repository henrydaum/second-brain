import { useEffect, useRef, useState } from "react";
import { CheckIcon, CopyIcon, FolderOpenIcon, MessageSquareIcon, MoreHorizontalIcon, CircleAlertIcon } from "lucide-react";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { nameOf } from "@/lib/files";
import { cn } from "@/lib/utils";

export function FileActionsMenu({ path, disabled, onMention, onReveal, className }: {
  path: string;
  disabled?: boolean;
  onMention?: () => void;
  onReveal?: () => void;
  className?: string;
}) {
  const leaving = useRef(false);
  const [copyState, setCopyState] = useState<"copied" | "failed" | null>(null);
  useEffect(() => {
    if (!copyState) return;
    const timer = window.setTimeout(() => setCopyState(null), 2500);
    return () => window.clearTimeout(timer);
  }, [copyState]);
  const feedback = copyState === "copied" ? "Path copied" : copyState === "failed" ? "Could not copy path. Clipboard access is unavailable." : "";
  return <DropdownMenu onOpenChange={(open) => { if (open) leaving.current = false; }}>
    <DropdownMenuTrigger asChild>
      <TooltipIconButton tooltip={feedback || "File actions"} aria-label={`Actions for ${nameOf(path)}`} disabled={disabled} className={cn("size-8", className)}>
        {copyState === "copied" ? <CheckIcon className="size-4" /> : copyState === "failed" ? <CircleAlertIcon className="text-destructive size-4" /> : <MoreHorizontalIcon className="size-4" />}
        <span role="status" className="sr-only">{feedback}</span>
      </TooltipIconButton>
    </DropdownMenuTrigger>
    <DropdownMenuContent align="end" onCloseAutoFocus={(event) => {
      if (leaving.current) event.preventDefault();
    }}>
      {onMention && <DropdownMenuItem onSelect={() => { leaving.current = true; onMention(); }}><MessageSquareIcon className="size-4" />Mention in chat</DropdownMenuItem>}
      <DropdownMenuItem onSelect={() => {
        setCopyState(null);
        void (async () => {
          try { await navigator.clipboard.writeText(path); setCopyState("copied"); }
          catch { setCopyState("failed"); }
        })();
      }}><CopyIcon className="size-4" />Copy path</DropdownMenuItem>
      {onReveal && <DropdownMenuItem onSelect={() => { leaving.current = true; onReveal(); }}><FolderOpenIcon className="size-4" />Open containing folder</DropdownMenuItem>}
    </DropdownMenuContent>
  </DropdownMenu>;
}
