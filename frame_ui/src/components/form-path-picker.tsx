import { Suspense, useEffect, useRef, useState } from "react";
import { FolderOpenIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LazyFileExplorerDialog, preloadFileExplorer } from "@/components/lazy-file-explorer";
import { insertFormPath } from "@/lib/form-paths";

export function FormPathPicker({ inputId, value, mode, disabled, identity, onChange }: {
  inputId: string;
  value: string;
  mode: string;
  disabled: boolean;
  identity: object;
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const selection = useRef({ start: 0, end: 0 });
  const openedFor = useRef(identity);
  const current = useRef(identity);
  current.current = identity;
  useEffect(() => { setOpen(false); }, [identity]);
  const field = () => document.getElementById(inputId) as HTMLInputElement | HTMLTextAreaElement | null;

  return <>
    <Button type="button" variant="outline" size="sm" className="mt-2" disabled={disabled}
      onPointerEnter={preloadFileExplorer} onFocus={preloadFileExplorer} onClick={() => {
        const input = field();
        selection.current = { start: input?.selectionStart ?? value.length, end: input?.selectionEnd ?? value.length };
        openedFor.current = identity;
        setMounted(true);
        setOpen(true);
      }}><FolderOpenIcon className="size-4" />Choose path</Button>
    {mounted && <Suspense fallback={null}>
      <LazyFileExplorerDialog open={open && openedFor.current === identity} onOpenChange={setOpen}
        onReturnFocus={() => {
          if (openedFor.current !== current.current) return;
          const input = field();
          input?.focus();
          if (input) input.setSelectionRange(selection.current.start, selection.current.end);
        }}
        onPick={(path) => {
          if (disabled || openedFor.current !== current.current) return;
          const { start, end } = selection.current;
          const next = insertFormPath(value, path, mode, start, end);
          const caret = mode === "json" ? next.length : start + path.length;
          selection.current = { start: caret, end: caret };
          onChange(next);
        }} />
    </Suspense>}
  </>;
}
