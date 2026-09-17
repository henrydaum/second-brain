import { useEffect, useRef, useState } from "react";
import { ChevronRightIcon } from "lucide-react";
import { hostBreadcrumbs } from "@/lib/file-explorer";

/** Breadcrumb links and a path editor share the same address field. */
export function ExplorerAddressBar({ directory, busy, onNavigate }: {
  directory: string;
  busy: boolean;
  onNavigate: (path: string) => Promise<boolean>;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(directory);
  const input = useRef<HTMLInputElement>(null);
  const editButton = useRef<HTMLButtonElement>(null);
  const wasEditing = useRef(false);
  const breadcrumbs = useRef<HTMLElement>(null);

  useEffect(() => {
    if (editing) {
      input.current?.focus();
      if (window.matchMedia?.("(pointer: coarse)").matches) {
        const end = input.current?.value.length ?? 0;
        input.current?.setSelectionRange(end, end);
        if (input.current) input.current.scrollLeft = input.current.scrollWidth;
      } else input.current?.select();
    }
    else {
      if (breadcrumbs.current) breadcrumbs.current.scrollLeft = breadcrumbs.current.scrollWidth;
      if (wasEditing.current) editButton.current?.focus();
    }
    wasEditing.current = editing;
  }, [editing, directory]);

  const edit = () => { setDraft(directory); setEditing(true); };
  return editing ? (
    <form className="min-w-0" onSubmit={(event) => {
      event.preventDefault();
      void onNavigate(draft.trim()).then((success) => { if (success) setEditing(false); });
    }}>
      <input ref={input} data-explorer-address-input aria-label="Host folder path"
        className="bg-background border-ring h-9 w-full min-w-0 rounded-md border px-3 text-base sm:text-sm outline-none ring-1 ring-ring"
        value={draft} onChange={(event) => setDraft(event.target.value)}
        onBlur={() => { if (!busy) setEditing(false); }}
        onKeyDown={(event) => {
          if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); setEditing(false); }
        }} />
    </form>
  ) : (
    <div className="bg-muted/40 flex h-9 min-w-0 items-center rounded-md border px-1">
      <nav ref={breadcrumbs} aria-label="Folder breadcrumbs" className="min-w-0 overflow-x-auto [scrollbar-width:none]">
        <ol className="flex w-max items-center text-base sm:text-sm">
          {hostBreadcrumbs(directory).map((crumb, index, crumbs) => <li key={crumb.path} className="flex shrink-0 items-center">
            {index > 0 && <ChevronRightIcon aria-hidden className="text-muted-foreground size-3.5" />}
            <button type="button" title={crumb.path} aria-current={index === crumbs.length - 1 ? "location" : undefined}
              disabled={busy}
              className="text-muted-foreground hover:bg-accent hover:text-foreground rounded px-2 py-1.5 outline-none focus-visible:ring-2 focus-visible:ring-ring aria-current:text-foreground"
              onClick={() => { if (index !== crumbs.length - 1) void onNavigate(crumb.path); }}>{crumb.label}</button>
          </li>)}
        </ol>
      </nav>
      <button ref={editButton} type="button" aria-label="Edit folder path" title="Edit folder path"
        className="h-full min-w-8 flex-1 rounded outline-none focus-visible:ring-2 focus-visible:ring-ring" onClick={edit} />
    </div>
  );
}
