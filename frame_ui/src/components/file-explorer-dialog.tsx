import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useAui } from "@assistant-ui/react";
import { ArrowLeftIcon, ArrowRightIcon, FolderIcon, FolderOpenIcon, RefreshCwIcon, StarIcon, SearchIcon } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { FileActionsMenu } from "@/components/file-actions-menu";
import type { ExplorerTarget } from "@/runtime/file-explorer-provider";
import { cn } from "@/lib/utils";
import { ExplorerAddressBar } from "@/components/explorer-address-bar";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { useSurfaceReveal } from "@/components/ui/use-surface-reveal";
import { FileKindIcon } from "@/components/file-kind-icon";
import { sdk } from "@/lib/client";
import { hostBreadcrumbs, mentionPath, parentHostPath, readDirectory, visibleEntries, type DirectoryEntry } from "@/lib/file-explorer";
import { forgetFile } from "@/lib/files";
import { forgetThumbnail } from "@/lib/thumbnails";
import { useFileActivity } from "@/runtime/file-activity-provider";

const inputClass = "bg-background min-w-0 rounded-md border px-3 py-2 text-base sm:text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring";
const messageOf = (error: unknown) => error instanceof Error ? error.message : "Could not load this folder.";

export function FileExplorerDialog({ open, onOpenChange, target, onPick, onReturnFocus }: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  target?: ExplorerTarget;
  onPick?: (path: string) => void;
  onReturnFocus?: () => void;
}) {
  const aui = useAui();
  const { view, viewing } = useFileActivity();
  const [directory, setDirectory] = useState("");
  const directoryRef = useRef("");
  const [history, setHistory] = useState<{ back: string[]; forward: string[] }>({ back: [], forward: [] });
  const [locations, setLocations] = useState<{ path: string; label: string }[]>([]);
  const [entries, setEntries] = useState<DirectoryEntry[]>([]);
  const [filter, setFilter] = useState("");
  const [revealed, setRevealed] = useState<string>();
  const [listBody, setListBody] = useState<HTMLDivElement | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const [locationFailure, setLocationFailure] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const directorySurface = useSurfaceReveal<HTMLUListElement>(directory);
  const request = useRef(0);
  const retry = useRef<() => void>(() => {});
  const mentionFocus = useRef(false);
  const previewButton = useRef<HTMLElement | null>(null);
  const wasViewing = useRef(false);
  const pendingTarget = useRef<ExplorerTarget | undefined>(target);
  const places = useRef(new Map<string, number>());
  const placeKey = JSON.stringify([directory, filter]);
  const revealPending = useRef(false);
  useEffect(() => { pendingTarget.current = target; }, [target]);

  const navigate = useCallback(async (path: string, validate = false, direction: "back" | "forward" | "new" = "new") => {
    const id = ++request.current;
    retry.current = () => void navigate(path, validate, direction);
    setBusy(true);
    setFailure(null);
    try {
      const result = await readDirectory(path, validate);
      if (id !== request.current) return false;
      const previous = directoryRef.current;
      if (previous && previous !== result.path) setHistory((held) => {
        if (direction === "back") return { back: held.back.slice(0, -1), forward: [...held.forward, previous] };
        if (direction === "forward") return { back: [...held.back, previous], forward: held.forward.slice(0, -1) };
        return { back: [...held.back, previous], forward: [] };
      });
      directoryRef.current = result.path;
      setDirectory(result.path);
      setEntries(result.entries);
      pendingTarget.current = undefined;
      return true;
    } catch (error) {
      if (id === request.current) setFailure(messageOf(error));
      return false;
    } finally {
      if (id === request.current) setBusy(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    const reveal = pendingTarget.current;
    if (reveal) { revealPending.current = true; setRevealed(reveal.path); setFilter(""); }
    const initialRequest = ++request.current;
    setBusy(true);
    setFailure(null);
    setLocationFailure(null);
    retry.current = () => setRefresh((value) => value + 1);
    // A failed shortcut read must not prevent browsing a known location.
    void Promise.allSettled([
      sdk<string>("paths.get", { name: "data" }),
      sdk<string[] | null>("config.read", { key: "fs_writable_dirs" }),
      sdk<string[] | null>("config.read", { key: "sync_directories" }),
      sdk<string>("paths.get", { name: "project" }),
    ]).then(([data, writable, synced, project]) => {
      if (!alive) return;
      const dataPath = data.status === "fulfilled" ? data.value : "";
      const folders = writable.status === "fulfilled" && Array.isArray(writable.value) ? writable.value : [];
      const syncFolders = synced.status === "fulfilled" && Array.isArray(synced.value) ? synced.value : [];
      const candidates = [
        { path: dataPath, label: "Data directory" },
        { path: project.status === "fulfilled" ? project.value : "", label: "Kernel root" },
        ...folders.map((path) => ({ path, label: "Writable" })),
        ...syncFolders.map((path) => ({ path, label: "Sync" })),
      ];
      const seen = new Set<string>();
      setLocations(candidates.filter(({ path }) => {
        if (typeof path !== "string" || !path || seen.has(path)) return false;
        seen.add(path);
        return true;
      }));
      if ([data, writable, synced, project].some((result) => result.status === "rejected")) setLocationFailure("Some folder shortcuts could not be loaded. Refresh to retry.");
      if (initialRequest !== request.current) return;
      const destination = reveal ? parentHostPath(reveal.path) : directoryRef.current || dataPath;
      if (destination) void navigate(destination);
      else {
        setBusy(false);
        setFailure("Could not discover the data folder. Retry or enter a host folder path.");
      }
    });
    return () => { alive = false; ++request.current; };
  }, [open, refresh, navigate, target]);

  useEffect(() => {
    if (wasViewing.current && !viewing && open) previewButton.current?.focus({ preventScroll: true });
    wasViewing.current = Boolean(viewing);
  }, [viewing, open]);

  useLayoutEffect(() => {
    if (open && !busy && listBody) listBody.scrollTop = places.current.get(placeKey) ?? 0;
  }, [open, busy, listBody, placeKey]);

  useEffect(() => {
    if (!open || busy || !revealed || !listBody || !revealPending.current) return;
    revealPending.current = false;
    const row = [...listBody.querySelectorAll<HTMLElement>("[data-explorer-path]")]
      .find((element) => element.dataset.explorerPath === revealed);
    row?.scrollIntoView?.({ block: "nearest" });
    row?.querySelector<HTMLButtonElement>("button")?.focus();
  }, [open, busy, revealed, entries, listBody]);

  const shown = useMemo(() => visibleEntries(entries, filter), [entries, filter]);
  const paths = shown.filter((entry) => !entry.is_dir).map((entry) => entry.path);
  const folderName = hostBreadcrumbs(directory).at(-1)?.label || "folder";
  const pick = (path: string) => {
    if (listBody) places.current.set(placeKey, listBody.scrollTop);
    onPick?.(path);
    ++request.current;
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={(next) => { if (!viewing) onOpenChange(next); }}>
      <DialogContent
        // Picker portals can be launched from a Settings form. Address-bar
        // submission must not bubble into that form and advance its step.
        onSubmit={(event) => event.stopPropagation()}
        className="sb-glass sb-glass-sheet flex h-[min(94dvh,54rem)] w-[min(calc(100vw-1rem),70rem)] max-w-none flex-col gap-0 overflow-hidden p-0 sm:max-w-none"
        overlayClassName="bg-black/45 backdrop-blur-[2px]"
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          (event.currentTarget as HTMLElement | null)?.focus();
        }}
        onEscapeKeyDown={(event) => { if (viewing || (event.target instanceof HTMLElement && event.target.hasAttribute("data-explorer-address-input"))) event.preventDefault(); }}
        onCloseAutoFocus={(event) => {
          if (onReturnFocus) { event.preventDefault(); onReturnFocus(); return; }
          if (!mentionFocus.current) return;
          event.preventDefault();
          mentionFocus.current = false;
          const composer = document.querySelector<HTMLTextAreaElement>('[data-slot="chat-composer-input"]');
          composer?.focus();
          if (composer) composer.setSelectionRange(composer.value.length, composer.value.length);
        }}
      >
        <header className="flex h-14 shrink-0 items-center gap-3 border-b ps-4 pe-14 sm:h-16 sm:ps-6">
          <span className="bg-primary text-primary-foreground flex size-8 items-center justify-center rounded-lg"><FolderOpenIcon className="size-4" /></span>
          <div className="min-w-0">
            <DialogTitle className="text-base">{onPick ? "Choose a path" : "File explorer"}</DialogTitle>
            <DialogDescription className="text-xs">{onPick ? "Select a file or folder on your Second Brain host" : "Files on your Second Brain host"}</DialogDescription>
          </div>
        </header>
        <div className="flex flex-col gap-2 border-b p-3 sm:p-4">
          <div data-slot="explorer-toolbar" className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-2 sm:grid-cols-[auto_minmax(0,1fr)_minmax(10rem,14rem)]">
            <div className="flex items-center gap-1">
              <TooltipIconButton tooltip="Back" disabled={busy || history.back.length === 0} onClick={() => void navigate(history.back.at(-1)!, false, "back")}><ArrowLeftIcon className="size-4" /></TooltipIconButton>
              <TooltipIconButton tooltip="Forward" className="hidden sm:inline-flex" disabled={busy || history.forward.length === 0} onClick={() => void navigate(history.forward.at(-1)!, false, "forward")}><ArrowRightIcon className="size-4" /></TooltipIconButton>
              <TooltipIconButton tooltip="Refresh folder" className="hidden sm:inline-flex" onClick={() => setRefresh((value) => value + 1)}><RefreshCwIcon className="size-4" /></TooltipIconButton>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <TooltipIconButton tooltip="Folder shortcuts"><StarIcon className="size-4" /></TooltipIconButton>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="max-h-80 max-w-[calc(100vw-2rem)] overflow-y-auto">
                  {locations.map(({ path, label }) => <DropdownMenuItem key={path} aria-label={`${label}: ${path}`} onSelect={() => void navigate(path, true)}>
                    <span className="min-w-0"><span className="block text-xs text-muted-foreground">{label}</span><span className="block truncate" title={path}>{path}</span></span>
                  </DropdownMenuItem>)}
                  {locations.length === 0 && <DropdownMenuItem disabled>No folder shortcuts available</DropdownMenuItem>}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            <div className="order-2 col-span-2 min-w-0 sm:order-none sm:col-span-1">
              <ExplorerAddressBar directory={directory} busy={busy} onNavigate={(path) => navigate(path, true)} />
            </div>
            <div className="relative order-1 min-w-0 sm:order-none">
              <SearchIcon aria-hidden className="text-muted-foreground pointer-events-none absolute start-3 top-2.5 size-4" />
              <input aria-label={`Search in ${folderName}`} placeholder={`Search in ${folderName}`} className={`${inputClass} h-9 w-full ps-9`} value={filter} onChange={(event) => setFilter(event.target.value)} />
            </div>
          </div>
          {locationFailure && <p role="status" className="text-muted-foreground text-xs">{locationFailure}</p>}
          {failure && <div role="alert" className="text-destructive flex items-center gap-2 text-sm"><span>{failure}</span><Button variant="outline" size="sm" onClick={() => retry.current()}>Retry</Button></div>}
        </div>
        <div ref={setListBody} data-slot="explorer-list" onScroll={(event) => {
          if (open && !busy) places.current.set(placeKey, event.currentTarget.scrollTop);
        }} className="bg-popover min-h-0 flex-1 overflow-y-auto p-2 sm:p-3" aria-busy={busy}>
          {!busy && revealed && directory === parentHostPath(revealed) && !entries.some((entry) => entry.path === revealed) && <p role="status" className="text-muted-foreground p-3 text-sm">The file is no longer in this folder.</p>}
          {busy && <p role="status" className="text-muted-foreground p-3 text-sm">Loading folder…</p>}
          {!busy && directory && shown.length === 0 && <p className="text-muted-foreground p-3 text-sm">{entries.length ? "No matching filenames." : "This folder is empty."}</p>}
          <ul ref={directorySurface} aria-label="Directory contents" className="sb-step-surface space-y-1" data-pending={busy || undefined}>
            {shown.map((entry) => <li key={entry.path} data-explorer-path={entry.path} className={cn("sb-row flex min-w-0 items-center gap-2 pe-2", entry.path === revealed && "ring-primary/50 ring-2 ring-inset")}>
              <button type="button" disabled={busy} className="sb-control flex min-w-0 flex-1 items-center gap-3 rounded-md px-3 py-3 text-start text-sm disabled:opacity-50" title={entry.path} onClick={(event) => {
                if (entry.is_dir) { void navigate(entry.path); return; }
                previewButton.current = event.currentTarget;
                for (const path of paths) { forgetFile(path); forgetThumbnail(path); }
                view(paths, paths.indexOf(entry.path), "explorer");
              }}>
                {entry.is_dir ? <FolderIcon aria-hidden className="text-primary fill-primary/10 size-5 shrink-0" /> : <FileKindIcon path={entry.path} className="text-muted-foreground size-5 shrink-0" />}
                <span className={cn("truncate", entry.is_dir && "font-medium")}>{entry.name}</span>
              </button>
              {onPick ? <Button variant="ghost" size="sm" disabled={busy} aria-label={`Select ${entry.name}`} onClick={() => pick(entry.path)}>Select</Button> : <FileActionsMenu path={entry.path} disabled={busy} onMention={() => {
                const composer = aui.composer();
                composer.setText(mentionPath(composer.getState().text, entry.path));
                mentionFocus.current = true;
                if (listBody) places.current.set(placeKey, listBody.scrollTop);
                ++request.current;
                onOpenChange(false);
              }} />}
            </li>)}
          </ul>
        </div>
        {onPick && <footer className="flex justify-end gap-2 border-t p-3">
          <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button disabled={busy || !directory} onClick={() => pick(directory)}>Use this folder</Button>
        </footer>}
      </DialogContent>
    </Dialog>
  );
}
