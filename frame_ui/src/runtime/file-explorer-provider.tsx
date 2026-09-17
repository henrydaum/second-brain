import { createContext, use, useCallback, useMemo, useState, Suspense, type PropsWithChildren } from "react";
import { LazyFileExplorerDialog, preloadFileExplorer } from "@/components/lazy-file-explorer";

export type ExplorerTarget = { path: string; request: number };
const ExplorerContext = createContext<{ openExplorer: (path?: string) => void } | null>(null);

export function useFileExplorer() {
  const context = use(ExplorerContext);
  if (!context) throw new Error("useFileExplorer outside FileExplorerProvider");
  return context;
}

/** One explorer shared by navigation and host-file surfaces. */
export function FileExplorerProvider({ children }: PropsWithChildren) {
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [target, setTarget] = useState<ExplorerTarget>();
  const openExplorer = useCallback((path?: string) => {
    preloadFileExplorer();
    setTarget((previous) => path ? { path, request: (previous?.request ?? 0) + 1 } : undefined);
    setMounted(true);
    setOpen(true);
  }, []);
  const value = useMemo(() => ({ openExplorer }), [openExplorer]);
  return <ExplorerContext value={value}>
    {children}
    {mounted && <Suspense fallback={null}>
      <LazyFileExplorerDialog open={open} onOpenChange={setOpen} target={target} />
    </Suspense>}
  </ExplorerContext>;
}
