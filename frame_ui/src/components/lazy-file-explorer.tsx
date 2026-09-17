import { lazyWithPreload } from "@/lib/lazy";

export const [LazyFileExplorerDialog, preloadFileExplorer] = lazyWithPreload(
  () => import("@/components/file-explorer-dialog"),
  (module) => module.FileExplorerDialog,
);
