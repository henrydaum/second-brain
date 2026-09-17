import { lazyWithPreload } from "@/lib/lazy";

export const [LazyFileViewerDialog, preloadFileViewer] = lazyWithPreload(
  () => import("@/components/file-viewer-dialog"),
  (module) => module.FileViewerDialog,
);
