import { lazyWithPreload } from "@/lib/lazy";

export const [LazyFilesDrawer, preloadFilesDrawer] = lazyWithPreload(
  () => import("@/components/files-drawer"),
  (module) => module.FilesDrawer,
);
