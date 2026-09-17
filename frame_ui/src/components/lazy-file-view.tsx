import { lazyWithPreload } from "@/lib/lazy";

export const [LazyFileView, preloadFileView] = lazyWithPreload(
  () => import("@/components/file-view"),
  (module) => module.FileView,
);
