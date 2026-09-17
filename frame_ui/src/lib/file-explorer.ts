import { sdk } from "@/lib/client";

export type DirectoryEntry = {
  path: string;
  name: string;
  is_dir: boolean;
  size: number | null;
  mtime: number | null;
};

/** Host paths, independent of the operating system running this browser. */
export function isAbsoluteHostPath(path: string): boolean {
  return path.startsWith("/") || /^[a-z]:[\\/]/i.test(path) || /^\\\\[^\\]+\\[^\\]+/.test(path);
}

export function parentHostPath(path: string): string {
  const windows = /^[a-z]:[\\/]/i.test(path) || path.startsWith("\\\\");
  const separator = windows && path.includes("\\") ? "\\" : "/";
  const normalized = windows ? path.replace(/[\\/]/g, separator) : path;
  const root = windows
    ? normalized.match(/^[a-z]:[\\/]|^[\\/]{2}[^\\/]+[\\/][^\\/]+[\\/]?/i)?.[0] ?? separator
    : normalized.startsWith("//")
      ? normalized.match(/^\/\/[^/]+\/[^/]+\/?/)?.[0] ?? "/"
      : "/";
  const trimmed = normalized.replace(windows ? /[\\/]+$/ : /\/+$/, "");
  if (trimmed.length <= root.replace(/[\\/]+$/, "").length) return root;
  const index = trimmed.lastIndexOf(separator);
  return index < root.length ? root : trimmed.slice(0, index);
}

export function visibleEntries(entries: DirectoryEntry[], filter: string): DirectoryEntry[] {
  const query = filter.toLocaleLowerCase();
  return entries.filter((entry) => entry.name.toLocaleLowerCase().includes(query))
    .sort((a, b) => Number(b.is_dir) - Number(a.is_dir) || a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));
}

/** Each breadcrumb keeps its full host path; drive and share roots stay intact. */
export function hostBreadcrumbs(path: string): { path: string; label: string }[] {
  if (!path) return [];
  const crumbs: { path: string; label: string }[] = [];
  let current = path;
  while (true) {
    const parent = parentHostPath(current);
    const trimmed = current.replace(/[\\/]+$/, "");
    const atRoot = parent.replace(/[\\/]+$/, "") === trimmed;
    crumbs.unshift({ path: current, label: atRoot ? current : trimmed.slice(trimmed.search(/[^\\/]+$/)) });
    if (atRoot) return crumbs;
    current = parent;
  }
}

export function mentionPath(draft: string, path: string): string {
  return `${draft}${draft && !draft.endsWith("\n") ? "\n" : ""}"${path}"\n`;
}

export async function readDirectory(path: string, validate = false): Promise<{ path: string; entries: DirectoryEntry[] }> {
  if (!isAbsoluteHostPath(path)) throw new Error("Enter an absolute path on the Second Brain host.");
  if (validate) {
    const stat = await sdk<{ path: string; is_dir: boolean }>("fs.stat", { path });
    if (!stat.is_dir) throw new Error("This path is a file. Enter a folder path.");
    path = stat.path;
  }
  return { path, entries: await sdk<DirectoryEntry[]>("fs.list", { path, details: true }) };
}
