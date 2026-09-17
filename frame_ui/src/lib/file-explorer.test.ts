import { describe, expect, it, vi } from "vitest";
vi.mock("@/lib/client", () => ({ sdk: vi.fn() }));
import { hostBreadcrumbs, isAbsoluteHostPath, mentionPath, parentHostPath, visibleEntries } from "@/lib/file-explorer";

describe("host paths", () => {
  it.each([
    ["/srv/data/notes", ["/", "/srv", "/srv/data", "/srv/data/notes"], ["/", "srv", "data", "notes"]],
    ["C:\\Users\\Henry", ["C:\\", "C:\\Users", "C:\\Users\\Henry"], ["C:\\", "Users", "Henry"]],
    ["\\\\host\\share\\notes", ["\\\\host\\share\\", "\\\\host\\share\\notes"], ["\\\\host\\share\\", "notes"]],
    ["/", ["/"], ["/"]],
    ["", [], []],
  ])("builds breadcrumbs for %s", (path, paths, labels) => {
    expect(hostBreadcrumbs(path).map((crumb) => crumb.path)).toEqual(paths);
    expect(hostBreadcrumbs(path).map((crumb) => crumb.label)).toEqual(labels);
  });
  it.each([
    ["C:\\", "C:\\"], ["C:\\Users\\Henry\\", "C:\\Users"],
    ["C:\\Users", "C:\\"], ["D:/work/notes", "D:/work"],
    ["D:/work", "D:/"], ["/", "/"], ["/srv/data/", "/srv"],
    ["/srv", "/"], ["\\\\host\\share\\folder", "\\\\host\\share\\"],
    ["\\\\host\\share\\", "\\\\host\\share\\"],
    ["//host/share/folder", "//host/share/"],
  ])("finds parent of %s", (path, parent) => {
    expect(isAbsoluteHostPath(path)).toBe(true);
    expect(parentHostPath(path)).toBe(parent);
  });
  it.each(["notes", "C:notes", "\\notes", "", "~", "\\\\host"])("rejects relative path %s", (path) => {
    expect(isAbsoluteHostPath(path)).toBe(false);
  });
});

it("filters names and puts folders first without modifying the listing", () => {
  const entries = [
    { path: "/a10", name: "a10", is_dir: false, size: 0, mtime: 0 },
    { path: "/a2", name: "a2", is_dir: false, size: 0, mtime: 0 },
    { path: "/za", name: "ZA", is_dir: true, size: 0, mtime: 0 },
    { path: "/b", name: "b", is_dir: false, size: 0, mtime: 0 },
  ];
  expect(visibleEntries(entries, "A").map((entry) => entry.name)).toEqual(["ZA", "a2", "a10"]);
  expect(entries[0].name).toBe("a10");
});

it("appends quoted paths on their own line without changing the draft", () => {
  expect(mentionPath("", "C:\\My files\\a.txt")).toBe('"C:\\My files\\a.txt"\n');
  expect(mentionPath("Please read", "/a")).toBe('Please read\n"/a"\n');
  expect(mentionPath("Please read\n", "/a")).toBe('Please read\n"/a"\n');
});
